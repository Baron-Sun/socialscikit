"""Multi-LLM Consensus Coding — run multiple LLMs, majority vote, agreement report.

Run 2–3 different LLMs independently on the same text segments. For each
segment the final theme set is determined by majority vote: a theme is
included only if ≥ ceil(n_coders / 2) coders assigned it.

The merged results are standard ``CodingResult`` objects so they flow
through the existing ``ConfidenceRanker → CodingReviewer → Exporter``
pipeline unchanged.
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from dataclasses import dataclass, field

from socialscikit.core.llm_client import LLMClient
from socialscikit.qualikit.coder import Coder, CodingReport, CodingResult
from socialscikit.qualikit.theme_definer import Theme


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SegmentConsensus:
    """Consensus result for a single text segment."""

    text_id: int
    text: str
    consensus_themes: list[str] = field(default_factory=list)
    consensus_confidences: dict[str, float] = field(default_factory=dict)
    agreement_rate: float = 0.0
    individual_results: list[CodingResult] = field(default_factory=list)
    vote_counts: dict[str, int] = field(default_factory=dict)

    def to_coding_result(self) -> CodingResult:
        """Convert to a standard CodingResult for downstream compatibility."""
        # Merge trigger words from all coders who assigned consensus themes
        merged_triggers: dict[str, list[str]] = {}
        merged_reasoning_parts: list[str] = []

        for theme in self.consensus_themes:
            triggers_set: set[str] = set()
            for r in self.individual_results:
                if theme in r.themes:
                    triggers_set.update(r.trigger_words.get(theme, []))
            merged_triggers[theme] = list(triggers_set)

        # Merge reasoning
        for r in self.individual_results:
            if r.reasoning:
                merged_reasoning_parts.append(r.reasoning)

        return CodingResult(
            text_id=self.text_id,
            text=self.text,
            themes=self.consensus_themes,
            confidences=self.consensus_confidences,
            trigger_words=merged_triggers,
            reasoning=" | ".join(merged_reasoning_parts) if merged_reasoning_parts else "",
        )


@dataclass
class ConsensusReport:
    """Full report from a consensus coding run."""

    segments: list[SegmentConsensus] = field(default_factory=list)
    n_coders: int = 0
    coder_models: list[str] = field(default_factory=list)
    n_total: int = 0
    n_coded: int = 0
    n_failed: int = 0
    overall_agreement: float = 0.0
    theme_distribution: dict[str, int] = field(default_factory=dict)
    per_coder_cost: list[float] = field(default_factory=list)
    total_cost: float = 0.0

    def to_coding_report(self) -> CodingReport:
        """Convert to a standard CodingReport for downstream compatibility."""
        results = [seg.to_coding_result() for seg in self.segments]
        return CodingReport(
            results=results,
            n_total=self.n_total,
            n_coded=self.n_coded,
            n_failed=self.n_failed,
            theme_distribution=dict(self.theme_distribution),
        )


# ---------------------------------------------------------------------------
# Consensus Coder
# ---------------------------------------------------------------------------


class ConsensusCoder:
    """Multi-LLM consensus coding engine.

    Instantiates a ``Coder`` per ``LLMClient``, runs all coders on the same
    segments, then merges results via majority vote.

    Usage::

        clients = [
            LLMClient(backend="openai", model="gpt-4o"),
            LLMClient(backend="anthropic", model="claude-sonnet-4-20250514"),
        ]
        consensus = ConsensusCoder(clients)
        report = consensus.code(texts, themes)
    """

    def __init__(
        self,
        llm_clients: list[LLMClient],
        majority_threshold: int | None = None,
    ):
        """
        Parameters
        ----------
        llm_clients : list[LLMClient]
            At least 2 LLM clients for consensus coding.
        majority_threshold : int or None
            Minimum number of coders that must agree for a theme to be included.
            Defaults to ``ceil(len(llm_clients) / 2)``.
        """
        if len(llm_clients) < 2:
            raise ValueError("Consensus coding requires at least 2 LLM clients.")

        self.llm_clients = llm_clients
        self.coders = [Coder(client) for client in llm_clients]
        self.n_coders = len(llm_clients)
        self.majority_threshold = majority_threshold or math.ceil(self.n_coders / 2)

    def code(
        self,
        texts: list[str],
        themes: list[Theme],
    ) -> ConsensusReport:
        """Synchronous consensus coding — runs coders sequentially.

        Parameters
        ----------
        texts : list[str]
            Text segments to code.
        themes : list[Theme]
            Theme definitions.

        Returns
        -------
        ConsensusReport
        """
        all_reports: list[CodingReport] = []
        for coder in self.coders:
            report = coder.code(texts, themes)
            all_reports.append(report)

        return self._merge_results(all_reports, texts)

    async def code_async(
        self,
        texts: list[str],
        themes: list[Theme],
        batch_size: int = 50,
    ) -> ConsensusReport:
        """Async consensus coding — runs coders in parallel.

        Parameters
        ----------
        texts : list[str]
        themes : list[Theme]
        batch_size : int
            Batch size for each coder's async coding.

        Returns
        -------
        ConsensusReport
        """
        tasks = [
            coder.code_async(texts, themes, batch_size)
            for coder in self.coders
        ]
        all_reports = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_reports: list[CodingReport] = []
        for r in all_reports:
            if isinstance(r, CodingReport):
                valid_reports.append(r)

        if len(valid_reports) < 2:
            raise RuntimeError(
                f"Only {len(valid_reports)}/{self.n_coders} coders succeeded. "
                "Need at least 2 for consensus."
            )

        return self._merge_results(valid_reports, texts)

    # ------------------------------------------------------------------
    # Internal: merge results
    # ------------------------------------------------------------------

    def _merge_results(
        self,
        all_reports: list[CodingReport],
        texts: list[str],
    ) -> ConsensusReport:
        """Apply majority vote to merge results from multiple coders."""
        n_coders = len(all_reports)
        threshold = self.majority_threshold
        n_total = len(texts)

        segments: list[SegmentConsensus] = []
        theme_dist: dict[str, int] = Counter()
        n_failed = 0
        agreement_sum = 0.0

        for idx in range(n_total):
            # Gather individual results for this segment
            individual: list[CodingResult] = []
            for report in all_reports:
                if idx < len(report.results):
                    individual.append(report.results[idx])
                else:
                    individual.append(CodingResult(text_id=idx, text=texts[idx]))

            # Majority vote
            consensus_themes, avg_confs, votes = self._majority_vote(
                individual, threshold,
            )

            # Agreement rate: for each theme mentioned by any coder,
            # agreement = votes / n_coders. Average across all mentioned themes.
            if votes:
                all_mentioned_themes = set(votes.keys())
                rates = [votes[t] / n_coders for t in all_mentioned_themes]
                seg_agreement = sum(rates) / len(rates)
            else:
                # All coders agree: no themes
                seg_agreement = 1.0

            agreement_sum += seg_agreement

            for t in consensus_themes:
                theme_dist[t] = theme_dist.get(t, 0) + 1

            if not any(r.themes for r in individual):
                n_failed += 1

            segments.append(SegmentConsensus(
                text_id=idx,
                text=texts[idx],
                consensus_themes=consensus_themes,
                consensus_confidences=avg_confs,
                agreement_rate=round(seg_agreement, 4),
                individual_results=individual,
                vote_counts=votes,
            ))

        overall_agreement = agreement_sum / n_total if n_total > 0 else 0.0

        # Cost tracking
        coder_models = []
        per_coder_cost = []
        for i, client in enumerate(self.llm_clients[:n_coders]):
            coder_models.append(f"{client.backend}:{client.model}")
            # Sum cost from call log
            cost = sum(log.cost_usd for log in client.call_log) if client.call_log else 0.0
            per_coder_cost.append(cost)

        return ConsensusReport(
            segments=segments,
            n_coders=n_coders,
            coder_models=coder_models,
            n_total=n_total,
            n_coded=n_total - n_failed,
            n_failed=n_failed,
            overall_agreement=round(overall_agreement, 4),
            theme_distribution=dict(theme_dist),
            per_coder_cost=per_coder_cost,
            total_cost=sum(per_coder_cost),
        )

    @staticmethod
    def _majority_vote(
        individual_results: list[CodingResult],
        threshold: int,
    ) -> tuple[list[str], dict[str, float], dict[str, int]]:
        """Compute majority vote for a single segment.

        Returns
        -------
        consensus_themes : list[str]
            Themes meeting the majority threshold.
        avg_confidences : dict[str, float]
            Average confidence among coders who assigned each consensus theme.
        vote_counts : dict[str, int]
            Theme -> number of coders who assigned it (for ALL mentioned themes).
        """
        # Count votes for each theme
        vote_counts: dict[str, int] = Counter()
        confidence_sums: dict[str, float] = {}
        confidence_counts: dict[str, int] = {}

        for result in individual_results:
            for theme in result.themes:
                vote_counts[theme] += 1
                confidence_sums[theme] = confidence_sums.get(theme, 0.0) + result.confidences.get(theme, 0.5)
                confidence_counts[theme] = confidence_counts.get(theme, 0) + 1

        # Filter by threshold
        consensus_themes = [
            t for t, count in vote_counts.items() if count >= threshold
        ]
        consensus_themes.sort()

        # Average confidence for consensus themes
        avg_confidences = {}
        for t in consensus_themes:
            avg_confidences[t] = round(
                confidence_sums[t] / confidence_counts[t], 3
            )

        return consensus_themes, avg_confidences, dict(vote_counts)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: ConsensusReport, lang: str = "zh") -> str:
        """Format a ConsensusReport as a human-readable string."""
        lines: list[str] = []

        if lang == "zh":
            lines.append("═══ 多模型共识编码报告 ═══")
            lines.append("")
            lines.append(f"编码模型数量：{report.n_coders}")
            lines.append(f"模型列表：{', '.join(report.coder_models)}")
            lines.append(f"总文本数：{report.n_total}")
            lines.append(f"成功编码：{report.n_coded}")
            lines.append(f"总体一致性：{report.overall_agreement:.2%}")
            lines.append(f"总费用：${report.total_cost:.4f}")
        else:
            lines.append("═══ Multi-LLM Consensus Coding Report ═══")
            lines.append("")
            lines.append(f"Number of coders: {report.n_coders}")
            lines.append(f"Models: {', '.join(report.coder_models)}")
            lines.append(f"Total segments: {report.n_total}")
            lines.append(f"Successfully coded: {report.n_coded}")
            lines.append(f"Overall agreement: {report.overall_agreement:.2%}")
            lines.append(f"Total cost: ${report.total_cost:.4f}")

        lines.append("")

        if report.theme_distribution:
            dist_label = "主题分布" if lang == "zh" else "Theme Distribution"
            lines.append(f"── {dist_label} ──")
            for theme, count in sorted(
                report.theme_distribution.items(), key=lambda x: -x[1],
            ):
                lines.append(f"  {theme}: {count}")
            lines.append("")

        return "\n".join(lines)
