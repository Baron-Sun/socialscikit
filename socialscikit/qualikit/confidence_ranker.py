"""Confidence-based ranking and triage of coding results.

Splits coding results into three tiers for review workflow:
- High (>0.85): auto-accept, spot-check recommended
- Medium (0.6-0.85): human review required (yellow)
- Low (<0.6): must be manually judged (red)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socialscikit.qualikit.coder import CodingResult, HIGH_CONFIDENCE, MEDIUM_CONFIDENCE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RankedResults:
    """Coding results split by confidence tier."""

    high: list[CodingResult] = field(default_factory=list)
    medium: list[CodingResult] = field(default_factory=list)
    low: list[CodingResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.high) + len(self.medium) + len(self.low)


# ---------------------------------------------------------------------------
# ConfidenceRanker
# ---------------------------------------------------------------------------


class ConfidenceRanker:
    """Rank and triage coding results by confidence.

    Usage::

        ranker = ConfidenceRanker()
        ranked = ranker.rank(report.results)
        # ranked.high → auto-accept
        # ranked.medium → human review
        # ranked.low → mandatory review
    """

    def __init__(
        self,
        high_threshold: float = HIGH_CONFIDENCE,
        medium_threshold: float = MEDIUM_CONFIDENCE,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def rank(self, results: list[CodingResult]) -> RankedResults:
        """Split results into confidence tiers."""
        ranked = RankedResults()

        for result in results:
            if not result.confidences:
                ranked.low.append(result)
                continue

            min_conf = min(result.confidences.values())
            if min_conf >= self.high_threshold:
                ranked.high.append(result)
            elif min_conf >= self.medium_threshold:
                ranked.medium.append(result)
            else:
                ranked.low.append(result)

        return ranked

    def summary(self, ranked: RankedResults) -> str:
        """Format a summary of the ranking distribution."""
        lines = [
            "═══ 置信度分级 ═══",
            f"高置信度（>{self.high_threshold}）：{len(ranked.high)} 条 — 可自动接受，建议抽查",
            f"中置信度（{self.medium_threshold}-{self.high_threshold}）：{len(ranked.medium)} 条 — 需要人工复核",
            f"低置信度（<{self.medium_threshold}）：{len(ranked.low)} 条 — 必须人工判断",
            f"总计：{ranked.total} 条",
        ]
        return "\n".join(lines)
