"""LLM-based segment classification for qualitative research.

Given segmented text and user-defined Research Questions (RQs) with optional
sub-themes, classifies each segment against the research framework.

Key design choices:
- ALL RQs processed in one prompt (LLM sees full framework simultaneously)
- A segment can match multiple RQs and sub-themes (multi-label)
- If user defines sub-themes → LLM must pick from those
- If no sub-themes → LLM generates labels automatically
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from socialscikit.core.llm_client import LLMClient
from socialscikit.qualikit.segmenter import TextPosition, TextSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ResearchQuestion:
    """A user-defined research question with optional sub-themes."""

    rq_id: str  # e.g. "RQ1"
    description: str  # e.g. "受访者对政策的态度"
    sub_themes: list[str] = field(default_factory=list)  # user-defined


@dataclass
class ExtractionResult:
    """LLM extraction result for a single (segment, RQ, sub-theme) match."""

    segment_id: int
    text: str
    rq_label: str  # which RQ this relates to
    sub_theme: str  # sub-theme label
    confidence: float  # 0.0 – 1.0
    reasoning: str = ""
    position: TextPosition | None = None


@dataclass
class ExtractionReport:
    """Summary of an extraction run."""

    results: list[ExtractionResult] = field(default_factory=list)
    n_segments_total: int = 0
    n_relevant: int = 0
    n_failed: int = 0
    rq_distribution: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM = (
    "你是一位经验丰富的质性研究助手。你的任务是将文本段落与研究框架中的研究问题"
    "和子主题进行匹配。一个段落可以同时与多个研究问题和子主题相关。"
    "请保持严谨，只在文本明确涉及时才标记。"
)

_CLASSIFICATION_PROMPT = """\
## 研究框架

{rq_framework}

## 待分析文本段落

{segments_block}

## 指令
逐段分析每个文本段落（编号 [N]），判断其与上述研究框架中的哪些研究问题和子主题相关。

规则：
1. 一个段落可以同时关联多个研究问题和多个子主题，每个匹配单独列出
2. 只在文本明确涉及时才标记，避免过度匹配
3. 不相关的段落不要包含在结果中
4. 如果研究问题指定了子主题列表，sub_theme 必须从中选择；如果标注了「自动生成」，请自行生成 2-8 字的子主题标签

仅返回 JSON，不要 markdown 代码块：
{{"matches": [{{"segment_id": 1, "rq_id": "RQ1", "sub_theme": "子主题名", "confidence": 0.85, "reasoning": "简要依据"}}]}}"""


# ---------------------------------------------------------------------------
# SegmentExtractor
# ---------------------------------------------------------------------------


class SegmentExtractor:
    """Classify segments against a research framework using an LLM.

    Usage::

        extractor = SegmentExtractor(llm_client)
        report = extractor.extract(segments, research_questions)
    """

    def __init__(
        self,
        llm: LLMClient,
        batch_size: int = 15,
        max_segment_chars: int = 500,
    ):
        self._llm = llm
        self._batch_size = batch_size
        self._max_segment_chars = max_segment_chars

    def extract(
        self,
        segments: list[TextSegment],
        research_questions: list[ResearchQuestion],
    ) -> ExtractionReport:
        """Classify segments against all RQs (synchronous).

        All RQs are sent in one prompt per batch so the LLM sees the
        full research framework simultaneously.  This produces better
        multi-RQ coverage than the old one-RQ-at-a-time approach.
        """
        report = ExtractionReport(n_segments_total=len(segments))
        seg_map = {s.segment_id: s for s in segments}
        rq_framework = self._format_rq_framework(research_questions)

        for batch_start in range(0, len(segments), self._batch_size):
            batch = segments[batch_start:batch_start + self._batch_size]
            batch_results = self._classify_batch(
                batch, research_questions, rq_framework,
            )
            for result in batch_results:
                seg = seg_map.get(result.segment_id)
                if seg:
                    result.position = seg.position
                report.results.append(result)

        # Count per-RQ distribution
        for rq in research_questions:
            report.rq_distribution[rq.rq_id] = sum(
                1 for r in report.results if r.rq_label == rq.rq_id
            )
        report.n_relevant = len(report.results)
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_rq_framework(self, rqs: list[ResearchQuestion]) -> str:
        """Build the research framework block for the prompt."""
        lines: list[str] = []
        for rq in rqs:
            lines.append(f"{rq.rq_id}：{rq.description}")
            if rq.sub_themes:
                lines.append(f"  子主题：{'、'.join(rq.sub_themes)}")
            else:
                lines.append("  子主题：（自动生成）")
            lines.append("")
        return "\n".join(lines)

    def _format_segments(self, segments: list[TextSegment]) -> str:
        """Format segments for the prompt."""
        lines: list[str] = []
        for seg in segments:
            text = seg.text
            if len(text) > self._max_segment_chars:
                text = text[:self._max_segment_chars] + "…"
            lines.append(f"[{seg.segment_id}] {text}")
            lines.append("")
        return "\n".join(lines)

    def _classify_batch(
        self,
        batch: list[TextSegment],
        rqs: list[ResearchQuestion],
        rq_framework: str,
    ) -> list[ExtractionResult]:
        """Send a batch of segments to LLM for classification."""
        segments_block = self._format_segments(batch)
        prompt = _CLASSIFICATION_PROMPT.format(
            rq_framework=rq_framework,
            segments_block=segments_block,
        )

        try:
            resp = self._llm.complete(
                prompt, system=_CLASSIFICATION_SYSTEM, max_tokens=4096,
            )
            return self._parse_response(resp.text, rqs, batch)
        except Exception as e:
            logger.warning("Classification batch failed: %s", e)
            return []

    def _parse_response(
        self,
        raw: str,
        rqs: list[ResearchQuestion],
        batch: list[TextSegment],
    ) -> list[ExtractionResult]:
        """Parse LLM JSON response into ExtractionResult list."""
        raw = raw.strip()

        # Strip markdown code fencing
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw).strip()

        # Try parsing JSON
        data = None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    pass

        if not data or not isinstance(data, dict):
            logger.warning("Could not parse response: %s", raw[:200])
            return []

        # Support both "matches" and legacy "relevant_segments" key
        items = data.get("matches", data.get("relevant_segments", []))
        if not isinstance(items, list):
            return []

        valid_rq_ids = {rq.rq_id for rq in rqs}
        valid_subs: dict[str, set[str]] = {
            rq.rq_id: set(rq.sub_themes) for rq in rqs if rq.sub_themes
        }
        seg_text = {s.segment_id: s.text for s in batch}

        results: list[ExtractionResult] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            seg_id = item.get("segment_id")
            if seg_id is None:
                continue
            try:
                seg_id = int(seg_id)
            except (ValueError, TypeError):
                continue

            # Get rq_id — fall back to single-RQ inference for compat
            rq_id = str(item.get("rq_id", item.get("rq_label", ""))).strip()
            if not rq_id and len(rqs) == 1:
                rq_id = rqs[0].rq_id
            if rq_id not in valid_rq_ids:
                continue

            sub_theme = str(item.get("sub_theme", "")).strip()

            # If predefined sub-themes exist for this RQ, validate
            if rq_id in valid_subs and sub_theme and sub_theme not in valid_subs[rq_id]:
                continue  # skip invalid sub-theme

            confidence = float(item.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            results.append(ExtractionResult(
                segment_id=seg_id,
                text=seg_text.get(seg_id, ""),
                rq_label=rq_id,
                sub_theme=sub_theme or "未分类",
                confidence=round(confidence, 3),
                reasoning=str(item.get("reasoning", "")).strip(),
            ))

        return results
