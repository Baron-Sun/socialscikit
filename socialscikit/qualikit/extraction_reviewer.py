"""Extraction result review — accept / edit / reject each extracted segment.

Provides a review session that tracks user decisions on each LLM-extracted
segment.  Users can also manually add segments that the LLM missed.
The final output is a DataFrame ready for export or downstream processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from socialscikit.qualikit.segment_extractor import (
    ExtractionReport,
    ExtractionResult,
    ResearchQuestion,
)
from socialscikit.qualikit.segmenter import TextPosition, TextSegment


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ReviewAction(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


@dataclass
class ReviewedExtraction:
    """An extraction result with user review status."""

    result: ExtractionResult
    action: ReviewAction = ReviewAction.PENDING
    edited_rq_label: str | None = None
    edited_sub_theme: str | None = None

    @property
    def final_rq_label(self) -> str:
        if self.action == ReviewAction.EDITED and self.edited_rq_label is not None:
            return self.edited_rq_label
        return self.result.rq_label

    @property
    def final_sub_theme(self) -> str:
        if self.action == ReviewAction.EDITED and self.edited_sub_theme is not None:
            return self.edited_sub_theme
        return self.result.sub_theme


@dataclass
class ExtractionReviewSession:
    """Manages review state for extraction results."""

    items: list[ReviewedExtraction] = field(default_factory=list)
    original_text: str = ""
    segments: list[TextSegment] = field(default_factory=list)
    research_questions: list[ResearchQuestion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ExtractionReviewer
# ---------------------------------------------------------------------------


class ExtractionReviewer:
    """Interactive review manager for extracted segments.

    Usage::

        reviewer = ExtractionReviewer()
        session = reviewer.create_session(report, original_text, segments, rqs)
        reviewer.accept(session, 0)
        reviewer.edit(session, 1, new_sub_theme="新子主题")
        reviewer.reject(session, 2)
        reviewer.accept_all_high(session)
        df = reviewer.export_to_dataframe(session)
    """

    def create_session(
        self,
        report: ExtractionReport,
        original_text: str,
        segments: list[TextSegment],
        research_questions: list[ResearchQuestion],
    ) -> ExtractionReviewSession:
        """Create a review session from extraction results."""
        items = [ReviewedExtraction(result=r) for r in report.results]
        # Sort by RQ label, then by position
        items.sort(key=lambda x: (
            x.result.rq_label,
            x.result.position.char_start if x.result.position else 0,
        ))
        return ExtractionReviewSession(
            items=items,
            original_text=original_text,
            segments=segments,
            research_questions=research_questions,
        )

    def accept(self, session: ExtractionReviewSession, index: int) -> ReviewedExtraction:
        """Accept an extraction result at the given index."""
        item = session.items[index]
        item.action = ReviewAction.ACCEPTED
        return item

    def reject(self, session: ExtractionReviewSession, index: int) -> ReviewedExtraction:
        """Reject an extraction result at the given index."""
        item = session.items[index]
        item.action = ReviewAction.REJECTED
        return item

    def edit(
        self,
        session: ExtractionReviewSession,
        index: int,
        new_rq_label: str | None = None,
        new_sub_theme: str | None = None,
    ) -> ReviewedExtraction:
        """Edit the labels of an extraction result."""
        item = session.items[index]
        item.action = ReviewAction.EDITED
        if new_rq_label is not None:
            item.edited_rq_label = new_rq_label
        if new_sub_theme is not None:
            item.edited_sub_theme = new_sub_theme
        return item

    def accept_all_high(
        self, session: ExtractionReviewSession, threshold: float = 0.85,
    ) -> int:
        """Accept all items with confidence >= threshold. Returns count."""
        count = 0
        for item in session.items:
            if item.action == ReviewAction.PENDING and item.result.confidence >= threshold:
                item.action = ReviewAction.ACCEPTED
                count += 1
        return count

    def add_manual(
        self,
        session: ExtractionReviewSession,
        segment_id: int,
        rq_label: str,
        sub_theme: str,
    ) -> ReviewedExtraction | None:
        """Manually add a segment that the LLM missed.

        Looks up the segment by ID from the session's segment list,
        creates a new ExtractionResult with confidence=1.0 (human).
        """
        seg = None
        for s in session.segments:
            if s.segment_id == segment_id:
                seg = s
                break
        if seg is None:
            return None

        result = ExtractionResult(
            segment_id=seg.segment_id,
            text=seg.text,
            rq_label=rq_label,
            sub_theme=sub_theme,
            confidence=1.0,
            reasoning="手动添加",
            position=seg.position,
        )
        reviewed = ReviewedExtraction(result=result, action=ReviewAction.ACCEPTED)
        session.items.append(reviewed)
        return reviewed

    def stats(self, session: ExtractionReviewSession) -> dict:
        """Return review progress statistics."""
        total = len(session.items)
        accepted = sum(1 for i in session.items if i.action == ReviewAction.ACCEPTED)
        edited = sum(1 for i in session.items if i.action == ReviewAction.EDITED)
        rejected = sum(1 for i in session.items if i.action == ReviewAction.REJECTED)
        pending = sum(1 for i in session.items if i.action == ReviewAction.PENDING)
        return {
            "total": total,
            "accepted": accepted,
            "edited": edited,
            "rejected": rejected,
            "pending": pending,
            "progress_pct": round((total - pending) / total * 100, 1) if total else 0,
        }

    def export_to_dataframe(self, session: ExtractionReviewSession) -> pd.DataFrame:
        """Export reviewed (non-rejected) results as a DataFrame.

        Columns match the required output schema for Excel export and
        downstream QualiKit pipeline compatibility.
        """
        rows: list[dict] = []
        for item in session.items:
            if item.action == ReviewAction.REJECTED:
                continue
            pos = item.result.position
            rows.append({
                "segment_id": item.result.segment_id,
                "text": item.result.text,
                "rq_label": item.final_rq_label,
                "sub_theme": item.final_sub_theme,
                "confidence": item.result.confidence,
                "line_start": pos.line_start if pos else 0,
                "line_end": pos.line_end if pos else 0,
                "char_start": pos.char_start if pos else 0,
                "char_end": pos.char_end if pos else 0,
                "paragraph_index": pos.paragraph_index if pos else 0,
                "review_status": item.action.value,
                "reasoning": item.result.reasoning,
                "evidence_span": getattr(item.result, "evidence_span", ""),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()
