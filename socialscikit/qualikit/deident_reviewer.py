"""Interactive de-identification review — accept, edit, or reject replacements.

Backs the Gradio review UI. Users can:
- Accept individual replacements
- Edit the replacement text
- Reject (restore original span)
- Bulk accept all / only high-confidence replacements
- Export a correspondence table

Important disclaimer:
    Automated de-identification is a first-pass tool. Manual review is
    required before IRB submission. This tool does not guarantee complete
    removal of identifying information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from socialscikit.qualikit.deidentifier import DeidentResult, ReplacementRecord


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ReviewAction(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EDITED = "edited"
    REJECTED = "rejected"


@dataclass
class ReviewedReplacement:
    """A replacement record with user review status."""

    record: ReplacementRecord
    action: ReviewAction = ReviewAction.PENDING
    edited_replacement: str | None = None  # set if action == EDITED

    @property
    def final_replacement(self) -> str:
        """The replacement text to use after review."""
        if self.action == ReviewAction.REJECTED:
            return self.record.original_span
        if self.action == ReviewAction.EDITED and self.edited_replacement is not None:
            return self.edited_replacement
        return self.record.replacement


@dataclass
class ReviewSession:
    """Manages the review state for a de-identification result."""

    items: list[ReviewedReplacement] = field(default_factory=list)
    original_texts: list[str] = field(default_factory=list)
    deidentified_texts: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DeidentReviewer
# ---------------------------------------------------------------------------


class DeidentReviewer:
    """Interactive review manager for de-identification results.

    Usage::

        reviewer = DeidentReviewer()
        session = reviewer.create_session(deident_result, original_texts)
        reviewer.accept(session, index=0)
        reviewer.reject(session, index=1)
        reviewer.edit(session, index=2, new_replacement="[PARTICIPANT_A]")
        final_texts = reviewer.apply(session)
    """

    def create_session(
        self,
        result: DeidentResult,
        original_texts: list[str],
    ) -> ReviewSession:
        """Create a review session from de-identification results.

        Parameters
        ----------
        result : DeidentResult
            Output from Deidentifier.process().
        original_texts : list[str]
            The original (pre-deidentification) texts.
        """
        items = [
            ReviewedReplacement(record=rec)
            for rec in result.replacement_log
        ]
        return ReviewSession(
            items=items,
            original_texts=list(original_texts),
            deidentified_texts=list(result.deidentified_texts),
        )

    def accept(self, session: ReviewSession, index: int) -> ReviewedReplacement:
        """Accept a replacement at the given index."""
        item = session.items[index]
        item.action = ReviewAction.ACCEPTED
        return item

    def reject(self, session: ReviewSession, index: int) -> ReviewedReplacement:
        """Reject a replacement (restore original text)."""
        item = session.items[index]
        item.action = ReviewAction.REJECTED
        return item

    def edit(
        self, session: ReviewSession, index: int, new_replacement: str,
    ) -> ReviewedReplacement:
        """Edit a replacement with custom text."""
        item = session.items[index]
        item.action = ReviewAction.EDITED
        item.edited_replacement = new_replacement
        return item

    def accept_all(self, session: ReviewSession) -> int:
        """Accept all pending replacements. Returns count accepted."""
        count = 0
        for item in session.items:
            if item.action == ReviewAction.PENDING:
                item.action = ReviewAction.ACCEPTED
                count += 1
        return count

    def accept_high_confidence(
        self, session: ReviewSession, threshold: float = 0.9,
    ) -> int:
        """Accept pending replacements above the confidence threshold."""
        count = 0
        for item in session.items:
            if item.action == ReviewAction.PENDING and item.record.confidence >= threshold:
                item.action = ReviewAction.ACCEPTED
                count += 1
        return count

    def apply(self, session: ReviewSession) -> list[str]:
        """Apply all reviewed replacements and return final texts.

        Pending items are treated as accepted (keeps the de-identified version).
        Rejected items restore the original span.
        Edited items use the edited replacement.
        """
        # Start from original texts and rebuild
        final_texts = list(session.original_texts)

        # Group records by text_id, sorted by position descending for safe replacement
        from collections import defaultdict
        by_text: dict[int, list[ReviewedReplacement]] = defaultdict(list)
        for item in session.items:
            by_text[item.record.text_id].append(item)

        for text_id, items in by_text.items():
            if text_id >= len(final_texts):
                continue
            text = final_texts[text_id]
            # Sort by position descending to replace from end to start
            sorted_items = sorted(items, key=lambda it: it.record.position[0], reverse=True)
            for item in sorted_items:
                start, end = item.record.position
                replacement = item.final_replacement
                text = text[:start] + replacement + text[end:]
            final_texts[text_id] = text

        return final_texts

    def stats(self, session: ReviewSession) -> dict[str, int]:
        """Return counts by review action."""
        counts: dict[str, int] = {a.value: 0 for a in ReviewAction}
        for item in session.items:
            counts[item.action.value] += 1
        return counts

    def is_complete(self, session: ReviewSession) -> bool:
        """True if no items are still pending."""
        return all(item.action != ReviewAction.PENDING for item in session.items)

    def export_correspondence_table(
        self, session: ReviewSession,
    ) -> list[dict[str, str]]:
        """Export a mapping of original spans to final replacements."""
        rows = []
        for item in session.items:
            rows.append({
                "原文": item.record.original_span,
                "替换为": item.final_replacement,
                "实体类型": item.record.entity_type,
                "操作": item.action.value,
                "置信度": f"{item.record.confidence:.2f}",
            })
        return rows
