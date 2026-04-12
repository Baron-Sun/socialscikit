"""Coding result review — interactive accept/reject/edit for each tier.

Three-tier review workflow:
- Tab 1 (High confidence): list view, spot-check, bulk "accept all" button
- Tab 2 (Medium confidence): one-by-one review with trigger word highlighting
- Tab 3 (Low confidence): one-by-one mandatory review, all must be processed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from socialscikit.qualikit.coder import CodingResult
from socialscikit.qualikit.confidence_ranker import RankedResults


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class CodingReviewAction(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"


@dataclass
class ReviewedCoding:
    """A coding result with review status."""

    result: CodingResult
    action: CodingReviewAction = CodingReviewAction.PENDING
    edited_themes: list[str] | None = None

    @property
    def final_themes(self) -> list[str]:
        if self.action == CodingReviewAction.REJECTED:
            return []
        if self.action == CodingReviewAction.EDITED and self.edited_themes is not None:
            return self.edited_themes
        return self.result.themes


@dataclass
class CodingReviewSession:
    """Manages review state for all coding results."""

    high: list[ReviewedCoding] = field(default_factory=list)
    medium: list[ReviewedCoding] = field(default_factory=list)
    low: list[ReviewedCoding] = field(default_factory=list)


# ---------------------------------------------------------------------------
# CodingReviewer
# ---------------------------------------------------------------------------


class CodingReviewer:
    """Interactive review manager for coded results.

    Usage::

        reviewer = CodingReviewer()
        session = reviewer.create_session(ranked_results)
        reviewer.accept_all_high(session)
        reviewer.accept(session, tier="medium", index=0)
        reviewer.reject(session, tier="low", index=0)
        reviewer.edit(session, tier="low", index=1, new_themes=["Theme A"])
    """

    def create_session(self, ranked: RankedResults) -> CodingReviewSession:
        """Create a review session from ranked results."""
        return CodingReviewSession(
            high=[ReviewedCoding(result=r) for r in ranked.high],
            medium=[ReviewedCoding(result=r) for r in ranked.medium],
            low=[ReviewedCoding(result=r) for r in ranked.low],
        )

    def _get_tier(self, session: CodingReviewSession, tier: str) -> list[ReviewedCoding]:
        tiers = {"high": session.high, "medium": session.medium, "low": session.low}
        if tier not in tiers:
            raise ValueError(f"Invalid tier: {tier}. Must be 'high', 'medium', or 'low'.")
        return tiers[tier]

    def accept(self, session: CodingReviewSession, tier: str, index: int) -> ReviewedCoding:
        """Accept a coding result."""
        items = self._get_tier(session, tier)
        items[index].action = CodingReviewAction.ACCEPTED
        return items[index]

    def reject(self, session: CodingReviewSession, tier: str, index: int) -> ReviewedCoding:
        """Reject a coding result (discard all assigned themes)."""
        items = self._get_tier(session, tier)
        items[index].action = CodingReviewAction.REJECTED
        return items[index]

    def edit(
        self, session: CodingReviewSession, tier: str, index: int,
        new_themes: list[str],
    ) -> ReviewedCoding:
        """Edit the assigned themes for a coding result."""
        items = self._get_tier(session, tier)
        items[index].action = CodingReviewAction.EDITED
        items[index].edited_themes = new_themes
        return items[index]

    def accept_all_high(self, session: CodingReviewSession) -> int:
        """Bulk accept all high-confidence results. Returns count."""
        count = 0
        for item in session.high:
            if item.action == CodingReviewAction.PENDING:
                item.action = CodingReviewAction.ACCEPTED
                count += 1
        return count

    def is_complete(self, session: CodingReviewSession) -> bool:
        """True if all low-confidence items have been reviewed."""
        return all(
            item.action != CodingReviewAction.PENDING
            for item in session.low
        )

    def stats(self, session: CodingReviewSession) -> dict:
        """Return review progress statistics."""
        def _tier_stats(items: list[ReviewedCoding]) -> dict:
            total = len(items)
            reviewed = sum(1 for i in items if i.action != CodingReviewAction.PENDING)
            return {"total": total, "reviewed": reviewed, "pending": total - reviewed}

        return {
            "high": _tier_stats(session.high),
            "medium": _tier_stats(session.medium),
            "low": _tier_stats(session.low),
        }

    def export_results(self, session: CodingReviewSession) -> list[dict]:
        """Export all reviewed results as a flat list of dicts."""
        rows = []
        for tier_name, items in [("high", session.high), ("medium", session.medium), ("low", session.low)]:
            for item in items:
                rows.append({
                    "text_id": item.result.text_id,
                    "text": item.result.text,
                    "themes": ", ".join(item.final_themes),
                    "confidence_tier": tier_name,
                    "review_action": item.action.value,
                    "original_themes": ", ".join(item.result.themes),
                    "reasoning": item.result.reasoning,
                })
        return rows
