"""Tests for socialscikit.qualikit.coding_reviewer."""

from __future__ import annotations

import pytest

from socialscikit.qualikit.coder import CodingResult
from socialscikit.qualikit.coding_reviewer import (
    CodingReviewAction,
    CodingReviewSession,
    CodingReviewer,
    ReviewedCoding,
)
from socialscikit.qualikit.confidence_ranker import ConfidenceRanker, RankedResults


@pytest.fixture()
def reviewer():
    return CodingReviewer()


@pytest.fixture()
def ranked():
    results = [
        CodingResult(text_id=0, text="a", themes=["A"], confidences={"A": 0.95}),
        CodingResult(text_id=1, text="b", themes=["B"], confidences={"B": 0.90}),
        CodingResult(text_id=2, text="c", themes=["A"], confidences={"A": 0.75}),
        CodingResult(text_id=3, text="d", themes=["B"], confidences={"B": 0.65}),
        CodingResult(text_id=4, text="e", themes=["A"], confidences={"A": 0.40}),
        CodingResult(text_id=5, text="f", themes=[], confidences={}),
    ]
    return ConfidenceRanker().rank(results)


@pytest.fixture()
def session(reviewer, ranked):
    return reviewer.create_session(ranked)


class TestCreateSession:
    def test_creates_session(self, session):
        assert isinstance(session, CodingReviewSession)
        assert len(session.high) == 2
        assert len(session.medium) == 2
        assert len(session.low) == 2

    def test_all_pending(self, session):
        for tier in [session.high, session.medium, session.low]:
            for item in tier:
                assert item.action == CodingReviewAction.PENDING


class TestAcceptRejectEdit:
    def test_accept(self, reviewer, session):
        item = reviewer.accept(session, "high", 0)
        assert item.action == CodingReviewAction.ACCEPTED
        assert item.final_themes == ["A"]

    def test_reject(self, reviewer, session):
        item = reviewer.reject(session, "medium", 0)
        assert item.action == CodingReviewAction.REJECTED
        assert item.final_themes == []

    def test_edit(self, reviewer, session):
        item = reviewer.edit(session, "low", 0, new_themes=["X", "Y"])
        assert item.action == CodingReviewAction.EDITED
        assert item.final_themes == ["X", "Y"]

    def test_invalid_tier(self, reviewer, session):
        with pytest.raises(ValueError):
            reviewer.accept(session, "invalid", 0)


class TestAcceptAllHigh:
    def test_accept_all(self, reviewer, session):
        count = reviewer.accept_all_high(session)
        assert count == 2
        for item in session.high:
            assert item.action == CodingReviewAction.ACCEPTED

    def test_skip_already_reviewed(self, reviewer, session):
        reviewer.reject(session, "high", 0)
        count = reviewer.accept_all_high(session)
        assert count == 1
        assert session.high[0].action == CodingReviewAction.REJECTED


class TestIsComplete:
    def test_not_complete_initially(self, reviewer, session):
        assert not reviewer.is_complete(session)

    def test_complete_after_low_reviewed(self, reviewer, session):
        for i in range(len(session.low)):
            reviewer.accept(session, "low", i)
        assert reviewer.is_complete(session)

    def test_medium_not_required(self, reviewer, session):
        """Medium tier doesn't block completion."""
        for i in range(len(session.low)):
            reviewer.accept(session, "low", i)
        # Medium still pending but is_complete should be True
        assert reviewer.is_complete(session)


class TestStats:
    def test_initial_stats(self, reviewer, session):
        stats = reviewer.stats(session)
        assert stats["high"]["total"] == 2
        assert stats["high"]["pending"] == 2
        assert stats["low"]["total"] == 2

    def test_after_review(self, reviewer, session):
        reviewer.accept_all_high(session)
        reviewer.accept(session, "medium", 0)
        stats = reviewer.stats(session)
        assert stats["high"]["reviewed"] == 2
        assert stats["medium"]["reviewed"] == 1


class TestExport:
    def test_export(self, reviewer, session):
        reviewer.accept_all_high(session)
        reviewer.accept(session, "medium", 0)
        reviewer.reject(session, "low", 0)
        rows = reviewer.export_results(session)
        assert len(rows) == 6
        assert "text_id" in rows[0]
        assert "themes" in rows[0]
        assert "review_action" in rows[0]

    def test_export_empty(self, reviewer):
        session = CodingReviewSession()
        rows = reviewer.export_results(session)
        assert rows == []


class TestReviewedCoding:
    def test_final_themes_accepted(self):
        rc = ReviewedCoding(
            result=CodingResult(text_id=0, text="t", themes=["A", "B"]),
            action=CodingReviewAction.ACCEPTED,
        )
        assert rc.final_themes == ["A", "B"]

    def test_final_themes_rejected(self):
        rc = ReviewedCoding(
            result=CodingResult(text_id=0, text="t", themes=["A"]),
            action=CodingReviewAction.REJECTED,
        )
        assert rc.final_themes == []

    def test_final_themes_edited(self):
        rc = ReviewedCoding(
            result=CodingResult(text_id=0, text="t", themes=["A"]),
            action=CodingReviewAction.EDITED,
            edited_themes=["X"],
        )
        assert rc.final_themes == ["X"]
