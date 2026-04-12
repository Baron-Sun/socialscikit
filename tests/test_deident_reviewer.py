"""Tests for socialscikit.qualikit.deident_reviewer."""

from __future__ import annotations

import pytest

from socialscikit.qualikit.deidentifier import Deidentifier
from socialscikit.qualikit.deident_reviewer import (
    DeidentReviewer,
    ReviewAction,
    ReviewedReplacement,
    ReviewSession,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def reviewer():
    return DeidentReviewer()


@pytest.fixture()
def sample_result():
    """Run deidentification on sample texts."""
    deident = Deidentifier()
    texts = [
        "Contact alice@example.com or call 555-123-4567.",
        "Email bob@test.org for help.",
    ]
    return deident.process(texts, entities=["EMAIL", "PHONE"]), texts


@pytest.fixture()
def session(reviewer, sample_result):
    result, texts = sample_result
    return reviewer.create_session(result, texts)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_creates_session(self, reviewer, sample_result):
        result, texts = sample_result
        session = reviewer.create_session(result, texts)
        assert isinstance(session, ReviewSession)
        assert len(session.items) == len(result.replacement_log)
        assert len(session.original_texts) == 2

    def test_all_pending_initially(self, session):
        for item in session.items:
            assert item.action == ReviewAction.PENDING


# ---------------------------------------------------------------------------
# Accept / Reject / Edit
# ---------------------------------------------------------------------------


class TestReviewActions:
    def test_accept(self, reviewer, session):
        item = reviewer.accept(session, 0)
        assert item.action == ReviewAction.ACCEPTED

    def test_reject(self, reviewer, session):
        item = reviewer.reject(session, 0)
        assert item.action == ReviewAction.REJECTED
        # final_replacement should be original span
        assert item.final_replacement == item.record.original_span

    def test_edit(self, reviewer, session):
        item = reviewer.edit(session, 0, "[PARTICIPANT_A]")
        assert item.action == ReviewAction.EDITED
        assert item.edited_replacement == "[PARTICIPANT_A]"
        assert item.final_replacement == "[PARTICIPANT_A]"

    def test_accept_all(self, reviewer, session):
        count = reviewer.accept_all(session)
        assert count == len(session.items)
        for item in session.items:
            assert item.action == ReviewAction.ACCEPTED

    def test_accept_high_confidence(self, reviewer, session):
        count = reviewer.accept_high_confidence(session, threshold=0.9)
        assert count >= 0
        for item in session.items:
            if item.record.confidence >= 0.9:
                assert item.action == ReviewAction.ACCEPTED

    def test_accept_all_skips_reviewed(self, reviewer, session):
        reviewer.reject(session, 0)
        count = reviewer.accept_all(session)
        # The rejected item should stay rejected
        assert session.items[0].action == ReviewAction.REJECTED
        assert count == len(session.items) - 1


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


class TestApply:
    def test_apply_all_accepted(self, reviewer, session):
        reviewer.accept_all(session)
        final = reviewer.apply(session)
        assert len(final) == 2
        # Should match the deidentified texts
        for f, d in zip(final, session.deidentified_texts):
            assert "example.com" not in f
            assert "test.org" not in f

    def test_apply_all_rejected(self, reviewer, session):
        for i in range(len(session.items)):
            reviewer.reject(session, i)
        final = reviewer.apply(session)
        # Should restore original texts
        assert "alice@example.com" in final[0]
        assert "bob@test.org" in final[1]

    def test_apply_mixed(self, reviewer, session):
        # Accept first, reject second
        reviewer.accept(session, 0)
        if len(session.items) > 1:
            reviewer.reject(session, 1)
        final = reviewer.apply(session)
        assert len(final) == 2

    def test_apply_edited(self, reviewer, session):
        reviewer.edit(session, 0, "[CUSTOM]")
        final = reviewer.apply(session)
        assert "[CUSTOM]" in final[0]

    def test_apply_pending_treated_as_accepted(self, reviewer, session):
        # Don't review anything — pending items keep deidentified version
        final = reviewer.apply(session)
        for text in final:
            assert "alice@example.com" not in text


# ---------------------------------------------------------------------------
# Stats & completion
# ---------------------------------------------------------------------------


class TestStats:
    def test_initial_stats(self, reviewer, session):
        stats = reviewer.stats(session)
        assert stats["pending"] == len(session.items)
        assert stats["accepted"] == 0

    def test_after_actions(self, reviewer, session):
        reviewer.accept(session, 0)
        if len(session.items) > 1:
            reviewer.reject(session, 1)
        stats = reviewer.stats(session)
        assert stats["accepted"] == 1
        assert stats["rejected"] == 1

    def test_not_complete_initially(self, reviewer, session):
        assert not reviewer.is_complete(session)

    def test_complete_after_all_reviewed(self, reviewer, session):
        reviewer.accept_all(session)
        assert reviewer.is_complete(session)


# ---------------------------------------------------------------------------
# Export correspondence table
# ---------------------------------------------------------------------------


class TestExport:
    def test_export(self, reviewer, session):
        reviewer.accept_all(session)
        table = reviewer.export_correspondence_table(session)
        assert len(table) == len(session.items)
        assert "原文" in table[0]
        assert "替换为" in table[0]
        assert "操作" in table[0]

    def test_export_empty(self, reviewer):
        session = ReviewSession()
        table = reviewer.export_correspondence_table(session)
        assert table == []


# ---------------------------------------------------------------------------
# ReviewedReplacement properties
# ---------------------------------------------------------------------------


class TestReviewedReplacement:
    def test_final_replacement_default(self, session):
        """Pending items return the deidentified replacement."""
        item = session.items[0]
        assert item.final_replacement == item.record.replacement

    def test_final_replacement_rejected(self, reviewer, session):
        reviewer.reject(session, 0)
        item = session.items[0]
        assert item.final_replacement == item.record.original_span
