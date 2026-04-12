"""Tests for socialscikit.qualikit.theme_reviewer."""

from __future__ import annotations

import pytest

from socialscikit.qualikit.theme_definer import Theme, ThemeSuggestion
from socialscikit.qualikit.theme_reviewer import (
    ThemeReviewSession,
    ThemeReviewer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SUGGESTIONS = [
    ThemeSuggestion(
        name="Financial Stress",
        description="Concerns about money and costs.",
        representative_texts=["I worry about bills."],
        estimated_coverage=0.3,
    ),
    ThemeSuggestion(
        name="Work-Life Balance",
        description="Difficulty balancing work and personal life.",
        representative_texts=["Long hours at work."],
        estimated_coverage=0.25,
    ),
    ThemeSuggestion(
        name="Community",
        description="Neighborhood and social connections.",
        representative_texts=["I volunteer locally."],
        estimated_coverage=0.2,
    ),
]


@pytest.fixture()
def reviewer():
    return ThemeReviewer()


@pytest.fixture()
def session(reviewer):
    return reviewer.create_session(SUGGESTIONS)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestCreateSession:
    def test_creates_session(self, session):
        assert isinstance(session, ThemeReviewSession)
        assert len(session.themes) == 3
        assert not session.locked

    def test_themes_from_suggestions(self, session):
        assert session.themes[0].name == "Financial Stress"
        assert session.themes[0].inclusion_examples == ["I worry about bills."]
        assert session.themes[0].exclusion_examples == []

    def test_empty_suggestions(self, reviewer):
        session = reviewer.create_session([])
        assert len(session.themes) == 0


# ---------------------------------------------------------------------------
# Edit operations
# ---------------------------------------------------------------------------


class TestEditTheme:
    def test_edit_name(self, reviewer, session):
        t = reviewer.edit_theme(session, 0, name="Money Worries")
        assert t.name == "Money Worries"

    def test_edit_description(self, reviewer, session):
        t = reviewer.edit_theme(session, 0, description="New description")
        assert t.description == "New description"

    def test_edit_examples(self, reviewer, session):
        reviewer.edit_theme(session, 0, exclusion_examples=["Not about money."])
        assert session.themes[0].exclusion_examples == ["Not about money."]

    def test_edit_out_of_range(self, reviewer, session):
        with pytest.raises(IndexError):
            reviewer.edit_theme(session, 99, name="X")

    def test_edit_locked(self, reviewer, session):
        # Add exclusion examples and lock
        for t in session.themes:
            t.exclusion_examples = ["example"]
        reviewer.lock(session)
        with pytest.raises(RuntimeError, match="locked"):
            reviewer.edit_theme(session, 0, name="X")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDeleteTheme:
    def test_delete(self, reviewer, session):
        deleted = reviewer.delete_theme(session, 1)
        assert deleted.name == "Work-Life Balance"
        assert len(session.themes) == 2

    def test_delete_out_of_range(self, reviewer, session):
        with pytest.raises(IndexError):
            reviewer.delete_theme(session, 10)

    def test_delete_locked(self, reviewer, session):
        for t in session.themes:
            t.exclusion_examples = ["ex"]
        reviewer.lock(session)
        with pytest.raises(RuntimeError):
            reviewer.delete_theme(session, 0)


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------


class TestAddTheme:
    def test_add(self, reviewer, session):
        idx = reviewer.add_theme(session, Theme(name="New", description="desc"))
        assert idx == 3
        assert len(session.themes) == 4
        assert session.themes[3].name == "New"

    def test_add_locked(self, reviewer, session):
        for t in session.themes:
            t.exclusion_examples = ["ex"]
        reviewer.lock(session)
        with pytest.raises(RuntimeError):
            reviewer.add_theme(session, Theme(name="X", description="d"))


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMergeThemes:
    def test_merge_two(self, reviewer, session):
        merged = reviewer.merge_themes(session, [0, 1], merged_name="Combined")
        assert merged.name == "Combined"
        assert len(session.themes) == 2  # was 3, removed 2, added 1
        # Examples combined
        assert "I worry about bills." in merged.inclusion_examples
        assert "Long hours at work." in merged.inclusion_examples

    def test_merge_needs_two(self, reviewer, session):
        with pytest.raises(ValueError):
            reviewer.merge_themes(session, [0], merged_name="X")

    def test_merge_invalid_index(self, reviewer, session):
        with pytest.raises(IndexError):
            reviewer.merge_themes(session, [0, 99], merged_name="X")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_exclusion_examples(self, reviewer, session):
        warnings = reviewer.validate_for_coding(session)
        # All themes lack exclusion examples
        assert len(warnings) >= 3
        assert any("排除示例" in w for w in warnings)

    def test_valid_after_adding_exclusions(self, reviewer, session):
        for t in session.themes:
            t.exclusion_examples = ["Not this."]
        warnings = reviewer.validate_for_coding(session)
        assert not any("排除示例" in w for w in warnings)

    def test_empty_name(self, reviewer, session):
        session.themes[0].name = ""
        warnings = reviewer.validate_for_coding(session)
        assert any("名称" in w for w in warnings)

    def test_empty_session(self, reviewer):
        session = ThemeReviewSession()
        warnings = reviewer.validate_for_coding(session)
        assert any("没有定义" in w for w in warnings)


# ---------------------------------------------------------------------------
# Lock / Unlock
# ---------------------------------------------------------------------------


class TestLock:
    def test_lock_with_valid_themes(self, reviewer, session):
        for t in session.themes:
            t.exclusion_examples = ["example"]
        warnings = reviewer.lock(session)
        assert session.locked
        # Warnings about exclusion examples should not be present
        assert not any("不能为空" in w for w in warnings)

    def test_lock_blocked_by_empty_name(self, reviewer, session):
        session.themes[0].name = ""
        warnings = reviewer.lock(session)
        assert not session.locked
        assert any("不能为空" in w for w in warnings)

    def test_lock_allows_missing_exclusion(self, reviewer, session):
        """Missing exclusion examples warn but don't block locking."""
        warnings = reviewer.lock(session)
        assert session.locked  # should still lock
        assert any("排除示例" in w for w in warnings)

    def test_unlock(self, reviewer, session):
        for t in session.themes:
            t.exclusion_examples = ["ex"]
        reviewer.lock(session)
        assert session.locked
        reviewer.unlock(session)
        assert not session.locked

    def test_get_themes_for_coding_unlocked(self, reviewer, session):
        with pytest.raises(RuntimeError, match="locked"):
            reviewer.get_themes_for_coding(session)

    def test_get_themes_for_coding(self, reviewer, session):
        for t in session.themes:
            t.exclusion_examples = ["ex"]
        reviewer.lock(session)
        themes = reviewer.get_themes_for_coding(session)
        assert len(themes) == 3
