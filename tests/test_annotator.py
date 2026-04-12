"""Tests for socialscikit.quantikit.annotator."""

import pandas as pd
import pytest

from socialscikit.quantikit.annotator import (
    Annotation,
    AnnotationSession,
    AnnotationSessionStats,
    AnnotationStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "text": [
            "I love this product",
            "Terrible experience",
            "It is okay",
            "Amazing quality",
            "Worst purchase ever",
        ],
    })


@pytest.fixture()
def session(sample_df):
    return AnnotationSession.from_dataframe(sample_df, labels=LABELS)


# ---------------------------------------------------------------------------
# Session creation
# ---------------------------------------------------------------------------


class TestSessionCreation:
    def test_from_dataframe(self, sample_df):
        s = AnnotationSession.from_dataframe(sample_df, labels=LABELS)
        assert s.current() is not None
        assert s.current().text == "I love this product"
        stats = s.stats()
        assert stats.total == 5
        assert stats.pending == 5

    def test_custom_text_col(self):
        df = pd.DataFrame({"content": ["hello", "world"]})
        s = AnnotationSession.from_dataframe(df, text_col="content", labels=["A", "B"])
        assert s.current().text == "hello"

    def test_invalid_text_col(self, sample_df):
        with pytest.raises(ValueError, match="not found"):
            AnnotationSession.from_dataframe(sample_df, text_col="nonexistent")

    def test_resume_partial_labels(self):
        df = pd.DataFrame({
            "text": ["text1", "text2", "text3"],
            "label": ["pos", None, "neg"],
        })
        s = AnnotationSession.from_dataframe(df, text_col="text", label_col="label")
        # Should skip to first pending (index 1)
        assert s.current().text == "text2"
        stats = s.stats()
        assert stats.labeled == 2
        assert stats.pending == 1

    def test_infer_labels_from_column(self):
        df = pd.DataFrame({
            "text": ["a", "b", "c"],
            "label": ["X", "Y", "X"],
        })
        s = AnnotationSession.from_dataframe(df, text_col="text", label_col="label")
        assert "X" in s.labels
        assert "Y" in s.labels

    def test_shuffle(self, sample_df):
        s = AnnotationSession.from_dataframe(sample_df, labels=LABELS, shuffle=True)
        # Still 5 items
        assert s.stats().total == 5


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_annotate_and_advance(self, session):
        item = session.annotate("positive")
        assert item.label == "positive"
        assert item.status == AnnotationStatus.LABELED
        assert session.current().text == "Terrible experience"

    def test_annotate_all(self, session):
        for label in ["positive", "negative", "neutral", "positive", "negative"]:
            session.annotate(label)
        assert session.current() is None
        assert session.is_complete

    def test_invalid_label(self, session):
        with pytest.raises(ValueError, match="Invalid label"):
            session.annotate("invalid_label")

    def test_annotate_past_end(self, session):
        for _ in range(5):
            session.annotate("positive")
        with pytest.raises(IndexError):
            session.annotate("positive")

    def test_annotator_note(self, session):
        item = session.annotate("positive", note="borderline case")
        assert item.annotator_note == "borderline case"

    def test_no_label_validation_when_empty(self):
        df = pd.DataFrame({"text": ["hello"]})
        s = AnnotationSession.from_dataframe(df, labels=[])
        # Empty labels list → no validation
        item = s.annotate("anything")
        assert item.label == "anything"


# ---------------------------------------------------------------------------
# Skip and Flag
# ---------------------------------------------------------------------------


class TestSkipAndFlag:
    def test_skip(self, session):
        item = session.skip()
        assert item.status == AnnotationStatus.SKIPPED
        assert session.current().text == "Terrible experience"

    def test_flag(self, session):
        item = session.flag(note="ambiguous")
        assert item.status == AnnotationStatus.FLAGGED
        assert item.annotator_note == "ambiguous"

    def test_skip_past_end(self, session):
        for _ in range(5):
            session.skip()
        with pytest.raises(IndexError):
            session.skip()

    def test_mixed_operations(self, session):
        session.annotate("positive")   # item 0
        session.skip()                  # item 1
        session.flag(note="unclear")    # item 2
        session.annotate("negative")   # item 3
        session.annotate("neutral")    # item 4

        stats = session.stats()
        assert stats.labeled == 3
        assert stats.skipped == 1
        assert stats.flagged == 1
        assert stats.pending == 0
        assert session.is_complete


# ---------------------------------------------------------------------------
# Undo
# ---------------------------------------------------------------------------


class TestUndo:
    def test_undo_returns_to_previous(self, session):
        session.annotate("positive")
        assert session.current().text == "Terrible experience"
        undone = session.undo()
        assert undone.text == "I love this product"
        assert undone.status == AnnotationStatus.PENDING
        assert undone.label is None

    def test_undo_nothing(self, session):
        result = session.undo()
        assert result is None

    def test_multiple_undos(self, session):
        session.annotate("positive")
        session.annotate("negative")
        session.annotate("neutral")

        session.undo()
        assert session.current().text == "It is okay"
        session.undo()
        assert session.current().text == "Terrible experience"

    def test_undo_skip(self, session):
        session.skip()
        undone = session.undo()
        assert undone.status == AnnotationStatus.PENDING

    def test_redo_after_undo(self, session):
        session.annotate("positive")
        session.undo()
        item = session.annotate("negative")
        assert item.label == "negative"


# ---------------------------------------------------------------------------
# Goto
# ---------------------------------------------------------------------------


class TestGoto:
    def test_goto_valid(self, session):
        item = session.goto(3)
        assert item.text == "Amazing quality"

    def test_goto_out_of_range(self, session):
        with pytest.raises(IndexError):
            session.goto(99)

    def test_goto_negative(self, session):
        with pytest.raises(IndexError):
            session.goto(-1)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_initial_stats(self, session):
        stats = session.stats()
        assert stats.total == 5
        assert stats.pending == 5
        assert stats.labeled == 0
        assert stats.progress_pct == 0.0

    def test_progress_pct(self, session):
        session.annotate("positive")
        session.annotate("negative")
        stats = session.stats()
        assert stats.progress_pct == 40.0

    def test_label_distribution(self, session):
        session.annotate("positive")
        session.annotate("positive")
        session.annotate("negative")
        stats = session.stats()
        assert stats.labels_distribution["positive"] == 2
        assert stats.labels_distribution["negative"] == 1

    def test_elapsed_time(self, session):
        stats = session.stats()
        assert stats.elapsed_seconds >= 0.0

    def test_empty_session_progress(self):
        stats = AnnotationSessionStats()
        assert stats.progress_pct == 0.0


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_labeled_only(self, session):
        session.annotate("positive")
        session.skip()
        session.annotate("negative")
        df = session.export(include_all=False)
        assert len(df) == 2
        assert "positive" in df["label"].values

    def test_export_all(self, session):
        session.annotate("positive")
        session.skip()
        session.flag()
        df = session.export(include_all=True)
        assert len(df) == 5  # all items: 1 labeled + 1 skipped + 1 flagged + 2 pending

    def test_export_empty(self, session):
        df = session.export()
        assert len(df) == 0

    def test_export_for_training(self, session):
        session.annotate("positive")
        session.skip()
        session.annotate("negative")
        df = session.export_for_training()
        assert len(df) == 2
        assert list(df.columns) == ["text", "label"]

    def test_export_columns(self, session):
        session.annotate("positive", note="test note")
        df = session.export()
        assert "idx" in df.columns
        assert "text" in df.columns
        assert "label" in df.columns
        assert "status" in df.columns
        assert "annotator_note" in df.columns
        assert df.iloc[0]["annotator_note"] == "test note"


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


class TestIsComplete:
    def test_not_complete(self, session):
        assert not session.is_complete

    def test_complete_after_all_labeled(self, session):
        for _ in range(5):
            session.annotate("positive")
        assert session.is_complete

    def test_complete_with_mixed(self, session):
        session.annotate("positive")
        session.skip()
        session.flag()
        session.annotate("negative")
        session.skip()
        assert session.is_complete
