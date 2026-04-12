"""Tests for socialscikit.ui.qualikit_app."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from socialscikit.ui.qualikit_app import (
    _apply_fixes,
    _check_theme_overlap,
    _deident_accept_all,
    _deident_accept_high,
    _load_and_validate,
    _run_deident,
    _suggest_themes,
    create_app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "text": [
            "Dr. Sarah Chen from Stanford University discussed the findings in detail with the research team",
            "The participant mentioned feeling stressed about finances and their ability to manage daily expenses",
            "Community engagement has improved since the new program was introduced last year in September",
            "I worry about my health insurance costs and whether I can afford the medications I need",
            "The interview took place at john@example.com office near downtown Portland Oregon",
        ],
        "speaker_id": ["S1", "S2", "S3", "S2", "S1"],
    })


@pytest.fixture()
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "test.csv"
    sample_df.to_csv(path, index=False)
    return path


class FakeFile:
    def __init__(self, path):
        self.name = str(path)


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_creates_blocks(self):
        app = create_app()
        assert app is not None


# ---------------------------------------------------------------------------
# Upload & Validate
# ---------------------------------------------------------------------------


class TestUpload:
    def test_no_file(self):
        df, summary, issues, preview = _load_and_validate(None)
        assert df is None

    def test_valid_csv(self, sample_csv):
        df, summary, issues, preview = _load_and_validate(FakeFile(sample_csv))
        assert df is not None
        assert len(df) == 5


# ---------------------------------------------------------------------------
# De-identification
# ---------------------------------------------------------------------------


class TestDeident:
    def test_run_deident(self, sample_df):
        session, stats, log_df, result = _run_deident(
            sample_df, "text", "EMAIL, PHONE, URL", "placeholder",
        )
        assert session is not None
        assert "检出" in stats or "未检" in stats

    def test_no_data(self):
        session, stats, log_df, result = _run_deident(None, "text", "EMAIL", "placeholder")
        assert session is None

    def test_bad_column(self, sample_df):
        session, stats, log_df, result = _run_deident(sample_df, "nonexistent", "EMAIL", "placeholder")
        assert session is None

    def test_accept_all(self, sample_df):
        session, _, _, _ = _run_deident(sample_df, "text", "EMAIL", "placeholder")
        if session is not None:
            session, msg = _deident_accept_all(session)
            assert "接受" in msg

    def test_accept_high(self, sample_df):
        session, _, _, _ = _run_deident(sample_df, "text", "EMAIL", "placeholder")
        if session is not None:
            session, msg = _deident_accept_high(session)
            assert "接受" in msg


# ---------------------------------------------------------------------------
# Theme suggestion
# ---------------------------------------------------------------------------


class TestThemes:
    def test_suggest_tfidf(self, sample_df):
        session, summary, table = _suggest_themes(
            sample_df, "text", 3, "tfidf", "openai", "", "",
        )
        assert session is not None
        assert "主题" in summary

    def test_no_data(self):
        session, summary, table = _suggest_themes(None, "text", 3, "tfidf", "", "", "")
        assert session is None

    def test_overlap_check(self, sample_df):
        session, _, _ = _suggest_themes(sample_df, "text", 3, "tfidf", "", "", "")
        result = _check_theme_overlap(session)
        assert isinstance(result, str)

    def test_overlap_no_session(self):
        result = _check_theme_overlap(None)
        assert "请先" in result
