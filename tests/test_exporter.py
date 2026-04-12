"""Tests for socialscikit.qualikit.exporter."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from socialscikit.qualikit.coder import CodingResult
from socialscikit.qualikit.exporter import ExportBundle, Exporter
from socialscikit.qualikit.theme_definer import Theme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THEMES = [
    Theme(name="Financial Stress", description="Money worries"),
    Theme(name="Work-Life Balance", description="Balancing work and personal life"),
    Theme(name="Health Concerns", description="Physical or mental health"),
]

RESULTS = [
    CodingResult(
        text_id=0, text="I can't pay my bills.",
        themes=["Financial Stress"],
        confidences={"Financial Stress": 0.95},
        trigger_words={"Financial Stress": ["bills", "pay"]},
        reasoning="Discusses bill payment.",
    ),
    CodingResult(
        text_id=1, text="Too many hours at work, never see family.",
        themes=["Work-Life Balance", "Health Concerns"],
        confidences={"Work-Life Balance": 0.88, "Health Concerns": 0.65},
        trigger_words={"Work-Life Balance": ["hours", "work", "family"]},
        reasoning="Work-family conflict with stress.",
    ),
    CodingResult(
        text_id=2, text="I feel anxious all the time.",
        themes=["Health Concerns"],
        confidences={"Health Concerns": 0.92},
        trigger_words={"Health Concerns": ["anxious"]},
        reasoning="Anxiety mentioned.",
    ),
    CodingResult(
        text_id=3, text="Nice weather today.",
        themes=[],
        confidences={},
    ),
]


@pytest.fixture()
def exporter():
    return Exporter()


# ---------------------------------------------------------------------------
# Excerpts Table
# ---------------------------------------------------------------------------


class TestExcerptsTable:
    def test_build(self, exporter):
        df = exporter.build_excerpts_table(RESULTS, source_label="interview_1.csv")
        assert isinstance(df, pd.DataFrame)
        # 1 + 2 + 1 + 0 = 4 rows (one per text-theme pair)
        assert len(df) == 4
        assert "主题" in df.columns
        assert "文本段落" in df.columns
        assert "来源" in df.columns

    def test_with_review_actions(self, exporter):
        actions = {0: "accepted", 1: "edited", 2: "accepted"}
        df = exporter.build_excerpts_table(RESULTS, review_actions=actions)
        assert df.iloc[0]["审核状态"] == "accepted"

    def test_empty_results(self, exporter):
        df = exporter.build_excerpts_table([])
        assert len(df) == 0

    def test_no_themes_assigned(self, exporter):
        results = [CodingResult(text_id=0, text="test", themes=[])]
        df = exporter.build_excerpts_table(results)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Co-occurrence Matrix
# ---------------------------------------------------------------------------


class TestCooccurrenceMatrix:
    def test_build(self, exporter):
        df = exporter.build_cooccurrence_matrix(RESULTS, THEMES)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        # Financial Stress appears in 1 text
        assert df.loc["Financial Stress", "Financial Stress"] == 1
        # Health Concerns appears in 2 texts
        assert df.loc["Health Concerns", "Health Concerns"] == 2
        # Work-Life Balance and Health Concerns co-occur in 1 text
        assert df.loc["Work-Life Balance", "Health Concerns"] == 1
        assert df.loc["Health Concerns", "Work-Life Balance"] == 1

    def test_no_cooccurrence(self, exporter):
        results = [
            CodingResult(text_id=0, text="a", themes=["Financial Stress"]),
            CodingResult(text_id=1, text="b", themes=["Health Concerns"]),
        ]
        df = exporter.build_cooccurrence_matrix(results, THEMES)
        assert df.loc["Financial Stress", "Health Concerns"] == 0

    def test_empty(self, exporter):
        df = exporter.build_cooccurrence_matrix([], THEMES)
        assert df.sum().sum() == 0


# ---------------------------------------------------------------------------
# Memo
# ---------------------------------------------------------------------------


class TestMemo:
    def test_generate(self, exporter):
        memo = exporter.generate_memo(RESULTS, THEMES)
        assert "质性编码分析备忘录" in memo
        assert "Financial Stress" in memo
        assert "频率" in memo
        assert "典型引用" in memo
        assert "研究者备注" in memo

    def test_empty_results(self, exporter):
        memo = exporter.generate_memo([], THEMES)
        assert "0" in memo  # total texts = 0

    def test_quote_truncation(self, exporter):
        long_text = "x" * 300
        results = [CodingResult(
            text_id=0, text=long_text,
            themes=["Financial Stress"], confidences={"Financial Stress": 0.9},
        )]
        memo = exporter.generate_memo(results, THEMES)
        assert "..." in memo


# ---------------------------------------------------------------------------
# Full export
# ---------------------------------------------------------------------------


class TestExport:
    def test_full_export(self, exporter):
        bundle = exporter.export(RESULTS, THEMES)
        assert isinstance(bundle, ExportBundle)
        assert isinstance(bundle.excerpts_df, pd.DataFrame)
        assert isinstance(bundle.cooccurrence_df, pd.DataFrame)
        assert "备忘录" in bundle.memo_text


# ---------------------------------------------------------------------------
# Save files
# ---------------------------------------------------------------------------


class TestSaveFiles:
    def test_save_excel(self, exporter, tmp_path):
        bundle = exporter.export(RESULTS, THEMES)
        path = exporter.save_excel(bundle, str(tmp_path / "output.xlsx"))
        assert os.path.exists(path)
        # Read back
        xl = pd.read_excel(path, sheet_name="摘录表")
        assert len(xl) == 4

    def test_save_excel_auto_path(self, exporter):
        bundle = exporter.export(RESULTS, THEMES)
        path = exporter.save_excel(bundle)
        assert os.path.exists(path)
        os.unlink(path)

    def test_save_memo(self, exporter, tmp_path):
        bundle = exporter.export(RESULTS, THEMES)
        path = exporter.save_memo(bundle, str(tmp_path / "memo.md"))
        assert os.path.exists(path)
        with open(path) as f:
            content = f.read()
        assert "备忘录" in content

    def test_save_memo_auto_path(self, exporter):
        bundle = exporter.export(RESULTS, THEMES)
        path = exporter.save_memo(bundle)
        assert os.path.exists(path)
        os.unlink(path)
