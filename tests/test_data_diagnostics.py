"""Tests for socialscikit.core.data_diagnostics."""

import pandas as pd
import pytest

from socialscikit.core.data_diagnostics import (
    DiagnosticsReport,
    format_report,
    generate_diagnostics,
)


# ---------------------------------------------------------------------------
# Basic report generation
# ---------------------------------------------------------------------------


class TestGenerateDiagnostics:
    def test_basic_report(self):
        df = pd.DataFrame({
            "text": [
                "This is a fairly long sentence that should have more than ten words in it for testing",
                "Another text entry with enough words to pass the short text check easily",
                "Third piece of text that is also reasonably long for our diagnostics test",
            ],
            "label": ["positive", "negative", "positive"],
        })
        report = generate_diagnostics(df)
        assert report.n_rows == 3
        assert report.n_columns == 2
        assert report.text_col == "text"
        assert report.label_col == "label"
        assert report.n_empty_texts == 0
        assert report.text_length_stats is not None
        assert report.text_length_stats.min > 0
        assert report.label_distribution is not None
        assert report.label_distribution.counts["positive"] == 2

    def test_auto_detect_columns(self):
        df = pd.DataFrame({
            "content": ["hello world this is a test sentence", "another test"],
            "category": ["A", "B"],
        })
        report = generate_diagnostics(df)
        assert report.text_col == "content"
        assert report.label_col == "category"

    def test_explicit_columns(self):
        df = pd.DataFrame({
            "my_text": ["some text here", "more text"],
            "my_label": ["X", "Y"],
        })
        report = generate_diagnostics(df, text_col="my_text", label_col="my_label")
        assert report.text_col == "my_text"
        assert report.label_col == "my_label"
        assert report.label_distribution is not None
        assert report.label_distribution.counts["X"] == 1

    def test_no_text_col(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        report = generate_diagnostics(df)
        assert report.text_col is None
        assert report.text_length_stats is None

    def test_no_label_col(self):
        df = pd.DataFrame({"text": ["hello", "world"]})
        report = generate_diagnostics(df)
        assert report.label_col is None
        assert report.label_distribution is None

    def test_file_size(self):
        df = pd.DataFrame({"text": ["a"]})
        report = generate_diagnostics(df, file_size_kb=42.5)
        assert report.file_size_kb == 42.5


# ---------------------------------------------------------------------------
# Empty text handling
# ---------------------------------------------------------------------------


class TestEmptyTexts:
    def test_counts_empty_texts(self):
        df = pd.DataFrame({"text": ["hello", "", None, "world", "  "]})
        report = generate_diagnostics(df)
        assert report.n_empty_texts == 3

    def test_all_empty(self):
        df = pd.DataFrame({"text": ["", None, "  "]})
        report = generate_diagnostics(df)
        assert report.n_empty_texts == 3
        assert report.text_length_stats is None


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


class TestDuplicates:
    def test_no_duplicates(self):
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        report = generate_diagnostics(df)
        assert report.n_duplicate_rows == 0
        assert report.duplicate_rate == 0.0

    def test_with_duplicates(self):
        df = pd.DataFrame({"text": ["a", "a", "b"]})
        report = generate_diagnostics(df)
        assert report.n_duplicate_rows == 1
        assert report.duplicate_rate == pytest.approx(1 / 3, abs=0.01)


# ---------------------------------------------------------------------------
# Label distribution
# ---------------------------------------------------------------------------


class TestLabelDistribution:
    def test_proportions(self):
        df = pd.DataFrame({
            "text": ["t1", "t2", "t3", "t4"],
            "label": ["A", "A", "A", "B"],
        })
        report = generate_diagnostics(df)
        assert report.label_distribution is not None
        assert report.label_distribution.proportions["A"] == 0.75
        assert report.label_distribution.proportions["B"] == 0.25


# ---------------------------------------------------------------------------
# Bias warnings
# ---------------------------------------------------------------------------


class TestBiasWarnings:
    def test_same_day_warning(self):
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(20)],
            "date": ["2024-03-15"] * 20,
        })
        report = generate_diagnostics(df, text_col="text")
        assert any("同一天" in w for w in report.bias_warnings)

    def test_one_week_warning(self):
        dates = [f"2024-03-{10 + i % 5}" for i in range(20)]
        df = pd.DataFrame({"text": [f"text {i}" for i in range(20)], "date": dates})
        report = generate_diagnostics(df, text_col="text")
        assert any("一周" in w for w in report.bias_warnings)

    def test_low_uniqueness_warning(self):
        df = pd.DataFrame({"text": ["same text"] * 10})
        report = generate_diagnostics(df)
        assert any("去重" in w for w in report.bias_warnings)

    def test_no_bias_warning(self):
        df = pd.DataFrame({
            "text": [f"unique text number {i} with enough variation" for i in range(20)],
        })
        report = generate_diagnostics(df)
        assert len(report.bias_warnings) == 0


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_format_output(self):
        df = pd.DataFrame({
            "text": [
                "This is a test sentence with more than ten words for length check",
                "Another sentence that is reasonably long for testing purposes here",
            ],
            "label": ["A", "B"],
        })
        report = generate_diagnostics(df, file_size_kb=10.5)
        output = format_report(report)
        assert "数据诊断报告" in output
        assert "2 行" in output
        assert "10.5 KB" in output
        assert "标签分布" in output

    def test_format_minimal(self):
        report = DiagnosticsReport(n_rows=0, n_columns=0)
        output = format_report(report)
        assert "0 行" in output
