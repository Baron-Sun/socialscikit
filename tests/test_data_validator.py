"""Tests for socialscikit.core.data_validator."""

import pandas as pd
import pytest

from socialscikit.core.data_validator import DataValidator, apply_auto_fixes


@pytest.fixture()
def validator():
    return DataValidator()


# ---------------------------------------------------------------------------
# QuantiKit mode
# ---------------------------------------------------------------------------


class TestQuantiKitValidation:
    def test_valid_data(self, validator):
        df = pd.DataFrame({
            "text": ["This is good", "This is bad", "Neutral text"],
            "label": ["positive", "negative", "neutral"],
        })
        report = validator.validate(df, mode="quantikit")
        assert report.is_valid
        assert report.suggested_text_col == "text"
        assert report.suggested_label_col == "label"
        assert report.n_rows == 3
        assert report.n_usable_rows == 3

    def test_no_text_column(self, validator):
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        report = validator.validate(df, mode="quantikit")
        assert not report.is_valid
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 1
        assert "文本列" in errors[0].message

    def test_empty_text_rows(self, validator):
        df = pd.DataFrame({
            "text": ["hello", "", None, "world"],
            "label": ["a", "b", "c", "d"],
        })
        report = validator.validate(df, mode="quantikit")
        assert report.is_valid  # warnings don't block
        warnings = [i for i in report.issues if i.severity == "warning" and "空文本" in i.message]
        assert len(warnings) == 1
        assert warnings[0].auto_fix_available
        assert report.n_usable_rows == 2

    def test_no_label_column(self, validator):
        df = pd.DataFrame({"text": ["hello", "world"]})
        report = validator.validate(df, mode="quantikit")
        assert report.is_valid
        infos = [i for i in report.issues if i.severity == "info" and "标签列" in i.message]
        assert len(infos) == 1

    def test_imbalanced_labels(self, validator):
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(60)],
            "label": ["majority"] * 50 + ["minority"] * 10,
        })
        report = validator.validate(df, mode="quantikit")
        # 50:10 = 5:1, exactly at boundary — depends on > 5
        assert report.is_valid

    def test_very_imbalanced_labels(self, validator):
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(61)],
            "label": ["majority"] * 55 + ["minority"] * 6,
        })
        report = validator.validate(df, mode="quantikit")
        warnings = [i for i in report.issues if "不均衡" in i.message]
        assert len(warnings) == 1

    def test_short_text_warning(self, validator):
        df = pd.DataFrame({
            "text": ["hi", "ok", "no", "yes", "go"],
        })
        report = validator.validate(df, mode="quantikit")
        warnings = [i for i in report.issues if "较短" in i.message]
        assert len(warnings) == 1

    def test_guesses_alternative_column_names(self, validator):
        df = pd.DataFrame({
            "content": ["some text", "more text"],
            "category": ["A", "B"],
        })
        report = validator.validate(df, mode="quantikit")
        assert report.suggested_text_col == "content"
        assert report.suggested_label_col == "category"


# ---------------------------------------------------------------------------
# QualiKit mode
# ---------------------------------------------------------------------------


class TestQualiKitValidation:
    def test_valid_qualikit_data(self, validator):
        df = pd.DataFrame({
            "text": [
                "I think the most important thing about this experience was how it changed my perspective on work life balance and personal growth",
                "When I started the program I was very skeptical but over time I realized that the community support made a huge difference in my daily routine",
            ],
            "speaker_id": ["S01", "S02"],
        })
        report = validator.validate(df, mode="qualikit")
        assert report.is_valid
        assert len(report.issues) == 0

    def test_no_speaker_column(self, validator):
        df = pd.DataFrame({"text": ["segment one", "segment two"]})
        report = validator.validate(df, mode="qualikit")
        warnings = [i for i in report.issues if "说话人" in i.message]
        assert len(warnings) == 1

    def test_pii_detection_email(self, validator):
        df = pd.DataFrame({
            "text": ["Contact me at john@example.com", "No PII here"],
            "speaker_id": ["S01", "S02"],
        })
        report = validator.validate(df, mode="qualikit")
        warnings = [i for i in report.issues if "个人信息" in i.message]
        assert len(warnings) == 1
        assert "邮箱" in warnings[0].message

    def test_pii_detection_phone(self, validator):
        df = pd.DataFrame({
            "text": ["Call me at 555-123-4567", "No PII here"],
            "speaker_id": ["S01", "S02"],
        })
        report = validator.validate(df, mode="qualikit")
        warnings = [i for i in report.issues if "个人信息" in i.message]
        assert len(warnings) == 1
        assert "电话" in warnings[0].message


# ---------------------------------------------------------------------------
# Auto-fix
# ---------------------------------------------------------------------------


class TestAutoFix:
    def test_removes_empty_text_rows(self, validator):
        df = pd.DataFrame({
            "text": ["hello", "", None, "world", "  "],
            "label": ["a", "b", "c", "d", "e"],
        })
        report = validator.validate(df, mode="quantikit")
        fixed = apply_auto_fixes(df, report)
        assert len(fixed) == 2
        assert list(fixed["text"]) == ["hello", "world"]

    def test_no_fix_needed(self, validator):
        df = pd.DataFrame({"text": ["hello", "world"]})
        report = validator.validate(df, mode="quantikit")
        fixed = apply_auto_fixes(df, report)
        assert len(fixed) == 2
