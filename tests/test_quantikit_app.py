"""Tests for socialscikit.ui.quantikit_app.

Tests the UI callback functions without launching the Gradio server.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from socialscikit.ui.quantikit_app import (
    _apply_fixes,
    _build_template_prompt,
    _create_annotation_session,
    _annotate_item,
    _skip_item,
    _flag_item,
    _undo_annotation,
    _ensure_text_placeholder,
    _export_annotations,
    _evaluate_prompt,
    _evaluate_results,
    _export_results,
    _extract_and_recommend,
    _extract_label_robust,
    _generate_smart_prompt,
    _insert_text_placeholder,
    _load_and_validate,
    _download_template,
    _match_label,
    _parse_examples_flexible,
    _parse_label_definitions,
    _run_classification,
    _test_variants,
    create_app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_csv(tmp_path):
    """Create a minimal CSV file for testing."""
    path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "text": [
            "I love this product, it is amazing and wonderful",
            "Terrible experience, would not recommend at all",
            "It is okay, nothing special about it really",
            "Amazing quality and great service overall",
            "Worst purchase ever made in my entire life",
        ],
        "label": ["positive", "negative", "neutral", "positive", "negative"],
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_df():
    return pd.DataFrame({
        "text": [
            "I love this product, it is amazing and wonderful",
            "Terrible experience, would not recommend at all",
            "It is okay, nothing special about it really",
            "Amazing quality and great service overall",
            "Worst purchase ever made in my entire life",
        ],
        "label": ["positive", "negative", "neutral", "positive", "negative"],
    })


class FakeFile:
    """Mimics a Gradio file object."""
    def __init__(self, path):
        self.name = str(path)


# ---------------------------------------------------------------------------
# Test: Robust label extraction
# ---------------------------------------------------------------------------


class TestExtractLabelRobust:
    """Test _extract_label_robust handles all common LLM response formats."""

    LABELS = ["W", "N", "R"]

    def test_plain_json(self):
        assert _extract_label_robust('{"label": "W", "confidence": 0.9}', self.LABELS) == "W"

    def test_json_with_markdown_fencing(self):
        raw = '```json\n{"label": "R", "confidence": 0.85}\n```'
        assert _extract_label_robust(raw, self.LABELS) == "R"

    def test_json_with_plain_fencing(self):
        raw = '```\n{"label": "N", "confidence": 0.7}\n```'
        assert _extract_label_robust(raw, self.LABELS) == "N"

    def test_bare_label(self):
        assert _extract_label_robust("W", self.LABELS) == "W"

    def test_bare_label_lowercase(self):
        assert _extract_label_robust("w", self.LABELS) == "W"

    def test_quoted_label(self):
        assert _extract_label_robust('"W"', self.LABELS) == "W"

    def test_label_in_sentence(self):
        raw = "该段落属于 W 类别，因为涉及社会福利。"
        assert _extract_label_robust(raw, self.LABELS) == "W"

    def test_label_with_chinese_expansion(self):
        """LLM returns 'W（福利）' — should map to 'W'."""
        raw = '{"label": "W（福利）", "confidence": 0.8}'
        assert _extract_label_robust(raw, self.LABELS) == "W"

    def test_multiline_with_explanation(self):
        raw = '{"label": "R", "confidence": 0.75}\n\n该段落涉及公民权利保障。'
        assert _extract_label_robust(raw, self.LABELS) == "R"

    def test_json_embedded_in_text(self):
        raw = '根据分析：{"label": "N", "confidence": 0.6}'
        assert _extract_label_robust(raw, self.LABELS) == "N"

    def test_unknown_label_returned_as_is(self):
        """If LLM returns something unmatchable, return it for debugging."""
        result = _extract_label_robust("completely_unknown", self.LABELS)
        assert result == "completely_unknown"


class TestMatchLabel:
    """Test _match_label handles variations in LLM label output."""

    LABELS = ["positive", "negative", "neutral"]

    def test_exact_match(self):
        assert _match_label("positive", self.LABELS) == "positive"

    def test_case_insensitive(self):
        assert _match_label("POSITIVE", self.LABELS) == "positive"

    def test_label_at_start(self):
        assert _match_label("positive (high confidence)", self.LABELS) == "positive"

    def test_label_embedded(self):
        assert _match_label("The sentiment is positive", self.LABELS) == "positive"

    def test_no_match(self):
        assert _match_label("xyz", self.LABELS) == "xyz"


class TestParseLabelDefinitions:
    """Test _parse_label_definitions handles Chinese and English colons."""

    def test_chinese_colon(self):
        defs = "W：福利相关内容\nN：中性内容\nR：权利相关内容"
        result = _parse_label_definitions(defs, ["W", "N", "R"])
        assert result == {"W": "福利相关内容", "N": "中性内容", "R": "权利相关内容"}

    def test_english_colon(self):
        defs = "W: welfare\nN: neutral\nR: rights"
        result = _parse_label_definitions(defs, ["W", "N", "R"])
        assert result == {"W": "welfare", "N": "neutral", "R": "rights"}

    def test_mixed_colons(self):
        defs = "W：福利\nN: neutral\nR = rights"
        result = _parse_label_definitions(defs, ["W", "N", "R"])
        assert len(result) == 3
        assert result["W"] == "福利"
        assert result["N"] == "neutral"
        assert result["R"] == "rights"

    def test_with_bullet_prefix(self):
        defs = "- W：福利\n- N：中性\n- R：权利"
        result = _parse_label_definitions(defs, ["W", "N", "R"])
        assert len(result) == 3
        assert result["W"] == "福利"

    def test_empty_returns_defaults(self):
        result = _parse_label_definitions("", ["W", "N", "R"])
        assert "W" in result
        assert "属于 W 类别的文本" in result["W"]

    def test_none_returns_defaults(self):
        result = _parse_label_definitions(None, ["pos", "neg"])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Test: App creation
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_creates_blocks(self):
        app = create_app()
        assert app is not None


# ---------------------------------------------------------------------------
# Test: Load & Validate
# ---------------------------------------------------------------------------


class TestLoadAndValidate:
    def test_no_file(self):
        df, summary, issues, preview = _load_and_validate(None)
        assert df is None
        assert "上传" in summary

    def test_valid_csv(self, sample_csv):
        df, summary, issues, preview = _load_and_validate(FakeFile(sample_csv))
        assert df is not None
        assert len(df) == 5
        assert preview is not None
        assert len(preview) == 5

    def test_invalid_file(self, tmp_path):
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("")
        df, summary, issues, preview = _load_and_validate(FakeFile(bad_file))
        assert df is None
        assert "失败" in summary or "上传" in summary


# ---------------------------------------------------------------------------
# Test: Apply fixes
# ---------------------------------------------------------------------------


class TestApplyFixes:
    def test_no_data(self):
        df, msg = _apply_fixes(None)
        assert df is None
        assert "没有" in msg

    def test_with_data(self, sample_df):
        df, msg = _apply_fixes(sample_df)
        assert df is not None
        assert "修复" in msg


# ---------------------------------------------------------------------------
# Test: Extract & Recommend (with budget)
# ---------------------------------------------------------------------------


class TestExtractAndRecommend:
    def test_no_data(self):
        feat, rec, budget, plot, tbl = _extract_and_recommend(
            None, "sentiment", 2, 0.8, "medium", "text", "label",
        )
        assert "上传" in feat
        assert plot is None

    def test_with_data(self, sample_df):
        feat, rec, budget, plot, tbl = _extract_and_recommend(
            sample_df, "sentiment", 3, 0.8, "medium", "text", "label",
        )
        assert "任务特征" in feat
        assert "方法推荐" in rec
        assert "标注预算" in budget
        assert "推荐标注量" in budget
        assert plot is not None
        assert tbl is not None
        assert "标注量" in tbl.columns

    def test_without_labels(self, sample_df):
        df = sample_df.drop(columns=["label"])
        feat, rec, budget, plot, tbl = _extract_and_recommend(
            df, "sentiment", 2, 0.8, "low", "text", "",
        )
        assert "任务特征" in feat


# ---------------------------------------------------------------------------
# Test: Annotation session
# ---------------------------------------------------------------------------


class TestAnnotation:
    def test_create_session(self, sample_df):
        session, stats, text, idx, msg = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        assert session is not None
        assert "总计：5" in stats
        assert text != ""

    def test_create_session_no_data(self):
        session, stats, text, idx, msg = _create_annotation_session(
            None, "text", "", "pos, neg", False,
        )
        assert session is None
        assert "上传" in stats

    def test_annotate_item(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        session, stats, text, idx, msg = _annotate_item(session, "positive")
        assert "已标注：1" in stats

    def test_annotate_empty_label(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        session, stats, text, idx, msg = _annotate_item(session, "  ")
        assert "请输入" in msg

    def test_skip_item(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        session, stats, text, idx, msg = _skip_item(session)
        assert "已跳过：1" in stats

    def test_flag_item(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        session, stats, text, idx, msg = _flag_item(session, "ambiguous")
        assert "已标记：1" in stats

    def test_undo(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        _annotate_item(session, "positive")
        session, stats, text, idx, msg = _undo_annotation(session)
        assert "撤销" in msg
        assert "已标注：0" in stats

    def test_undo_nothing(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        session, stats, text, idx, msg = _undo_annotation(session)
        assert "没有" in msg

    def test_export_annotations(self, sample_df):
        session, _, _, _, _ = _create_annotation_session(
            sample_df, "text", "", "positive, negative, neutral", False,
        )
        _annotate_item(session, "positive")
        _annotate_item(session, "negative")
        df, msg = _export_annotations(session, False)
        assert df is not None
        assert len(df) == 2

    def test_export_no_session(self):
        df, msg = _export_annotations(None, False)
        assert df is None


# ---------------------------------------------------------------------------
# Test: Prompt Builder
# ---------------------------------------------------------------------------


class TestGenerateSmartPrompt:
    def test_basic_no_api(self):
        """Without API key, falls back to template assembly."""
        result = _generate_smart_prompt(
            "Classify sentiment",
            "positive, negative, neutral",
            "positive: good\nnegative: bad\nneutral: meh",
            "", "", "openai", "gpt-4o-mini", "",
        )
        assert "positive" in result
        assert "negative" in result

    def test_no_labels(self):
        result = _generate_smart_prompt("", "", "", "", "", "openai", "gpt-4o-mini", "")
        assert "请先" in result and "类别" in result

    def test_with_task_desc_no_api(self):
        result = _generate_smart_prompt(
            "对政策文档分类",
            "W, N, R",
            "W: 福利\nN: 中性\nR: 权利",
            "", "", "openai", "gpt-4o-mini", "",
        )
        assert "W" in result

    def test_auto_definitions(self):
        result = _generate_smart_prompt(
            "", "positive, negative", "", "", "",
            "openai", "gpt-4o-mini", "",
        )
        assert "positive" in result


class TestEvaluatePrompt:
    def test_no_prompt(self):
        result = _evaluate_prompt("", "W, N, R", "openai", "gpt-4o-mini", "")
        assert "请先" in result

    def test_missing_text_placeholder(self):
        result = _evaluate_prompt(
            "请分类以下文本。",
            "W, N, R", "openai", "gpt-4o-mini", "",
        )
        assert "{text}" in result  # warns about missing placeholder

    def test_good_prompt(self):
        result = _evaluate_prompt(
            "请分类：\n\n待分类文本：{text}\n\n请输出 W/N/R。",
            "W, N, R", "openai", "gpt-4o-mini", "",
        )
        assert "通过" in result or "检查" in result

    def test_text_after_output(self):
        result = _evaluate_prompt(
            "请输出 JSON 格式。\n待分类文本：{text}",
            "W, N, R", "openai", "gpt-4o-mini", "",
        )
        assert "输出指令之后" in result or "之前" in result


class TestInsertTextPlaceholder:
    def test_inserts_before_output(self):
        prompt = "请分类。\n请输出 JSON 格式。"
        result = _insert_text_placeholder(prompt)
        assert "{text}" in result
        text_pos = result.index("{text}")
        output_pos = result.index("请输出")
        assert text_pos < output_pos

    def test_inserts_at_end_if_no_output(self):
        prompt = "请分类以下文本。"
        result = _insert_text_placeholder(prompt)
        assert "{text}" in result


class TestParseExamplesFlexible:
    """Test the flexible example parser that accepts JSON, line-by-line, etc."""

    def test_none_returns_none(self):
        assert _parse_examples_flexible(None, ["W", "R"]) is None

    def test_empty_string_returns_none(self):
        assert _parse_examples_flexible("", ["W", "R"]) is None
        assert _parse_examples_flexible("   ", ["W", "R"]) is None

    def test_json_format(self):
        raw = '{"W": ["福利政策文本"], "R": ["权利相关文本"]}'
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is not None
        assert "W" in result
        assert "R" in result
        assert result["W"] == ["福利政策文本"]

    def test_json_single_strings(self):
        """JSON with string values instead of lists."""
        raw = '{"W": "福利", "R": "权利"}'
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is not None
        assert result["W"] == ["福利"]

    def test_line_by_line_chinese_colon(self):
        raw = "W：加大低保补贴\nR：保障公民知情权"
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is not None
        assert "W" in result
        assert result["W"] == ["加大低保补贴"]
        assert result["R"] == ["保障公民知情权"]

    def test_line_by_line_english_colon(self):
        raw = "positive: Great product!\nnegative: Terrible."
        result = _parse_examples_flexible(raw, ["positive", "negative"])
        assert result is not None
        assert "positive" in result
        assert "negative" in result

    def test_bullet_format(self):
        raw = "- W：福利文本\n- R：权利文本"
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is not None
        assert "W" in result

    def test_multiple_examples_per_label(self):
        raw = "W：文本1\nW：文本2\nR：文本3"
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is not None
        assert len(result["W"]) == 2
        assert len(result["R"]) == 1

    def test_unrecognized_labels_skipped(self):
        raw = "X：不存在的标签\nW：存在的标签"
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is not None
        assert "W" in result
        assert "X" not in result

    def test_gibberish_returns_none(self):
        raw = "some random text without any label structure"
        result = _parse_examples_flexible(raw, ["W", "R"])
        assert result is None


class TestEnsureTextPlaceholder:
    """Test the {text} placement validator/fixer."""

    def test_already_correct(self):
        prompt = "请分类：\n待分类文本：{text}\n请输出标签。"
        result = _ensure_text_placeholder(prompt)
        assert "{text}" in result

    def test_missing_placeholder_gets_inserted(self):
        prompt = "请分类。\n请输出 JSON 格式。"
        result = _ensure_text_placeholder(prompt)
        assert "{text}" in result

    def test_placeholder_after_output_gets_fixed(self):
        prompt = "请输出 JSON 格式：\n{text}"
        result = _ensure_text_placeholder(prompt)
        assert "{text}" in result
        text_pos = result.index("{text}")
        output_pos = result.lower().index("json")
        assert text_pos < output_pos

    def test_no_output_instruction(self):
        prompt = "请分类以下文本。"
        result = _ensure_text_placeholder(prompt)
        assert "{text}" in result


class TestBuildTemplatePrompt:
    """Test the template-based prompt builder (no LLM needed)."""

    def test_basic(self):
        result = _build_template_prompt(
            task_desc="对情感分类",
            labels=["positive", "negative"],
            label_definitions={"positive": "正面", "negative": "负面"},
            inclusion_examples=None,
            exclusion_examples=None,
        )
        assert "{text}" in result
        assert "positive" in result
        assert "negative" in result
        # {text} must be before output instruction
        text_pos = result.index("{text}")
        assert "请只输出" in result
        output_pos = result.index("请只输出")
        assert text_pos < output_pos

    def test_with_examples(self):
        result = _build_template_prompt(
            task_desc=None,
            labels=["W", "R"],
            label_definitions={"W": "福利", "R": "权利"},
            inclusion_examples={"W": ["低保补贴"], "R": ["知情权"]},
            exclusion_examples={"W": ["虽然提到保障但是权利"]},
        )
        assert "低保补贴" in result
        assert "知情权" in result
        assert "虽然提到保障但是权利" in result
        assert "{text}" in result

    def test_no_task_desc_uses_default(self):
        result = _build_template_prompt(
            task_desc="",
            labels=["A", "B"],
            label_definitions={"A": "类A", "B": "类B"},
            inclusion_examples=None,
            exclusion_examples=None,
        )
        assert "请将文本分类" in result
        assert "A" in result

    def test_text_before_output(self):
        """The fundamental invariant: {text} always before output instruction."""
        result = _build_template_prompt(
            task_desc="分类",
            labels=["X", "Y"],
            label_definitions={"X": "X类", "Y": "Y类"},
            inclusion_examples=None,
            exclusion_examples=None,
        )
        text_pos = result.index("{text}")
        output_pos = result.index("请只输出")
        assert text_pos < output_pos


class TestTestVariants:
    def test_no_data(self):
        result, detail = _test_variants(
            None, "text", "label",
            "prompt", "", "", "",
            "positive, negative", "openai", "gpt-4o-mini", "",
        )
        assert "上传" in result
        assert detail is None

    def test_no_api_key(self, sample_df):
        result, detail = _test_variants(
            sample_df, "text", "label",
            "prompt", "", "", "",
            "positive, negative", "openai", "gpt-4o-mini", "",
        )
        assert "API Key" in result
        assert detail is None

    def test_no_label_col(self, sample_df):
        result, detail = _test_variants(
            sample_df, "text", "",
            "prompt", "", "", "",
            "positive, negative", "openai", "gpt-4o-mini", "sk-test",
        )
        assert "标签列" in result
        assert detail is None


class TestRunClassification:
    def test_no_data(self):
        summary, df = _run_classification(
            None, "text", "label", "positive, negative",
            "my prompt", "openai", "gpt-4o-mini", "",
        )
        assert "上传" in summary

    def test_no_api_key(self, sample_df):
        summary, df = _run_classification(
            sample_df, "text", "label", "positive, negative",
            "my prompt", "openai", "gpt-4o-mini", "",
        )
        assert "API Key" in summary

    def test_no_labels(self, sample_df):
        summary, df = _run_classification(
            sample_df, "text", "label", "",
            "my prompt", "openai", "gpt-4o-mini", "sk-test",
        )
        assert "类别" in summary

    def test_all_labeled_skips(self, sample_df):
        """When all rows have labels, should report nothing to classify."""
        summary, df = _run_classification(
            sample_df, "text", "label", "positive, negative",
            "my prompt", "openai", "gpt-4o-mini", "sk-test",
        )
        assert "已有标签" in summary


# ---------------------------------------------------------------------------
# Test: Evaluate
# ---------------------------------------------------------------------------


class TestEvaluate:
    def test_no_data(self):
        result = _evaluate_results(None, None, "label")
        assert "分类" in result

    def test_no_label_col(self):
        df = pd.DataFrame({"text": ["a"]})
        result_df = pd.DataFrame({"predicted_label": ["pos"]})
        result = _evaluate_results(result_df, df, "label")
        assert "未找到" in result

    def test_evaluation(self, sample_df):
        result_df = pd.DataFrame({
            "predicted_label": ["positive", "negative", "neutral", "positive", "positive"],
        })
        result = _evaluate_results(result_df, sample_df, "label")
        assert "F1" in result or "Accuracy" in result


# ---------------------------------------------------------------------------
# Test: Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_no_data(self):
        result = _export_results(None)
        assert result is None

    def test_export(self):
        df = pd.DataFrame({"text": ["a"], "predicted_label": ["pos"]})
        path = _export_results(df)
        assert path is not None
        assert os.path.exists(path)
        loaded = pd.read_csv(path)
        assert len(loaded) == 1
        os.unlink(path)


# ---------------------------------------------------------------------------
# Test: Template
# ---------------------------------------------------------------------------


class TestTemplate:
    def test_download_template(self):
        path = _download_template()
        assert os.path.exists(path)
        assert "quantikit" in path.lower() or "template" in path.lower()
