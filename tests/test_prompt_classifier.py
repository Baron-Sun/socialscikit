"""Tests for socialscikit.quantikit.prompt_classifier."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from socialscikit.core.llm_client import LLMResponse
from socialscikit.quantikit.prompt_classifier import (
    BatchClassificationReport,
    ClassificationResult,
    PromptClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABELS = ["positive", "negative", "neutral"]


def _mock_llm_response(label: str, confidence: float = 0.9) -> LLMResponse:
    return LLMResponse(text=json.dumps({"label": label, "confidence": confidence}))


def _make_mock_llm(responses: list[LLMResponse]) -> MagicMock:
    mock = MagicMock()
    mock.complete.side_effect = responses
    return mock


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_json_response(self):
        raw = '{"label": "positive", "confidence": 0.95}'
        result = PromptClassifier._parse_response(raw, set(LABELS))
        assert result["label"] == "positive"
        assert result["confidence"] == 0.95

    def test_json_no_confidence(self):
        raw = '{"label": "negative"}'
        result = PromptClassifier._parse_response(raw, set(LABELS))
        assert result["label"] == "negative"
        assert result["confidence"] is None

    def test_json_embedded_in_text(self):
        raw = 'The answer is {"label": "neutral", "confidence": 0.8} based on analysis.'
        result = PromptClassifier._parse_response(raw, set(LABELS))
        assert result["label"] == "neutral"
        assert result["confidence"] == 0.8

    def test_plain_label(self):
        raw = "positive"
        result = PromptClassifier._parse_response(raw, set(LABELS))
        assert result["label"] == "positive"

    def test_label_in_sentence(self):
        raw = "I think this is positive because it expresses happiness."
        result = PromptClassifier._parse_response(raw, set(LABELS))
        assert result["label"] == "positive"

    def test_case_insensitive_match(self):
        raw = "NEGATIVE"
        result = PromptClassifier._parse_response(raw, set(LABELS))
        assert result["label"] == "negative"

    def test_unknown_response(self):
        raw = "I cannot determine the category."
        result = PromptClassifier._parse_response(raw, {"cat_a", "cat_b"})
        # Falls back to raw text truncated
        assert result["confidence"] is None


# ---------------------------------------------------------------------------
# Zero-shot classification (sync)
# ---------------------------------------------------------------------------


class TestZeroShotClassify:
    def test_basic_classification(self):
        mock_llm = _make_mock_llm([
            _mock_llm_response("positive", 0.95),
            _mock_llm_response("negative", 0.88),
        ])
        classifier = PromptClassifier(mock_llm)
        report = classifier.classify(
            texts=["I love this!", "Terrible product."],
            labels=LABELS,
        )
        assert report.n_total == 2
        assert report.n_classified == 2
        assert report.n_failed == 0
        assert report.results[0].predicted_label == "positive"
        assert report.results[1].predicted_label == "negative"
        assert report.results[0].confidence == 0.95

    def test_with_label_definitions(self):
        mock_llm = _make_mock_llm([_mock_llm_response("neutral")])
        classifier = PromptClassifier(mock_llm)
        report = classifier.classify(
            texts=["It arrived on time."],
            labels=LABELS,
            label_definitions={
                "positive": "Expresses satisfaction",
                "negative": "Expresses dissatisfaction",
                "neutral": "Neither positive nor negative",
            },
        )
        assert report.results[0].predicted_label == "neutral"

    def test_handles_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = RuntimeError("API error")
        classifier = PromptClassifier(mock_llm)
        report = classifier.classify(texts=["test"], labels=LABELS)
        assert report.n_failed == 1
        assert report.results[0].predicted_label == "UNKNOWN"

    def test_label_distribution(self):
        mock_llm = _make_mock_llm([
            _mock_llm_response("positive"),
            _mock_llm_response("positive"),
            _mock_llm_response("negative"),
        ])
        classifier = PromptClassifier(mock_llm)
        report = classifier.classify(
            texts=["a", "b", "c"], labels=LABELS,
        )
        assert report.label_distribution["positive"] == 2
        assert report.label_distribution["negative"] == 1


# ---------------------------------------------------------------------------
# Few-shot classification (sync)
# ---------------------------------------------------------------------------


class TestFewShotClassify:
    def test_with_examples(self):
        mock_llm = _make_mock_llm([_mock_llm_response("positive", 0.92)])
        classifier = PromptClassifier(mock_llm)
        report = classifier.classify(
            texts=["Great experience!"],
            labels=LABELS,
            examples={
                "positive": ["Love it!", "Excellent!"],
                "negative": ["Hate it.", "Awful."],
                "neutral": ["It's okay."],
            },
        )
        assert report.results[0].predicted_label == "positive"
        # Verify few-shot prompt was used (check the call args)
        call_args = mock_llm.complete.call_args
        prompt_used = call_args[0][0]
        assert "Examples" in prompt_used
        assert "Love it!" in prompt_used


# ---------------------------------------------------------------------------
# Custom prompt
# ---------------------------------------------------------------------------


class TestCustomPrompt:
    def test_custom_prompt_template(self):
        mock_llm = _make_mock_llm([_mock_llm_response("negative")])
        classifier = PromptClassifier(
            mock_llm,
            custom_prompt="Classify into {labels}: {text}",
            custom_system="You are a classifier.",
        )
        report = classifier.classify(texts=["Bad product"], labels=LABELS)
        assert report.results[0].predicted_label == "negative"
        call_args = mock_llm.complete.call_args
        assert call_args.kwargs["system"] == "You are a classifier."


# ---------------------------------------------------------------------------
# Async classification
# ---------------------------------------------------------------------------


class TestAsyncClassify:
    def test_async_basic(self):
        mock_llm = MagicMock()
        mock_llm.batch_complete = AsyncMock(side_effect=[
            [_mock_llm_response("positive")],
            [_mock_llm_response("negative")],
        ])
        classifier = PromptClassifier(mock_llm)
        report = asyncio.get_event_loop().run_until_complete(
            classifier.classify_async(
                texts=["good", "bad"],
                labels=LABELS,
            )
        )
        assert report.n_total == 2
        assert report.results[0].predicted_label == "positive"
        assert report.results[1].predicted_label == "negative"

    def test_async_with_error(self):
        mock_llm = MagicMock()
        mock_llm.batch_complete = AsyncMock(side_effect=[
            [_mock_llm_response("positive")],
            RuntimeError("API error"),
        ])
        classifier = PromptClassifier(mock_llm)
        report = asyncio.get_event_loop().run_until_complete(
            classifier.classify_async(texts=["ok", "fail"], labels=LABELS)
        )
        assert report.n_total == 2
        assert report.n_failed == 1


# ---------------------------------------------------------------------------
# Export to DataFrame
# ---------------------------------------------------------------------------


class TestToDataFrame:
    def test_basic_export(self):
        report = BatchClassificationReport(
            results=[
                ClassificationResult(text="a", predicted_label="pos", confidence=0.9),
                ClassificationResult(text="b", predicted_label="neg", confidence=0.8),
            ],
            n_total=2, n_classified=2, n_failed=0,
        )
        df = PromptClassifier.to_dataframe(report)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["text", "predicted_label", "confidence"]
        assert df.iloc[0]["predicted_label"] == "pos"

    def test_empty_report(self):
        report = BatchClassificationReport()
        df = PromptClassifier.to_dataframe(report)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------


class TestBuildReport:
    def test_report_aggregation(self):
        results = [
            ClassificationResult(text="a", predicted_label="X"),
            ClassificationResult(text="b", predicted_label="X"),
            ClassificationResult(text="c", predicted_label="Y"),
            ClassificationResult(text="d", predicted_label="UNKNOWN"),
        ]
        report = PromptClassifier._build_report(results)
        assert report.n_total == 4
        assert report.n_classified == 3
        assert report.n_failed == 1
        assert report.label_distribution == {"X": 2, "Y": 1}
