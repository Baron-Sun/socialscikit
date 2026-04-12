"""Tests for socialscikit.qualikit.coder."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from socialscikit.qualikit.coder import (
    Coder,
    CodingReport,
    CodingResult,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
)
from socialscikit.qualikit.theme_definer import Theme


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

THEMES = [
    Theme(
        name="Financial Stress",
        description="Concerns about money, bills, and costs.",
        inclusion_examples=["I can't pay my rent."],
        exclusion_examples=["I enjoy my hobbies."],
    ),
    Theme(
        name="Work-Life Balance",
        description="Difficulty balancing work and personal life.",
        inclusion_examples=["I work too many hours."],
        exclusion_examples=["I love my flexible schedule."],
    ),
    Theme(
        name="Health Concerns",
        description="Physical or mental health worries.",
        inclusion_examples=["I feel anxious all the time."],
        exclusion_examples=["I exercise daily."],
    ),
]


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM client that returns pre-set responses."""
    mock = MagicMock()
    call_count = [0]

    def complete_side_effect(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        resp = MagicMock()
        resp.text = responses[idx % len(responses)]
        return resp

    mock.complete.side_effect = complete_side_effect
    return mock


# ---------------------------------------------------------------------------
# CodingResult
# ---------------------------------------------------------------------------


class TestCodingResult:
    def test_high_confidence(self):
        r = CodingResult(
            text_id=0, text="test",
            themes=["A"], confidences={"A": 0.95},
        )
        assert r.confidence_tier == "high"

    def test_medium_confidence(self):
        r = CodingResult(
            text_id=0, text="test",
            themes=["A"], confidences={"A": 0.75},
        )
        assert r.confidence_tier == "medium"

    def test_low_confidence(self):
        r = CodingResult(
            text_id=0, text="test",
            themes=["A"], confidences={"A": 0.3},
        )
        assert r.confidence_tier == "low"

    def test_empty_confidence(self):
        r = CodingResult(text_id=0, text="test")
        assert r.confidence_tier == "low"

    def test_multi_theme_tier(self):
        """Tier is based on minimum confidence."""
        r = CodingResult(
            text_id=0, text="test",
            themes=["A", "B"], confidences={"A": 0.95, "B": 0.5},
        )
        assert r.confidence_tier == "low"


# ---------------------------------------------------------------------------
# CodingReport
# ---------------------------------------------------------------------------


class TestCodingReport:
    def test_empty_report(self):
        r = CodingReport()
        assert r.high_confidence_count == 0
        assert r.medium_confidence_count == 0
        assert r.low_confidence_count == 0

    def test_report_counts(self):
        r = CodingReport(results=[
            CodingResult(text_id=0, text="a", themes=["A"], confidences={"A": 0.95}),
            CodingResult(text_id=1, text="b", themes=["A"], confidences={"A": 0.75}),
            CodingResult(text_id=2, text="c", themes=["A"], confidences={"A": 0.3}),
        ])
        assert r.high_confidence_count == 1
        assert r.medium_confidence_count == 1
        assert r.low_confidence_count == 1


# ---------------------------------------------------------------------------
# Synchronous coding
# ---------------------------------------------------------------------------


class TestCodeSync:
    def test_basic_coding(self):
        response = json.dumps({
            "themes": [{
                "name": "Financial Stress",
                "confidence": 0.9,
                "trigger_words": ["bills", "rent"],
                "reasoning": "Text discusses financial concerns.",
            }],
        })
        mock_llm = _make_mock_llm([response])
        coder = Coder(mock_llm)
        report = coder.code(
            texts=["I can't afford to pay my bills and rent."],
            themes=THEMES,
        )
        assert report.n_total == 1
        assert report.n_coded == 1
        assert report.n_failed == 0
        assert "Financial Stress" in report.results[0].themes
        assert report.results[0].confidences["Financial Stress"] == 0.9

    def test_multi_theme_coding(self):
        response = json.dumps({
            "themes": [
                {"name": "Financial Stress", "confidence": 0.85, "trigger_words": ["bills"]},
                {"name": "Health Concerns", "confidence": 0.7, "trigger_words": ["anxious"]},
            ],
        })
        mock_llm = _make_mock_llm([response])
        coder = Coder(mock_llm)
        report = coder.code(
            texts=["Bills make me anxious and stressed."],
            themes=THEMES,
        )
        result = report.results[0]
        assert len(result.themes) == 2
        assert "Financial Stress" in result.themes
        assert "Health Concerns" in result.themes

    def test_no_themes_assigned(self):
        response = json.dumps({"themes": []})
        mock_llm = _make_mock_llm([response])
        coder = Coder(mock_llm)
        report = coder.code(texts=["The sky is blue."], themes=THEMES)
        assert report.results[0].themes == []

    def test_invalid_theme_name_filtered(self):
        response = json.dumps({
            "themes": [
                {"name": "Nonexistent Theme", "confidence": 0.9},
                {"name": "Financial Stress", "confidence": 0.8},
            ],
        })
        mock_llm = _make_mock_llm([response])
        coder = Coder(mock_llm)
        report = coder.code(texts=["Money problems."], themes=THEMES)
        # Only valid theme should be in results
        assert "Financial Stress" in report.results[0].themes
        assert len(report.results[0].themes) == 1

    def test_multiple_texts(self):
        responses = [
            json.dumps({"themes": [{"name": "Financial Stress", "confidence": 0.9}]}),
            json.dumps({"themes": [{"name": "Work-Life Balance", "confidence": 0.8}]}),
        ]
        mock_llm = _make_mock_llm(responses)
        coder = Coder(mock_llm)
        report = coder.code(
            texts=["Can't pay bills.", "Too many work hours."],
            themes=THEMES,
        )
        assert report.n_total == 2
        assert report.n_coded == 2
        assert report.theme_distribution.get("Financial Stress", 0) == 1
        assert report.theme_distribution.get("Work-Life Balance", 0) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_malformed_json_response(self):
        mock_llm = _make_mock_llm(["not valid json at all"])
        coder = Coder(mock_llm)
        report = coder.code(texts=["Test text."], themes=THEMES)
        # Should not crash, just produce empty result
        assert report.n_total == 1
        assert report.results[0].themes == []

    def test_llm_exception(self):
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = Exception("API error")
        coder = Coder(mock_llm)
        report = coder.code(texts=["Test text."], themes=THEMES)
        assert report.n_failed == 1


# ---------------------------------------------------------------------------
# Theme name matching
# ---------------------------------------------------------------------------


class TestThemeNameMatching:
    def test_exact_match(self):
        result = Coder._match_theme_name("Financial Stress", {"Financial Stress", "Health"})
        assert result == "Financial Stress"

    def test_case_insensitive(self):
        result = Coder._match_theme_name("financial stress", {"Financial Stress", "Health"})
        assert result == "Financial Stress"

    def test_substring_match(self):
        result = Coder._match_theme_name("Financial", {"Financial Stress", "Health"})
        assert result == "Financial Stress"

    def test_no_match(self):
        result = Coder._match_theme_name("Nonexistent", {"Financial Stress", "Health"})
        assert result is None


# ---------------------------------------------------------------------------
# Format themes
# ---------------------------------------------------------------------------


class TestFormatThemes:
    def test_format_themes(self):
        formatted = Coder._format_themes(THEMES)
        assert "Financial Stress" in formatted
        assert "Work-Life Balance" in formatted
        assert "Include" in formatted
        assert "Exclude" in formatted
