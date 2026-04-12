"""Tests for socialscikit.qualikit.theme_definer."""

from __future__ import annotations

import pytest

from socialscikit.qualikit.theme_definer import (
    Theme,
    ThemeDefiner,
    ThemeSuggestion,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_TEXTS = [
    "I feel stressed about paying bills and rent every month.",
    "My job is very demanding and I work long hours.",
    "The commute to work takes over two hours each day.",
    "I worry about my children's education and future.",
    "Healthcare costs are rising and insurance is expensive.",
    "My manager is supportive and helps me grow professionally.",
    "I enjoy spending time with my family on weekends.",
    "The neighborhood has become unsafe with rising crime.",
    "I struggle to balance work and personal life.",
    "Public transportation in my area is unreliable.",
    "I am concerned about saving enough for retirement.",
    "My colleagues are friendly and we have good teamwork.",
    "Grocery prices have increased significantly this year.",
    "I feel isolated and have few friends in my area.",
    "The schools in my district are underfunded.",
    "I appreciate the flexibility of remote work arrangements.",
    "Housing costs in the city are unaffordable.",
    "I volunteer at the local community center regularly.",
    "Traffic congestion makes daily commuting exhausting.",
    "I am worried about climate change and its effects.",
]


@pytest.fixture()
def definer():
    return ThemeDefiner()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class TestDataClasses:
    def test_theme_suggestion(self):
        ts = ThemeSuggestion(
            name="Financial Stress",
            description="Concerns about money and costs.",
            representative_texts=["text1", "text2"],
            estimated_coverage=0.25,
        )
        assert ts.name == "Financial Stress"
        assert ts.estimated_coverage == 0.25

    def test_theme(self):
        t = Theme(
            name="Work-Life Balance",
            description="Difficulty balancing work and personal life.",
            inclusion_examples=["I struggle to balance work and life."],
            exclusion_examples=["I love my job."],
        )
        assert t.name == "Work-Life Balance"
        assert len(t.inclusion_examples) == 1
        assert len(t.exclusion_examples) == 1


# ---------------------------------------------------------------------------
# TF-IDF theme suggestion
# ---------------------------------------------------------------------------


class TestTfidfSuggestion:
    def test_suggest_themes(self, definer):
        suggestions = definer.suggest_themes(SAMPLE_TEXTS, n_themes=4, method="tfidf")
        assert len(suggestions) > 0
        assert len(suggestions) <= 4
        for s in suggestions:
            assert s.name != ""
            assert s.description != ""
            assert len(s.representative_texts) > 0
            assert 0 < s.estimated_coverage <= 1.0

    def test_coverage_sums_to_one(self, definer):
        suggestions = definer.suggest_themes(SAMPLE_TEXTS, n_themes=4, method="tfidf")
        total = sum(s.estimated_coverage for s in suggestions)
        assert 0.9 <= total <= 1.1  # approximately 1.0

    def test_few_texts(self, definer):
        suggestions = definer.suggest_themes(
            ["hello world", "goodbye world"], n_themes=2, method="tfidf",
        )
        assert len(suggestions) <= 2

    def test_single_text(self, definer):
        suggestions = definer.suggest_themes(["just one text"], n_themes=3, method="tfidf")
        assert len(suggestions) == 1

    def test_empty_input(self, definer):
        suggestions = definer.suggest_themes([], n_themes=5, method="tfidf")
        assert suggestions == []

    def test_n_themes_capped_by_text_count(self, definer):
        texts = ["text a", "text b", "text c"]
        suggestions = definer.suggest_themes(texts, n_themes=10, method="tfidf")
        assert len(suggestions) <= 3

    def test_representative_texts_from_input(self, definer):
        suggestions = definer.suggest_themes(SAMPLE_TEXTS, n_themes=3, method="tfidf")
        for s in suggestions:
            for rt in s.representative_texts:
                assert rt in SAMPLE_TEXTS

    def test_sorted_by_coverage(self, definer):
        suggestions = definer.suggest_themes(SAMPLE_TEXTS, n_themes=4, method="tfidf")
        for i in range(len(suggestions) - 1):
            assert suggestions[i].estimated_coverage >= suggestions[i + 1].estimated_coverage


# ---------------------------------------------------------------------------
# Overlap assessment
# ---------------------------------------------------------------------------


class TestOverlapAssessment:
    def test_no_overlap(self, definer):
        themes = [
            Theme(name="Financial Stress", description="Concerns about money bills costs"),
            Theme(name="Nature Environment", description="Weather climate outdoor activities"),
        ]
        warnings = definer.assess_overlap(themes)
        # These are quite different, should be no warnings
        assert len(warnings) == 0

    def test_high_overlap(self, definer):
        themes = [
            Theme(name="Financial Stress", description="Concerns about paying bills and financial difficulties"),
            Theme(name="Economic Hardship", description="Financial difficulties and trouble paying bills"),
        ]
        warnings = definer.assess_overlap(themes)
        assert len(warnings) >= 1
        assert warnings[0]["overlap_pct"] > 40

    def test_single_theme(self, definer):
        themes = [Theme(name="A", description="desc")]
        warnings = definer.assess_overlap(themes)
        assert warnings == []

    def test_empty(self, definer):
        warnings = definer.assess_overlap([])
        assert warnings == []

    def test_warning_message(self, definer):
        themes = [
            Theme(name="A", description="financial stress money bills"),
            Theme(name="B", description="financial stress money costs"),
        ]
        warnings = definer.assess_overlap(themes)
        if warnings:
            assert "重叠" in warnings[0]["message"]


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


class TestLLMFallback:
    def test_fallback_to_tfidf_when_no_llm(self, definer):
        """method='llm' without client should fall back to tfidf."""
        suggestions = definer.suggest_themes(SAMPLE_TEXTS, n_themes=3, method="llm")
        assert len(suggestions) > 0
