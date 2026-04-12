"""Tests for socialscikit.quantikit.prompt_optimizer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pandas as pd
import pytest

from socialscikit.core.llm_client import LLMResponse
from socialscikit.quantikit.prompt_optimizer import PromptOptimizer, PromptVariant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def optimizer():
    return PromptOptimizer()


@pytest.fixture()
def label_defs():
    return {
        "positive": "Expresses approval, satisfaction, or positive emotion.",
        "negative": "Expresses disapproval, dissatisfaction, or negative emotion.",
        "neutral": "Neither clearly positive nor negative.",
    }


@pytest.fixture()
def inclusion_examples():
    return {
        "positive": ["Great product, love it!", "Best purchase I ever made."],
        "negative": ["Terrible quality, broke in a day.", "Worst experience ever."],
        "neutral": ["It arrived on time.", "The product is average."],
    }


@pytest.fixture()
def exclusion_examples():
    return {
        "positive": ["It's okay I guess.", "Not bad but not great."],
        "negative": ["Could be better but it works.", "I expected more."],
        "neutral": ["I absolutely love this!", "This is horrible."],
    }


@pytest.fixture()
def eval_df():
    return pd.DataFrame({
        "text": [
            "This is amazing!",
            "I hate this product",
            "It works fine",
            "Absolutely wonderful",
            "Terrible service",
            "Nothing special",
        ],
        "label": ["positive", "negative", "neutral", "positive", "negative", "neutral"],
    })


# ---------------------------------------------------------------------------
# generate_initial
# ---------------------------------------------------------------------------


class TestGenerateInitial:
    def test_basic_prompt(self, optimizer, label_defs, inclusion_examples, exclusion_examples):
        prompt = optimizer.generate_initial(
            task_description="Classify product reviews by sentiment.",
            label_definitions=label_defs,
            inclusion_examples=inclusion_examples,
            exclusion_examples=exclusion_examples,
        )
        assert "Classify product reviews" in prompt
        assert "positive" in prompt
        assert "negative" in prompt
        assert "neutral" in prompt
        assert "{text}" in prompt  # placeholder preserved

    def test_includes_exclusion_examples(self, optimizer, label_defs, inclusion_examples, exclusion_examples):
        prompt = optimizer.generate_initial(
            task_description="Sentiment classification",
            label_definitions=label_defs,
            inclusion_examples=inclusion_examples,
            exclusion_examples=exclusion_examples,
        )
        assert "okay I guess" in prompt

    def test_no_exclusion_warning(self, optimizer, label_defs, inclusion_examples):
        prompt = optimizer.generate_initial(
            task_description="Sentiment classification",
            label_definitions=label_defs,
            inclusion_examples=inclusion_examples,
            exclusion_examples=None,
        )
        assert "Dunivin" in prompt  # should mention the recommendation

    def test_no_inclusion_examples(self, optimizer, label_defs):
        prompt = optimizer.generate_initial(
            task_description="Classify texts",
            label_definitions=label_defs,
        )
        assert "positive" in prompt
        assert "No inclusion examples" in prompt

    def test_label_definitions_formatted(self, optimizer, label_defs):
        prompt = optimizer.generate_initial(
            task_description="Test task",
            label_definitions=label_defs,
        )
        assert "**positive**" in prompt
        assert "**negative**" in prompt


# ---------------------------------------------------------------------------
# generate_variants (mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateVariants:
    def test_requires_llm(self, optimizer):
        with pytest.raises(RuntimeError, match="LLM client"):
            optimizer.generate_variants("some prompt")

    def test_generates_variants(self):
        mock_llm = MagicMock()
        variants_json = json.dumps([
            {"style": "concise", "prompt_text": "Short version of the prompt."},
            {"style": "detailed", "prompt_text": "Detailed version of the prompt."},
            {"style": "structured", "prompt_text": "Step-by-step version."},
        ])
        mock_llm.complete.return_value = LLMResponse(text=variants_json)

        opt = PromptOptimizer(llm_client=mock_llm)
        variants = opt.generate_variants("original prompt", n=3)

        assert len(variants) == 4  # original + 3 generated
        assert variants[0].style == "original"
        assert variants[1].style == "concise"

    def test_handles_malformed_llm_response(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = LLMResponse(text="not valid json")

        opt = PromptOptimizer(llm_client=mock_llm)
        variants = opt.generate_variants("original prompt", n=3)
        # Should fall back to just the original
        assert len(variants) == 1
        assert variants[0].style == "original"


# ---------------------------------------------------------------------------
# select_examples (TF-IDF mode, no LLM needed)
# ---------------------------------------------------------------------------


class TestSelectExamples:
    def test_tfidf_selection(self, optimizer):
        df = pd.DataFrame({
            "text": [
                "I love this product so much",
                "Great quality and fast shipping",
                "Amazing experience overall",
                "Best purchase I ever made in my life",
                "Really happy with this item",
                "Terrible product broke immediately",
                "Worst customer service ever experienced",
                "Complete waste of money and time",
            ],
            "label": ["pos", "pos", "pos", "pos", "pos", "neg", "neg", "neg"],
        })
        selected = optimizer.select_examples(df, label_col="label", n_per_class=2)
        assert "pos" in selected
        assert "neg" in selected
        assert len(selected["pos"]) == 2
        assert len(selected["neg"]) == 2

    def test_fewer_texts_than_requested(self, optimizer):
        df = pd.DataFrame({
            "text": ["only one text here"],
            "label": ["A"],
        })
        selected = optimizer.select_examples(df, label_col="label", n_per_class=3)
        assert len(selected["A"]) == 1

    def test_multiple_classes(self, optimizer):
        df = pd.DataFrame({
            "text": [f"text {i}" for i in range(12)],
            "label": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
        })
        selected = optimizer.select_examples(df, label_col="label", n_per_class=2)
        assert len(selected) == 3
        for label in ["A", "B", "C"]:
            assert len(selected[label]) == 2


# ---------------------------------------------------------------------------
# select_examples (LLM mode, mocked)
# ---------------------------------------------------------------------------


class TestSelectExamplesLLM:
    def test_llm_selection(self):
        mock_llm = MagicMock()
        mock_llm.complete.return_value = LLMResponse(
            text=json.dumps(["text A1", "text A2"])
        )
        opt = PromptOptimizer(llm_client=mock_llm)
        df = pd.DataFrame({
            "text": ["text A1", "text A2", "text A3", "text A4", "text A5"],
            "label": ["A"] * 5,
        })
        selected = opt.select_examples(df, label_col="label", n_per_class=2, method="llm")
        assert len(selected["A"]) == 2


# ---------------------------------------------------------------------------
# evaluate_and_select (mocked LLM)
# ---------------------------------------------------------------------------


class TestEvaluateAndSelect:
    def test_requires_llm(self, optimizer, eval_df):
        variants = [PromptVariant(prompt_text="test", style="original")]
        with pytest.raises(RuntimeError, match="LLM client"):
            optimizer.evaluate_and_select(variants, eval_df, label_col="label")

    def test_evaluates_variants(self, eval_df):
        mock_llm = MagicMock()
        # Mock returns correct labels for all texts
        labels = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        mock_llm.complete.side_effect = [
            LLMResponse(text=l) for l in labels * 2  # 2 variants
        ]

        opt = PromptOptimizer(llm_client=mock_llm)
        variants = [
            PromptVariant(prompt_text="prompt A {text}", style="original"),
            PromptVariant(prompt_text="prompt B {text}", style="concise"),
        ]
        result = opt.evaluate_and_select(variants, eval_df, label_col="label")

        assert result.best_variant is not None
        assert result.best_variant.f1_score is not None
        assert result.best_variant.f1_score > 0
        assert len(result.all_variants) == 2
        assert result.eval_summary is not None
        assert len(result.eval_summary) == 2

    def test_picks_best_variant(self, eval_df):
        mock_llm = MagicMock()
        # Variant A: all wrong
        wrong = [LLMResponse(text="wrong")] * 6
        # Variant B: all correct
        correct = [LLMResponse(text=l) for l in eval_df["label"].tolist()]
        mock_llm.complete.side_effect = wrong + correct

        opt = PromptOptimizer(llm_client=mock_llm)
        variants = [
            PromptVariant(prompt_text="bad prompt {text}", style="bad"),
            PromptVariant(prompt_text="good prompt {text}", style="good"),
        ]
        result = opt.evaluate_and_select(variants, eval_df, label_col="label")

        assert result.best_variant.style == "good"
        assert result.best_variant.f1_score > result.all_variants[0].f1_score


# ---------------------------------------------------------------------------
# Macro F1 helper
# ---------------------------------------------------------------------------


class TestMacroF1:
    def test_perfect_prediction(self):
        preds = ["A", "B", "C", "A"]
        trues = ["A", "B", "C", "A"]
        f1 = PromptOptimizer._macro_f1(preds, trues)
        assert f1 == 1.0

    def test_all_wrong(self):
        preds = ["B", "C", "A"]
        trues = ["A", "B", "C"]
        f1 = PromptOptimizer._macro_f1(preds, trues)
        assert f1 == 0.0

    def test_partial_correct(self):
        preds = ["A", "A", "B"]
        trues = ["A", "B", "B"]
        f1 = PromptOptimizer._macro_f1(preds, trues)
        assert 0.0 < f1 < 1.0

    def test_case_insensitive(self):
        preds = ["Positive", "NEGATIVE"]
        trues = ["positive", "negative"]
        f1 = PromptOptimizer._macro_f1(preds, trues)
        assert f1 == 1.0

    def test_empty(self):
        assert PromptOptimizer._macro_f1([], []) == 0.0
