"""Tests for socialscikit.quantikit.feature_extractor."""

import pandas as pd
import pytest

from socialscikit.quantikit.feature_extractor import (
    BUDGET_LEVELS,
    DOWNSTREAM_USES,
    TASK_TYPES,
    FeatureExtractor,
    TaskFeatures,
)


@pytest.fixture()
def extractor():
    return FeatureExtractor()


@pytest.fixture()
def labeled_df():
    return pd.DataFrame({
        "text": [
            "I love this product it is amazing and wonderful",
            "This is terrible I hate it so much never again",
            "Pretty good overall I would recommend this to friends",
            "Worst experience I have ever had with customer service",
            "Average quality nothing special but does the job fine",
            "Absolutely fantastic five stars highly recommended for everyone",
        ],
        "label": ["positive", "negative", "positive", "negative", "neutral", "positive"],
    })


@pytest.fixture()
def unlabeled_df():
    return pd.DataFrame({
        "text": [
            "The policy reform has wide implications for social welfare",
            "Economic indicators suggest a downturn in the next quarter",
            "Community engagement programs are showing positive results overall",
        ],
    })


# ---------------------------------------------------------------------------
# Basic feature extraction
# ---------------------------------------------------------------------------


class TestBasicExtraction:
    def test_sample_count(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        assert f.n_samples == 6

    def test_labeled_count(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        assert f.n_labeled == 6

    def test_unlabeled_count(self, extractor, unlabeled_df):
        f = extractor.extract(unlabeled_df)
        assert f.n_labeled == 0

    def test_n_classes(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        assert f.n_classes == 3

    def test_n_classes_from_user_input(self, extractor, unlabeled_df):
        f = extractor.extract(unlabeled_df, user_inputs={"n_classes": 5})
        assert f.n_classes == 5

    def test_label_balance_ratio(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        # positive=3, negative=2, neutral=1 => ratio = 3/1 = 3.0
        assert f.label_balance_ratio == 3.0

    def test_avg_text_length(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        assert f.avg_text_length_tokens > 0

    def test_language_default_en(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        assert f.language == "en"

    def test_is_multilingual_false(self, extractor, labeled_df):
        f = extractor.extract(labeled_df)
        assert f.is_multilingual is False


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------


class TestColumnDetection:
    def test_detects_content_col(self, extractor):
        df = pd.DataFrame({"content": ["some text here"], "category": ["A"]})
        f = extractor.extract(df)
        assert f.n_samples == 1
        assert f.n_labeled == 1
        assert f.n_classes == 1

    def test_explicit_columns(self, extractor):
        df = pd.DataFrame({"my_text": ["hello world test"], "my_label": ["X"]})
        f = extractor.extract(df, text_col="my_text", label_col="my_label")
        assert f.n_labeled == 1
        assert f.avg_text_length_tokens > 0

    def test_no_matching_columns(self, extractor):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        f = extractor.extract(df)
        assert f.n_labeled == 0
        assert f.avg_text_length_tokens == 0


# ---------------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------------


class TestUserInputs:
    def test_task_type(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"task_type": "sentiment"})
        assert f.task_type == "sentiment"

    def test_invalid_task_type_falls_back(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"task_type": "banana"})
        assert f.task_type == "custom"

    def test_target_f1(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"target_f1": 0.92})
        assert f.target_f1 == 0.92

    def test_budget_level(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"budget_level": "low"})
        assert f.budget_level == "low"

    def test_invalid_budget_falls_back(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"budget_level": "unlimited"})
        assert f.budget_level == "medium"

    def test_downstream_use(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"downstream_use": "hypothesis_testing"})
        assert f.downstream_use == "hypothesis_testing"

    def test_class_boundary_clarity(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"class_boundary_clarity": "ambiguous"})
        assert f.class_boundary_clarity == "ambiguous"

    def test_domain_specificity(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"domain_specificity": "domain_specific"})
        assert f.domain_specificity == "domain_specific"

    def test_annotation_agreement(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs={"annotation_agreement": 0.75})
        assert f.annotation_agreement == 0.75


# ---------------------------------------------------------------------------
# Advanced features
# ---------------------------------------------------------------------------


class TestAdvancedFeatures:
    def test_text_diversity_disabled_by_default(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, enable_advanced=False)
        assert f.text_diversity is None

    def test_text_diversity_enabled(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, enable_advanced=True)
        assert f.text_diversity is not None
        assert 0.0 <= f.text_diversity <= 1.0

    def test_text_diversity_needs_multiple_texts(self, extractor):
        df = pd.DataFrame({"text": ["single"]})
        f = extractor.extract(df, enable_advanced=True)
        # Only 1 text — can't compute diversity
        assert f.text_diversity is None or f.text_diversity == 0.0


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_values(self):
        f = TaskFeatures()
        assert f.target_f1 == 0.80
        assert f.budget_level == "medium"
        assert f.task_type == "custom"
        assert f.language == "en"
        assert f.is_multilingual is False

    def test_constants(self):
        assert "sentiment" in TASK_TYPES
        assert "low" in BUDGET_LEVELS
        assert "hypothesis_testing" in DOWNSTREAM_USES


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe(self, extractor):
        df = pd.DataFrame({"text": [], "label": []})
        f = extractor.extract(df)
        assert f.n_samples == 0
        assert f.n_labeled == 0

    def test_all_nan_labels(self, extractor):
        df = pd.DataFrame({"text": ["a", "b"], "label": [None, None]})
        f = extractor.extract(df)
        assert f.n_labeled == 0
        assert f.n_classes == 0

    def test_empty_texts_filtered(self, extractor):
        df = pd.DataFrame({"text": ["hello world test", "", "  ", "another valid sentence"]})
        f = extractor.extract(df)
        assert f.avg_text_length_tokens > 0

    def test_none_user_inputs(self, extractor, labeled_df):
        f = extractor.extract(labeled_df, user_inputs=None)
        assert f.task_type == "custom"
