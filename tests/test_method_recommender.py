"""Tests for socialscikit.quantikit.method_recommender."""

import pytest

from socialscikit.quantikit.feature_extractor import TaskFeatures
from socialscikit.quantikit.method_recommender import MethodRecommender


@pytest.fixture()
def recommender():
    return MethodRecommender()


# ---------------------------------------------------------------------------
# Rule 1: No labels, simple task, ≤3 classes → zero_shot
# ---------------------------------------------------------------------------


class TestRule1ZeroShot:
    def test_sentiment_no_labels(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=2, task_type="sentiment", target_f1=0.75)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "zero_shot"
        assert rec.confidence == "high"
        assert "Ziems" in rec.literature_support[0]

    def test_sentiment_3_classes(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=3, task_type="sentiment", target_f1=0.80)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "zero_shot"

    def test_alternative_is_few_shot(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=2, task_type="sentiment", target_f1=0.75)
        rec = recommender.recommend(f)
        assert rec.alternative_method == "few_shot"


# ---------------------------------------------------------------------------
# Rule 2: No labels, complex task → few_shot
# ---------------------------------------------------------------------------


class TestRule2FewShot:
    def test_framing_no_labels(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=5, task_type="framing", target_f1=0.75)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"
        assert rec.confidence == "medium"
        assert any("Chae" in ref for ref in rec.literature_support)

    def test_many_classes_no_labels(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=10, task_type="topic", target_f1=0.70)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"

    def test_sentiment_but_many_classes(self, recommender):
        """sentiment with >3 classes should trigger Rule 2, not Rule 1."""
        f = TaskFeatures(n_labeled=0, n_classes=5, task_type="sentiment", target_f1=0.75)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"


# ---------------------------------------------------------------------------
# Rule 3: No labels + high target F1 → cold start
# ---------------------------------------------------------------------------


class TestRule3ColdStart:
    def test_high_target_no_labels(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=3, task_type="framing", target_f1=0.90)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "fine_tune"
        assert rec.cold_start_recommendation is not None
        assert any("Carlson" in ref for ref in rec.literature_support)

    def test_cold_start_report_fields(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=4, task_type="topic", target_f1=0.90)
        rec = recommender.recommend(f)
        cs = rec.cold_start_recommendation
        assert cs is not None
        assert cs.minimum_n > 0
        assert cs.recommended_n > cs.minimum_n
        assert cs.diminishing_returns_n > cs.recommended_n
        assert cs.marginal_gain_after > 0
        assert "标注量" in cs.message

    def test_cold_start_simple_task(self, recommender):
        """Simple task with high target still triggers cold start."""
        f = TaskFeatures(n_labeled=0, n_classes=2, task_type="sentiment", target_f1=0.90)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "fine_tune"
        assert rec.cold_start_recommendation is not None
        # Simple task should have lower minimum_n
        assert rec.cold_start_recommendation.minimum_n == 200

    def test_cold_start_many_classes(self, recommender):
        f = TaskFeatures(n_labeled=0, n_classes=10, task_type="custom", target_f1=0.88)
        rec = recommender.recommend(f)
        cs = rec.cold_start_recommendation
        assert cs is not None
        assert cs.minimum_n == 500  # complex task needs more data


# ---------------------------------------------------------------------------
# Rule 4: Small labeled set (1–199) → few_shot
# ---------------------------------------------------------------------------


class TestRule4SmallLabeled:
    def test_50_labels(self, recommender):
        f = TaskFeatures(n_labeled=50, n_classes=3, task_type="framing")
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"
        assert rec.confidence == "high"

    def test_199_labels(self, recommender):
        f = TaskFeatures(n_labeled=199, n_classes=5, task_type="topic")
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"

    def test_1_label(self, recommender):
        f = TaskFeatures(n_labeled=1, n_classes=2, task_type="sentiment")
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"


# ---------------------------------------------------------------------------
# Rule 5: Medium labeled set (200–499) → few_shot + compare
# ---------------------------------------------------------------------------


class TestRule5MediumLabeled:
    def test_200_labels(self, recommender):
        f = TaskFeatures(n_labeled=200, n_classes=3, task_type="framing")
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"
        assert rec.confidence == "medium"
        assert rec.alternative_method == "fine_tune"
        assert any("Do" in ref for ref in rec.literature_support)

    def test_499_labels(self, recommender):
        f = TaskFeatures(n_labeled=499, n_classes=4, task_type="topic")
        rec = recommender.recommend(f)
        assert rec.recommended_method == "few_shot"
        assert "交叉区间" in rec.reasoning


# ---------------------------------------------------------------------------
# Rule 6: Large labeled (≥500), moderate target → fine_tune
# ---------------------------------------------------------------------------


class TestRule6LargeLabeled:
    def test_500_labels(self, recommender):
        f = TaskFeatures(n_labeled=500, n_classes=3, task_type="sentiment", target_f1=0.82)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "fine_tune"
        assert rec.confidence == "high"

    def test_1000_labels(self, recommender):
        f = TaskFeatures(n_labeled=1000, n_classes=5, target_f1=0.80)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "fine_tune"

    def test_multilingual_mentions_xlm(self, recommender):
        f = TaskFeatures(
            n_labeled=600, n_classes=3, task_type="sentiment",
            target_f1=0.80, is_multilingual=True,
        )
        rec = recommender.recommend(f)
        assert rec.recommended_method == "fine_tune"
        assert "XLM-RoBERTa" in rec.reasoning


# ---------------------------------------------------------------------------
# Rule 7: Large labeled (≥500), high target → active_learning
# ---------------------------------------------------------------------------


class TestRule7ActiveLearning:
    def test_500_labels_high_target(self, recommender):
        f = TaskFeatures(n_labeled=500, n_classes=3, target_f1=0.90)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "active_learning"
        assert rec.confidence == "high"
        assert any("Montgomery" in ref for ref in rec.literature_support)

    def test_2000_labels_high_target(self, recommender):
        f = TaskFeatures(n_labeled=2000, n_classes=8, target_f1=0.88)
        rec = recommender.recommend(f)
        assert rec.recommended_method == "active_learning"


# ---------------------------------------------------------------------------
# Cross-cutting: sensitivity analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:
    def test_hypothesis_testing_triggers(self, recommender):
        f = TaskFeatures(
            n_labeled=100, n_classes=2, task_type="sentiment",
            downstream_use="hypothesis_testing",
        )
        rec = recommender.recommend(f)
        assert rec.sensitivity_analysis_suggested is True

    def test_descriptive_does_not_trigger(self, recommender):
        f = TaskFeatures(
            n_labeled=100, n_classes=2, task_type="sentiment",
            downstream_use="descriptive",
        )
        rec = recommender.recommend(f)
        assert rec.sensitivity_analysis_suggested is False

    def test_no_downstream_use(self, recommender):
        f = TaskFeatures(n_labeled=100, n_classes=2, task_type="sentiment")
        rec = recommender.recommend(f)
        assert rec.sensitivity_analysis_suggested is False


# ---------------------------------------------------------------------------
# Cross-cutting: multilingual
# ---------------------------------------------------------------------------


class TestMultilingual:
    def test_multilingual_note_in_reasoning(self, recommender):
        f = TaskFeatures(
            n_labeled=0, n_classes=2, task_type="sentiment",
            target_f1=0.75, is_multilingual=True,
        )
        rec = recommender.recommend(f)
        assert "多语言" in rec.reasoning

    def test_multilingual_active_learning(self, recommender):
        f = TaskFeatures(
            n_labeled=800, n_classes=4, target_f1=0.90, is_multilingual=True,
        )
        rec = recommender.recommend(f)
        assert "XLM-RoBERTa" in rec.reasoning


# ---------------------------------------------------------------------------
# Edge: all recommendations have required fields
# ---------------------------------------------------------------------------


class TestRecommendationCompleteness:
    @pytest.mark.parametrize("n_labeled,n_classes,task_type,target_f1", [
        (0, 2, "sentiment", 0.75),
        (0, 5, "framing", 0.70),
        (0, 3, "topic", 0.92),
        (50, 3, "sentiment", 0.80),
        (300, 4, "moral", 0.80),
        (600, 3, "sentiment", 0.80),
        (600, 3, "sentiment", 0.90),
    ])
    def test_all_fields_populated(self, recommender, n_labeled, n_classes, task_type, target_f1):
        f = TaskFeatures(
            n_labeled=n_labeled, n_classes=n_classes,
            task_type=task_type, target_f1=target_f1,
        )
        rec = recommender.recommend(f)
        assert rec.recommended_method in {"zero_shot", "few_shot", "fine_tune", "active_learning"}
        assert rec.confidence in {"high", "medium", "low"}
        assert len(rec.reasoning) > 0
        assert len(rec.literature_support) > 0
        assert len(rec.estimated_cost) > 0
        assert rec.estimated_performance[0] > 0
