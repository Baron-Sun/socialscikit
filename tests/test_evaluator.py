"""Tests for socialscikit.quantikit.evaluator."""

import pandas as pd
import pytest

from socialscikit.quantikit.evaluator import (
    ConfusionMatrix,
    EvaluationReport,
    Evaluator,
    PerClassMetrics,
)


@pytest.fixture()
def evaluator():
    return Evaluator()


# ---------------------------------------------------------------------------
# Perfect predictions
# ---------------------------------------------------------------------------


class TestPerfect:
    def test_all_correct(self, evaluator):
        trues = ["A", "B", "C", "A", "B"]
        preds = ["A", "B", "C", "A", "B"]
        r = evaluator.evaluate(trues, preds)
        assert r.accuracy == 1.0
        assert r.macro_f1 == 1.0
        assert r.weighted_f1 == 1.0
        assert r.cohens_kappa == 1.0
        assert r.n_correct == 5

    def test_binary_perfect(self, evaluator):
        trues = ["pos", "neg", "pos", "neg"]
        preds = ["pos", "neg", "pos", "neg"]
        r = evaluator.evaluate(trues, preds)
        assert r.accuracy == 1.0
        assert r.cohens_kappa == 1.0


# ---------------------------------------------------------------------------
# All wrong
# ---------------------------------------------------------------------------


class TestAllWrong:
    def test_completely_wrong(self, evaluator):
        trues = ["A", "B", "C"]
        preds = ["B", "C", "A"]
        r = evaluator.evaluate(trues, preds)
        assert r.accuracy == 0.0
        assert r.macro_f1 == 0.0
        assert r.n_correct == 0

    def test_kappa_negative_possible(self, evaluator):
        """Kappa can be negative when agreement is worse than chance."""
        trues = ["A", "A", "B", "B"]
        preds = ["B", "B", "A", "A"]
        r = evaluator.evaluate(trues, preds)
        assert r.cohens_kappa < 0


# ---------------------------------------------------------------------------
# Partial correctness
# ---------------------------------------------------------------------------


class TestPartial:
    def test_mixed_results(self, evaluator):
        trues = ["pos", "neg", "pos", "neg", "neutral"]
        preds = ["pos", "pos", "pos", "neg", "neutral"]
        r = evaluator.evaluate(trues, preds)
        assert 0 < r.accuracy < 1.0
        assert 0 < r.macro_f1 < 1.0
        assert r.n_correct == 4
        assert r.n_total == 5

    def test_per_class_metrics(self, evaluator):
        trues = ["A", "A", "B", "B"]
        preds = ["A", "B", "B", "B"]
        r = evaluator.evaluate(trues, preds)

        a_metrics = next(pc for pc in r.per_class if pc.label == "A")
        b_metrics = next(pc for pc in r.per_class if pc.label == "B")

        # A: TP=1, FP=0, FN=1 → P=1.0, R=0.5, F1=0.6667
        assert a_metrics.precision == 1.0
        assert a_metrics.recall == 0.5
        assert 0.66 < a_metrics.f1 < 0.67
        assert a_metrics.support == 2

        # B: TP=2, FP=1, FN=0 → P=0.6667, R=1.0, F1=0.8
        assert 0.66 < b_metrics.precision < 0.67
        assert b_metrics.recall == 1.0
        assert b_metrics.f1 == 0.8
        assert b_metrics.support == 2


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_binary_cm(self, evaluator):
        trues = ["pos", "pos", "neg", "neg"]
        preds = ["pos", "neg", "neg", "neg"]
        r = evaluator.evaluate(trues, preds, labels=["neg", "pos"])
        cm = r.confusion_matrix
        assert cm is not None
        assert cm.labels == ["neg", "pos"]
        # cm[0] = true:neg → [pred:neg, pred:pos] = [2, 0]
        # cm[1] = true:pos → [pred:neg, pred:pos] = [1, 1]
        assert cm.matrix[0] == [2, 0]
        assert cm.matrix[1] == [1, 1]

    def test_cm_to_dataframe(self, evaluator):
        trues = ["A", "B", "A"]
        preds = ["A", "A", "A"]
        r = evaluator.evaluate(trues, preds)
        df = r.confusion_matrix.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert "pred:A" in df.columns
        assert "true:A" in df.index


# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_moderate_agreement(self, evaluator):
        """Typical moderate agreement scenario."""
        trues = ["A"] * 20 + ["B"] * 20
        preds = ["A"] * 15 + ["B"] * 5 + ["B"] * 18 + ["A"] * 2
        r = evaluator.evaluate(trues, preds)
        # Should be moderate (0.4–0.7 range)
        assert 0.3 < r.cohens_kappa < 0.8

    def test_random_agreement(self, evaluator):
        """Nearly random predictions → kappa near 0."""
        import random
        random.seed(42)
        trues = ["A"] * 50 + ["B"] * 50
        preds = [random.choice(["A", "B"]) for _ in range(100)]
        r = evaluator.evaluate(trues, preds)
        assert -0.2 < r.cohens_kappa < 0.3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_inputs(self, evaluator):
        r = evaluator.evaluate([], [])
        assert r.n_total == 0
        assert r.accuracy == 0.0
        assert r.macro_f1 == 0.0

    def test_single_sample(self, evaluator):
        r = evaluator.evaluate(["A"], ["A"])
        assert r.accuracy == 1.0
        assert r.n_total == 1

    def test_length_mismatch(self, evaluator):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluator.evaluate(["A", "B"], ["A"])

    def test_explicit_labels(self, evaluator):
        """Explicit labels include classes not in data."""
        trues = ["A", "A"]
        preds = ["A", "A"]
        r = evaluator.evaluate(trues, preds, labels=["A", "B", "C"])
        assert len(r.per_class) == 3
        b_metrics = next(pc for pc in r.per_class if pc.label == "B")
        assert b_metrics.support == 0
        assert b_metrics.f1 == 0.0

    def test_whitespace_handling(self, evaluator):
        trues = [" A ", "B"]
        preds = ["A", " B"]
        r = evaluator.evaluate(trues, preds)
        assert r.accuracy == 1.0


# ---------------------------------------------------------------------------
# evaluate_df
# ---------------------------------------------------------------------------


class TestEvaluateDF:
    def test_from_dataframe(self, evaluator):
        df = pd.DataFrame({
            "true_label": ["A", "B", "A", "B"],
            "pred_label": ["A", "B", "B", "B"],
        })
        r = evaluator.evaluate_df(df, true_col="true_label", pred_col="pred_label")
        assert r.n_total == 4
        assert r.n_correct == 3


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_format_output(self, evaluator):
        trues = ["pos", "neg", "pos", "neg", "neutral"]
        preds = ["pos", "neg", "neg", "neg", "neutral"]
        r = evaluator.evaluate(trues, preds)
        output = Evaluator.format_report(r)
        assert "分类评估报告" in output
        assert "Macro-F1" in output
        assert "Kappa" in output
        assert "混淆矩阵" in output
        assert "pos" in output

    def test_format_empty(self):
        output = Evaluator.format_report(EvaluationReport())
        assert "0" in output
