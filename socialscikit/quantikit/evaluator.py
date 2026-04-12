"""Classification evaluation — F1, Cohen's Kappa, confusion matrix, and per-class metrics.

Provides both programmatic access and formatted reports suitable for
Gradio UI or CLI display.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PerClassMetrics:
    """Precision / recall / F1 for a single class."""

    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of true instances


@dataclass
class ConfusionMatrix:
    """Row = true label, column = predicted label."""

    labels: list[str]
    matrix: list[list[int]]  # matrix[i][j] = count(true=labels[i], pred=labels[j])

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.matrix,
            index=[f"true:{l}" for l in self.labels],
            columns=[f"pred:{l}" for l in self.labels],
        )


@dataclass
class EvaluationReport:
    """Full evaluation result."""

    # Aggregate metrics
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohens_kappa: float = 0.0

    # Per-class detail
    per_class: list[PerClassMetrics] = field(default_factory=list)

    # Confusion matrix
    confusion_matrix: ConfusionMatrix | None = None

    # Counts
    n_total: int = 0
    n_correct: int = 0


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Evaluate classification predictions against ground-truth labels.

    Usage::

        evaluator = Evaluator()
        report = evaluator.evaluate(
            true_labels=["pos", "neg", "pos"],
            pred_labels=["pos", "pos", "pos"],
        )
        print(report.macro_f1, report.cohens_kappa)
        print(evaluator.format_report(report))
    """

    def evaluate(
        self,
        true_labels: list[str],
        pred_labels: list[str],
        labels: list[str] | None = None,
    ) -> EvaluationReport:
        """Compute all evaluation metrics.

        Parameters
        ----------
        true_labels : list[str]
            Ground-truth labels.
        pred_labels : list[str]
            Predicted labels (same length as true_labels).
        labels : list[str] or None
            Explicit label order. If None, derived from the union of
            true and predicted labels, sorted alphabetically.

        Returns
        -------
        EvaluationReport
        """
        if len(true_labels) != len(pred_labels):
            raise ValueError(
                f"Length mismatch: {len(true_labels)} true vs {len(pred_labels)} predicted."
            )

        n = len(true_labels)
        if n == 0:
            return EvaluationReport()

        # Normalise
        trues = [str(t).strip() for t in true_labels]
        preds = [str(p).strip() for p in pred_labels]

        # Label set
        if labels is None:
            labels = sorted(set(trues) | set(preds))
        label_to_idx = {l: i for i, l in enumerate(labels)}
        k = len(labels)

        # Build confusion matrix
        cm = [[0] * k for _ in range(k)]
        n_correct = 0
        for t, p in zip(trues, preds):
            ti = label_to_idx.get(t)
            pi = label_to_idx.get(p)
            if ti is not None and pi is not None:
                cm[ti][pi] += 1
                if ti == pi:
                    n_correct += 1

        # Per-class metrics
        per_class: list[PerClassMetrics] = []
        for i, label in enumerate(labels):
            tp = cm[i][i]
            fp = sum(cm[j][i] for j in range(k)) - tp
            fn = sum(cm[i][j] for j in range(k)) - tp
            support = sum(cm[i][j] for j in range(k))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_class.append(PerClassMetrics(
                label=label,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                support=support,
            ))

        # Macro F1
        macro_f1 = sum(pc.f1 for pc in per_class) / k if k > 0 else 0.0

        # Weighted F1
        total_support = sum(pc.support for pc in per_class)
        weighted_f1 = (
            sum(pc.f1 * pc.support for pc in per_class) / total_support
            if total_support > 0
            else 0.0
        )

        # Accuracy
        accuracy = n_correct / n if n > 0 else 0.0

        # Cohen's Kappa
        kappa = self._cohens_kappa(cm, n, k)

        return EvaluationReport(
            accuracy=round(accuracy, 4),
            macro_f1=round(macro_f1, 4),
            weighted_f1=round(weighted_f1, 4),
            cohens_kappa=round(kappa, 4),
            per_class=per_class,
            confusion_matrix=ConfusionMatrix(labels=labels, matrix=cm),
            n_total=n,
            n_correct=n_correct,
        )

    # ------------------------------------------------------------------
    # Cohen's Kappa
    # ------------------------------------------------------------------

    @staticmethod
    def _cohens_kappa(cm: list[list[int]], n: int, k: int) -> float:
        """Compute Cohen's Kappa from confusion matrix.

        κ = (p_o - p_e) / (1 - p_e)

        where p_o = observed agreement, p_e = expected agreement by chance.
        """
        if n == 0:
            return 0.0

        p_o = sum(cm[i][i] for i in range(k)) / n

        # Expected agreement
        p_e = 0.0
        for i in range(k):
            row_sum = sum(cm[i][j] for j in range(k))  # true count for class i
            col_sum = sum(cm[j][i] for j in range(k))  # pred count for class i
            p_e += (row_sum / n) * (col_sum / n)

        if p_e >= 1.0:
            return 0.0

        return (p_o - p_e) / (1 - p_e)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(report: EvaluationReport) -> str:
        """Format an EvaluationReport as a human-readable string."""
        lines: list[str] = []
        lines.append("═══ 分类评估报告 ═══")
        lines.append("")
        lines.append(f"总样本：{report.n_total}   正确：{report.n_correct}")
        lines.append(f"准确率 (Accuracy)：{report.accuracy:.4f}")
        lines.append(f"宏平均 F1 (Macro-F1)：{report.macro_f1:.4f}")
        lines.append(f"加权 F1 (Weighted-F1)：{report.weighted_f1:.4f}")
        lines.append(f"Cohen's Kappa：{report.cohens_kappa:.4f}")
        lines.append("")

        # Per-class table
        if report.per_class:
            lines.append("── 各类别指标 ──")
            lines.append(f"{'类别':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
            lines.append("─" * 62)
            for pc in report.per_class:
                lines.append(
                    f"{pc.label:<20} {pc.precision:>10.4f} {pc.recall:>10.4f} "
                    f"{pc.f1:>10.4f} {pc.support:>10}"
                )
            lines.append("")

        # Confusion matrix
        if report.confusion_matrix:
            lines.append("── 混淆矩阵 ──")
            cm = report.confusion_matrix
            header = f"{'':>15}" + "".join(f"{l:>12}" for l in cm.labels)
            lines.append(header)
            for i, row_label in enumerate(cm.labels):
                row = f"{row_label:>15}" + "".join(f"{cm.matrix[i][j]:>12}" for j in range(len(cm.labels)))
                lines.append(row)
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Convenience: evaluate from DataFrame
    # ------------------------------------------------------------------

    def evaluate_df(
        self,
        df: pd.DataFrame,
        true_col: str,
        pred_col: str,
        labels: list[str] | None = None,
    ) -> EvaluationReport:
        """Evaluate from a DataFrame with true and predicted columns."""
        return self.evaluate(
            true_labels=df[true_col].tolist(),
            pred_labels=df[pred_col].tolist(),
            labels=labels,
        )
