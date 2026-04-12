"""Inter-Coder Reliability (ICR) — Cohen's Kappa, Krippendorff's Alpha, Jaccard agreement.

Compute agreement metrics between human-human or human-LLM coders.
Supports both single-label (QuantiKit) and multi-label (QualiKit) coding.

References:
- Cohen, J. (1960). A coefficient of agreement for nominal scales.
  Educational and Psychological Measurement, 20(1), 37–46.
- Krippendorff, K. (2011). Computing Krippendorff's Alpha-Reliability.
  https://repository.upenn.edu/asc_papers/43
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer
  agreement for categorical data. Biometrics, 33(1), 159–174.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ICRResult:
    """Result for a single metric computation."""

    metric_name: str  # "cohens_kappa" | "krippendorffs_alpha" | "jaccard_agreement"
    value: float
    interpretation: str  # e.g. "Moderate agreement"
    n_coders: int = 2
    n_items: int = 0
    n_categories: int = 0


@dataclass
class PerCategoryAgreement:
    """Agreement breakdown for a single category/theme."""

    category: str
    observed_agreement: float
    expected_agreement: float
    specific_agreement: float  # category-specific agreement


@dataclass
class ICRReport:
    """Full inter-coder reliability report."""

    results: list[ICRResult] = field(default_factory=list)
    per_category: list[PerCategoryAgreement] = field(default_factory=list)
    pairwise_matrix: list[list[float]] | None = None
    coder_labels: list[str] = field(default_factory=list)
    summary_text: str = ""


# ---------------------------------------------------------------------------
# ICR Calculator
# ---------------------------------------------------------------------------


class ICRCalculator:
    """Compute inter-coder reliability metrics.

    Supports two input modes:
    1. Two label lists (simple 2-coder case, single-label)
    2. Multi-label theme sets (QualiKit qualitative coding)

    Also supports Krippendorff's Alpha with a reliability matrix for 2+ coders.

    Usage::

        calc = ICRCalculator()

        # Single-label (QuantiKit)
        report = calc.compute_all(
            coder1_labels=["pos", "neg", "pos"],
            coder2_labels=["pos", "pos", "pos"],
        )

        # Multi-label (QualiKit)
        report = calc.compute_all_multilabel(
            coder1_themes=[{"economy", "policy"}, {"health"}],
            coder2_themes=[{"economy"}, {"health", "education"}],
        )
    """

    # ------------------------------------------------------------------
    # Cohen's Kappa (2 coders, single-label)
    # ------------------------------------------------------------------

    def compute_cohens_kappa(
        self,
        coder1_labels: list[str],
        coder2_labels: list[str],
        labels: list[str] | None = None,
    ) -> ICRResult:
        """Cohen's Kappa for two coders with single-label classification.

        Parameters
        ----------
        coder1_labels, coder2_labels : list[str]
            Labels assigned by each coder (same length).
        labels : list[str] or None
            Explicit label set. If None, derived from the union.

        Returns
        -------
        ICRResult
        """
        if len(coder1_labels) != len(coder2_labels):
            raise ValueError(
                f"Length mismatch: {len(coder1_labels)} vs {len(coder2_labels)}."
            )

        n = len(coder1_labels)
        if n == 0:
            return ICRResult(
                metric_name="cohens_kappa", value=0.0,
                interpretation="No data", n_items=0,
            )

        c1 = [str(x).strip() for x in coder1_labels]
        c2 = [str(x).strip() for x in coder2_labels]

        if labels is None:
            labels = sorted(set(c1) | set(c2))
        label_idx = {l: i for i, l in enumerate(labels)}
        k = len(labels)

        # Build confusion matrix
        cm = [[0] * k for _ in range(k)]
        for a, b in zip(c1, c2):
            ai = label_idx.get(a)
            bi = label_idx.get(b)
            if ai is not None and bi is not None:
                cm[ai][bi] += 1

        # Observed agreement
        p_o = sum(cm[i][i] for i in range(k)) / n

        # Expected agreement by chance
        p_e = 0.0
        for i in range(k):
            row_sum = sum(cm[i][j] for j in range(k))
            col_sum = sum(cm[j][i] for j in range(k))
            p_e += (row_sum / n) * (col_sum / n)

        if p_e >= 1.0:
            kappa = 0.0
        else:
            kappa = (p_o - p_e) / (1.0 - p_e)

        return ICRResult(
            metric_name="cohens_kappa",
            value=round(kappa, 4),
            interpretation=self.interpret_kappa(kappa),
            n_coders=2,
            n_items=n,
            n_categories=k,
        )

    # ------------------------------------------------------------------
    # Krippendorff's Alpha (2+ coders, handles missing data)
    # ------------------------------------------------------------------

    def compute_krippendorffs_alpha(
        self,
        reliability_matrix: list[list[str | None]],
        data_type: str = "nominal",
    ) -> ICRResult:
        """Krippendorff's Alpha for 2+ coders.

        Parameters
        ----------
        reliability_matrix : list[list[str | None]]
            Shape: (n_items, n_coders). Cell = label or None for missing.
        data_type : str
            "nominal" (default). Others reserved for future.

        Returns
        -------
        ICRResult
        """
        if not reliability_matrix:
            return ICRResult(
                metric_name="krippendorffs_alpha", value=0.0,
                interpretation="No data", n_items=0,
            )

        n_items = len(reliability_matrix)
        n_coders = len(reliability_matrix[0]) if reliability_matrix else 0

        if data_type != "nominal":
            raise ValueError(
                f"Currently only 'nominal' data type is supported, got '{data_type}'."
            )

        # Collect all categories
        all_values: set[str] = set()
        for row in reliability_matrix:
            for v in row:
                if v is not None:
                    all_values.add(str(v))

        if len(all_values) <= 1:
            # All same value or empty — perfect agreement (by convention)
            return ICRResult(
                metric_name="krippendorffs_alpha", value=1.0,
                interpretation=self.interpret_alpha(1.0),
                n_coders=n_coders, n_items=n_items,
                n_categories=len(all_values),
            )

        # For each item, count the number of non-missing values
        # and the value frequencies
        item_counts: list[dict[str, int]] = []
        item_mu: list[int] = []  # number of coders per item (non-missing)

        for row in reliability_matrix:
            counts: dict[str, int] = Counter()
            mu = 0
            for v in row:
                if v is not None:
                    counts[str(v)] += 1
                    mu += 1
            item_counts.append(counts)
            item_mu.append(mu)

        # Only keep items where at least 2 coders provided values
        valid_items = [i for i in range(n_items) if item_mu[i] >= 2]

        if not valid_items:
            return ICRResult(
                metric_name="krippendorffs_alpha", value=0.0,
                interpretation="Insufficient data (need ≥2 coders per item)",
                n_coders=n_coders, n_items=0,
            )

        # Observed disagreement
        # D_o = (1 / sum_of_pairable_values) * sum_over_items(
        #     1/(m_u - 1) * sum_c_k( n_uc * n_uk * delta(c,k) )
        # )
        total_pairable = sum(item_mu[i] * (item_mu[i] - 1) for i in valid_items)

        if total_pairable == 0:
            return ICRResult(
                metric_name="krippendorffs_alpha", value=0.0,
                interpretation="Insufficient data",
                n_coders=n_coders, n_items=0,
            )

        d_observed = 0.0
        for i in valid_items:
            mu = item_mu[i]
            counts = item_counts[i]
            # Sum over all pairs of categories (c, k) where c != k
            for c, n_c in counts.items():
                for k, n_k in counts.items():
                    if c != k:
                        # For nominal: delta(c, k) = 1 when c != k
                        d_observed += n_c * n_k

        d_observed /= total_pairable

        # Expected disagreement
        # Marginal frequencies across all items
        n_total_values = sum(item_mu[i] for i in valid_items)
        marginal: Counter = Counter()
        for i in valid_items:
            for v, cnt in item_counts[i].items():
                marginal[v] += cnt

        d_expected = 0.0
        for c in marginal:
            for k in marginal:
                if c != k:
                    d_expected += marginal[c] * marginal[k]

        d_expected /= (n_total_values * (n_total_values - 1))

        if d_expected == 0:
            alpha = 1.0
        else:
            alpha = 1.0 - d_observed / d_expected

        return ICRResult(
            metric_name="krippendorffs_alpha",
            value=round(alpha, 4),
            interpretation=self.interpret_alpha(alpha),
            n_coders=n_coders,
            n_items=len(valid_items),
            n_categories=len(all_values),
        )

    # ------------------------------------------------------------------
    # Multi-label Jaccard agreement (QualiKit)
    # ------------------------------------------------------------------

    def compute_multilabel_agreement(
        self,
        coder1_themes: list[set[str]],
        coder2_themes: list[set[str]],
    ) -> ICRResult:
        """Jaccard-based agreement for multi-label coding.

        For each segment, agreement = |intersection| / |union|.
        If both coders assign no themes, agreement = 1.0 (both agree: no themes).
        Returns the average across all segments.

        Parameters
        ----------
        coder1_themes, coder2_themes : list[set[str]]
            Theme sets per segment (same length).

        Returns
        -------
        ICRResult
        """
        if len(coder1_themes) != len(coder2_themes):
            raise ValueError(
                f"Length mismatch: {len(coder1_themes)} vs {len(coder2_themes)}."
            )

        n = len(coder1_themes)
        if n == 0:
            return ICRResult(
                metric_name="jaccard_agreement", value=0.0,
                interpretation="No data", n_items=0,
            )

        total_jaccard = 0.0
        for s1, s2 in zip(coder1_themes, coder2_themes):
            if not s1 and not s2:
                total_jaccard += 1.0
            elif not s1 or not s2:
                total_jaccard += 0.0
            else:
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                total_jaccard += intersection / union if union > 0 else 0.0

        avg_jaccard = total_jaccard / n
        all_themes = set()
        for s in coder1_themes + coder2_themes:
            all_themes |= s

        return ICRResult(
            metric_name="jaccard_agreement",
            value=round(avg_jaccard, 4),
            interpretation=self._interpret_jaccard(avg_jaccard),
            n_coders=2,
            n_items=n,
            n_categories=len(all_themes),
        )

    # ------------------------------------------------------------------
    # Per-category agreement (single-label)
    # ------------------------------------------------------------------

    def _compute_per_category(
        self,
        coder1_labels: list[str],
        coder2_labels: list[str],
        labels: list[str],
    ) -> list[PerCategoryAgreement]:
        """Compute category-specific agreement for each label."""
        n = len(coder1_labels)
        if n == 0:
            return []

        result = []
        for cat in labels:
            # Binary: does coder assign this category or not?
            c1_binary = [1 if l == cat else 0 for l in coder1_labels]
            c2_binary = [1 if l == cat else 0 for l in coder2_labels]

            # Observed agreement for this category
            agree = sum(1 for a, b in zip(c1_binary, c2_binary) if a == b)
            p_o = agree / n

            # Expected agreement
            p1 = sum(c1_binary) / n
            p2 = sum(c2_binary) / n
            p_e = p1 * p2 + (1 - p1) * (1 - p2)

            # Specific agreement: proportion of cases where both say "yes"
            both_yes = sum(1 for a, b in zip(c1_binary, c2_binary) if a == 1 and b == 1)
            either_yes = sum(1 for a, b in zip(c1_binary, c2_binary) if a == 1 or b == 1)
            specific = both_yes / either_yes if either_yes > 0 else 0.0

            result.append(PerCategoryAgreement(
                category=cat,
                observed_agreement=round(p_o, 4),
                expected_agreement=round(p_e, 4),
                specific_agreement=round(specific, 4),
            ))

        return result

    # ------------------------------------------------------------------
    # Comprehensive reports
    # ------------------------------------------------------------------

    def compute_all(
        self,
        coder1_labels: list[str],
        coder2_labels: list[str],
        labels: list[str] | None = None,
    ) -> ICRReport:
        """Compute Cohen's Kappa, Krippendorff's Alpha, and per-category.

        Parameters
        ----------
        coder1_labels, coder2_labels : list[str]
        labels : list[str] or None

        Returns
        -------
        ICRReport
        """
        if labels is None:
            labels = sorted(
                set(str(x).strip() for x in coder1_labels)
                | set(str(x).strip() for x in coder2_labels)
            )

        kappa_result = self.compute_cohens_kappa(coder1_labels, coder2_labels, labels)

        # Build reliability matrix for Krippendorff's Alpha
        c1 = [str(x).strip() for x in coder1_labels]
        c2 = [str(x).strip() for x in coder2_labels]
        reliability_matrix = [[a, b] for a, b in zip(c1, c2)]
        alpha_result = self.compute_krippendorffs_alpha(reliability_matrix, "nominal")

        per_category = self._compute_per_category(c1, c2, labels)

        report = ICRReport(
            results=[kappa_result, alpha_result],
            per_category=per_category,
            coder_labels=["Coder 1", "Coder 2"],
        )
        report.summary_text = self.format_report(report)
        return report

    def compute_all_multilabel(
        self,
        coder1_themes: list[set[str]],
        coder2_themes: list[set[str]],
        all_themes: list[str] | None = None,
    ) -> ICRReport:
        """Full report for multi-label coding comparison.

        Computes Jaccard agreement and per-theme binary Kappa.

        Parameters
        ----------
        coder1_themes, coder2_themes : list[set[str]]
        all_themes : list[str] or None

        Returns
        -------
        ICRReport
        """
        jaccard_result = self.compute_multilabel_agreement(coder1_themes, coder2_themes)

        if all_themes is None:
            all_t: set[str] = set()
            for s in coder1_themes + coder2_themes:
                all_t |= s
            all_themes = sorted(all_t)

        # Per-theme binary Kappa
        per_category = []
        per_theme_kappas = []
        for theme in all_themes:
            c1_binary = ["yes" if theme in s else "no" for s in coder1_themes]
            c2_binary = ["yes" if theme in s else "no" for s in coder2_themes]
            kappa_r = self.compute_cohens_kappa(c1_binary, c2_binary, ["yes", "no"])
            per_theme_kappas.append(kappa_r)

            # Specific agreement for this theme
            n = len(coder1_themes)
            both_yes = sum(
                1 for s1, s2 in zip(coder1_themes, coder2_themes)
                if theme in s1 and theme in s2
            )
            either_yes = sum(
                1 for s1, s2 in zip(coder1_themes, coder2_themes)
                if theme in s1 or theme in s2
            )
            specific = both_yes / either_yes if either_yes > 0 else 0.0

            per_category.append(PerCategoryAgreement(
                category=theme,
                observed_agreement=round(
                    sum(1 for a, b in zip(c1_binary, c2_binary) if a == b) / n, 4
                ) if n > 0 else 0.0,
                expected_agreement=0.0,  # not as meaningful for multi-label
                specific_agreement=round(specific, 4),
            ))

        # Average per-theme Kappa as a summary metric
        valid_kappas = [r.value for r in per_theme_kappas if r.n_items > 0]
        avg_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else 0.0
        avg_kappa_result = ICRResult(
            metric_name="avg_per_theme_kappa",
            value=round(avg_kappa, 4),
            interpretation=self.interpret_kappa(avg_kappa),
            n_coders=2,
            n_items=len(coder1_themes),
            n_categories=len(all_themes),
        )

        report = ICRReport(
            results=[jaccard_result, avg_kappa_result],
            per_category=per_category,
            coder_labels=["Coder 1", "Coder 2"],
        )
        report.summary_text = self.format_report(report, multilabel=True)
        return report

    # ------------------------------------------------------------------
    # Interpretation scales
    # ------------------------------------------------------------------

    @staticmethod
    def interpret_kappa(value: float) -> str:
        """Landis & Koch (1977) interpretation scale for Kappa."""
        if value < 0:
            return "Poor agreement"
        elif value < 0.21:
            return "Slight agreement"
        elif value < 0.41:
            return "Fair agreement"
        elif value < 0.61:
            return "Moderate agreement"
        elif value < 0.81:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    @staticmethod
    def interpret_alpha(value: float) -> str:
        """Krippendorff interpretation scale for Alpha."""
        if value < 0.667:
            return "Unreliable — discard or recode"
        elif value < 0.8:
            return "Tentatively reliable"
        else:
            return "Reliable"

    @staticmethod
    def _interpret_jaccard(value: float) -> str:
        """Interpret average Jaccard agreement."""
        if value < 0.2:
            return "Poor agreement"
        elif value < 0.4:
            return "Fair agreement"
        elif value < 0.6:
            return "Moderate agreement"
        elif value < 0.8:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_report(
        report: ICRReport,
        lang: str = "zh",
        multilabel: bool = False,
    ) -> str:
        """Format an ICRReport as a human-readable string."""
        lines: list[str] = []

        if lang == "zh":
            lines.append("═══ 编码者间信度报告 ═══")
        else:
            lines.append("═══ Inter-Coder Reliability Report ═══")
        lines.append("")

        for r in report.results:
            label_map = {
                "cohens_kappa": "Cohen's Kappa",
                "krippendorffs_alpha": "Krippendorff's Alpha",
                "jaccard_agreement": "Jaccard Agreement (avg)" if lang == "en"
                    else "Jaccard 一致性（均值）",
                "avg_per_theme_kappa": "Avg Per-Theme Kappa" if lang == "en"
                    else "各主题 Kappa 均值",
            }
            name = label_map.get(r.metric_name, r.metric_name)
            lines.append(f"{name}: {r.value:.4f}")
            if lang == "zh":
                lines.append(f"  解释: {r.interpretation}")
            else:
                lines.append(f"  Interpretation: {r.interpretation}")
            if r.n_items > 0:
                items_label = "项目数" if lang == "zh" else "Items"
                cats_label = "类别数" if lang == "zh" else "Categories"
                lines.append(f"  {items_label}: {r.n_items}  |  {cats_label}: {r.n_categories}")
            lines.append("")

        if report.per_category:
            if multilabel:
                header_label = "各主题一致性" if lang == "zh" else "Per-Theme Agreement"
            else:
                header_label = "各类别一致性" if lang == "zh" else "Per-Category Agreement"
            lines.append(f"── {header_label} ──")

            cat_label = "类别" if lang == "zh" else "Category"
            obs_label = "观测一致" if lang == "zh" else "Observed"
            spec_label = "特定一致" if lang == "zh" else "Specific"
            lines.append(f"{cat_label:<24} {obs_label:>10} {spec_label:>10}")
            lines.append("─" * 46)
            for pc in report.per_category:
                lines.append(
                    f"{pc.category:<24} {pc.observed_agreement:>10.4f} "
                    f"{pc.specific_agreement:>10.4f}"
                )
            lines.append("")

        return "\n".join(lines)
