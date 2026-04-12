"""Method recommendation engine — rule-based with literature citations.

Every decision rule is annotated with its literature source. The engine
takes ``TaskFeatures`` and returns a ``MethodRecommendation`` with a
user-facing explanation, estimated performance, cost, and alternatives.

Literature:
- Ziems et al. (2024). Can large language models transform computational
  social science? Computational Linguistics, 50(1), 237-291.
- Chae & Davidson (2025). Large language models for text classification:
  From zero-shot learning to instruction-tuning. SMR.
- Do, Ollion & Shen (2024). The augmented social scientist. SMR, 53(3).
- Carlson et al. (2026). The use of LLMs to annotate data in management
  research. Strategic Management Journal.
- Montgomery et al. (2024). Improving probabilistic models in text
  classification via active learning. APSR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socialscikit.quantikit.feature_extractor import TaskFeatures


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ColdStartReport:
    """Recommendation when no labeled data exists but high F1 is requested."""

    message: str  # user-facing explanation
    minimum_n: int  # lowest viable annotation count
    minimum_expected_f1: tuple[float, float]  # (point, uncertainty)
    recommended_n: int
    recommended_expected_f1: tuple[float, float]
    diminishing_returns_n: int  # annotation count where marginal gain flattens
    marginal_gain_after: float  # F1 gain per 100 samples after that point


@dataclass
class MethodRecommendation:
    """Full recommendation result."""

    recommended_method: str  # "zero_shot" | "few_shot" | "fine_tune" | "active_learning"
    confidence: str  # "high" | "medium" | "low"
    reasoning: str  # user-facing explanation (non-technical)
    literature_support: list[str] = field(default_factory=list)
    estimated_performance: tuple[float, float] = (0.0, 0.0)  # (point_estimate, uncertainty)
    estimated_cost: str = ""  # e.g. "$0.50 per 1000 samples"
    alternative_method: str | None = None
    alternative_reasoning: str | None = None
    cold_start_recommendation: ColdStartReport | None = None
    sensitivity_analysis_suggested: bool = False


# ---------------------------------------------------------------------------
# Literature references (reusable constants)
# ---------------------------------------------------------------------------

_REF_ZIEMS = (
    "Ziems et al. (2024). Can large language models transform "
    "computational social science? Computational Linguistics, 50(1), 237-291."
)
_REF_CHAE = (
    "Chae & Davidson (2025). Large language models for text classification: "
    "From zero-shot learning to instruction-tuning. SMR."
)
_REF_DO = (
    "Do, Ollion & Shen (2024). The augmented social scientist. SMR, 53(3)."
)
_REF_CARLSON = (
    "Carlson et al. (2026). The use of LLMs to annotate data in "
    "management research. Strategic Management Journal."
)
_REF_MONTGOMERY = (
    "Montgomery et al. (2024). Improving probabilistic models in text "
    "classification via active learning. APSR."
)

# Simple task types where zero-shot LLMs perform reasonably well
_SIMPLE_TASKS = {"sentiment"}

# ---------------------------------------------------------------------------
# Cost estimates (rough, per 1000 samples)
# ---------------------------------------------------------------------------

_COST_ESTIMATES = {
    "zero_shot": "$0.50–2.00 per 1000 samples (API cost)",
    "few_shot": "$1.00–4.00 per 1000 samples (API cost)",
    "fine_tune": "$5–20 compute + ~$0.10 per 1000 samples inference",
    "active_learning": "$5–20 compute + annotation labor cost",
}


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


class MethodRecommender:
    """Rule-based method recommendation engine.

    Decision rules are derived from the CSS literature. Each rule is
    documented in the code with its source.

    Usage::

        from socialscikit.quantikit.feature_extractor import TaskFeatures
        features = TaskFeatures(n_labeled=0, n_classes=2, task_type="sentiment")
        rec = MethodRecommender().recommend(features)
        print(rec.recommended_method, rec.reasoning)
    """

    def recommend(self, features: TaskFeatures) -> MethodRecommendation:
        """Generate a method recommendation based on task features.

        The decision tree is evaluated top-down; the first matching rule wins.
        """

        # --- Check sensitivity analysis flag ---
        # Carlson et al. (2026): hypothesis testing downstream use warrants
        # sensitivity analysis to ensure robustness of classification-based findings.
        sensitivity = features.downstream_use == "hypothesis_testing"

        # --- Multilingual check ---
        # Standard practice: multilingual data requires XLM-RoBERTa for fine-tuning.
        # This doesn't change the method but is noted in reasoning.
        multilingual_note = ""
        if features.is_multilingual:
            multilingual_note = "（检测到多语言数据，fine-tuning 将使用 XLM-RoBERTa。）"

        # ===============================================================
        # RULE 1: No labels, simple task (e.g. sentiment), ≤3 classes
        # → Zero-shot LLM
        # Source: Ziems et al. 2024 (κ ≈ 0.55 on simple tasks)
        # ===============================================================
        if (
            features.n_labeled == 0
            and features.task_type in _SIMPLE_TASKS
            and features.n_classes <= 3
            and features.target_f1 <= 0.85
        ):
            return MethodRecommendation(
                recommended_method="zero_shot",
                confidence="high",
                reasoning=(
                    "您的任务为情感分类（≤3 类），且无标注数据。"
                    "研究表明 LLM 在简单情感任务上零样本表现较好（预期 F1 ≈ 0.70–0.80）。"
                    "建议直接使用零样本分类。" + multilingual_note
                ),
                literature_support=[_REF_ZIEMS],
                estimated_performance=(0.75, 0.05),
                estimated_cost=_COST_ESTIMATES["zero_shot"],
                alternative_method="few_shot",
                alternative_reasoning="如有少量标注数据（10–30 条），few-shot 可进一步提升效果。",
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # RULE 2: No labels, complex/multi-class task
        # → Few-shot LLM + suggest annotation
        # Source: Chae & Davidson 2025 (zero-shot drops on fine-grained tasks)
        # ===============================================================
        if (
            features.n_labeled == 0
            and (features.task_type not in _SIMPLE_TASKS or features.n_classes > 3)
            and features.target_f1 <= 0.85
        ):
            return MethodRecommendation(
                recommended_method="few_shot",
                confidence="medium",
                reasoning=(
                    "您的任务较复杂（多类别或非情感任务），且无标注数据。"
                    "零样本在细粒度任务上效果显著下降（Chae & Davidson 2025），"
                    "建议先手动标注 10–30 条示例，使用 few-shot 分类。" + multilingual_note
                ),
                literature_support=[_REF_CHAE, _REF_ZIEMS],
                estimated_performance=(0.65, 0.08),
                estimated_cost=_COST_ESTIMATES["few_shot"],
                alternative_method="zero_shot",
                alternative_reasoning="如不便标注，也可尝试零样本，但预期精度较低。",
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # RULE 3: No labels + high target F1 (> 0.85)
        # → Cold start: recommend annotation budget
        # Source: Carlson et al. 2026; Chae & Davidson 2025
        # ===============================================================
        if features.n_labeled == 0 and features.target_f1 > 0.85:
            cold_start = self._build_cold_start(features)
            return MethodRecommendation(
                recommended_method="fine_tune",
                confidence="medium",
                reasoning=(
                    f"您的目标精度较高（F1 > {features.target_f1:.2f}），但当前没有标注数据。"
                    "直接使用 LLM 零样本分类在此类任务上预期 F1 ≈ 0.65–0.75。"
                    "如需达到目标精度，建议进行监督学习（fine-tuning），需要先标注数据。"
                    + multilingual_note
                ),
                literature_support=[_REF_CARLSON, _REF_CHAE, _REF_ZIEMS],
                estimated_performance=(0.70, 0.08),
                estimated_cost=_COST_ESTIMATES["fine_tune"],
                alternative_method="few_shot",
                alternative_reasoning="如暂时不便标注大量数据，可先用 few-shot 获得初步结果。",
                cold_start_recommendation=cold_start,
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # RULE 4: Small labeled set (1–199)
        # → Few-shot LLM (use labeled data as examples)
        # Source: Ziems et al. 2024
        # ===============================================================
        if 0 < features.n_labeled < 200:
            return MethodRecommendation(
                recommended_method="few_shot",
                confidence="high",
                reasoning=(
                    f"您有 {features.n_labeled} 条标注数据，适合用作 few-shot 示例。"
                    "在小样本场景下，LLM few-shot 通常优于传统模型（Ziems et al. 2024）。"
                    + multilingual_note
                ),
                literature_support=[_REF_ZIEMS],
                estimated_performance=(0.75, 0.05),
                estimated_cost=_COST_ESTIMATES["few_shot"],
                alternative_method="fine_tune",
                alternative_reasoning="如能补充标注至 500+ 条，fine-tuning 可能获得更高精度。",
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # RULE 5: Medium labeled set (200–499)
        # → Few-shot and fine-tune in parallel, compare
        # Source: Do et al. 2024
        # ===============================================================
        if 200 <= features.n_labeled < 500:
            return MethodRecommendation(
                recommended_method="few_shot",
                confidence="medium",
                reasoning=(
                    f"您有 {features.n_labeled} 条标注数据，处于 few-shot 和 fine-tuning 的交叉区间。"
                    "建议同时运行两种方法并在验证集上对比（Do et al. 2024）。"
                    + multilingual_note
                ),
                literature_support=[_REF_DO],
                estimated_performance=(0.78, 0.05),
                estimated_cost=_COST_ESTIMATES["few_shot"],
                alternative_method="fine_tune",
                alternative_reasoning="如验证集上 fine-tuning 优于 few-shot，可切换为 fine-tuning。",
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # RULE 6: Large labeled set (≥500), moderate target
        # → Fine-tune RoBERTa (or XLM-RoBERTa if multilingual)
        # Source: Do et al. 2024
        # ===============================================================
        if features.n_labeled >= 500 and features.target_f1 <= 0.85:
            model_name = "XLM-RoBERTa" if features.is_multilingual else "RoBERTa"
            return MethodRecommendation(
                recommended_method="fine_tune",
                confidence="high",
                reasoning=(
                    f"您有 {features.n_labeled} 条标注数据，足以支撑 fine-tuning。"
                    f"建议使用 {model_name} 进行微调（Do et al. 2024），"
                    f"在此数据量下预期 F1 ≈ 0.82–0.88。" + multilingual_note
                ),
                literature_support=[_REF_DO],
                estimated_performance=(0.85, 0.03),
                estimated_cost=_COST_ESTIMATES["fine_tune"],
                alternative_method="few_shot",
                alternative_reasoning="如计算资源有限，LLM few-shot 也可作为轻量替代。",
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # RULE 7: Large labeled set (≥500), high target (>0.85)
        # → Fine-tune + active learning
        # Source: Montgomery et al. 2024
        # ===============================================================
        if features.n_labeled >= 500 and features.target_f1 > 0.85:
            model_name = "XLM-RoBERTa" if features.is_multilingual else "RoBERTa"
            return MethodRecommendation(
                recommended_method="active_learning",
                confidence="high",
                reasoning=(
                    f"您有 {features.n_labeled} 条标注数据，目标精度较高（F1 > 0.85）。"
                    f"建议使用 {model_name} fine-tuning 结合主动学习（Montgomery et al. 2024），"
                    "主动学习可优先标注模型最不确定的样本，以最少标注量最大化精度提升。"
                    + multilingual_note
                ),
                literature_support=[_REF_MONTGOMERY, _REF_DO],
                estimated_performance=(0.87, 0.03),
                estimated_cost=_COST_ESTIMATES["active_learning"],
                alternative_method="fine_tune",
                alternative_reasoning="如不需要主动学习流程，直接 fine-tuning 也可获得较好结果。",
                sensitivity_analysis_suggested=sensitivity,
            )

        # ===============================================================
        # Fallback: shouldn't normally reach here
        # ===============================================================
        return MethodRecommendation(
            recommended_method="few_shot",
            confidence="low",
            reasoning=(
                "无法根据当前特征明确推荐方法，默认建议使用 few-shot 分类。"
                "请补充更多任务信息以获得更准确的推荐。"
            ),
            literature_support=[_REF_ZIEMS],
            estimated_performance=(0.70, 0.10),
            estimated_cost=_COST_ESTIMATES["few_shot"],
            sensitivity_analysis_suggested=sensitivity,
        )

    # ------------------------------------------------------------------
    # Cold-start report builder
    # ------------------------------------------------------------------

    def _build_cold_start(self, features: TaskFeatures) -> ColdStartReport:
        """Build annotation budget recommendation for cold-start scenarios.

        Estimates are based on typical learning curves from CSS benchmark
        datasets (Carlson et al. 2026; Do et al. 2024).
        """
        # Heuristics based on task complexity
        if features.n_classes <= 3 and features.task_type in _SIMPLE_TASKS:
            min_n, min_f1 = 200, (0.78, 0.05)
            rec_n, rec_f1 = 500, (0.84, 0.03)
            dim_n, dim_gain = 1000, 0.005
        elif features.n_classes <= 5:
            min_n, min_f1 = 300, (0.75, 0.05)
            rec_n, rec_f1 = 800, (0.83, 0.03)
            dim_n, dim_gain = 1500, 0.005
        else:
            min_n, min_f1 = 500, (0.72, 0.06)
            rec_n, rec_f1 = 1200, (0.82, 0.04)
            dim_n, dim_gain = 2000, 0.003

        message = (
            f"检测到：您的任务目标精度 F1 > {features.target_f1:.2f}，"
            "但当前没有标注数据。\n\n"
            "直接使用 LLM 零样本分类在此类任务上预期 F1 ≈ 0.65–0.75"
            "（Chae & Davidson 2025；Ziems et al. 2024）。\n\n"
            f"如需达到 F1 > {features.target_f1:.2f}，建议进行监督学习（fine-tuning）。\n"
            "根据您的任务特征，推荐标注量估算：\n\n"
            f"  最低可用：约 {min_n} 条 → 预期 F1 ≈ {min_f1[0]:.2f}（±{min_f1[1]:.2f}）\n"
            f"  推荐标注：约 {rec_n} 条 → 预期 F1 ≈ {rec_f1[0]:.2f}（±{rec_f1[1]:.2f}）\n"
            f"  边际收益趋平：约 {dim_n} 条后每 100 条提升 < {dim_gain:.3f}"
        )

        return ColdStartReport(
            message=message,
            minimum_n=min_n,
            minimum_expected_f1=min_f1,
            recommended_n=rec_n,
            recommended_expected_f1=rec_f1,
            diminishing_returns_n=dim_n,
            marginal_gain_after=dim_gain,
        )
