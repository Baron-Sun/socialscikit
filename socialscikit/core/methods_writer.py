"""Methods Section Auto-generation — template-based methods paragraph for papers.

Generates a Methods paragraph draft from pipeline metadata. Template-based
(no LLM calls) so every number and method name is deterministic and verifiable.

Outputs both English and Chinese versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Pipeline metadata
# ---------------------------------------------------------------------------


@dataclass
class QuantiKitPipelineMetadata:
    """Metadata collected from a QuantiKit analysis session."""

    dataset_name: str = ""
    n_samples: int = 0
    n_classes: int = 0
    class_labels: list[str] = field(default_factory=list)
    classification_method: str = ""  # "zero-shot" | "few-shot" | "fine-tune-local" | "fine-tune-api"
    model_name: str = ""
    model_backend: str = ""  # "openai" | "anthropic" | "ollama"
    n_annotations: int = 0
    prompt_optimization_used: bool = False
    n_prompt_variants: int = 0
    # Evaluation metrics
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    cohens_kappa: float = 0.0
    # ICR metrics (from ICR module, if run)
    icr_kappa: float = 0.0
    icr_alpha: float = 0.0


@dataclass
class QualiKitPipelineMetadata:
    """Metadata collected from a QualiKit analysis session."""

    dataset_name: str = ""
    n_segments: int = 0
    deidentification_performed: bool = False
    n_pii_detected: int = 0
    n_themes: int = 0
    theme_names: list[str] = field(default_factory=list)
    coding_model_name: str = ""
    coding_model_backend: str = ""
    # Consensus
    consensus_coding_used: bool = False
    n_consensus_models: int = 0
    consensus_model_names: list[str] = field(default_factory=list)
    consensus_agreement: float = 0.0
    # Confidence tiers
    n_high_confidence: int = 0
    n_medium_confidence: int = 0
    n_low_confidence: int = 0
    # Review
    n_accepted: int = 0
    n_rejected: int = 0
    n_edited: int = 0
    # ICR
    icr_jaccard: float = 0.0
    icr_per_theme_kappa: float = 0.0


@dataclass
class MethodsSection:
    """Generated methods section output."""

    text_en: str = ""
    text_zh: str = ""
    metadata_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Methods Writer
# ---------------------------------------------------------------------------


class MethodsWriter:
    """Generate a Methods section paragraph for academic papers.

    Uses template-based generation with slot-filling from pipeline metadata.
    No LLM calls — purely deterministic to ensure accuracy.

    Usage::

        writer = MethodsWriter()
        meta = QuantiKitPipelineMetadata(
            n_samples=5000, n_classes=3, classification_method="few-shot",
            model_name="gpt-4o", accuracy=0.87, macro_f1=0.85,
        )
        section = writer.generate_quantikit_methods(meta)
        print(section.text_en)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_quantikit_methods(
        self, metadata: QuantiKitPipelineMetadata,
    ) -> MethodsSection:
        """Generate methods section for a QuantiKit text classification analysis."""
        return MethodsSection(
            text_en=self._build_quantikit_en(metadata),
            text_zh=self._build_quantikit_zh(metadata),
            metadata_used=self._meta_to_dict(metadata),
        )

    def generate_qualikit_methods(
        self, metadata: QualiKitPipelineMetadata,
    ) -> MethodsSection:
        """Generate methods section for a QualiKit qualitative coding analysis."""
        return MethodsSection(
            text_en=self._build_qualikit_en(metadata),
            text_zh=self._build_qualikit_zh(metadata),
            metadata_used=self._meta_to_dict(metadata),
        )

    # ------------------------------------------------------------------
    # QuantiKit — English
    # ------------------------------------------------------------------

    def _build_quantikit_en(self, m: QuantiKitPipelineMetadata) -> str:
        parts: list[str] = []

        # Opening
        n_str = f"{m.n_samples:,}" if m.n_samples else "N"
        classes_str = f"{m.n_classes}" if m.n_classes else "multiple"
        labels_str = (
            f" ({', '.join(m.class_labels)})" if m.class_labels else ""
        )
        parts.append(
            f"We classified {n_str} text samples into {classes_str} "
            f"categories{labels_str} using SocialSciKit (Sun et al., 2026)."
        )

        # Classification method
        method_desc = self._method_desc_en(m.classification_method, m.model_name, m.model_backend)
        if method_desc:
            parts.append(method_desc)

        # Annotations
        if m.n_annotations > 0:
            parts.append(
                f"A total of {m.n_annotations} samples were manually annotated "
                f"to serve as the training/evaluation set."
            )

        # Prompt optimization
        if m.prompt_optimization_used:
            opt_str = (
                f" across {m.n_prompt_variants} prompt variants"
                if m.n_prompt_variants > 1 else ""
            )
            parts.append(
                f"Automated Prompt Engineering (APE) was applied to optimize "
                f"classification prompts{opt_str}."
            )

        # Evaluation
        eval_parts = []
        if m.accuracy > 0:
            eval_parts.append(f"accuracy of {m.accuracy:.2%}")
        if m.macro_f1 > 0:
            eval_parts.append(f"macro-F1 of {m.macro_f1:.4f}")
        if m.weighted_f1 > 0:
            eval_parts.append(f"weighted-F1 of {m.weighted_f1:.4f}")
        if m.cohens_kappa > 0:
            eval_parts.append(f"Cohen's Kappa of {m.cohens_kappa:.4f}")
        if eval_parts:
            parts.append(
                f"The model achieved {', '.join(eval_parts)} on the held-out "
                f"evaluation set."
            )

        # ICR
        icr_parts = []
        if m.icr_kappa > 0:
            icr_parts.append(f"Cohen's Kappa = {m.icr_kappa:.4f}")
        if m.icr_alpha > 0:
            icr_parts.append(f"Krippendorff's Alpha = {m.icr_alpha:.4f}")
        if icr_parts:
            parts.append(
                f"Inter-coder reliability was assessed ({', '.join(icr_parts)})."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # QuantiKit — Chinese
    # ------------------------------------------------------------------

    def _build_quantikit_zh(self, m: QuantiKitPipelineMetadata) -> str:
        parts: list[str] = []

        n_str = f"{m.n_samples:,}" if m.n_samples else "N"
        classes_str = f"{m.n_classes}" if m.n_classes else "多个"
        labels_str = (
            f"（{', '.join(m.class_labels)}）" if m.class_labels else ""
        )
        parts.append(
            f"本研究使用 SocialSciKit（Sun et al., 2026）对 {n_str} 条文本进行"
            f" {classes_str} 分类{labels_str}。"
        )

        method_desc = self._method_desc_zh(m.classification_method, m.model_name, m.model_backend)
        if method_desc:
            parts.append(method_desc)

        if m.n_annotations > 0:
            parts.append(f"共人工标注 {m.n_annotations} 条样本作为训练/评估数据。")

        if m.prompt_optimization_used:
            opt_str = f"，共测试 {m.n_prompt_variants} 种 Prompt 变体" if m.n_prompt_variants > 1 else ""
            parts.append(f"使用自动提示工程（APE）优化分类 Prompt{opt_str}。")

        eval_parts = []
        if m.accuracy > 0:
            eval_parts.append(f"准确率 {m.accuracy:.2%}")
        if m.macro_f1 > 0:
            eval_parts.append(f"宏平均 F1 = {m.macro_f1:.4f}")
        if m.weighted_f1 > 0:
            eval_parts.append(f"加权 F1 = {m.weighted_f1:.4f}")
        if m.cohens_kappa > 0:
            eval_parts.append(f"Cohen's Kappa = {m.cohens_kappa:.4f}")
        if eval_parts:
            parts.append(f"模型在评估集上的表现为：{', '.join(eval_parts)}。")

        icr_parts = []
        if m.icr_kappa > 0:
            icr_parts.append(f"Cohen's Kappa = {m.icr_kappa:.4f}")
        if m.icr_alpha > 0:
            icr_parts.append(f"Krippendorff's Alpha = {m.icr_alpha:.4f}")
        if icr_parts:
            parts.append(f"编码者间信度检验：{', '.join(icr_parts)}。")

        return "".join(parts)

    # ------------------------------------------------------------------
    # QualiKit — English
    # ------------------------------------------------------------------

    def _build_qualikit_en(self, m: QualiKitPipelineMetadata) -> str:
        parts: list[str] = []

        n_str = f"{m.n_segments:,}" if m.n_segments else "N"
        parts.append(
            f"Qualitative coding was performed on {n_str} text segments "
            f"using SocialSciKit (Sun et al., 2026)."
        )

        # De-identification
        if m.deidentification_performed:
            pii_str = (
                f", detecting and masking {m.n_pii_detected} personally "
                f"identifiable items" if m.n_pii_detected > 0 else ""
            )
            parts.append(
                f"Prior to coding, all texts were de-identified{pii_str}."
            )

        # Research framework
        if m.n_themes > 0:
            themes_str = (
                f" ({', '.join(m.theme_names[:5])}{'...' if len(m.theme_names) > 5 else ''})"
                if m.theme_names else ""
            )
            parts.append(
                f"A coding framework with {m.n_themes} themes was defined{themes_str}."
            )

        # Coding method
        if m.consensus_coding_used and m.n_consensus_models >= 2:
            models_str = ", ".join(m.consensus_model_names) if m.consensus_model_names else f"{m.n_consensus_models} models"
            parts.append(
                f"Multi-LLM consensus coding was employed: {models_str} independently "
                f"coded each segment, and themes were retained only when a majority "
                f"of coders agreed (overall agreement: {m.consensus_agreement:.2%})."
            )
        elif m.coding_model_name:
            backend_str = f" ({m.coding_model_backend})" if m.coding_model_backend else ""
            parts.append(
                f"LLM-assisted coding was performed using {m.coding_model_name}"
                f"{backend_str}."
            )

        # Confidence tiers
        total_conf = m.n_high_confidence + m.n_medium_confidence + m.n_low_confidence
        if total_conf > 0:
            parts.append(
                f"Coding confidence was categorized into three tiers: "
                f"high ({m.n_high_confidence}), medium ({m.n_medium_confidence}), "
                f"and low ({m.n_low_confidence})."
            )

        # Human review
        total_reviewed = m.n_accepted + m.n_rejected + m.n_edited
        if total_reviewed > 0:
            parts.append(
                f"Human review was conducted on all coded segments: "
                f"{m.n_accepted} accepted, {m.n_rejected} rejected, "
                f"and {m.n_edited} manually edited."
            )

        # ICR
        icr_parts = []
        if m.icr_jaccard > 0:
            icr_parts.append(f"Jaccard agreement = {m.icr_jaccard:.4f}")
        if m.icr_per_theme_kappa > 0:
            icr_parts.append(f"average per-theme Kappa = {m.icr_per_theme_kappa:.4f}")
        if icr_parts:
            parts.append(
                f"Inter-coder reliability between human review and LLM coding "
                f"was assessed ({', '.join(icr_parts)})."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # QualiKit — Chinese
    # ------------------------------------------------------------------

    def _build_qualikit_zh(self, m: QualiKitPipelineMetadata) -> str:
        parts: list[str] = []

        n_str = f"{m.n_segments:,}" if m.n_segments else "N"
        parts.append(
            f"本研究使用 SocialSciKit（Sun et al., 2026）对 {n_str} 条文本段落"
            f"进行质性编码。"
        )

        if m.deidentification_performed:
            pii_str = f"，共检测并脱敏 {m.n_pii_detected} 项个人信息" if m.n_pii_detected > 0 else ""
            parts.append(f"编码前对所有文本进行了脱敏处理{pii_str}。")

        if m.n_themes > 0:
            themes_str = (
                f"（{', '.join(m.theme_names[:5])}{'...' if len(m.theme_names) > 5 else ''}）"
                if m.theme_names else ""
            )
            parts.append(f"定义了包含 {m.n_themes} 个主题的编码框架{themes_str}。")

        if m.consensus_coding_used and m.n_consensus_models >= 2:
            models_str = "、".join(m.consensus_model_names) if m.consensus_model_names else f"{m.n_consensus_models} 个模型"
            parts.append(
                f"采用多模型共识编码策略：{models_str}分别独立编码，"
                f"仅保留多数模型一致的主题标签（总体一致性：{m.consensus_agreement:.2%}）。"
            )
        elif m.coding_model_name:
            backend_str = f"（{m.coding_model_backend}）" if m.coding_model_backend else ""
            parts.append(f"使用 {m.coding_model_name}{backend_str} 进行 LLM 辅助编码。")

        total_conf = m.n_high_confidence + m.n_medium_confidence + m.n_low_confidence
        if total_conf > 0:
            parts.append(
                f"编码置信度分为三档：高（{m.n_high_confidence} 条）、"
                f"中（{m.n_medium_confidence} 条）、低（{m.n_low_confidence} 条）。"
            )

        total_reviewed = m.n_accepted + m.n_rejected + m.n_edited
        if total_reviewed > 0:
            parts.append(
                f"对全部编码进行人工审核：接受 {m.n_accepted} 条、"
                f"拒绝 {m.n_rejected} 条、手动编辑 {m.n_edited} 条。"
            )

        icr_parts = []
        if m.icr_jaccard > 0:
            icr_parts.append(f"Jaccard 一致性 = {m.icr_jaccard:.4f}")
        if m.icr_per_theme_kappa > 0:
            icr_parts.append(f"各主题平均 Kappa = {m.icr_per_theme_kappa:.4f}")
        if icr_parts:
            parts.append(f"人工审核与 LLM 编码的编码者间信度：{', '.join(icr_parts)}。")

        return "".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _method_desc_en(method: str, model: str, backend: str) -> str:
        """Build the classification method sentence (English)."""
        model_str = model or "the language model"
        backend_str = f" ({backend})" if backend else ""

        method_map = {
            "zero-shot": (
                f"A zero-shot prompting approach was used with "
                f"{model_str}{backend_str}, where no labeled examples were "
                f"provided in the prompt."
            ),
            "few-shot": (
                f"A few-shot prompting approach was used with "
                f"{model_str}{backend_str}, where labeled examples were "
                f"included in the prompt for in-context learning."
            ),
            "fine-tune-local": (
                f"The model {model_str} was fine-tuned locally on the "
                f"annotated training data."
            ),
            "fine-tune-api": (
                f"The model {model_str}{backend_str} was fine-tuned via "
                f"the provider's API on the annotated training data."
            ),
        }
        return method_map.get(method, "")

    @staticmethod
    def _method_desc_zh(method: str, model: str, backend: str) -> str:
        """Build the classification method sentence (Chinese)."""
        model_str = model or "语言模型"
        backend_str = f"（{backend}）" if backend else ""

        method_map = {
            "zero-shot": (
                f"采用零样本提示（zero-shot prompting）方法，"
                f"使用 {model_str}{backend_str}，Prompt 中不包含标注示例。"
            ),
            "few-shot": (
                f"采用少样本提示（few-shot prompting）方法，"
                f"使用 {model_str}{backend_str}，Prompt 中包含标注示例进行上下文学习。"
            ),
            "fine-tune-local": (
                f"在标注训练数据上对 {model_str} 进行本地微调。"
            ),
            "fine-tune-api": (
                f"通过 API 在标注训练数据上对 {model_str}{backend_str} 进行微调。"
            ),
        }
        return method_map.get(method, "")

    @staticmethod
    def _meta_to_dict(metadata: object) -> dict:
        """Convert a metadata dataclass to a plain dict."""
        if hasattr(metadata, "__dataclass_fields__"):
            from dataclasses import asdict
            return asdict(metadata)
        return {}
