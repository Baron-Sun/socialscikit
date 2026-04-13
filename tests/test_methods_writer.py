"""Tests for the Methods Section Auto-generation module."""

import pytest

from socialscikit.core.methods_writer import (
    MethodsSection,
    MethodsWriter,
    QuantiKitPipelineMetadata,
    QualiKitPipelineMetadata,
)


@pytest.fixture
def writer():
    return MethodsWriter()


# =====================================================================
# QuantiKit Methods
# =====================================================================


class TestQuantiKitMethods:
    def test_full_metadata_en(self, writer):
        meta = QuantiKitPipelineMetadata(
            dataset_name="twitter_sentiment",
            n_samples=5000,
            n_classes=3,
            class_labels=["positive", "negative", "neutral"],
            classification_method="few-shot",
            model_name="gpt-4o",
            model_backend="openai",
            n_annotations=500,
            prompt_optimization_used=True,
            n_prompt_variants=5,
            accuracy=0.87,
            macro_f1=0.85,
            weighted_f1=0.86,
            cohens_kappa=0.80,
            icr_kappa=0.78,
            icr_alpha=0.82,
        )
        section = writer.generate_quantikit_methods(meta)
        assert isinstance(section, MethodsSection)
        assert "5,000" in section.text_en
        assert "3 categories" in section.text_en
        assert "few-shot" in section.text_en
        assert "gpt-4o" in section.text_en
        assert "500 samples" in section.text_en
        assert "APE" in section.text_en or "Prompt Engineering" in section.text_en
        assert "87" in section.text_en  # accuracy
        assert "0.85" in section.text_en  # macro f1
        assert "Kappa" in section.text_en
        assert "Alpha" in section.text_en
        assert section.metadata_used  # not empty

    def test_full_metadata_zh(self, writer):
        meta = QuantiKitPipelineMetadata(
            n_samples=3000,
            n_classes=2,
            class_labels=["正面", "负面"],
            classification_method="zero-shot",
            model_name="gpt-4o-mini",
            model_backend="openai",
            accuracy=0.92,
            macro_f1=0.90,
        )
        section = writer.generate_quantikit_methods(meta)
        assert "SocialSciKit" in section.text_zh
        assert "3,000" in section.text_zh
        assert "零样本" in section.text_zh
        assert "92" in section.text_zh

    def test_minimal_metadata(self, writer):
        meta = QuantiKitPipelineMetadata(n_samples=100, n_classes=2)
        section = writer.generate_quantikit_methods(meta)
        assert "100" in section.text_en
        assert "2 categories" in section.text_en
        assert "SocialSciKit" in section.text_en
        # Should not contain eval metrics
        assert "accuracy" not in section.text_en.lower() or "N/A" in section.text_en

    def test_zero_shot_method(self, writer):
        meta = QuantiKitPipelineMetadata(
            classification_method="zero-shot", model_name="gpt-4o",
        )
        section = writer.generate_quantikit_methods(meta)
        assert "zero-shot" in section.text_en.lower()

    def test_finetune_method(self, writer):
        meta = QuantiKitPipelineMetadata(
            classification_method="fine-tune-api", model_name="gpt-4o",
            model_backend="openai",
        )
        section = writer.generate_quantikit_methods(meta)
        assert "fine-tuned" in section.text_en.lower()

    def test_with_icr(self, writer):
        meta = QuantiKitPipelineMetadata(icr_kappa=0.75, icr_alpha=0.80)
        section = writer.generate_quantikit_methods(meta)
        assert "Inter-coder" in section.text_en
        assert "0.75" in section.text_en
        assert "0.80" in section.text_en

    def test_without_eval(self, writer):
        meta = QuantiKitPipelineMetadata(n_samples=50)
        section = writer.generate_quantikit_methods(meta)
        assert "achieved" not in section.text_en.lower()


# =====================================================================
# QualiKit Methods
# =====================================================================


class TestQualiKitMethods:
    def test_full_metadata_en(self, writer):
        meta = QualiKitPipelineMetadata(
            dataset_name="interviews",
            n_segments=200,
            deidentification_performed=True,
            n_pii_detected=45,
            n_themes=6,
            theme_names=["theme1", "theme2", "theme3", "theme4", "theme5", "theme6"],
            coding_model_name="gpt-4o",
            coding_model_backend="openai",
            n_high_confidence=120,
            n_medium_confidence=50,
            n_low_confidence=30,
            n_accepted=150,
            n_rejected=10,
            n_edited=40,
            icr_jaccard=0.78,
            icr_per_theme_kappa=0.72,
        )
        section = writer.generate_qualikit_methods(meta)
        assert "200" in section.text_en
        assert "de-identified" in section.text_en.lower()
        assert "45" in section.text_en
        assert "6 themes" in section.text_en
        assert "gpt-4o" in section.text_en
        assert "high (120)" in section.text_en
        assert "150 accepted" in section.text_en
        assert "Jaccard" in section.text_en

    def test_full_metadata_zh(self, writer):
        meta = QualiKitPipelineMetadata(
            n_segments=100,
            n_themes=4,
            theme_names=["主题1", "主题2", "主题3", "主题4"],
            coding_model_name="gpt-4o-mini",
            n_high_confidence=60,
            n_medium_confidence=30,
            n_low_confidence=10,
        )
        section = writer.generate_qualikit_methods(meta)
        assert "SocialSciKit" in section.text_zh
        assert "100" in section.text_zh
        assert "4 个主题" in section.text_zh

    def test_with_consensus(self, writer):
        meta = QualiKitPipelineMetadata(
            n_segments=50,
            consensus_coding_used=True,
            n_consensus_models=3,
            consensus_model_names=["gpt-4o", "claude-sonnet", "llama3"],
            consensus_agreement=0.85,
        )
        section = writer.generate_qualikit_methods(meta)
        assert "consensus" in section.text_en.lower() or "Multi-LLM" in section.text_en
        assert "gpt-4o" in section.text_en
        assert "85" in section.text_en

    def test_with_deident(self, writer):
        meta = QualiKitPipelineMetadata(
            n_segments=30,
            deidentification_performed=True,
            n_pii_detected=12,
        )
        section = writer.generate_qualikit_methods(meta)
        assert "de-identified" in section.text_en.lower()
        assert "12" in section.text_en

    def test_without_consensus(self, writer):
        meta = QualiKitPipelineMetadata(
            n_segments=50,
            coding_model_name="claude-sonnet-4-20250514",
            coding_model_backend="anthropic",
        )
        section = writer.generate_qualikit_methods(meta)
        assert "consensus" not in section.text_en.lower()
        assert "claude" in section.text_en.lower()


# =====================================================================
# Edge Cases
# =====================================================================


class TestEdgeCases:
    def test_empty_quantikit_metadata(self, writer):
        meta = QuantiKitPipelineMetadata()
        section = writer.generate_quantikit_methods(meta)
        assert isinstance(section, MethodsSection)
        assert len(section.text_en) > 0
        assert len(section.text_zh) > 0

    def test_empty_qualikit_metadata(self, writer):
        meta = QualiKitPipelineMetadata()
        section = writer.generate_qualikit_methods(meta)
        assert isinstance(section, MethodsSection)
        assert len(section.text_en) > 0
        assert len(section.text_zh) > 0

    def test_metadata_to_dict(self, writer):
        meta = QuantiKitPipelineMetadata(n_samples=100, accuracy=0.9)
        section = writer.generate_quantikit_methods(meta)
        assert section.metadata_used["n_samples"] == 100
        assert section.metadata_used["accuracy"] == 0.9

    def test_methods_section_fields(self, writer):
        meta = QuantiKitPipelineMetadata(n_samples=10)
        section = writer.generate_quantikit_methods(meta)
        assert hasattr(section, "text_en")
        assert hasattr(section, "text_zh")
        assert hasattr(section, "metadata_used")
        assert isinstance(section.metadata_used, dict)
