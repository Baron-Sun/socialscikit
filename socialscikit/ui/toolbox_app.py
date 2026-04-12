"""Toolbox tab callbacks — standalone ICR, Consensus Coding, and Methods Generator."""

from __future__ import annotations

import json
import logging
import tempfile

import pandas as pd

from socialscikit.core.icr import ICRCalculator
from socialscikit.core.llm_client import LLMClient
from socialscikit.core.methods_writer import (
    MethodsWriter,
    QuantiKitPipelineMetadata,
    QualiKitPipelineMetadata,
)
from socialscikit.qualikit.consensus import ConsensusCoder
from socialscikit.qualikit.theme_definer import Theme

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ICR Calculator
# ---------------------------------------------------------------------------


def _compute_icr_from_files(file1, col1, file2, col2, mode):
    """Standalone ICR: upload two CSV files, pick label columns, compute agreement.

    Parameters
    ----------
    file1, file2 : uploaded CSV files (Gradio File objects)
    col1, col2 : column names containing labels
    mode : "single-label" or "multi-label"
    """
    if file1 is None or file2 is None:
        return "Please upload both CSV files."

    try:
        df1 = pd.read_csv(file1.name if hasattr(file1, "name") else file1)
        df2 = pd.read_csv(file2.name if hasattr(file2, "name") else file2)
    except Exception as e:
        return f"Failed to read CSV: {e}"

    c1 = (col1 or "label").strip()
    c2 = (col2 or "label").strip()

    if c1 not in df1.columns:
        return f"Column '{c1}' not found in File 1. Available: {', '.join(df1.columns[:10])}"
    if c2 not in df2.columns:
        return f"Column '{c2}' not found in File 2. Available: {', '.join(df2.columns[:10])}"

    labels1 = df1[c1].astype(str).tolist()
    labels2 = df2[c2].astype(str).tolist()

    min_len = min(len(labels1), len(labels2))
    if min_len == 0:
        return "No data found in the specified columns."
    labels1 = labels1[:min_len]
    labels2 = labels2[:min_len]

    calc = ICRCalculator()

    if mode == "multi-label":
        # For multi-label: split comma-separated values into sets
        themes1 = [set(s.strip() for s in v.split(",") if s.strip()) for v in labels1]
        themes2 = [set(s.strip() for s in v.split(",") if s.strip()) for v in labels2]
        report = calc.compute_all_multilabel(themes1, themes2)
    else:
        report = calc.compute_all(labels1, labels2)

    return report.summary_text


# ---------------------------------------------------------------------------
# Consensus Coding
# ---------------------------------------------------------------------------


def _run_standalone_consensus(
    data_file, text_col, themes_text,
    b1, m1, k1, b2, m2, k2, b3, m3, k3,
):
    """Standalone consensus coding: upload CSV + define themes, configure 2-3 LLMs."""
    if data_file is None:
        return "Please upload a data file.", None, ""

    try:
        df = pd.read_csv(data_file.name if hasattr(data_file, "name") else data_file)
    except Exception as e:
        return f"Failed to read CSV: {e}", None, ""

    text_col = (text_col or "text").strip()
    if text_col not in df.columns:
        return f"Column '{text_col}' not found. Available: {', '.join(df.columns[:10])}", None, ""

    texts = df[text_col].dropna().astype(str).tolist()
    if not texts:
        return "No texts found in the specified column.", None, ""

    # Parse themes (one per line)
    if not themes_text or not themes_text.strip():
        return "Please define at least one theme (one per line).", None, ""

    theme_lines = [line.strip() for line in themes_text.strip().split("\n") if line.strip()]
    themes = []
    for line in theme_lines:
        # Support "name: description" or just "name"
        if ":" in line:
            name, desc = line.split(":", 1)
            themes.append(Theme(name=name.strip(), description=desc.strip()))
        else:
            themes.append(Theme(name=line, description=""))

    # Build LLM clients
    clients = []
    for backend, model, api_key in [(b1, m1, k1), (b2, m2, k2), (b3, m3, k3)]:
        if model and model.strip():
            if not api_key and backend != "ollama":
                continue
            clients.append(LLMClient(
                backend=backend, model=model.strip(),
                api_key=api_key.strip() if api_key else None,
            ))

    if len(clients) < 2:
        return "Consensus coding requires at least 2 valid LLMs configured.", None, ""

    try:
        consensus = ConsensusCoder(clients)
        report = consensus.code(texts, themes)
    except Exception as e:
        return f"Consensus coding failed: {e}", None, ""

    # Summary
    summary = ConsensusCoder.format_report(report, lang="en")

    # Results table
    rows = []
    for seg in report.segments:
        rows.append({
            "ID": seg.text_id,
            "Text": (seg.text[:80] + "...") if len(seg.text) > 80 else seg.text,
            "Consensus Themes": ", ".join(seg.consensus_themes),
            "Agreement": f"{seg.agreement_rate:.2%}",
            "Votes": "; ".join(f"{t}: {c}/{report.n_coders}" for t, c in seg.vote_counts.items()),
        })
    results_df = pd.DataFrame(rows) if rows else None

    # Agreement
    agreement = f"Overall agreement: {report.overall_agreement:.2%}\n"
    agreement += f"Models: {', '.join(report.coder_models)}\n"
    agreement += f"Total cost: ${report.total_cost:.4f}"

    return summary, results_df, agreement


# ---------------------------------------------------------------------------
# Methods Section Generator
# ---------------------------------------------------------------------------


def _generate_methods_from_log(log_file):
    """Import pipeline log JSON and auto-generate methods section."""
    if log_file is None:
        return "Please upload a pipeline log JSON file.", ""

    try:
        path = log_file.name if hasattr(log_file, "name") else log_file
        with open(path, "r", encoding="utf-8") as f:
            log = json.load(f)
    except Exception as e:
        return f"Failed to read log file: {e}", ""

    pipeline = log.get("pipeline", "")
    writer = MethodsWriter()

    if pipeline == "quantikit":
        meta = QuantiKitPipelineMetadata()
        meta.n_samples = log.get("n_samples", 0)
        meta.n_classes = log.get("n_classes", 0)
        meta.class_labels = log.get("class_labels", [])
        meta.model_name = log.get("model_name", "")
        meta.model_backend = log.get("model_backend", "")
        meta.classification_method = log.get("classification_method", "")
        meta.n_annotations = log.get("n_annotations", 0)
        meta.accuracy = log.get("accuracy", 0.0)
        meta.macro_f1 = log.get("macro_f1", 0.0)
        meta.weighted_f1 = log.get("weighted_f1", 0.0)
        meta.cohens_kappa = log.get("cohens_kappa", 0.0)
        section = writer.generate_quantikit_methods(meta)

    elif pipeline == "qualikit":
        meta = QualiKitPipelineMetadata()
        meta.n_segments = log.get("n_segments", 0)
        meta.n_themes = log.get("n_themes", 0)
        meta.theme_names = log.get("theme_names", [])
        meta.coding_model_name = log.get("coding_model_name", "")
        meta.coding_model_backend = log.get("coding_model_backend", "")
        meta.consensus_coding_used = log.get("consensus_coding_used", False)
        meta.n_consensus_models = log.get("n_consensus_models", 0)
        meta.consensus_model_names = log.get("consensus_model_names", [])
        meta.consensus_agreement = log.get("consensus_agreement", 0.0)
        meta.deidentification_performed = log.get("deidentification_performed", False)
        # Confidence tiers
        meta.n_high_confidence = log.get("n_high_confidence", 0)
        meta.n_medium_confidence = log.get("n_medium_confidence", 0)
        meta.n_low_confidence = log.get("n_low_confidence", 0)
        # Review stats
        review = log.get("review_stats", {})
        meta.n_accepted = review.get("accepted", 0)
        meta.n_rejected = review.get("rejected", 0)
        meta.n_edited = review.get("edited", 0)
        section = writer.generate_qualikit_methods(meta)

    else:
        return (
            f"Unknown pipeline type '{pipeline}'. "
            "Expected 'quantikit' or 'qualikit' in the log JSON.",
            "",
        )

    return section.text_en, section.text_zh


def _generate_methods_from_form(
    pipeline_type,
    # QuantiKit fields
    qt_n_samples, qt_n_classes, qt_class_labels, qt_model_name,
    qt_accuracy, qt_macro_f1, qt_cohens_kappa,
    # QualiKit fields
    ql_n_segments, ql_n_themes, ql_theme_names, ql_model_name,
    ql_consensus_used, ql_n_consensus_models,
):
    """Fallback: generate methods section from manually filled form fields."""
    writer = MethodsWriter()

    if pipeline_type == "QuantiKit":
        meta = QuantiKitPipelineMetadata()
        meta.n_samples = int(qt_n_samples or 0)
        meta.n_classes = int(qt_n_classes or 0)
        if qt_class_labels:
            meta.class_labels = [s.strip() for s in qt_class_labels.split(",") if s.strip()]
        meta.model_name = qt_model_name or ""
        meta.accuracy = float(qt_accuracy or 0)
        meta.macro_f1 = float(qt_macro_f1 or 0)
        meta.cohens_kappa = float(qt_cohens_kappa or 0)
        section = writer.generate_quantikit_methods(meta)

    elif pipeline_type == "QualiKit":
        meta = QualiKitPipelineMetadata()
        meta.n_segments = int(ql_n_segments or 0)
        meta.n_themes = int(ql_n_themes or 0)
        if ql_theme_names:
            meta.theme_names = [s.strip() for s in ql_theme_names.split(",") if s.strip()]
        meta.coding_model_name = ql_model_name or ""
        meta.consensus_coding_used = bool(ql_consensus_used)
        meta.n_consensus_models = int(ql_n_consensus_models or 0)
        section = writer.generate_qualikit_methods(meta)

    else:
        return "Please select a pipeline type.", ""

    return section.text_en, section.text_zh
