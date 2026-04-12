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

# Max number of LLM slots pre-created in the Consensus UI
MAX_LLM_SLOTS = 5


# ---------------------------------------------------------------------------
# ICR Calculator
# ---------------------------------------------------------------------------


def _icr_on_upload(file):
    """When a CSV is uploaded, return its column names for the CheckboxGroup."""
    if file is None:
        return [], ""
    try:
        import gradio as gr
        df = pd.read_csv(file.name if hasattr(file, "name") else file)
        cols = df.columns.tolist()
        return gr.update(choices=cols, value=[]), f"{len(df)} rows, {len(cols)} columns"
    except Exception as e:
        return [], f"Failed to read: {e}"


def _compute_icr(file, selected_cols, mode):
    """Compute ICR from one CSV with N selected coder columns.

    Auto-selects metric:
    - 2 coders → Cohen's Kappa + Krippendorff's Alpha + per-category
    - 3+ coders → Krippendorff's Alpha only (Cohen's Kappa is 2-coder only)

    Parameters
    ----------
    file : uploaded CSV
    selected_cols : list[str] — selected column names (each = one coder)
    mode : "single-label" or "multi-label"
    """
    if file is None:
        return "Please upload a CSV file."

    if not selected_cols or len(selected_cols) < 2:
        return "Please select at least 2 coder columns."

    try:
        df = pd.read_csv(file.name if hasattr(file, "name") else file)
    except Exception as e:
        return f"Failed to read CSV: {e}"

    for col in selected_cols:
        if col not in df.columns:
            return f"Column '{col}' not found."

    n_coders = len(selected_cols)
    calc = ICRCalculator()

    if mode == "multi-label":
        # Multi-label: comma-separated values → sets
        if n_coders == 2:
            themes1 = [
                set(s.strip() for s in str(v).split(",") if s.strip())
                for v in df[selected_cols[0]].fillna("")
            ]
            themes2 = [
                set(s.strip() for s in str(v).split(",") if s.strip())
                for v in df[selected_cols[1]].fillna("")
            ]
            report = calc.compute_all_multilabel(themes1, themes2)
            report.coder_labels = selected_cols
            report.summary_text = calc.format_report(report, multilabel=True)
            return report.summary_text
        else:
            # 3+ coders multi-label: pairwise Jaccard average
            from socialscikit.core.icr import ICRReport, ICRResult
            all_themes_per_coder = []
            for col in selected_cols:
                themes = [
                    set(s.strip() for s in str(v).split(",") if s.strip())
                    for v in df[col].fillna("")
                ]
                all_themes_per_coder.append(themes)

            # Pairwise Jaccard
            from itertools import combinations
            pair_jaccards = []
            pair_labels = []
            for i, j in combinations(range(n_coders), 2):
                r = calc.compute_multilabel_agreement(
                    all_themes_per_coder[i], all_themes_per_coder[j]
                )
                pair_jaccards.append(r.value)
                pair_labels.append(f"{selected_cols[i]} vs {selected_cols[j]}: {r.value:.4f}")

            avg_jaccard = sum(pair_jaccards) / len(pair_jaccards)
            lines = [
                f"═══ Inter-Coder Reliability Report ({n_coders} coders, multi-label) ═══",
                "",
                f"Coders: {', '.join(selected_cols)}",
                f"Items: {len(df)}",
                "",
                f"Average pairwise Jaccard: {avg_jaccard:.4f}  ({calc._interpret_jaccard(avg_jaccard)})",
                "",
                "Pairwise breakdown:",
            ]
            for lbl in pair_labels:
                lines.append(f"  {lbl}")
            return "\n".join(lines)

    else:
        # Single-label mode
        if n_coders == 2:
            labels1 = df[selected_cols[0]].astype(str).tolist()
            labels2 = df[selected_cols[1]].astype(str).tolist()
            report = calc.compute_all(labels1, labels2)
            report.coder_labels = selected_cols
            report.summary_text = calc.format_report(report)
            return report.summary_text
        else:
            # 3+ coders: Krippendorff's Alpha only
            # Build reliability matrix: (n_items, n_coders)
            from socialscikit.core.icr import ICRReport, ICRResult
            reliability_matrix = []
            for _, row in df.iterrows():
                item = []
                for col in selected_cols:
                    val = row[col]
                    if pd.isna(val) or str(val).strip() == "":
                        item.append(None)
                    else:
                        item.append(str(val).strip())
                reliability_matrix.append(item)

            alpha_result = calc.compute_krippendorffs_alpha(reliability_matrix)

            # Also compute pairwise Cohen's Kappa for reference
            from itertools import combinations
            pair_kappas = []
            pair_labels = []
            for i, j in combinations(range(n_coders), 2):
                c_i = df[selected_cols[i]].astype(str).tolist()
                c_j = df[selected_cols[j]].astype(str).tolist()
                r = calc.compute_cohens_kappa(c_i, c_j)
                pair_kappas.append(r.value)
                pair_labels.append(
                    f"{selected_cols[i]} vs {selected_cols[j]}: "
                    f"κ = {r.value:.4f} ({r.interpretation})"
                )

            # Collect all categories
            all_cats = set()
            for row in reliability_matrix:
                for v in row:
                    if v is not None:
                        all_cats.add(v)

            lines = [
                f"═══ Inter-Coder Reliability Report ({n_coders} coders) ═══",
                "",
                f"Coders: {', '.join(selected_cols)}",
                f"Items: {len(reliability_matrix)}",
                f"Categories: {len(all_cats)}  ({', '.join(sorted(all_cats)[:10])}{'...' if len(all_cats) > 10 else ''})",
                "",
                f"Krippendorff's Alpha: {alpha_result.value:.4f}  ({alpha_result.interpretation})",
                "",
                "Pairwise Cohen's Kappa:",
            ]
            for lbl in pair_labels:
                lines.append(f"  {lbl}")

            avg_kappa = sum(pair_kappas) / len(pair_kappas) if pair_kappas else 0
            lines.append(f"\nAverage pairwise Kappa: {avg_kappa:.4f}  ({calc.interpret_kappa(avg_kappa)})")

            return "\n".join(lines)


# ---------------------------------------------------------------------------
# Consensus Coding
# ---------------------------------------------------------------------------


def _run_standalone_consensus(data_file, text_col, themes_text, *llm_args):
    """Standalone consensus coding with variable number of LLMs.

    llm_args is a flat tuple: (b1, m1, k1, b2, m2, k2, ..., bN, mN, kN)
    for up to MAX_LLM_SLOTS slots.
    """
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
        if ":" in line:
            name, desc = line.split(":", 1)
            themes.append(Theme(name=name.strip(), description=desc.strip()))
        else:
            themes.append(Theme(name=line, description=""))

    # Build LLM clients from variable-length args (groups of 3)
    clients = []
    args = list(llm_args)
    for i in range(0, len(args), 3):
        if i + 2 >= len(args):
            break
        backend, model, api_key = args[i], args[i + 1], args[i + 2]
        if model and str(model).strip():
            if not api_key and backend != "ollama":
                continue
            clients.append(LLMClient(
                backend=backend, model=str(model).strip(),
                api_key=str(api_key).strip() if api_key else None,
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
        meta.n_high_confidence = log.get("n_high_confidence", 0)
        meta.n_medium_confidence = log.get("n_medium_confidence", 0)
        meta.n_low_confidence = log.get("n_low_confidence", 0)
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
    qt_n_samples, qt_n_classes, qt_class_labels, qt_model_name,
    qt_accuracy, qt_macro_f1, qt_cohens_kappa,
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
