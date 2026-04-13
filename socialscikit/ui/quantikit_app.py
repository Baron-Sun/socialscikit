"""Gradio Web UI for QuantiKit — full text classification workflow.

Steps:
1. Upload data -> validation + diagnostics
2. Feature extraction -> method recommendation + budget estimate
3. Annotation (optional manual labeling)
4. Configure classification (prompt / fine-tune)
5. Evaluation
6. Export results

Launch via CLI: ``socialscikit quantikit``
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from socialscikit.core.data_diagnostics import format_report, generate_diagnostics
from socialscikit.core.data_loader import DataLoadError, get_template_path, load_file
from socialscikit.core.data_validator import DataValidator, ValidationReport, apply_auto_fixes
from socialscikit.core.llm_client import LLMClient
from socialscikit.quantikit.annotator import AnnotationSession, AnnotationStatus
from socialscikit.quantikit.budget_recommender import BudgetRecommender
from socialscikit.quantikit.evaluator import Evaluator
from socialscikit.quantikit.feature_extractor import TASK_TYPES, FeatureExtractor, TaskFeatures
from socialscikit.quantikit.method_recommender import MethodRecommender
from socialscikit.quantikit.prompt_classifier import PromptClassifier
from socialscikit.quantikit.prompt_optimizer import PromptOptimizer, PromptVariant
from socialscikit.core import charts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust label extraction (knows valid labels → much higher accuracy)
# ---------------------------------------------------------------------------


def _extract_label_robust(raw: str, valid_labels: list[str]) -> str:
    """Extract a classification label from LLM response, using valid labels
    to guide matching.

    Parsing order:
    1. Strip markdown code fencing (```json ... ```)
    2. Try JSON parse → extract "label" key
    3. Regex for "label": "..." pattern
    4. Exact match against valid labels (case-insensitive)
    5. Substring match — check if any valid label appears in the response
    6. Fallback to first non-empty line
    """
    raw = raw.strip()

    # 0. Strip markdown code fencing
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    # 1. Try full JSON parse
    for text in (cleaned, raw):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "label" in data:
                extracted = str(data["label"]).strip()
                return _match_label(extracted, valid_labels)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 2. Regex for {"label": "..."} pattern embedded in text
    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if m:
        return _match_label(m.group(1).strip(), valid_labels)

    # 3. Exact match (response IS a valid label)
    for label in valid_labels:
        if raw.strip().lower() == label.lower():
            return label

    # 4. First line is a valid label
    first_line = raw.split("\n")[0].strip().strip('"').strip("'").strip()
    for label in valid_labels:
        if first_line.lower() == label.lower():
            return label

    # 5. Any valid label appears as a standalone word in the response
    for label in valid_labels:
        # Match label as whole word or surrounded by common delimiters
        pattern = r'(?:^|[\s"\'（(,，:：])' + re.escape(label) + r'(?:[\s"\'）),，:：.]|$)'
        if re.search(pattern, raw, re.IGNORECASE):
            return label

    # 6. Fallback — return cleaned first line (for debugging visibility)
    return first_line[:50] if first_line else raw[:50]


def _match_label(extracted: str, valid_labels: list[str]) -> str:
    """Match an extracted label string against valid labels.

    Handles case differences and common LLM variations like
    returning the full definition instead of just the code.
    """
    # Exact match (case-insensitive)
    for vl in valid_labels:
        if extracted.lower() == vl.lower():
            return vl

    # Label appears at the start (e.g., "W（福利）" → "W")
    for vl in valid_labels:
        if extracted.lower().startswith(vl.lower()):
            return vl

    # Label appears anywhere in the extracted string
    for vl in valid_labels:
        if vl.lower() in extracted.lower():
            return vl

    # No match — return as-is for debugging
    return extracted


def _macro_f1_standalone(preds: list[str], trues: list[str]) -> float:
    """Compute macro F1 from string predictions and true labels."""
    preds_norm = [p.strip().lower() for p in preds]
    trues_norm = [t.strip().lower() for t in trues]
    all_labels = set(trues_norm)

    f1_scores = []
    for label in all_labels:
        tp = sum(1 for p, t in zip(preds_norm, trues_norm) if p == label and t == label)
        fp = sum(1 for p, t in zip(preds_norm, trues_norm) if p == label and t != label)
        fn = sum(1 for p, t in zip(preds_norm, trues_norm) if p != label and t == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0


def _parse_label_definitions(
    label_defs_str: str | None,
    labels: list[str],
) -> dict[str, str]:
    """Parse label definitions from user input text.

    Accepts both English ``:`` and Chinese ``：`` colons, and flexible
    formats like ``W: 定义...`` or ``- W：定义...`` or ``W = 定义...``.
    Falls back to auto-generated placeholder definitions if parsing fails.
    """
    label_definitions: dict[str, str] = {}
    if label_defs_str and label_defs_str.strip():
        for line in label_defs_str.strip().split("\n"):
            line = line.strip().lstrip("-").lstrip("•").lstrip("*").strip()
            if not line:
                continue
            # Try splitting on Chinese colon, English colon, or equals sign
            sep = None
            for candidate in ("：", ":", "="):
                if candidate in line:
                    sep = candidate
                    break
            if sep:
                k, v = line.split(sep, 1)
                k = k.strip()
                v = v.strip()
                if k and v:
                    label_definitions[k] = v

    if not label_definitions:
        label_definitions = {l: f"属于 {l} 类别的文本" for l in labels}

    return label_definitions


def _insert_text_placeholder(prompt: str) -> str:
    """Insert {text} placeholder BEFORE the last output format instruction.

    The text to classify must appear BEFORE the output instruction so the LLM
    reads it before starting to generate the response. Appending at the very
    end (after an output instruction like "输出 JSON...") causes the LLM to
    skip the text entirely.
    """
    lines = prompt.strip().split("\n")

    # Find the last line that looks like an output/response instruction
    output_keywords = [
        "输出", "output", "respond", "返回", "json", "格式",
        "请只输出", "classification", "label",
    ]
    insert_before = len(lines)  # default: end
    for i in range(len(lines) - 1, max(len(lines) - 6, -1), -1):
        lower = lines[i].strip().lower()
        if any(kw in lower for kw in output_keywords):
            insert_before = i
            break

    lines.insert(insert_before, "\n待分类文本：{text}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared state (per-session via gr.State)
# ---------------------------------------------------------------------------

_validator = DataValidator()
_extractor = FeatureExtractor()
_recommender = MethodRecommender()
_budget_recommender = BudgetRecommender()


# ---------------------------------------------------------------------------
# Step 1: Upload & Validate
# ---------------------------------------------------------------------------


def _load_and_validate(file, mode="quantikit"):
    """Load file, run validation, return (df, report_text, issues_df, raw_df_preview)."""
    if file is None:
        return None, "请上传数据文件。", None, None

    try:
        df = load_file(file.name)
    except DataLoadError as e:
        return None, f"加载失败：{e}", None, None

    report = _validator.validate(df, mode=mode)

    # Format issues table
    issues_rows = []
    for issue in report.issues:
        issues_rows.append({
            "级别": {"error": "❌ 错误", "warning": "⚠️ 警告", "info": "ℹ️ 信息"}[issue.severity],
            "位置": issue.field,
            "说明": issue.message,
            "可自动修复": "是" if issue.auto_fix_available else "否",
        })
    issues_df = pd.DataFrame(issues_rows) if issues_rows else None

    # Diagnostics
    file_size = os.path.getsize(file.name) / 1024
    diag = generate_diagnostics(
        df,
        text_col=report.suggested_text_col,
        label_col=report.suggested_label_col,
        file_size_kb=file_size,
    )
    diag_text = format_report(diag)

    summary = report.summary + "\n\n" + diag_text
    preview = df.head(10)

    return df, summary, issues_df, preview


def _load_and_select_columns(file, mode="quantikit"):
    """Load file and return column selectors.

    Returns (df, summary, issues_df, preview,
             text_col_update, label_col_update).
    The last two are gr.update() dicts for Dropdown components.
    """
    import gradio as gr

    df, summary, issues_df, preview = _load_and_validate(file, mode=mode)

    if df is None:
        empty = gr.update(choices=[], value=None)
        return df, summary, issues_df, preview, empty, empty

    cols = list(df.columns)
    report = _validator.validate(df, mode=mode)
    text_val = report.suggested_text_col or (cols[0] if cols else None)
    label_val = report.suggested_label_col or (cols[1] if len(cols) > 1 else None)

    text_update = gr.update(choices=cols, value=text_val)
    label_update = gr.update(choices=cols, value=label_val)

    return df, summary, issues_df, preview, text_update, label_update


def _confirm_columns(df_state, text_col, label_col, mode="quantikit"):
    """Re-run validation and diagnostics with user-selected columns.

    Returns (summary, issues_df, preview).
    """
    if df_state is None:
        return "请先上传数据。", None, None

    # Re-validate — temporarily rename columns so validator picks them up
    df = df_state.copy()
    issues_rows = []

    # Check text column
    if not text_col or text_col not in df.columns:
        issues_rows.append({
            "级别": "❌ 错误",
            "位置": "文本列",
            "说明": f"所选文本列 '{text_col}' 不存在。",
            "可自动修复": "否",
        })
        text_col_valid = None
    else:
        text_col_valid = text_col
        empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
        n_empty = int(empty_mask.sum())
        if n_empty > 0:
            pct = n_empty / len(df) * 100
            issues_rows.append({
                "级别": "⚠️ 警告",
                "位置": text_col,
                "说明": f"发现 {n_empty} 行空文本（占 {pct:.1f}%）",
                "可自动修复": "是",
            })

    # Check label column
    label_col_valid = None
    if label_col and label_col in df.columns:
        label_col_valid = label_col
        vals = df[label_col].dropna()
        n_unique = vals.nunique()
        if n_unique < 2:
            issues_rows.append({
                "级别": "⚠️ 警告",
                "位置": label_col,
                "说明": f"标签列仅有 {n_unique} 个唯一值，分类至少需要 2 个类别。",
                "可自动修复": "否",
            })
        # Label distribution info
        dist = vals.value_counts()
        if len(dist) > 0:
            ratio = dist.iloc[0] / dist.iloc[-1] if dist.iloc[-1] > 0 else float("inf")
            if ratio > 5:
                issues_rows.append({
                    "级别": "⚠️ 警告",
                    "位置": label_col,
                    "说明": f"标签分布不均衡（最多类:最少类 = {ratio:.1f}:1），可能影响模型性能。",
                    "可自动修复": "否",
                })

    issues_df = pd.DataFrame(issues_rows) if issues_rows else None

    # Diagnostics
    diag = generate_diagnostics(
        df,
        text_col=text_col_valid,
        label_col=label_col_valid,
        file_size_kb=0,
    )
    diag_text = format_report(diag)

    n_rows = len(df)
    status = "✅ 列映射已确认" if text_col_valid else "❌ 请选择有效的文本列"
    summary = (
        f"{status}\n"
        f"文本列: {text_col_valid or '未设置'} | 标签列: {label_col_valid or '未设置'}\n"
        f"总行数: {n_rows}\n\n{diag_text}"
    )
    preview = df.head(10)
    return summary, issues_df, preview


def _apply_fixes(df_state):
    """Apply auto-fixes to the current DataFrame."""
    if df_state is None:
        return None, "没有数据可修复。"
    report = _validator.validate(df_state, mode="quantikit")
    fixed = apply_auto_fixes(df_state, report)
    new_report = _validator.validate(fixed, mode="quantikit")
    return fixed, f"已修复。{new_report.summary}（剩余 {len(fixed)} 行）"


# ---------------------------------------------------------------------------
# Step 2: Feature Extraction, Method Recommendation & Budget
# ---------------------------------------------------------------------------


def _extract_and_recommend(
    df_state, task_type, n_classes, target_f1, budget_level,
    text_col, label_col,
):
    """Extract features, generate recommendation, and estimate annotation budget."""
    if df_state is None:
        return "请先上传数据。", "", "", None, None

    user_inputs = {
        "task_type": task_type,
        "target_f1": float(target_f1),
        "budget_level": budget_level,
    }
    if n_classes:
        user_inputs["n_classes"] = int(n_classes)

    features = _extractor.extract(
        df_state,
        user_inputs=user_inputs,
        text_col=text_col or None,
        label_col=label_col or None,
    )

    rec = _recommender.recommend(features)

    # Format feature summary
    feat_lines = [
        "═══ 任务特征 ═══",
        f"样本数：{features.n_samples}",
        f"已标注：{features.n_labeled}",
        f"类别数：{features.n_classes}",
        f"平均文本长度：{features.avg_text_length_tokens} tokens",
        f"语言：{features.language}{'（多语言）' if features.is_multilingual else ''}",
        f"任务类型：{features.task_type}",
        f"目标 F1：{features.target_f1}",
    ]
    feat_text = "\n".join(feat_lines)

    # Format recommendation
    rec_lines = [
        "═══ 方法推荐 ═══",
        f"推荐方法：{rec.recommended_method}（置信度：{rec.confidence}）",
        "",
        f"推荐理由：{rec.reasoning}",
        "",
        f"预期表现：F1 ≈ {rec.estimated_performance[0]:.2f}（±{rec.estimated_performance[1]:.2f}）",
        f"预估费用：{rec.estimated_cost}",
        "",
        f"文献支持：",
    ]
    for ref in rec.literature_support:
        rec_lines.append(f"  • {ref}")

    if rec.alternative_method:
        rec_lines.append(f"\n备选方案：{rec.alternative_method}")
        rec_lines.append(f"  {rec.alternative_reasoning}")

    if rec.cold_start_recommendation:
        rec_lines.append(f"\n{rec.cold_start_recommendation.message}")

    if rec.sensitivity_analysis_suggested:
        rec_lines.append("\n⚠️ 建议进行敏感性分析（用于假设检验场景）")

    rec_text = "\n".join(rec_lines)

    # Budget recommendation
    budget_report = _budget_recommender.recommend(features, target_f1=float(target_f1))
    budget_lines = [
        "═══ 标注预算推荐 ═══",
        f"推荐标注量：{budget_report.recommended_n} 条",
        f"80% 置信区间：[{budget_report.confidence_interval[0]}, {budget_report.confidence_interval[1]}] 条",
        f"估算依据：{'经验拟合' if budget_report.estimation_basis == 'empirical' else '先验估计'}",
    ]
    if budget_report.prior_source:
        budget_lines.append(f"数据来源：{budget_report.prior_source}")
    if budget_report.update_after_n:
        budget_lines.append(f"建议标注 {budget_report.update_after_n} 条后重新估算。")
    budget_text = "\n".join(budget_lines)

    # Marginal returns chart + table
    curve = budget_report.marginal_returns_curve
    if curve:
        budget_plot = _plot_marginal_curve(
            curve, budget_report.recommended_n, float(target_f1),
        )
        curve_table = _build_curve_table(curve, budget_report.recommended_n)
    else:
        budget_plot = None
        curve_table = None

    return feat_text, rec_text, budget_text, budget_plot, curve_table


def _build_curve_table(curve, recommended_n):
    """Build a DataFrame with ~8 key points from the marginal curve."""
    if not curve:
        return None
    # Pick ~8 evenly spaced points including first and last
    n_points = min(8, len(curve))
    step = max(1, (len(curve) - 1) // (n_points - 1))
    indices = list(range(0, len(curve), step))
    if indices[-1] != len(curve) - 1:
        indices.append(len(curve) - 1)

    rows = []
    for i in indices:
        n, f1 = curve[i]
        mark = " *" if n >= recommended_n and (not rows or rows[-1]["标注量"] < recommended_n) else ""
        rows.append({
            "标注量": n,
            "预估 F1": f"{f1:.3f}",
            "": "← 推荐" if mark else "",
        })
    return pd.DataFrame(rows)


def _plot_marginal_curve(curve, recommended_n, target_f1):
    """Generate a high-res matplotlib figure for the marginal returns curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "STHeiti", "Microsoft YaHei",
        "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    ns = [p[0] for p in curve]
    f1s = [p[1] for p in curve]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Curve + fill
    ax.plot(ns, f1s, color="#4A90D9", linewidth=2.2, label="预估 F1", zorder=3)
    ax.fill_between(ns, f1s, alpha=0.06, color="#4A90D9")

    # Target line
    ax.axhline(y=target_f1, color="#c0392b", linewidth=1, linestyle="--",
               alpha=0.7, label=f"目标 F1 = {target_f1:.2f}", zorder=2)

    # Recommended N marker
    rec_f1 = None
    for n, f1 in curve:
        if n >= recommended_n:
            rec_f1 = f1
            break
    if rec_f1 is None and curve:
        rec_f1 = curve[-1][1]

    ax.plot(recommended_n, rec_f1, "o", color="#c0392b", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)

    # Smart label placement — avoid overlapping the curve
    x_range = max(ns) - min(ns)
    y_range = max(f1s) - min(f1s) if len(set(f1s)) > 1 else 0.1
    txt_x = recommended_n + x_range * 0.06
    txt_y = rec_f1 + y_range * 0.08
    # If label would go above chart, flip down
    if txt_y > max(f1s):
        txt_y = rec_f1 - y_range * 0.10

    ax.annotate(
        f"推荐: {recommended_n:,} 条",
        xy=(recommended_n, rec_f1),
        xytext=(txt_x, txt_y),
        fontsize=9, color="#c0392b", fontweight="500",
        arrowprops=dict(arrowstyle="-", color="#c0392b", lw=0.8, connectionstyle="arc3,rad=0.15"),
    )

    ax.set_xlabel("标注数据量", fontsize=10.5, color="#444", labelpad=8)
    ax.set_ylabel("预估 Macro-F1", fontsize=10.5, color="#444", labelpad=8)
    ax.legend(fontsize=9, frameon=False, loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ddd")
    ax.spines["bottom"].set_color("#ddd")
    ax.tick_params(colors="#666", labelsize=9, length=3)
    ax.grid(axis="y", color="#f0f0f0", linewidth=0.6)
    ax.set_xlim(left=0)

    fig.tight_layout(pad=1.2)
    return fig


# ---------------------------------------------------------------------------
# Step 3: Annotation
# ---------------------------------------------------------------------------


def _create_annotation_session(df_state, text_col, label_col, labels_str, shuffle):
    """Create an annotation session from uploaded data."""
    if df_state is None:
        return None, "请先上传数据。", "", "", ""

    labels = [l.strip() for l in labels_str.split(",") if l.strip()] if labels_str.strip() else None

    try:
        session = AnnotationSession.from_dataframe(
            df_state,
            text_col=text_col or "text",
            labels=labels,
            label_col=label_col if label_col and label_col in df_state.columns else None,
            shuffle=shuffle,
        )
    except ValueError as e:
        return None, f"创建标注会话失败：{e}", "", "", ""

    current = session.current()
    stats = session.stats()
    stats_text = (
        f"总计：{stats.total} 条 | 已标注：{stats.labeled} | "
        f"已跳过：{stats.skipped} | 已标记：{stats.flagged} | "
        f"待标注：{stats.pending} | 进度：{stats.progress_pct}%"
    )
    current_text = current.text if current else "（已完成全部标注）"
    current_idx = f"第 {session._cursor + 1}/{stats.total} 条" if current else ""

    return session, stats_text, current_text, current_idx, ""


def _annotate_item(session_state, label):
    """Label current item and advance."""
    if session_state is None:
        return None, "请先创建标注会话。", "", "", ""
    if not label.strip():
        return session_state, _get_stats_text(session_state), \
            _get_current_text(session_state), _get_current_idx(session_state), "请输入标签。"
    try:
        session_state.annotate(label.strip())
    except (IndexError, ValueError) as e:
        return session_state, _get_stats_text(session_state), \
            _get_current_text(session_state), _get_current_idx(session_state), str(e)

    return session_state, _get_stats_text(session_state), \
        _get_current_text(session_state), _get_current_idx(session_state), ""


def _skip_item(session_state):
    """Skip current item."""
    if session_state is None:
        return None, "请先创建标注会话。", "", "", ""
    try:
        session_state.skip()
    except IndexError as e:
        return session_state, _get_stats_text(session_state), \
            _get_current_text(session_state), _get_current_idx(session_state), str(e)

    return session_state, _get_stats_text(session_state), \
        _get_current_text(session_state), _get_current_idx(session_state), ""


def _flag_item(session_state, note):
    """Flag current item as ambiguous."""
    if session_state is None:
        return None, "请先创建标注会话。", "", "", ""
    try:
        session_state.flag(note=note)
    except IndexError as e:
        return session_state, _get_stats_text(session_state), \
            _get_current_text(session_state), _get_current_idx(session_state), str(e)

    return session_state, _get_stats_text(session_state), \
        _get_current_text(session_state), _get_current_idx(session_state), ""


def _undo_annotation(session_state):
    """Undo the last annotation action."""
    if session_state is None:
        return None, "请先创建标注会话。", "", "", ""
    result = session_state.undo()
    msg = "已撤销。" if result else "没有可撤销的操作。"

    return session_state, _get_stats_text(session_state), \
        _get_current_text(session_state), _get_current_idx(session_state), msg


def _export_annotations(session_state, include_all):
    """Export annotation results as DataFrame (display-friendly)."""
    if session_state is None:
        return None, "请先创建标注会话。"
    df = session_state.export(include_all=include_all)
    if df.empty:
        return None, "没有标注数据可导出。"

    # Truncate text for display and reorder columns
    display_df = df.copy()
    if "text" in display_df.columns:
        display_df["text"] = display_df["text"].astype(str).apply(
            lambda t: t[:80] + "…" if len(t) > 80 else t
        )
    # Put label & status before text
    cols = list(display_df.columns)
    preferred = []
    for c in ["label", "status", "text", "annotator_note"]:
        if c in cols:
            preferred.append(c)
            cols.remove(c)
    preferred.extend(cols)
    display_df = display_df[preferred]

    return display_df, f"导出 {len(df)} 条标注数据。"


def _download_annotations_csv(session_state, include_all):
    """Export annotations as a downloadable CSV file (full text, no truncation)."""
    if session_state is None:
        return None
    df = session_state.export(include_all=include_all)
    if df.empty:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix="_annotations.csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    return tmp.name


def _update_main_df_from_annotations(session_state, df_state, text_col, label_col):
    """Merge annotations back into the main DataFrame."""
    if session_state is None or df_state is None:
        return df_state, "无数据可合并。"
    export_df = session_state.export_for_training()
    if export_df.empty:
        return df_state, "没有标注数据可合并。"
    # Update label column in the main df
    tc = text_col or "text"
    lc = label_col or "label"
    if lc not in df_state.columns:
        df_state[lc] = None
    text_to_label = dict(zip(export_df["text"], export_df["label"]))
    df_state[lc] = df_state[tc].map(lambda t: text_to_label.get(str(t), df_state.loc[df_state[tc] == t, lc].values[0] if t in df_state[tc].values else None))
    n_updated = export_df.shape[0]
    return df_state, f"已将 {n_updated} 条标注合并到主数据集的 '{lc}' 列。"


# Annotation helpers
def _get_stats_text(session):
    if session is None:
        return ""
    stats = session.stats()
    return (
        f"总计：{stats.total} 条 | 已标注：{stats.labeled} | "
        f"已跳过：{stats.skipped} | 已标记：{stats.flagged} | "
        f"待标注：{stats.pending} | 进度：{stats.progress_pct}%"
    )


def _make_annotation_chart(session):
    """Generate annotation progress donut chart for the dashboard.

    Returns (progress_fig,) for .then() chaining.
    """
    if session is None:
        return None
    try:
        stats = session.stats()
        return charts.plot_annotation_progress(
            stats.labeled, stats.skipped, stats.flagged, stats.pending,
        )
    except Exception:
        return None


def _get_current_text(session):
    if session is None:
        return ""
    current = session.current()
    return current.text if current else "（已完成全部标注）"


def _get_current_idx(session):
    if session is None:
        return ""
    current = session.current()
    stats = session.stats()
    return f"第 {session._cursor + 1}/{stats.total} 条" if current else "完成"


# ---------------------------------------------------------------------------
# Step 4a: Prompt Builder — generate, preview, optimize
# ---------------------------------------------------------------------------


def _parse_examples_flexible(
    raw: str | None, labels: list[str],
) -> dict[str, list[str]] | None:
    """Parse examples from user input — accepts multiple formats.

    Supported formats:
    1. JSON: ``{"W": ["example1"], "R": ["example2"]}``
    2. Line-by-line: ``W: example text`` or ``W：example text``
    3. Free text with label prefix: ``W - example text``

    Returns ``{label: [texts]}`` or None if empty / unparseable.
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # 1. Try JSON first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            result = {}
            for k, v in parsed.items():
                k = str(k).strip()
                if isinstance(v, list):
                    result[k] = [str(x).strip() for x in v if str(x).strip()]
                elif isinstance(v, str) and v.strip():
                    result[k] = [v.strip()]
            if result:
                return result
    except (json.JSONDecodeError, TypeError):
        pass

    # 2. Line-by-line parsing
    result: dict[str, list[str]] = {}
    labels_lower = {l.lower(): l for l in labels}

    for line in text.split("\n"):
        line = line.strip().lstrip("-").lstrip("•").lstrip("*").strip()
        if not line:
            continue

        # Try splitting on various separators
        matched_label = None
        example_text = None
        for sep in ("：", ":", "-", "→", "->"):
            if sep in line:
                parts = line.split(sep, 1)
                candidate = parts[0].strip()
                # Check if the part before separator matches a label
                if candidate.lower() in labels_lower:
                    matched_label = labels_lower[candidate.lower()]
                    example_text = parts[1].strip()
                    break
                # Also check if label appears at start
                for lbl in labels:
                    if candidate.lower().startswith(lbl.lower()):
                        matched_label = lbl
                        example_text = parts[1].strip()
                        break
                if matched_label:
                    break

        if matched_label and example_text:
            result.setdefault(matched_label, []).append(example_text)

    return result if result else None


def _generate_smart_prompt(
    task_desc, labels_str, label_defs_str, examples_str, excl_str,
    backend, model, api_key,
):
    """Generate a classification prompt from user inputs.

    Design principles:
    - Accept ANY reasonable input format (loose parsing)
    - Produce a strict, well-structured prompt (strict output)
    - {text} placeholder is an internal detail — users never see it
    - Always validate and fix the output before returning

    References:
    - Zhou et al. (2022) APE: automatic prompt generation + selection
    - Lo (2023) CLEAR framework: context-rich prompt structure
    - Dunivin (2024): exclusion criteria improve accuracy
    """
    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not labels:
        return "⚠️ 请先在「类别」中输入标签（逗号分隔），例如：W, N, R"

    # ---- Parse all inputs with flexible formats ----
    label_definitions = _parse_label_definitions(label_defs_str, labels)
    inclusion_examples = _parse_examples_flexible(examples_str, labels)
    exclusion_examples = _parse_examples_flexible(excl_str, labels)

    # ---- LLM-powered prompt generation (when API available) ----
    if api_key or backend == "ollama":
        try:
            llm = LLMClient(backend=backend, model=model, api_key=api_key or None)

            system = (
                "你是一位专业的 prompt engineer，擅长为社会科学文本分类任务设计高质量 prompt。\n"
                "用户会提供任务描述、类别定义和示例。请生成一个可以直接使用的分类 prompt。\n\n"
                "生成的 prompt 必须满足：\n"
                "1. 开头用清晰的一两句话说明分类任务\n"
                "2. 列出每个类别的定义，包含边界条件和容易混淆的区分规则\n"
                "3. 如果有正例/反例，融入 prompt 中作为指导\n"
                "4. 包含决策规则：遇到模糊情况如何判断\n"
                "5. 在需要插入待分类文本的位置，放置占位符 {text}（必须保留大括号）\n"
                "6. 最后给出输出格式要求\n\n"
                "结构要求（严格按此顺序）：\n"
                "  [任务说明] → [类别定义] → [决策规则] → [示例] → "
                "[待分类文本：{text}] → [输出格式指令]\n\n"
                "注意：{text} 必须在输出格式指令之前！\n"
                "用与任务描述相同的语言。只返回 prompt 本身，不要额外说明。"
            )

            parts = []
            if task_desc and task_desc.strip():
                parts.append(f"## 任务描述\n{task_desc.strip()}")

            parts.append(
                "## 类别定义\n"
                + "\n".join(f"- {k}: {v}" for k, v in label_definitions.items())
            )

            if inclusion_examples:
                ex_lines = []
                for lbl, texts in inclusion_examples.items():
                    for t in texts:
                        ex_lines.append(f"  {lbl}: {t}")
                parts.append("## 正例\n" + "\n".join(ex_lines))

            if exclusion_examples:
                ex_lines = []
                for lbl, texts in exclusion_examples.items():
                    for t in texts:
                        ex_lines.append(f"  {lbl}: {t}")
                parts.append("## 反例（容易误判的边界案例）\n" + "\n".join(ex_lines))

            resp = llm.complete("\n\n".join(parts), system=system, max_tokens=2048)
            generated = resp.text.strip()

            # Ensure proper {text} placement
            generated = _ensure_text_placeholder(generated)
            return generated

        except Exception as e:
            logger.warning("LLM prompt generation failed: %s — falling back to template", e)

    # ---- Fallback: template assembly (no API key) ----
    return _build_template_prompt(
        task_desc, labels, label_definitions,
        inclusion_examples, exclusion_examples,
    )


def _ensure_text_placeholder(prompt: str) -> str:
    """Ensure prompt has {text} in the correct position (before output instruction).

    This is an INTERNAL function — users never interact with {text} directly.
    """
    if "{text}" in prompt:
        # Check placement — must be before output instructions
        text_pos = prompt.index("{text}")
        output_keywords = ["输出", "返回", "json", "output", "respond", "格式"]
        last_output_pos = -1
        for kw in output_keywords:
            p = prompt.lower().rfind(kw)
            if p > last_output_pos and p > text_pos:
                last_output_pos = p

        if last_output_pos > text_pos:
            # {text} is correctly before output instruction — good
            return prompt

        # {text} is AFTER output instruction — fix it
        prompt = prompt.replace("{text}", "").strip()
        # Fall through to re-insert

    return _insert_text_placeholder(prompt)


def _build_template_prompt(
    task_desc: str | None,
    labels: list[str],
    label_definitions: dict[str, str],
    inclusion_examples: dict[str, list[str]] | None,
    exclusion_examples: dict[str, list[str]] | None,
) -> str:
    """Build a classification prompt from structured components (no LLM needed).

    Always produces a well-structured prompt with proper {text} placement.
    """
    parts: list[str] = []

    # Task description
    task = (
        task_desc.strip()
        if task_desc and task_desc.strip()
        else f"请将文本分类为以下类别之一：{', '.join(labels)}"
    )
    parts.append(task)

    # Category definitions
    parts.append("\n类别定义：")
    for label, defn in label_definitions.items():
        parts.append(f"- {label}：{defn}")

    # Inclusion examples
    if inclusion_examples:
        parts.append("\n正例：")
        for label, texts in inclusion_examples.items():
            for t in texts:
                parts.append(f"  {label}：「{t}」")

    # Exclusion examples
    if exclusion_examples:
        parts.append("\n反例（不属于该类别的易混淆案例）：")
        for label, texts in exclusion_examples.items():
            for t in texts:
                parts.append(f"  不是 {label}：「{t}」")

    # Text placeholder (before output instruction)
    parts.append("\n待分类文本：{text}")

    # Output instruction (after text)
    parts.append(f"\n请只输出以下类别之一：{', '.join(labels)}")

    return "\n".join(parts)


def _evaluate_prompt(prompt_text, labels_str, backend, model, api_key):
    """Evaluate prompt quality and provide improvement suggestions.

    Uses LLM to analyze the prompt and give actionable feedback.
    Without API key, provides rule-based checks.

    Returns (evaluation_text,).
    """
    if not prompt_text or not prompt_text.strip():
        return "请先在上方生成或输入 Prompt。"

    prompt = prompt_text.strip()
    labels = [l.strip() for l in labels_str.split(",") if l.strip()]

    # ---- Rule-based checks (always run) ----
    issues: list[str] = []
    suggestions: list[str] = []

    if "{text}" not in prompt:
        issues.append("❌ 缺少 {text} 占位符 — 分类时无法插入待分类文本")
        suggestions.append("在 Prompt 中添加 {text}（放在输出指令之前）")

    if not labels:
        issues.append("❌ 未定义类别标签")

    has_output_instruction = any(
        kw in prompt.lower()
        for kw in ["输出", "output", "respond", "返回", "json", "label"]
    )
    if not has_output_instruction:
        issues.append("⚠️ 没有明确的输出格式指令")
        suggestions.append("添加输出格式要求，如：请只输出类别标签")

    # Check {text} placement — must be before output instruction
    if "{text}" in prompt:
        text_pos = prompt.index("{text}")
        output_pos = -1
        for kw in ["输出", "返回", "json", "respond", "output"]:
            p = prompt.lower().rfind(kw)
            if p > output_pos:
                output_pos = p
        if output_pos > 0 and text_pos > output_pos:
            issues.append(
                "⚠️ {text} 在输出指令之后 — LLM 可能先生成响应再看到文本"
            )
            suggestions.append("将 {text} 移到输出格式指令之前")

    has_definitions = False
    for label in labels:
        if label in prompt:
            has_definitions = True
    if labels and not has_definitions:
        issues.append("⚠️ Prompt 中未出现类别标签")

    word_count = len(prompt)
    if word_count < 50:
        issues.append("⚠️ Prompt 太短，可能缺少足够的指导信息")
    elif word_count > 2000:
        suggestions.append("Prompt 较长（{} 字），考虑精简以降低 token 消耗".format(word_count))

    # ---- LLM evaluation (when API key available) ----
    llm_assessment = ""
    if api_key or backend == "ollama":
        try:
            llm = LLMClient(backend=backend, model=model, api_key=api_key or None)
            eval_system = (
                "你是一位 prompt engineering 专家。请对以下用于文本分类的 Prompt 进行评估。\n"
                "评估维度：\n"
                "1. 任务描述是否清晰\n"
                "2. 类别定义是否有边界条件和区分规则\n"
                "3. 是否有处理模糊案例的决策规则\n"
                "4. {text} 占位符位置是否合理\n"
                "5. 输出格式指令是否明确\n\n"
                "请给出：\n"
                "- 总体评分（1-10）\n"
                "- 2-3 个主要优点\n"
                "- 2-3 个具体改进建议，每个建议包含：\n"
                "  · 问题：指出当前 Prompt 中具体哪句话/哪部分有什么不足\n"
                "  · 建议改为：给出具体的改写示例\n\n"
                "用中文回答，简洁务实。"
            )
            eval_prompt = (
                f"类别标签：{', '.join(labels)}\n\n"
                f"待评估的 Prompt：\n---\n{prompt}\n---"
            )
            resp = llm.complete(eval_prompt, system=eval_system, max_tokens=1024)
            llm_assessment = resp.text.strip()
        except Exception as e:
            llm_assessment = f"（LLM 评估失败：{e}）"

    # ---- Build output ----
    parts: list[str] = []
    if issues:
        parts.append("🔍 基础检查：\n" + "\n".join(f"  {i}" for i in issues))
    else:
        parts.append("✅ 基础检查通过（格式正确）")

    if suggestions:
        parts.append("💡 改进建议：\n" + "\n".join(f"  • {s}" for s in suggestions))

    if llm_assessment:
        parts.append(f"\n📝 LLM 详细评估：\n{llm_assessment}")
    elif not api_key and backend != "ollama":
        parts.append("\n（填写 API Key 可获得 LLM 详细评估和改进建议）")

    parts.append(
        "\n💡 提示：您可以将当前 Prompt 复制到下方的对比版本栏中进行修改，"
        "然后点击「测试对比」在已标注数据上对比不同版本的效果。"
    )

    return "\n\n".join(parts)


def _test_variants(
    df_state, text_col, label_col,
    prompt_text, v1_text, v2_text, v3_text,
    labels_str, backend, model, api_key,
):
    """Evaluate prompt variants on labeled data with full detail output.

    Does inline classification (not via evaluate_and_select) so we capture
    raw LLM responses and can show users exactly what each variant predicted.

    Returns (summary_text, detail_dataframe).
    """
    if df_state is None:
        return "请先上传数据。", None
    if not api_key and backend != "ollama":
        return "请输入 API Key。", None

    label_col = (label_col or "").strip()
    if not label_col or label_col not in df_state.columns:
        return "需要标签列进行评估。请确保数据中已有真实标签。", None

    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not labels:
        return "请输入类别标签。", None

    # Collect non-empty prompts to test
    candidate_names: list[str] = []
    candidate_prompts: list[str] = []
    if prompt_text and prompt_text.strip():
        candidate_names.append("当前 Prompt")
        candidate_prompts.append(prompt_text.strip())
    for i, v in enumerate([v1_text, v2_text, v3_text], 1):
        if v and v.strip():
            candidate_names.append(f"版本 {i}")
            candidate_prompts.append(v.strip())

    if not candidate_prompts:
        return "没有可测试的 Prompt。请先生成或输入 Prompt。", None

    # Evaluation subset — use labeled rows only
    eval_df = df_state.dropna(subset=[label_col])
    n_total_labeled = len(eval_df)
    eval_df = eval_df.head(50)
    n_eval = len(eval_df)
    if n_eval < 5:
        return (
            f"标签数据不足（当前 {n_total_labeled} 条，至少需要 5 条），无法测试。",
            None,
        )

    texts = eval_df[text_col].astype(str).tolist()
    true_labels = eval_df[label_col].astype(str).tolist()
    llm = LLMClient(backend=backend, model=model, api_key=api_key or None)

    # Prepare detail table rows — one row per test text
    detail_rows: list[dict] = []
    for idx in range(n_eval):
        t = texts[idx]
        detail_rows.append({
            "文本（前60字）": t[:60] + ("…" if len(t) > 60 else ""),
            "真实标签": true_labels[idx],
        })

    # Classify with each prompt variant
    scores: list[tuple[str, float, float]] = []  # (name, f1, acc)
    prompt_preview = ""
    # Store first few raw responses for the first variant (for diagnostic)
    first_variant_raw_samples: list[tuple[str, str, str]] = []  # (true, pred, raw)
    # Track which variants were auto-fixed
    auto_fixed_names: list[str] = []

    for c_idx, (name, prompt_tmpl) in enumerate(
        zip(candidate_names, candidate_prompts)
    ):
        preds: list[str] = []

        # Ensure {text} placeholder exists and is correctly placed
        if "{text}" not in prompt_tmpl:
            logger.info("Prompt '%s' missing {text} — auto-inserting.", name)
            prompt_tmpl = _ensure_text_placeholder(prompt_tmpl)
            auto_fixed_names.append(name)

        for t_idx, text_item in enumerate(texts):
            filled = prompt_tmpl.replace("{text}", text_item)

            # Save a preview of the first filled prompt (once)
            if c_idx == 0 and t_idx == 0:
                prompt_preview = filled

            try:
                resp = llm.complete(filled, max_tokens=150)
                raw = resp.text.strip()
                label = _extract_label_robust(raw, labels)
            except Exception as e:
                raw = str(e)
                label = "ERROR"
                logger.warning("Classification failed: %s", e)

            preds.append(label)

            # Save first 5 raw samples from first variant for diagnostics
            if c_idx == 0 and t_idx < 5:
                first_variant_raw_samples.append(
                    (true_labels[t_idx], label, raw[:120])
                )

        # Populate detail table columns for this variant
        for idx in range(n_eval):
            detail_rows[idx][f"{name}"] = preds[idx]
            match = preds[idx].strip().lower() == true_labels[idx].strip().lower()
            detail_rows[idx][f"{name} ✓"] = "✅" if match else "❌"

        # Calculate metrics
        correct = sum(
            1 for p, t in zip(preds, true_labels)
            if p.strip().lower() == t.strip().lower()
        )
        acc = correct / n_eval if n_eval else 0.0
        f1 = _macro_f1_standalone(preds, true_labels)
        scores.append((name, round(f1, 4), round(acc, 4)))

    # ---- Build summary text ----
    lines = [
        f"在 {n_eval} 条已标注数据上测试"
        + (f"（共 {n_total_labeled} 条标注数据，取前 50 条测试）"
           if n_total_labeled > 50 else "")
        + "：\n",
    ]

    best_name = max(scores, key=lambda s: s[1])[0]
    for name, f1, acc in scores:
        marker = " ✅ 推荐" if name == best_name else ""
        lines.append(f"  {name:12s}  F1 = {f1:.3f}  Acc = {acc:.3f}{marker}")

    lines.append(f"\n推荐「{best_name}」，请点击对应的「采用」按钮。")

    # ---- Warn about auto-fixed variants ----
    if auto_fixed_names:
        lines.append(
            f"\n⚠️ 以下版本缺少 {{text}} 占位符，已自动插入：{', '.join(auto_fixed_names)}"
        )

    detail_df = pd.DataFrame(detail_rows)
    return "\n".join(lines), detail_df


# ---------------------------------------------------------------------------
# Step 4b: Execute Prompt Classification
# ---------------------------------------------------------------------------


def _run_classification(
    df_state, text_col, label_col, labels_str,
    prompt_text,
    backend, model, api_key,
):
    """Run classification on UNLABELED texts only using the finalized prompt."""
    if df_state is None:
        return "请先上传数据。", None

    if not api_key and backend != "ollama":
        return "请输入 API Key。", None

    if not text_col:
        return "请指定文本列名。", None

    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not labels:
        return "请输入类别标签（逗号分隔）。", None

    custom_prompt = prompt_text.strip() if prompt_text and prompt_text.strip() else None

    # Pre-flight validation: ensure {text} placeholder is present and correct
    if custom_prompt:
        custom_prompt = _ensure_text_placeholder(custom_prompt)

    llm = LLMClient(backend=backend, model=model, api_key=api_key or None)

    # ---- Determine which rows need classification ----
    lc = (label_col or "").strip()
    if lc and lc in df_state.columns:
        unlabeled_mask = (
            df_state[lc].isna()
            | df_state[lc].astype(str).str.strip().isin(["", "nan", "None"])
        )
        n_already = int((~unlabeled_mask).sum())
    else:
        unlabeled_mask = pd.Series(True, index=df_state.index)
        n_already = 0

    classify_df = df_state[unlabeled_mask]
    texts = classify_df[text_col].dropna().astype(str).tolist()

    if not texts:
        return "所有文本都已有标签，无需分类。", None

    # ---- Classify unlabeled texts ----
    classifier = PromptClassifier(llm, custom_prompt=custom_prompt)
    report = classifier.classify(texts=texts, labels=labels)

    # ---- Build combined result DataFrame ----
    result_rows = []
    pred_iter = iter(report.results)
    for _, row in df_state.iterrows():
        text_val = str(row[text_col]) if pd.notna(row[text_col]) else ""
        if n_already > 0 and not unlabeled_mask.loc[row.name]:
            # Already labeled — keep existing label
            result_rows.append({
                "text": text_val,
                "predicted_label": str(row[lc]),
                "source": "既有标签",
            })
        else:
            r = next(pred_iter, None)
            result_rows.append({
                "text": text_val,
                "predicted_label": r.predicted_label if r else "UNKNOWN",
                "source": "LLM 预测",
            })

    result_df = pd.DataFrame(result_rows)

    summary = (
        f"分类完成！\n"
        f"新分类：{report.n_classified} 条成功，{report.n_failed} 条失败\n"
        f"跳过已有标签：{n_already} 条\n"
        f"新分类标签分布：{report.label_distribution}"
    )
    return summary, result_df


def _run_finetune(
    df_state, text_col, label_col,
    model_name, batch_size, num_epochs, learning_rate,
):
    """Run fine-tuning with HuggingFace transformers."""
    if df_state is None:
        return "请先上传数据。", None

    if not label_col or label_col not in df_state.columns:
        return "Fine-tuning 需要标签列。请确保数据中包含标签。", None

    labeled = df_state.dropna(subset=[label_col])
    if len(labeled) < 20:
        return f"标注数据不足（当前 {len(labeled)} 条，至少需要 20 条）。请先在标注页完成更多标注。", None

    try:
        from socialscikit.quantikit.classifier import Classifier, TrainConfig

        config = TrainConfig(
            model_name=model_name,
            batch_size=int(batch_size),
            num_epochs=int(num_epochs),
            learning_rate=float(learning_rate),
        )
        clf = Classifier()
        result = clf.train(
            train_df=labeled,
            text_col=text_col,
            label_col=label_col,
            config=config,
        )

        # Predict on full dataset
        all_texts = df_state[text_col].dropna().astype(str).tolist()
        pred_result = clf.predict(all_texts)
        result_df = clf.to_dataframe(all_texts, pred_result)

        summary = (
            f"Fine-tuning 完成！\n"
            f"最佳 F1：{result.best_eval_f1:.4f}（第 {result.best_epoch} 轮）\n"
            f"最佳 Loss：{result.best_eval_loss:.4f}\n"
            f"模型保存至：{result.model_path}\n"
            f"已对全部 {len(all_texts)} 条文本生成预测。"
        )
        return summary, result_df

    except ImportError:
        return "Fine-tuning 需要安装 transformers, datasets, torch。请运行：pip install transformers datasets torch", None
    except Exception as e:
        return f"Fine-tuning 失败：{e}", None


# ---------------------------------------------------------------------------
# Step 4c: API Fine-tuning (OpenAI)
# ---------------------------------------------------------------------------


def _start_api_finetune(
    df_state, text_col, label_col, labels_str, label_defs_str,
    api_key, base_model, n_epochs, suffix,
):
    """Submit an API fine-tuning job to OpenAI.

    Returns (status_text, job_id_str).  Non-blocking — does not wait for
    training to complete.
    """
    if df_state is None:
        return "请先上传数据。", ""
    if not api_key:
        return "请输入 OpenAI API Key。", ""
    if not text_col or text_col not in df_state.columns:
        return f"文本列 '{text_col}' 不存在。", ""
    if not label_col or label_col not in df_state.columns:
        return f"标签列 '{label_col}' 不存在。请先标注数据。", ""

    labels = [l.strip() for l in labels_str.split(",") if l.strip()]
    if not labels:
        return "请输入类别标签。", ""

    # Parse optional label definitions
    label_defs = None
    if label_defs_str and label_defs_str.strip():
        label_defs = {}
        for line in label_defs_str.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                label_defs[k.strip()] = v.strip()

    try:
        from socialscikit.quantikit.api_finetuner import APIFineTuneConfig, APIFineTuner

        ft = APIFineTuner(api_key=api_key)
        jsonl_path = ft.prepare_jsonl(
            df_state, text_col, label_col, labels,
            label_definitions=label_defs,
        )
        file_id = ft.upload_file(jsonl_path)
        config = APIFineTuneConfig(
            model=base_model,
            n_epochs=n_epochs or "auto",
            suffix=suffix or "",
        )
        job_id = ft.create_job(file_id, config)
        return (
            f"训练任务已提交！\n"
            f"任务 ID: {job_id}\n"
            f"文件 ID: {file_id}\n"
            f"基础模型: {base_model}\n\n"
            f"训练通常需要 10-60 分钟。请点击「刷新状态」查看进度。"
        ), job_id

    except ImportError:
        return "API Fine-tuning 需要 openai 库。请运行：pip install openai", ""
    except Exception as e:
        return f"提交失败：{e}", ""


def _check_api_ft_status(
    job_id_state, df_state, text_col, api_key,
):
    """Check job status; if succeeded, auto-predict on all texts.

    Returns (status_text, result_df_or_None, result_df_or_None).
    """
    if not job_id_state:
        return "没有正在进行的训练任务。", None, None
    if not api_key:
        return "请输入 OpenAI API Key。", None, None

    try:
        from socialscikit.quantikit.api_finetuner import APIFineTuner

        ft = APIFineTuner(api_key=api_key)
        status = ft.check_status(job_id_state)
        status_text = APIFineTuner.format_status(status)

        if status.status == "succeeded" and status.fine_tuned_model:
            # Auto-predict
            texts = df_state[text_col].astype(str).tolist()
            status_text += f"\n\n正在使用 {status.fine_tuned_model} 对 {len(texts)} 条文本进行预测..."
            preds = ft.predict(texts, status.fine_tuned_model)
            import pandas as pd
            result_df = pd.DataFrame({
                "text": texts,
                "predicted_label": preds,
            })
            status_text += "\n预测完成！"
            return status_text, result_df, result_df

        return status_text, None, None

    except Exception as e:
        return f"查询失败：{e}", None, None


def _cancel_api_ft_job(job_id_state, api_key):
    """Cancel a running fine-tuning job."""
    if not job_id_state:
        return "没有正在进行的训练任务。"
    if not api_key:
        return "请输入 OpenAI API Key。"
    try:
        from socialscikit.quantikit.api_finetuner import APIFineTuner

        ft = APIFineTuner(api_key=api_key)
        status = ft.cancel_job(job_id_state)
        return APIFineTuner.format_status(status)
    except Exception as e:
        return f"取消失败：{e}"


# ---------------------------------------------------------------------------
# Step 5: Evaluation
# ---------------------------------------------------------------------------


def _evaluate_results(result_df_state, df_state, label_col, pred_col="predicted_label"):
    """Evaluate predictions against ground truth.

    Returns
    -------
    tuple
        (text_report, metrics_html, confusion_fig, per_class_fig)
    """
    _empty = ("", "", None, None)
    if result_df_state is None or df_state is None:
        return ("请先运行分类。", *_empty[1:])
    if not label_col or label_col not in df_state.columns:
        return ("未找到标签列，无法评估。请确认数据中包含真实标签。", *_empty[1:])

    true_labels = df_state[label_col].dropna().astype(str).tolist()
    pred_labels = result_df_state[pred_col].tolist()

    # Align lengths (in case of NaN drops)
    min_len = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]

    evaluator = Evaluator()
    report = evaluator.evaluate(true_labels, pred_labels)

    text = Evaluator.format_report(report)

    # Metric summary cards
    metrics_html = charts.format_eval_metrics_html(
        report.accuracy, report.macro_f1, report.weighted_f1,
        report.cohens_kappa, report.n_total, report.n_correct,
    )

    # Confusion matrix chart
    cm_fig = None
    if report.confusion_matrix:
        try:
            cm_fig = charts.plot_confusion_matrix(
                report.confusion_matrix.labels, report.confusion_matrix.matrix,
            )
        except Exception as e:
            logger.warning("Failed to plot confusion matrix: %s", e)

    # Per-class metrics chart
    pc_fig = None
    if report.per_class:
        try:
            pc_data = [
                {"label": pc.label, "precision": pc.precision,
                 "recall": pc.recall, "f1": pc.f1, "support": pc.support}
                for pc in report.per_class
            ]
            pc_fig = charts.plot_per_class_metrics(pc_data)
        except Exception as e:
            logger.warning("Failed to plot per-class metrics: %s", e)

    return text, metrics_html, cm_fig, pc_fig


# ---------------------------------------------------------------------------
# Step 5b: Export Pipeline Log
# ---------------------------------------------------------------------------


def _export_pipeline_log(result_df_state, df_state, label_col, pred_col="predicted_label"):
    """Export QuantiKit pipeline metadata as JSON for the Toolbox Methods Generator."""
    import json, tempfile

    if result_df_state is None or df_state is None:
        return None

    log = {"pipeline": "quantikit", "n_samples": int(len(df_state))}

    # Class info from predictions
    if "predicted_label" in result_df_state.columns:
        labels = result_df_state["predicted_label"].dropna().unique().tolist()
        log["n_classes"] = len(labels)
        log["class_labels"] = [str(l) for l in sorted(labels)]

    # Evaluation metrics (re-compute if ground truth available)
    if label_col and label_col in df_state.columns:
        true_labels = df_state[label_col].dropna().astype(str).tolist()
        pred_labels = result_df_state[pred_col].tolist()
        min_len = min(len(true_labels), len(pred_labels))
        true_labels = true_labels[:min_len]
        pred_labels = pred_labels[:min_len]

        evaluator = Evaluator()
        report = evaluator.evaluate(true_labels, pred_labels)
        log["accuracy"] = report.accuracy
        log["macro_f1"] = report.macro_f1
        log["weighted_f1"] = report.weighted_f1
        log["cohens_kappa"] = report.cohens_kappa

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix="_quantikit_log.json", delete=False, encoding="utf-8",
    )
    json.dump(log, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Step 6: Export
# ---------------------------------------------------------------------------


def _export_results(result_df_state):
    """Export results as CSV file for download."""
    if result_df_state is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    result_df_state.to_csv(tmp.name, index=False)
    return tmp.name


# ---------------------------------------------------------------------------
# Template download
# ---------------------------------------------------------------------------


def _download_template():
    """Return the QuantiKit template CSV path."""
    return str(get_template_path("quantikit"))


# ---------------------------------------------------------------------------
# Build Gradio App
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    """Create the QuantiKit Gradio application."""

    _force_light_js = """
    () => {
        const forceLightMode = () => {
            document.documentElement.classList.remove('dark');
            document.body.classList.remove('dark');
        };
        forceLightMode();
        new MutationObserver(forceLightMode).observe(document.body, {
            attributes: true, attributeFilter: ['class']
        });
        new MutationObserver(forceLightMode).observe(document.documentElement, {
            attributes: true, attributeFilter: ['class']
        });
    }
    """

    _force_light_css = """
    :root, :host { color-scheme: light only !important; }
    .dark, body.dark, html.dark, :root.dark, .dark :root, .dark :host {
        --body-background-fill: #ffffff !important;
        --body-text-color: #333333 !important;
        --body-text-color-subdued: #777777 !important;
        --background-fill-primary: #ffffff !important;
        --background-fill-secondary: #fafafa !important;
        --block-background-fill: #ffffff !important;
        --block-border-color: #f0f0f0 !important;
        --block-label-text-color: #444444 !important;
        --block-title-text-color: #333333 !important;
        --panel-background-fill: #ffffff !important;
        --input-background-fill: #ffffff !important;
        --input-border-color: #e5e7eb !important;
        --input-placeholder-color: #aaaaaa !important;
        --border-color-primary: #e5e7eb !important;
        --neutral-50: #fafafa !important; --neutral-100: #f5f5f5 !important;
        --neutral-200: #e5e5e5 !important; --neutral-300: #d4d4d4 !important;
        --neutral-400: #a3a3a3 !important; --neutral-500: #737373 !important;
        --neutral-600: #525252 !important; --neutral-700: #404040 !important;
        --neutral-800: #262626 !important; --neutral-900: #171717 !important;
        --color-accent: #333 !important; --color-accent-soft: #f5f5f5 !important;
        --text-color: #333333 !important; --text-color-subdued: #777777 !important;
        --shadow-drop: none !important; --shadow-drop-lg: none !important;
        --button-secondary-background-fill: #ffffff !important;
        --table-even-background-fill: #ffffff !important;
        --table-odd-background-fill: #fafafa !important;
        --loader-color: #4A90D9 !important;
        --error-background-fill: #fef2f2 !important;
        --error-border-color: #fca5a5 !important;
        --error-text-color: #991b1b !important;
        color-scheme: light only !important;
    }
    /* Progress bar */
    .progress-bar { background: #4A90D9 !important; }
    .progress-bar-wrap { background: #e5e7eb !important; border: 1px solid #d1d5db !important; }
    .eta-bar { background: rgba(74, 144, 217, 0.20) !important; opacity: 1 !important; }
    .progress-text, .progress-level, .progress-level-inner { color: #333 !important; }
    .meta-text, .meta-text-center { color: #666 !important; }
    .wrap.generating { border-color: #4A90D9 !important; }
    .dark .progress-bar { background: #4A90D9 !important; }
    .dark .progress-bar-wrap { background: #e5e7eb !important; }
    .dark .eta-bar { background: rgba(74, 144, 217, 0.20) !important; opacity: 1 !important; }
    .dark .meta-text, .dark .meta-text-center { color: #666 !important; }
    /* Toast */
    .toast-body { background: #fff !important; color: #333 !important; }
    .toast-body.info { border-left: 4px solid #4A90D9 !important; }
    .dark .toast-body { background: #fff !important; color: #333 !important; }
    .dark .gradio-container, .dark .block, .dark .panel, .dark .form, .dark .tabitem {
        background: #ffffff !important; color: #333333 !important;
    }
    .dark textarea, .dark input[type="text"], .dark input[type="password"],
    .dark input[type="number"], .dark select {
        background: #ffffff !important; color: #333333 !important; border-color: #e0e0e0 !important;
    }
    .dark textarea::placeholder, .dark input::placeholder { color: #aaaaaa !important; }
    .dark label span { color: #333 !important; }
    .dark .table-wrap, .dark .table-wrap th, .dark .table-wrap td {
        color: #333333 !important;
    }
    """

    with gr.Blocks(
        title="SocialSciKit — QuantiKit",
        theme=gr.themes.Soft(),
        js=_force_light_js,
        css=_force_light_css,
    ) as app:
        gr.Markdown("# SocialSciKit — QuantiKit\n零代码文本分类工具，面向社会科学研究者。")

        # Shared state
        df_state = gr.State(None)
        result_df_state = gr.State(None)
        annotation_session_state = gr.State(None)

        # =============================================================
        # Tab 1: Upload & Validate
        # =============================================================
        with gr.Tab("1. 数据上传"):
            gr.Markdown(
                "请上传 CSV 或 Excel 文件。文件必须包含一列文本数据。"
            )
            with gr.Row():
                file_input = gr.File(label="上传数据文件", file_types=[".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt"])
                template_btn = gr.Button("下载模板")
                template_file = gr.File(label="模板文件", visible=False)

            summary_box = gr.Textbox(label="诊断报告", lines=15, interactive=False)
            issues_table = gr.Dataframe(label="问题列表", interactive=False)
            preview_table = gr.Dataframe(label="数据预览（前 10 行）", interactive=False)

            with gr.Row():
                auto_fix_btn = gr.Button("自动修复", variant="secondary")
                fix_status = gr.Textbox(label="修复状态", interactive=False)

            # Events
            file_input.change(
                fn=_load_and_validate,
                inputs=[file_input],
                outputs=[df_state, summary_box, issues_table, preview_table],
            )
            auto_fix_btn.click(
                fn=_apply_fixes,
                inputs=[df_state],
                outputs=[df_state, fix_status],
            )
            template_btn.click(fn=_download_template, outputs=[template_file])

        # =============================================================
        # Tab 2: Method Recommendation + Budget
        # =============================================================
        with gr.Tab("2. 方法推荐"):
            gr.Markdown("配置任务参数，获取方法推荐和标注预算建议。")
            with gr.Row():
                task_type = gr.Dropdown(
                    choices=sorted(TASK_TYPES), value="sentiment",
                    label="任务类型",
                )
                n_classes = gr.Number(label="类别数量", value=2, precision=0)
                target_f1 = gr.Slider(0.5, 1.0, value=0.80, step=0.05, label="目标 F1")
                budget_level = gr.Dropdown(
                    choices=["low", "medium", "high"], value="medium",
                    label="预算水平",
                )
            with gr.Row():
                text_col_input = gr.Textbox(label="文本列名", value="text")
                label_col_input = gr.Textbox(label="标签列名（可选）", value="label")

            recommend_btn = gr.Button("生成推荐", variant="primary")
            features_box = gr.Textbox(label="任务特征", lines=8, interactive=False)
            recommendation_box = gr.Textbox(label="方法推荐", lines=15, interactive=False)
            budget_box = gr.Textbox(label="标注预算推荐", lines=8, interactive=False)
            with gr.Row():
                budget_plot = gr.Plot(label="边际收益曲线", scale=3)
                budget_curve_tbl = gr.Dataframe(label="关键节点", interactive=False, scale=1)

            recommend_btn.click(
                fn=_extract_and_recommend,
                inputs=[df_state, task_type, n_classes, target_f1, budget_level,
                        text_col_input, label_col_input],
                outputs=[features_box, recommendation_box, budget_box, budget_plot, budget_curve_tbl],
            )

        # =============================================================
        # Tab 3: Annotation
        # =============================================================
        with gr.Tab("3. 标注"):
            gr.Markdown(
                "手动标注数据（可选）。适用于需要少量标注数据进行 few-shot 或 fine-tuning 的场景。\n"
                "支持标注、跳过、标记疑问和撤销操作。"
            )

            with gr.Row():
                ann_text_col = gr.Textbox(label="文本列名", value="text")
                ann_label_col = gr.Textbox(label="已有标签列（可选）", value="")
                ann_labels = gr.Textbox(label="标签列表（逗号分隔）", value="positive, negative, neutral")
                ann_shuffle = gr.Checkbox(label="随机顺序", value=False)

            create_session_btn = gr.Button("创建标注会话", variant="primary")
            ann_stats = gr.Textbox(label="标注进度", interactive=False)

            with gr.Group():
                ann_idx_display = gr.Textbox(label="当前条目", interactive=False)
                ann_text_display = gr.Textbox(label="待标注文本", lines=5, interactive=False)

            with gr.Row():
                ann_label_input = gr.Textbox(label="输入标签", scale=3)
                ann_submit_btn = gr.Button("标注", variant="primary", scale=1)
                ann_skip_btn = gr.Button("跳过", variant="secondary", scale=1)

            with gr.Row():
                ann_flag_note = gr.Textbox(label="疑问备注", placeholder="标记原因...", scale=3)
                ann_flag_btn = gr.Button("标记疑问", variant="secondary", scale=1)
                ann_undo_btn = gr.Button("撤销", variant="secondary", scale=1)

            ann_msg = gr.Textbox(label="消息", interactive=False)

            with gr.Accordion("导出标注", open=False):
                with gr.Row():
                    ann_include_all = gr.Checkbox(label="包含跳过/标记的条目", value=False)
                    ann_export_btn = gr.Button("导出标注数据")
                    ann_merge_btn = gr.Button("合并标注到主数据集", variant="secondary")
                ann_export_table = gr.Dataframe(label="标注结果", interactive=False)
                ann_merge_msg = gr.Textbox(label="合并状态", interactive=False)

            # Events
            session_outputs = [annotation_session_state, ann_stats, ann_text_display, ann_idx_display, ann_msg]

            create_session_btn.click(
                fn=_create_annotation_session,
                inputs=[df_state, ann_text_col, ann_label_col, ann_labels, ann_shuffle],
                outputs=session_outputs,
            )
            ann_submit_btn.click(
                fn=_annotate_item,
                inputs=[annotation_session_state, ann_label_input],
                outputs=session_outputs,
            )
            ann_skip_btn.click(
                fn=_skip_item,
                inputs=[annotation_session_state],
                outputs=session_outputs,
            )
            ann_flag_btn.click(
                fn=_flag_item,
                inputs=[annotation_session_state, ann_flag_note],
                outputs=session_outputs,
            )
            ann_undo_btn.click(
                fn=_undo_annotation,
                inputs=[annotation_session_state],
                outputs=session_outputs,
            )
            ann_export_btn.click(
                fn=_export_annotations,
                inputs=[annotation_session_state, ann_include_all],
                outputs=[ann_export_table, ann_msg],
            )
            ann_merge_btn.click(
                fn=_update_main_df_from_annotations,
                inputs=[annotation_session_state, df_state, ann_text_col, ann_label_col],
                outputs=[df_state, ann_merge_msg],
            )

        # =============================================================
        # Tab 4: Classification
        # =============================================================
        with gr.Tab("4. 分类"):
            gr.Markdown("配置 LLM 并运行分类。支持 Prompt 分类和 Fine-tuning 两种方式。")

            with gr.Row():
                backend = gr.Dropdown(
                    choices=["openai", "anthropic", "ollama"], value="openai",
                    label="LLM 后端",
                )
                model_name = gr.Textbox(label="模型名称", value="gpt-4o-mini")
                api_key_input = gr.Textbox(label="API Key", type="password")

            with gr.Row():
                cls_text_col = gr.Textbox(label="文本列名", value="text")
                cls_label_col = gr.Textbox(label="标签列名（评估/优化用）", value="label")
                cls_labels = gr.Textbox(label="类别标签（逗号分隔）", value="positive, negative, neutral")

            with gr.Tabs():
                # --- Sub-tab: Prompt Classification (guided workflow) ---
                with gr.Tab("Prompt 分类"):
                    gr.Markdown(
                        "### 第一步：设计 Prompt\n"
                        "描述你的分类任务，填写类别定义和示例。系统将调用 LLM 生成 Prompt。"
                    )
                    sa_task_desc = gr.Textbox(
                        label="任务描述（用自然语言描述你要做什么）", lines=3,
                        placeholder="例如：我需要对文本的情感倾向进行分类……",
                    )
                    label_defs_input = gr.Textbox(
                        label="类别定义（每行一个，格式：标签：定义 或 标签: 定义）", lines=4,
                        placeholder="positive：表达正面情感\nnegative：表达负面情感",
                    )
                    examples_input = gr.Textbox(
                        label="正例（可选 — 每行一个，格式：标签：示例文本）", lines=3,
                        placeholder="positive：Great product, I love it!\nnegative：Terrible, waste of money.\nneutral：It's okay, nothing special.",
                    )
                    excl_input = gr.Textbox(
                        label="反例（可选 — 容易误判的边界案例，格式同上）", lines=2,
                        placeholder="positive：虽然提到了好但整体是负面的\nnegative：有批评但总体是正面的",
                    )
                    gen_prompt_btn = gr.Button("生成 Prompt（调用 LLM）", variant="primary")

                    gr.Markdown("### 当前 Prompt\n下方即为最终分类指令，可直接编辑。")
                    prompt_box = gr.Textbox(
                        label="当前 Prompt（可编辑）", lines=10, interactive=True,
                        placeholder="点击「生成 Prompt」由 LLM 自动生成，或直接在此输入 …",
                    )

                    gr.Markdown(
                        "### 第二步：优化 & 对比（可选）\n"
                        "先评估 Prompt 质量获取改进建议，再手动编辑不同版本进行对比测试。"
                    )
                    with gr.Row():
                        sa_eval_btn = gr.Button("评估 Prompt 质量", variant="secondary")
                        sa_copy_btn = gr.Button("复制到对比栏 →", variant="secondary", size="sm")
                    sa_eval_msg = gr.Textbox(label="评估结果 & 改进建议", lines=8, interactive=False)
                    gr.Markdown(
                        "**对比版本**（将当前 Prompt 复制过来修改，测试不同写法的效果）："
                    )
                    with gr.Row():
                        sa_v1 = gr.Textbox(label="对比版本 1", lines=6, interactive=True, scale=1)
                        sa_v2 = gr.Textbox(label="对比版本 2", lines=6, interactive=True, scale=1)
                        sa_v3 = gr.Textbox(label="对比版本 3", lines=6, interactive=True, scale=1)
                    with gr.Row():
                        sa_use1 = gr.Button("采用版本 1", variant="secondary", size="sm")
                        sa_use2 = gr.Button("采用版本 2", variant="secondary", size="sm")
                        sa_use3 = gr.Button("采用版本 3", variant="secondary", size="sm")
                    sa_test_btn = gr.Button("测试对比", variant="primary")
                    opt_summary = gr.Textbox(label="测试结果", lines=6, interactive=False)
                    sa_test_detail = gr.Dataframe(
                        label="详细预测对比（每条文本的真实标签 vs 各变体预测）",
                        interactive=False, wrap=True,
                    )

                    gr.Markdown(
                        "### 第三步：执行分类\n"
                        "只会对未标注的文本分类，已有标签的不会重新分类。"
                    )
                    classify_btn = gr.Button("开始分类", variant="primary")
                    cls_summary = gr.Textbox(label="分类结果", lines=4, interactive=False)
                    cls_results = gr.Dataframe(label="分类详情", interactive=False)

                # --- Sub-tab: Fine-tuning ---
                with gr.Tab("Fine-tuning"):
                    gr.Markdown(
                        "使用 HuggingFace Transformers 进行模型微调。\n"
                        "需要已标注的数据（至少 20 条）。需要安装 transformers, datasets, torch。"
                    )
                    with gr.Row():
                        ft_model = gr.Dropdown(
                            choices=["roberta-base", "xlm-roberta-base", "bert-base-uncased", "bert-base-chinese"],
                            value="roberta-base",
                            label="预训练模型",
                        )
                        ft_batch_size = gr.Number(label="Batch Size", value=16, precision=0)
                        ft_epochs = gr.Number(label="训练轮数", value=5, precision=0)
                        ft_lr = gr.Number(label="学习率", value=2e-5)

                    ft_btn = gr.Button("开始 Fine-tuning", variant="primary")
                    ft_summary = gr.Textbox(label="训练结果", lines=8, interactive=False)
                    ft_results = gr.Dataframe(label="预测结果", interactive=False)

                # --- Sub-tab: API Fine-tuning ---
                with gr.Tab("API Fine-tuning"):
                    gr.Markdown(
                        "**在 OpenAI 服务器上微调模型。**\n\n"
                        "- 需要 OpenAI API Key（需启用付费账户）\n"
                        "- 至少 10 条标注数据（建议 50-100+ 条）\n"
                        "- 训练时间约 10-60 分钟，训练完成后自动对全部文本进行预测\n"
                        "- 费用参考：gpt-4o-mini 约 $0.003/1K tokens"
                    )
                    with gr.Row():
                        aft_key = gr.Textbox(label="OpenAI API Key", type="password")
                        aft_model = gr.Dropdown(
                            choices=["gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-3.5-turbo-0125"],
                            value="gpt-4o-mini-2024-07-18", label="基础模型",
                        )
                    with gr.Row():
                        aft_epochs = gr.Dropdown(
                            choices=["auto", "1", "2", "3", "4", "5"],
                            value="auto", label="训练轮数",
                        )
                        aft_suffix = gr.Textbox(label="模型后缀（可选）", placeholder="my-classifier")
                    aft_btn = gr.Button("提交训练任务", variant="primary")
                    aft_status = gr.Textbox(label="训练状态", lines=6, interactive=False)
                    with gr.Row():
                        aft_check = gr.Button("刷新状态", variant="secondary")
                        aft_cancel = gr.Button("取消训练", variant="secondary")
                    aft_results = gr.Dataframe(label="预测结果", interactive=False)
                    aft_job_id = gr.State("")

            # Events — Prompt workflow
            gen_prompt_btn.click(
                fn=_generate_smart_prompt,
                inputs=[sa_task_desc, cls_labels, label_defs_input, examples_input, excl_input,
                        backend, model_name, api_key_input],
                outputs=[prompt_box],
            )
            sa_eval_btn.click(
                fn=_evaluate_prompt,
                inputs=[prompt_box, cls_labels, backend, model_name, api_key_input],
                outputs=[sa_eval_msg],
            )
            sa_copy_btn.click(
                fn=lambda p: (p, p, p),
                inputs=[prompt_box],
                outputs=[sa_v1, sa_v2, sa_v3],
            )
            sa_test_btn.click(
                fn=_test_variants,
                inputs=[df_state, cls_text_col, cls_label_col,
                        prompt_box, sa_v1, sa_v2, sa_v3,
                        cls_labels, backend, model_name, api_key_input],
                outputs=[opt_summary, sa_test_detail],
            )
            def _adopt_variant(v, name):
                if not v or not v.strip():
                    gr.Warning(f"{name} 为空，无法采用。")
                    raise gr.Error(f"{name} 为空")
                gr.Info(f"✅ 已采用{name}作为当前 Prompt")
                return v

            sa_use1.click(fn=lambda v: _adopt_variant(v, "版本 1"), inputs=[sa_v1], outputs=[prompt_box])
            sa_use2.click(fn=lambda v: _adopt_variant(v, "版本 2"), inputs=[sa_v2], outputs=[prompt_box])
            sa_use3.click(fn=lambda v: _adopt_variant(v, "版本 3"), inputs=[sa_v3], outputs=[prompt_box])
            classify_btn.click(
                fn=_run_classification,
                inputs=[df_state, cls_text_col, cls_label_col, cls_labels,
                        prompt_box,
                        backend, model_name, api_key_input],
                outputs=[cls_summary, result_df_state],
            )
            result_df_state.change(
                fn=lambda x: x,
                inputs=[result_df_state],
                outputs=[cls_results],
            )

            # Events — Fine-tuning
            def _ft_wrapper(df_state, text_col, label_col, model, bs, epochs, lr):
                summary, result_df = _run_finetune(
                    df_state, text_col, label_col, model, bs, epochs, lr,
                )
                return summary, result_df, result_df

            ft_btn.click(
                fn=_ft_wrapper,
                inputs=[df_state, cls_text_col, cls_label_col,
                        ft_model, ft_batch_size, ft_epochs, ft_lr],
                outputs=[ft_summary, ft_results, result_df_state],
            )

            # Events — API Fine-tuning
            aft_btn.click(
                fn=_start_api_finetune,
                inputs=[df_state, cls_text_col, cls_label_col, cls_labels,
                        label_defs_input, aft_key, aft_model, aft_epochs, aft_suffix],
                outputs=[aft_status, aft_job_id],
            )
            aft_check.click(
                fn=_check_api_ft_status,
                inputs=[aft_job_id, df_state, cls_text_col, aft_key],
                outputs=[aft_status, aft_results, result_df_state],
            )
            aft_cancel.click(
                fn=_cancel_api_ft_job,
                inputs=[aft_job_id, aft_key],
                outputs=[aft_status],
            )

        # =============================================================
        # Tab 5: Evaluation
        # =============================================================
        eval_report_state = gr.State(None)

        with gr.Tab("5. 评估"):
            gr.Markdown("将分类结果与真实标签对比，计算评估指标。")
            eval_label_col = gr.Textbox(label="真实标签列名", value="label")
            eval_btn = gr.Button("运行评估", variant="primary")
            eval_output = gr.Textbox(label="评估报告", lines=20, interactive=False)

            eval_btn.click(
                fn=_evaluate_results,
                inputs=[result_df_state, df_state, eval_label_col],
                outputs=[eval_output, eval_report_state],
            )

            # --- Inter-Coder Reliability ---
            with gr.Accordion("编码者间信度 (ICR)", open=False):
                gr.Markdown("上传第二编码者（或人工标注）的标签 CSV，计算编码者间一致性。")
                with gr.Row():
                    icr_file = gr.File(label="第二编码者标签（CSV）", file_types=[".csv"])
                    icr_second_col = gr.Textbox(label="标签列名", value="label")
                icr_btn = gr.Button("计算编码者间信度", variant="secondary")
                icr_output = gr.Textbox(label="信度报告", lines=14, interactive=False)

                icr_btn.click(
                    fn=_compute_icr,
                    inputs=[result_df_state, icr_file, icr_second_col],
                    outputs=[icr_output],
                )

        # =============================================================
        # Tab 6: Export
        # =============================================================
        with gr.Tab("6. 导出"):
            gr.Markdown("导出分类结果为 CSV 文件。")
            export_btn = gr.Button("导出结果", variant="primary")
            export_file = gr.File(label="下载结果")

            export_btn.click(
                fn=_export_results,
                inputs=[result_df_state],
                outputs=[export_file],
            )

            # --- Methods Section Generator ---
            with gr.Accordion("方法论段落生成", open=False):
                gr.Markdown("根据分析流程自动生成论文方法论段落草稿。复制后按需编辑。")
                methods_btn = gr.Button("生成方法论段落", variant="secondary")
                methods_en = gr.Textbox(label="Methods (English)", lines=8, interactive=True)
                methods_zh = gr.Textbox(label="方法论（中文）", lines=8, interactive=True)

                methods_btn.click(
                    fn=_generate_qt_methods,
                    inputs=[result_df_state, df_state, eval_report_state],
                    outputs=[methods_en, methods_zh],
                )

    return app


# ---------------------------------------------------------------------------
# Launch helper (called by CLI)
# ---------------------------------------------------------------------------


def launch(port: int = 7860, share: bool = False) -> None:
    """Launch the QuantiKit Gradio app."""
    app = create_app()
    app.launch(server_port=port, share=share)
