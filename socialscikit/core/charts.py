"""Academic-style matplotlib charts for SocialSciKit dashboards.

All charts follow a consistent visual language:
- White background, subtle gridlines
- Blue (#4A90D9) primary palette with complementary greens/oranges
- CJK font fallback for Chinese label support
- Clean spines, readable tick labels

Every ``plot_*`` function returns a ``matplotlib.figure.Figure`` ready for
``gr.Plot``.  The ``format_*_html`` helpers produce styled HTML cards for
``gr.HTML``.
"""

from __future__ import annotations

import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as mticker      # noqa: E402
import numpy as np                       # noqa: E402

logger = logging.getLogger(__name__)

# ======================================================================
# Palette & style
# ======================================================================

PALETTE = {
    "primary":   "#4A90D9",
    "secondary": "#5BA88D",
    "accent":    "#E8734A",
    "warning":   "#F5A623",
    "purple":    "#9B7ED8",
    "neutral":   "#8B9DC3",
    "bg":        "#FFFFFF",
    "text":      "#333333",
    "text_sec":  "#666666",
    "grid":      "#F0F0F0",
    "spine":     "#DDDDDD",
    "card_bg":   "#F8F9FA",
}

# Categorical palette (up to 10 classes)
CAT_COLORS = [
    "#4A90D9", "#5BA88D", "#E8734A", "#F5A623", "#9B7ED8",
    "#E85D75", "#5BCBCF", "#B8CC4A", "#D4A76A", "#8B9DC3",
]

CJK_FONTS = [
    "PingFang SC", "Heiti SC", "STHeiti",
    "Microsoft YaHei", "SimHei", "DejaVu Sans",
]


def _setup_style() -> None:
    """Apply the global academic-style defaults."""
    plt.rcParams.update({
        "font.sans-serif": CJK_FONTS,
        "axes.unicode_minus": False,
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["bg"],
        "axes.edgecolor": PALETTE["spine"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text_sec"],
        "ytick.color": PALETTE["text_sec"],
    })


def _clean_ax(ax: plt.Axes, *, keep_left: bool = True,
              keep_bottom: bool = True) -> None:
    """Remove top/right spines and style remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["spine"])
    ax.spines["bottom"].set_color(PALETTE["spine"])
    ax.tick_params(colors=PALETTE["text_sec"], labelsize=9, length=3)


# ======================================================================
# QuantiKit — Evaluation charts
# ======================================================================


def plot_confusion_matrix(
    labels: list[str],
    matrix: list[list[int]],
    lang: str = "zh",
) -> plt.Figure:
    """Confusion matrix heatmap with annotated counts.

    Parameters
    ----------
    labels : list[str]
        Class labels (row = true, col = predicted).
    matrix : list[list[int]]
        Confusion matrix as nested list.
    """
    _setup_style()
    k = len(labels)

    # Empty / degenerate input — return a placeholder
    if k == 0 or not matrix:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
        fig.patch.set_facecolor("white")
        msg = "无数据" if lang == "zh" else "No data"
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color=PALETTE["text_sec"])
        ax.axis("off")
        return fig

    arr = np.array(matrix, dtype=float)

    # Normalise for colour (row-wise, avoid /0)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = arr / row_sums

    figsize = (max(4.5, 0.9 * k + 2), max(4, 0.9 * k + 1.5))
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")

    cmap = plt.cm.Blues
    im = ax.imshow(norm, interpolation="nearest", cmap=cmap, aspect="auto",
                   vmin=0, vmax=1)

    # Annotate cells
    thresh = norm.max() / 2.0
    for i in range(k):
        for j in range(k):
            color = "white" if norm[i, j] > thresh else PALETTE["text"]
            count = int(arr[i, j])
            pct = norm[i, j] * 100
            text = f"{count}\n({pct:.0f}%)" if count > 0 else "0"
            fontsize = 11 if k <= 6 else 9
            ax.text(j, i, text, ha="center", va="center",
                    color=color, fontsize=fontsize, fontweight="500")

    # Axis labels
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    tick_fontsize = 9 if k <= 8 else 7
    ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=45 if k > 5 else 0,
                       ha="right" if k > 5 else "center")
    ax.set_yticklabels(labels, fontsize=tick_fontsize)

    xlabel = "预测标签" if lang == "zh" else "Predicted"
    ylabel = "真实标签" if lang == "zh" else "True"
    title = "混淆矩阵" if lang == "zh" else "Confusion Matrix"
    ax.set_xlabel(xlabel, fontsize=10.5, color=PALETTE["text"], labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10.5, color=PALETTE["text"], labelpad=8)
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=12)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.06)
    cbar.ax.tick_params(labelsize=8, colors=PALETTE["text_sec"])
    cbar_label = "行归一化比例" if lang == "zh" else "Row-normalised"
    cbar.set_label(cbar_label, fontsize=9, color=PALETTE["text_sec"])

    fig.tight_layout(pad=1.5)
    return fig


def plot_per_class_metrics(
    per_class: list[dict[str, Any]],
    lang: str = "zh",
) -> plt.Figure:
    """Grouped horizontal bar chart of Precision / Recall / F1 per class.

    Parameters
    ----------
    per_class : list of dict
        Each dict has keys: label, precision, recall, f1, support.
    """
    _setup_style()
    n = len(per_class)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 2), dpi=150)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    labels = [pc["label"] for pc in per_class]
    precision = [pc["precision"] for pc in per_class]
    recall = [pc["recall"] for pc in per_class]
    f1 = [pc["f1"] for pc in per_class]

    y = np.arange(n)
    bar_h = 0.25

    figsize = (6, max(3, 0.6 * n + 1.5))
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")

    bars_p = ax.barh(y + bar_h, precision, bar_h, color=PALETTE["primary"],
                     label="Precision", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_r = ax.barh(y, recall, bar_h, color=PALETTE["secondary"],
                     label="Recall", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_f = ax.barh(y - bar_h, f1, bar_h, color=PALETTE["accent"],
                     label="F1", alpha=0.85, edgecolor="white", linewidth=0.5)

    # Value labels
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            w = bar.get_width()
            if w > 0.05:
                ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{w:.2f}", va="center", fontsize=8,
                        color=PALETTE["text_sec"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    title = "各类别指标" if lang == "zh" else "Per-Class Metrics"
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _clean_ax(ax)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)

    fig.tight_layout(pad=1.2)
    return fig


def plot_label_distribution(
    true_counts: dict[str, int],
    pred_counts: dict[str, int],
    lang: str = "zh",
) -> plt.Figure:
    """Side-by-side vertical bar chart of true vs predicted distributions.

    Parameters
    ----------
    true_counts, pred_counts : dict[str, int]
        Label → count mappings.
    """
    _setup_style()
    all_labels = sorted(set(true_counts) | set(pred_counts))
    n = len(all_labels)
    if n == 0:
        fig, ax = plt.subplots(figsize=(5, 2), dpi=150)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    true_vals = [true_counts.get(l, 0) for l in all_labels]
    pred_vals = [pred_counts.get(l, 0) for l in all_labels]

    x = np.arange(n)
    w = 0.35

    figsize = (max(5, 0.7 * n + 2), 3.5)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")

    true_label = "真实" if lang == "zh" else "True"
    pred_label = "预测" if lang == "zh" else "Predicted"

    ax.bar(x - w / 2, true_vals, w, color=PALETTE["primary"], alpha=0.8,
           label=true_label, edgecolor="white", linewidth=0.5)
    ax.bar(x + w / 2, pred_vals, w, color=PALETTE["accent"], alpha=0.8,
           label=pred_label, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9,
                       rotation=45 if n > 5 else 0,
                       ha="right" if n > 5 else "center")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    title = "标签分布对比" if lang == "zh" else "Label Distribution"
    ylabel = "数量" if lang == "zh" else "Count"
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=10)
    ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["text_sec"])
    ax.legend(fontsize=8, frameon=False)
    _clean_ax(ax)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)

    fig.tight_layout(pad=1.2)
    return fig


# ======================================================================
# Annotation progress chart
# ======================================================================


def plot_annotation_progress(
    labeled: int,
    skipped: int,
    flagged: int,
    pending: int,
    lang: str = "zh",
) -> plt.Figure:
    """Donut chart showing annotation session progress."""
    _setup_style()

    sizes = [labeled, skipped, flagged, pending]
    colors = [PALETTE["primary"], PALETTE["warning"], PALETTE["purple"], "#E0E0E0"]
    if lang == "zh":
        seg_labels = ["已标注", "已跳过", "已标记", "待标注"]
    else:
        seg_labels = ["Labeled", "Skipped", "Flagged", "Pending"]

    # Filter out zero segments
    filtered = [(s, c, l) for s, c, l in zip(sizes, colors, seg_labels) if s > 0]
    if not filtered:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return fig

    sizes_f, colors_f, labels_f = zip(*filtered)
    total = sum(sizes_f)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    fig.patch.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        sizes_f, labels=labels_f, colors=colors_f,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9, color=PALETTE["text"]),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("600")

    # Centre text
    pct = (labeled / total * 100) if total > 0 else 0
    ax.text(0, 0, f"{pct:.0f}%", ha="center", va="center",
            fontsize=22, fontweight="700", color=PALETTE["primary"])
    progress_text = "完成" if lang == "zh" else "Done"
    ax.text(0, -0.15, progress_text, ha="center", va="center",
            fontsize=9, color=PALETTE["text_sec"])

    title = "标注进度" if lang == "zh" else "Annotation Progress"
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=12, y=1.02)

    fig.tight_layout(pad=0.5)
    return fig


# ======================================================================
# QualiKit — Extraction / Review charts
# ======================================================================


def plot_confidence_histogram(
    confidences: list[float],
    lang: str = "zh",
) -> plt.Figure:
    """Histogram of extraction confidence scores with tier shading."""
    _setup_style()
    if not confidences:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=150)
    fig.patch.set_facecolor("white")

    # Background tier shading
    ax.axvspan(0, 0.5, alpha=0.06, color="#E8734A")      # low
    ax.axvspan(0.5, 0.75, alpha=0.06, color="#F5A623")    # medium
    ax.axvspan(0.75, 1.0, alpha=0.06, color="#5BA88D")    # high

    ax.hist(confidences, bins=20, range=(0, 1), color=PALETTE["primary"],
            alpha=0.8, edgecolor="white", linewidth=0.5)

    # Tier labels
    tier_y = ax.get_ylim()[1] * 0.92
    if lang == "zh":
        ax.text(0.25, tier_y, "低", ha="center", fontsize=8,
                color=PALETTE["accent"], alpha=0.7)
        ax.text(0.625, tier_y, "中", ha="center", fontsize=8,
                color=PALETTE["warning"], alpha=0.7)
        ax.text(0.875, tier_y, "高", ha="center", fontsize=8,
                color=PALETTE["secondary"], alpha=0.7)
    else:
        ax.text(0.25, tier_y, "Low", ha="center", fontsize=8,
                color=PALETTE["accent"], alpha=0.7)
        ax.text(0.625, tier_y, "Med", ha="center", fontsize=8,
                color=PALETTE["warning"], alpha=0.7)
        ax.text(0.875, tier_y, "High", ha="center", fontsize=8,
                color=PALETTE["secondary"], alpha=0.7)

    # Median line
    median = float(np.median(confidences))
    ax.axvline(median, color=PALETTE["accent"], linewidth=1.2, linestyle="--",
               alpha=0.8)
    median_label = f"中位数 {median:.2f}" if lang == "zh" else f"Median {median:.2f}"
    ax.text(median + 0.02, ax.get_ylim()[1] * 0.8, median_label,
            fontsize=8, color=PALETTE["accent"])

    title = "置信度分布" if lang == "zh" else "Confidence Distribution"
    xlabel = "置信度" if lang == "zh" else "Confidence"
    ylabel = "频次" if lang == "zh" else "Count"
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["text_sec"])
    ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["text_sec"])
    ax.set_xlim(0, 1)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _clean_ax(ax)

    fig.tight_layout(pad=1.2)
    return fig


def plot_theme_distribution(
    theme_counts: dict[str, int],
    lang: str = "zh",
) -> plt.Figure:
    """Horizontal bar chart of theme / research question frequencies."""
    _setup_style()
    if not theme_counts:
        fig, ax = plt.subplots(figsize=(5, 2), dpi=150)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    # Sort ascending for horizontal bars (bottom to top)
    sorted_items = sorted(theme_counts.items(), key=lambda x: x[1])
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    n = len(labels)

    colors = [CAT_COLORS[i % len(CAT_COLORS)] for i in range(n)]

    figsize = (6, max(2.5, 0.45 * n + 1))
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor("white")

    bars = ax.barh(range(n), values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5, height=0.6)

    # Value labels
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                str(v), va="center", fontsize=9, color=PALETTE["text_sec"])

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    title = "主题分布" if lang == "zh" else "Theme Distribution"
    xlabel = "提取数" if lang == "zh" else "Extractions"
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=10)
    ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["text_sec"])
    _clean_ax(ax)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.6)

    fig.tight_layout(pad=1.2)
    return fig


def plot_review_progress(
    accepted: int,
    edited: int,
    rejected: int,
    pending: int,
    lang: str = "zh",
) -> plt.Figure:
    """Donut chart of review action distribution."""
    _setup_style()

    sizes = [accepted, edited, rejected, pending]
    colors_list = [PALETTE["secondary"], PALETTE["primary"],
                   PALETTE["accent"], "#E0E0E0"]
    if lang == "zh":
        seg_labels = ["接受", "编辑", "拒绝", "待审"]
    else:
        seg_labels = ["Accepted", "Edited", "Rejected", "Pending"]

    filtered = [(s, c, l) for s, c, l in zip(sizes, colors_list, seg_labels) if s > 0]
    if not filtered:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    sizes_f, colors_f, labels_f = zip(*filtered)
    total = sum(sizes_f)
    done = accepted + edited + rejected

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    fig.patch.set_facecolor("white")

    wedges, texts, autotexts = ax.pie(
        sizes_f, labels=labels_f, colors=colors_f,
        autopct=lambda pct: f"{pct:.0f}%" if pct > 5 else "",
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
        textprops=dict(fontsize=9, color=PALETTE["text"]),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("600")

    pct = (done / total * 100) if total > 0 else 0
    ax.text(0, 0, f"{pct:.0f}%", ha="center", va="center",
            fontsize=22, fontweight="700", color=PALETTE["secondary"])
    progress_text = "已审查" if lang == "zh" else "Reviewed"
    ax.text(0, -0.15, progress_text, ha="center", va="center",
            fontsize=9, color=PALETTE["text_sec"])

    title = "审查进度" if lang == "zh" else "Review Progress"
    ax.set_title(title, fontsize=12, fontweight="600", color=PALETTE["text"],
                 pad=12, y=1.02)

    fig.tight_layout(pad=0.5)
    return fig


# ======================================================================
# HTML metric cards
# ======================================================================

_CARD_CSS = (
    "display:inline-flex;flex-direction:column;align-items:center;"
    "justify-content:center;background:{bg};border-radius:10px;"
    "padding:18px 16px 14px;text-align:center;border-left:4px solid {color};"
    "min-width:120px;flex:1;"
)

_GRID_CSS = (
    "display:flex;flex-wrap:wrap;gap:12px;margin:8px 0 16px;"
)


def _metric_card(value: str, label: str, color: str) -> str:
    style = _CARD_CSS.format(bg=PALETTE["card_bg"], color=color)
    return (
        f'<div style="{style}">'
        f'<div style="font-size:26px;font-weight:700;color:{color};'
        f'line-height:1.2;">{value}</div>'
        f'<div style="font-size:11px;color:{PALETTE["text_sec"]};'
        f'margin-top:6px;letter-spacing:0.3px;">{label}</div>'
        f'</div>'
    )


def format_eval_metrics_html(
    accuracy: float,
    macro_f1: float,
    weighted_f1: float,
    cohens_kappa: float,
    n_total: int,
    n_correct: int,
    lang: str = "zh",
) -> str:
    """Render evaluation metrics as styled HTML cards.

    Returns
    -------
    str
        HTML string for ``gr.HTML``.
    """
    if lang == "zh":
        lbl_acc = "准确率"
        lbl_mf1 = "Macro F1"
        lbl_wf1 = "Weighted F1"
        lbl_kap = "Cohen's κ"
        lbl_tot = "总样本"
        lbl_cor = "正确数"
    else:
        lbl_acc = "Accuracy"
        lbl_mf1 = "Macro F1"
        lbl_wf1 = "Weighted F1"
        lbl_kap = "Cohen's κ"
        lbl_tot = "Total"
        lbl_cor = "Correct"

    cards = [
        _metric_card(f"{accuracy:.4f}", lbl_acc, PALETTE["primary"]),
        _metric_card(f"{macro_f1:.4f}", lbl_mf1, PALETTE["secondary"]),
        _metric_card(f"{weighted_f1:.4f}", lbl_wf1, PALETTE["accent"]),
        _metric_card(f"{cohens_kappa:.4f}", lbl_kap, PALETTE["purple"]),
        _metric_card(str(n_total), lbl_tot, PALETTE["neutral"]),
        _metric_card(str(n_correct), lbl_cor, PALETTE["primary"]),
    ]

    return f'<div style="{_GRID_CSS}">{"".join(cards)}</div>'


def format_review_stats_html(
    total: int,
    accepted: int,
    edited: int,
    rejected: int,
    pending: int,
    lang: str = "zh",
) -> str:
    """Render review progress as styled HTML cards."""
    if lang == "zh":
        labels = ["总条目", "已接受", "已编辑", "已拒绝", "待审"]
    else:
        labels = ["Total", "Accepted", "Edited", "Rejected", "Pending"]

    colors = [PALETTE["neutral"], PALETTE["secondary"], PALETTE["primary"],
              PALETTE["accent"], "#AAAAAA"]
    values = [str(total), str(accepted), str(edited), str(rejected), str(pending)]

    cards = [_metric_card(v, l, c) for v, l, c in zip(values, labels, colors)]
    return f'<div style="{_GRID_CSS}">{"".join(cards)}</div>'


def format_annotation_stats_html(
    total: int,
    labeled: int,
    skipped: int,
    flagged: int,
    pending: int,
    elapsed: float,
    label_dist: dict[str, int] | None = None,
    lang: str = "zh",
) -> str:
    """Render annotation session stats as styled HTML cards."""
    if lang == "zh":
        labels = ["总数", "已标注", "已跳过", "已标记", "待标注"]
    else:
        labels = ["Total", "Labeled", "Skipped", "Flagged", "Pending"]

    colors = [PALETTE["neutral"], PALETTE["primary"], PALETTE["warning"],
              PALETTE["purple"], "#AAAAAA"]
    values = [str(total), str(labeled), str(skipped), str(flagged), str(pending)]

    cards = [_metric_card(v, l, c) for v, l, c in zip(values, labels, colors)]

    # Progress percentage card
    pct = (labeled / total * 100) if total > 0 else 0
    pct_label = "完成率" if lang == "zh" else "Progress"
    cards.append(_metric_card(f"{pct:.0f}%", pct_label, PALETTE["secondary"]))

    # Time card
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    time_str = f"{mins}:{secs:02d}"
    time_label = "用时" if lang == "zh" else "Time"
    cards.append(_metric_card(time_str, time_label, PALETTE["accent"]))

    html = f'<div style="{_GRID_CSS}">{"".join(cards)}</div>'

    # Add label distribution bar if available
    if label_dist:
        html += _mini_label_bar(label_dist, lang)

    return html


def _mini_label_bar(label_dist: dict[str, int], lang: str = "zh") -> str:
    """Inline stacked bar showing label distribution."""
    total = sum(label_dist.values())
    if total == 0:
        return ""

    segments = []
    for i, (label, count) in enumerate(sorted(label_dist.items())):
        pct = count / total * 100
        color = CAT_COLORS[i % len(CAT_COLORS)]
        segments.append(
            f'<div style="flex:{pct};background:{color};height:18px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:10px;color:white;font-weight:500;min-width:30px;"'
            f' title="{label}: {count}">'
            f'{label[:8]}' if pct > 10 else
            f'<div style="flex:{pct};background:{color};height:18px;'
            f'min-width:4px;" title="{label}: {count}"></div>'
        )

    dist_label = "标签分布" if lang == "zh" else "Label Dist."
    return (
        f'<div style="margin-top:8px;">'
        f'<div style="font-size:10px;color:{PALETTE["text_sec"]};margin-bottom:4px;">'
        f'{dist_label}</div>'
        f'<div style="display:flex;border-radius:4px;overflow:hidden;">'
        f'{"".join(segments)}</div></div>'
    )
