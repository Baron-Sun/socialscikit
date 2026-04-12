"""Data quality diagnostics — generates a comprehensive report on the uploaded dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

try:
    from langdetect import detect as _detect_lang
    from langdetect.lang_detect_exception import LangDetectException
except ImportError:
    _detect_lang = None
    LangDetectException = Exception

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _enc = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TextLengthStats:
    """Token-level length distribution of the text column."""

    min: int = 0
    max: int = 0
    mean: float = 0.0
    median: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0


@dataclass
class LabelDistribution:
    """Per-label counts and proportions."""

    counts: dict[str, int] = field(default_factory=dict)
    proportions: dict[str, float] = field(default_factory=dict)


@dataclass
class DiagnosticsReport:
    """Full data quality report."""

    # Scale
    n_rows: int = 0
    n_columns: int = 0
    file_size_kb: float | None = None

    # Text
    text_col: str | None = None
    text_length_stats: TextLengthStats | None = None
    n_empty_texts: int = 0

    # Labels
    label_col: str | None = None
    label_distribution: LabelDistribution | None = None

    # Duplicates
    n_duplicate_rows: int = 0
    duplicate_rate: float = 0.0

    # Language
    language_distribution: dict[str, int] = field(default_factory=dict)
    primary_language: str | None = None

    # Potential bias
    bias_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    if _enc is not None:
        return len(_enc.encode(text))
    return len(text.split())


def _detect_languages(texts: pd.Series, sample_size: int = 100) -> dict[str, int]:
    """Detect language distribution from a sample."""
    if _detect_lang is None:
        return {}
    sample = texts.dropna().sample(min(sample_size, len(texts)), random_state=42)
    langs: list[str] = []
    for t in sample:
        try:
            langs.append(_detect_lang(str(t)))
        except LangDetectException:
            continue
    return dict(Counter(langs).most_common())


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def generate_diagnostics(
    df: pd.DataFrame,
    text_col: str | None = None,
    label_col: str | None = None,
    file_size_kb: float | None = None,
) -> DiagnosticsReport:
    """Generate a comprehensive diagnostics report.

    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
    text_col : str or None
        Name of the text column. If None, tries to auto-detect.
    label_col : str or None
        Name of the label column. If None, tries to auto-detect.
    file_size_kb : float or None
        Original file size in KB (for display only).

    Returns
    -------
    DiagnosticsReport
    """
    report = DiagnosticsReport(
        n_rows=len(df),
        n_columns=len(df.columns),
        file_size_kb=file_size_kb,
    )

    # --- Auto-detect columns if not provided ---
    if text_col is None:
        text_col = _guess_col(df, {"text", "content", "body", "message", "sentence", "comment"})
    if label_col is None:
        label_col = _guess_col(df, {"label", "labels", "class", "category", "tag", "sentiment"})

    report.text_col = text_col
    report.label_col = label_col

    # --- Text analysis ---
    if text_col and text_col in df.columns:
        texts = df[text_col].astype(str)
        empty_mask = df[text_col].isna() | (texts.str.strip() == "")
        report.n_empty_texts = int(empty_mask.sum())

        valid_texts = texts[~empty_mask]
        if len(valid_texts) > 0:
            token_counts = valid_texts.apply(_count_tokens)
            report.text_length_stats = TextLengthStats(
                min=int(token_counts.min()),
                max=int(token_counts.max()),
                mean=round(float(token_counts.mean()), 1),
                median=round(float(token_counts.median()), 1),
                p25=round(float(token_counts.quantile(0.25)), 1),
                p75=round(float(token_counts.quantile(0.75)), 1),
                p95=round(float(token_counts.quantile(0.95)), 1),
            )

            # Language detection
            lang_dist = _detect_languages(valid_texts)
            report.language_distribution = lang_dist
            if lang_dist:
                report.primary_language = next(iter(lang_dist))

    # --- Label distribution ---
    if label_col and label_col in df.columns:
        counts = df[label_col].value_counts()
        total = counts.sum()
        report.label_distribution = LabelDistribution(
            counts=counts.to_dict(),
            proportions={k: round(v / total, 4) for k, v in counts.items()},
        )

    # --- Duplicates ---
    n_dup = int(df.duplicated().sum())
    report.n_duplicate_rows = n_dup
    report.duplicate_rate = round(n_dup / len(df), 4) if len(df) > 0 else 0.0

    # --- Bias warnings ---
    report.bias_warnings = _check_bias(df, text_col)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _guess_col(df: pd.DataFrame, hints: set[str]) -> str | None:
    for col in df.columns:
        if col.strip().lower() in hints:
            return col
    return None


def _check_bias(df: pd.DataFrame, text_col: str | None) -> list[str]:
    """Heuristic checks for potential sampling bias."""
    warnings: list[str] = []

    # Check for timestamp column clustering
    for col in df.columns:
        if col.strip().lower() in {"date", "timestamp", "created_at", "time"}:
            try:
                dates = pd.to_datetime(df[col], errors="coerce").dropna()
                if len(dates) > 10:
                    date_range = (dates.max() - dates.min()).days
                    if date_range <= 1:
                        warnings.append(
                            "所有数据来自同一天，可能存在时间采样偏差，"
                            "建议确认是否需要更广时间范围的数据。"
                        )
                    elif date_range <= 7:
                        warnings.append(
                            "数据集中在一周内，建议确认时间范围是否满足研究需求。"
                        )
            except Exception:
                pass

    # Check text diversity — if many duplicates
    if text_col and text_col in df.columns:
        texts = df[text_col].dropna().astype(str)
        if len(texts) > 0:
            unique_rate = texts.nunique() / len(texts)
            if unique_rate < 0.8:
                warnings.append(
                    f"文本去重率为 {unique_rate:.1%}，较多重复文本可能影响分析结果，"
                    "建议去重后使用。"
                )

    return warnings


def format_report(report: DiagnosticsReport) -> str:
    """Format a DiagnosticsReport as a human-readable string.

    Useful for CLI output or Gradio display.
    """
    lines: list[str] = []
    lines.append("═══ 数据诊断报告 ═══")
    lines.append("")

    # Scale
    lines.append(f"📊 数据规模：{report.n_rows} 行 × {report.n_columns} 列")
    if report.file_size_kb is not None:
        lines.append(f"   文件大小：{report.file_size_kb:.1f} KB")
    lines.append(f"   空文本行：{report.n_empty_texts}")
    lines.append("")

    # Text length
    if report.text_length_stats:
        s = report.text_length_stats
        lines.append("📝 文本长度分布（tokens）：")
        lines.append(f"   最小 {s.min} | P25 {s.p25} | 中位数 {s.median} | P75 {s.p75} | P95 {s.p95} | 最大 {s.max}")
        lines.append(f"   平均 {s.mean} tokens")
        lines.append("")

    # Label distribution
    if report.label_distribution:
        lines.append("🏷️  标签分布：")
        for label, count in report.label_distribution.counts.items():
            pct = report.label_distribution.proportions[label] * 100
            lines.append(f"   {label}: {count} ({pct:.1f}%)")
        lines.append("")

    # Duplicates
    lines.append(f"🔁 重复行：{report.n_duplicate_rows} ({report.duplicate_rate:.1%})")
    lines.append("")

    # Language
    if report.language_distribution:
        lines.append(f"🌐 语言分布（采样检测）：")
        for lang, count in report.language_distribution.items():
            lines.append(f"   {lang}: {count}")
        lines.append("")

    # Bias warnings
    if report.bias_warnings:
        lines.append("⚠️  潜在偏差提示：")
        for w in report.bias_warnings:
            lines.append(f"   • {w}")
        lines.append("")

    return "\n".join(lines)
