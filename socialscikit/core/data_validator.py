"""Foolproof data validation — detects issues and offers actionable fixes."""

from __future__ import annotations

import re
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
    tiktoken = None
    _enc = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    severity: str  # "error" | "warning" | "info"
    field: str  # column name or location
    message: str  # user-facing explanation (non-technical)
    auto_fix_available: bool = False
    auto_fix_description: str | None = None


@dataclass
class ValidationReport:
    is_valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: str = ""
    suggested_text_col: str | None = None
    suggested_label_col: str | None = None
    n_rows: int = 0
    n_usable_rows: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEXT_COL_HINTS = {"text", "content", "body", "message", "sentence", "comment", "review", "post"}
_LABEL_COL_HINTS = {"label", "labels", "class", "category", "tag", "sentiment", "target"}
_SPEAKER_COL_HINTS = {"speaker", "speaker_id", "respondent", "participant", "interviewee"}
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")


def _guess_column(df: pd.DataFrame, hints: set[str]) -> str | None:
    """Return the first column whose lowered name matches a hint set."""
    for col in df.columns:
        if col.strip().lower() in hints:
            return col
    return None


def _count_tokens(text: str) -> int:
    if _enc is not None:
        return len(_enc.encode(text))
    return len(text.split())


def _detect_language(texts: pd.Series, sample_size: int = 50) -> str | None:
    """Detect majority language from a sample of texts."""
    if _detect_lang is None:
        return None
    sample = texts.dropna().sample(min(sample_size, len(texts)), random_state=42)
    langs: list[str] = []
    for t in sample:
        try:
            langs.append(_detect_lang(str(t)))
        except LangDetectException:
            continue
    if not langs:
        return None
    from collections import Counter

    return Counter(langs).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class DataValidator:
    """Validate a DataFrame for QuantiKit or QualiKit workflows.

    Usage::

        report = DataValidator().validate(df, mode="quantikit")
        for issue in report.issues:
            print(issue.severity, issue.message)
    """

    def validate(self, df: pd.DataFrame, mode: str = "quantikit") -> ValidationReport:
        """Run all checks and return a ValidationReport.

        Parameters
        ----------
        df : pd.DataFrame
            The loaded data.
        mode : str
            ``"quantikit"`` or ``"qualikit"``.
        """
        issues: list[ValidationIssue] = []
        text_col = _guess_column(df, _TEXT_COL_HINTS)
        label_col = _guess_column(df, _LABEL_COL_HINTS)

        # --- Common checks ---
        if text_col is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="columns",
                    message=(
                        "未找到文本列。请确认数据中包含名为 'text' 的列，"
                        "或在下方选择包含文本内容的列。"
                    ),
                )
            )
        else:
            # Empty text rows
            empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
            n_empty = int(empty_mask.sum())
            if n_empty > 0:
                pct = n_empty / len(df) * 100
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        field=text_col,
                        message=f"发现 {n_empty} 行空文本（占 {pct:.1f}%），是否自动删除？",
                        auto_fix_available=True,
                        auto_fix_description="删除空文本行",
                    )
                )

            # Text length stats (only on non-empty texts)
            valid_texts = df.loc[~empty_mask, text_col].astype(str)

            if len(valid_texts) > 0:
                word_counts = valid_texts.str.split().str.len()
                median_words = word_counts.median()
                if median_words < 10:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            field=text_col,
                            message=(
                                "文本较短（中位数不足 10 词），分类难度可能较高，"
                                "建议检查数据完整性。"
                            ),
                        )
                    )

                # Token length check
                token_counts = valid_texts.apply(_count_tokens)
                n_long = int((token_counts > 512).sum())
                if n_long > 0:
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            field=text_col,
                            message=(
                                f"{n_long} 条文本超过 512 tokens，"
                                "fine-tuning 时将自动截断，建议确认截断不影响关键信息。"
                            ),
                        )
                    )

                # Language detection
                lang = _detect_language(valid_texts)
                if lang and lang != "en":
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            field=text_col,
                            message=(
                                f"检测到主要语言为 {lang}，"
                                "将自动推荐多语言模型（XLM-RoBERTa）。"
                            ),
                        )
                    )

        # --- QuantiKit-specific checks ---
        if mode == "quantikit":
            issues.extend(self._check_quantikit(df, label_col))

        # --- QualiKit-specific checks ---
        elif mode == "qualikit":
            issues.extend(self._check_qualikit(df, text_col))

        # --- Build report ---
        has_error = any(i.severity == "error" for i in issues)
        n_usable = len(df)
        if text_col is not None:
            empty_mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
            n_usable = int((~empty_mask).sum())

        if has_error:
            summary = "数据存在必须修复的问题，请查看下方详情。"
        elif issues:
            summary = f"数据基本可用，但有 {len(issues)} 条建议，请查看。"
        else:
            summary = "数据格式正确，可以继续。"

        return ValidationReport(
            is_valid=not has_error,
            issues=issues,
            summary=summary,
            suggested_text_col=text_col,
            suggested_label_col=label_col,
            n_rows=len(df),
            n_usable_rows=n_usable,
        )

    # ------------------------------------------------------------------
    # QuantiKit checks
    # ------------------------------------------------------------------

    def _check_quantikit(
        self, df: pd.DataFrame, label_col: str | None
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if label_col is None:
            issues.append(
                ValidationIssue(
                    severity="info",
                    field="columns",
                    message=(
                        "未发现标签列，将进入零样本/少样本模式。"
                        "如需监督学习，请提供标签列。"
                    ),
                )
            )
        else:
            # Label distribution imbalance
            counts = df[label_col].value_counts()
            if len(counts) >= 2:
                ratio = counts.iloc[0] / counts.iloc[-1]
                if ratio > 5:
                    issues.append(
                        ValidationIssue(
                            severity="warning",
                            field=label_col,
                            message=(
                                f"标签分布不均衡（最多类:最少类 = {ratio:.1f}:1），"
                                "可能影响模型性能，建议补充少数类样本。"
                            ),
                        )
                    )

        return issues

    # ------------------------------------------------------------------
    # QualiKit checks
    # ------------------------------------------------------------------

    def _check_qualikit(
        self, df: pd.DataFrame, text_col: str | None
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        speaker_col = _guess_column(df, _SPEAKER_COL_HINTS)
        if speaker_col is None:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    field="columns",
                    message="未发现说话人列，脱敏和主题分析将视所有文本为单一来源。",
                )
            )

        # PII scan
        if text_col is not None:
            texts_combined = df[text_col].dropna().astype(str)
            n_email = int(texts_combined.str.contains(_EMAIL_RE).sum())
            n_phone = int(texts_combined.str.contains(_PHONE_RE).sum())
            n_pii = n_email + n_phone
            if n_pii > 0:
                details = []
                if n_email:
                    details.append(f"{n_email} 处疑似邮箱")
                if n_phone:
                    details.append(f"{n_phone} 处疑似电话号码")
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        field=text_col,
                        message=(
                            f"检测到可能的个人信息（{', '.join(details)}），"
                            "建议在分析前运行脱敏模块。"
                        ),
                    )
                )

        return issues


# ---------------------------------------------------------------------------
# Auto-fix helpers
# ---------------------------------------------------------------------------


def apply_auto_fixes(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    """Apply all auto-fixable issues and return a cleaned DataFrame.

    Currently supports:
    - Removing rows with empty text
    """
    df = df.copy()
    text_col = report.suggested_text_col
    for issue in report.issues:
        if not issue.auto_fix_available:
            continue
        if issue.auto_fix_description == "删除空文本行" and text_col is not None:
            mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
            df = df[~mask].reset_index(drop=True)
    return df
