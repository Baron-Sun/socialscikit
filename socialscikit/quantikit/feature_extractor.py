"""Feature extraction for method recommendation.

Extracts basic features automatically from data + user inputs.
Advanced features (semantic diversity, domain specificity, etc.) are optional
and require explicit activation.
"""

from __future__ import annotations

from dataclasses import dataclass

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

# Recognised task types for the method recommender
TASK_TYPES = {"sentiment", "framing", "moral", "topic", "stance", "custom"}

# Budget levels
BUDGET_LEVELS = {"low", "medium", "high"}

# Downstream use types
DOWNSTREAM_USES = {"hypothesis_testing", "descriptive", "prediction"}


@dataclass
class TaskFeatures:
    """Features describing a classification task.

    Basic features are extracted automatically; advanced features require
    ``enable_advanced=True`` and may need extra compute or user input.
    """

    # === Basic features (auto-extracted) ===
    n_samples: int = 0
    n_labeled: int = 0
    label_balance_ratio: float | None = None  # max_class / min_class
    avg_text_length_tokens: int = 0
    language: str = "en"
    is_multilingual: bool = False
    n_classes: int = 0
    task_type: str = "custom"
    target_f1: float = 0.80
    budget_level: str = "medium"

    # === Advanced features (optional) ===
    text_diversity: float | None = None
    domain_specificity: str | None = None  # "general" | "domain_specific"
    annotation_agreement: float | None = None  # Cohen's Kappa
    class_boundary_clarity: str | None = None  # "clear" | "ambiguous"
    downstream_use: str | None = None  # "hypothesis_testing" | "descriptive" | "prediction"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    if _enc is not None:
        return len(_enc.encode(text))
    return len(text.split())


def _detect_language(texts: pd.Series, sample_size: int = 50) -> tuple[str, bool]:
    """Return (primary_language, is_multilingual)."""
    if _detect_lang is None:
        return "en", False
    sample = texts.dropna().sample(min(sample_size, len(texts)), random_state=42)
    langs: list[str] = []
    for t in sample:
        try:
            langs.append(_detect_lang(str(t)))
        except LangDetectException:
            continue
    if not langs:
        return "en", False
    from collections import Counter

    counter = Counter(langs)
    primary = counter.most_common(1)[0][0]
    is_multi = len(counter) > 1 and counter.most_common(1)[0][1] / len(langs) < 0.9
    return primary, is_multi


def _compute_text_diversity(texts: pd.Series) -> float:
    """Compute semantic diversity via sentence embeddings.

    Returns a 0-1 score where 1 = maximally diverse.
    Uses mean pairwise cosine distance on a sample.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
    except ImportError:
        return 0.5  # fallback

    sample = texts.dropna().sample(min(200, len(texts)), random_state=42).tolist()
    if len(sample) < 2:
        return 0.0

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf = vectorizer.fit_transform(sample)
    sim_matrix = cosine_similarity(tfidf)

    # Mean off-diagonal similarity -> diversity = 1 - similarity
    n = sim_matrix.shape[0]
    total_sim = (sim_matrix.sum() - n) / (n * (n - 1))
    return round(float(1 - total_sim), 4)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Extract task features from data and user inputs for method recommendation.

    Basic features are always computed. Advanced features require
    ``enable_advanced=True`` and cost extra compute time.

    Usage::

        extractor = FeatureExtractor()
        features = extractor.extract(
            df,
            user_inputs={"task_type": "sentiment", "target_f1": 0.85},
        )
    """

    def extract(
        self,
        df: pd.DataFrame,
        user_inputs: dict | None = None,
        text_col: str | None = None,
        label_col: str | None = None,
        enable_advanced: bool = False,
    ) -> TaskFeatures:
        """Extract features from the data and user-supplied metadata.

        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        user_inputs : dict or None
            User-provided metadata. Recognised keys:

            - ``task_type`` (str): one of TASK_TYPES
            - ``target_f1`` (float): desired F1 score, 0.0–1.0
            - ``budget_level`` (str): "low" / "medium" / "high"
            - ``n_classes`` (int): number of classes (if no label column)
            - ``class_boundary_clarity`` (str): "clear" / "ambiguous"
            - ``domain_specificity`` (str): "general" / "domain_specific"
            - ``annotation_agreement`` (float): Cohen's Kappa if available
            - ``downstream_use`` (str): one of DOWNSTREAM_USES

        text_col : str or None
            Text column name. Auto-detected if None.
        label_col : str or None
            Label column name. Auto-detected if None.
        enable_advanced : bool
            If True, compute expensive advanced features (text diversity).
        """
        user_inputs = user_inputs or {}
        features = TaskFeatures()

        # --- Auto-detect columns ---
        if text_col is None:
            text_col = self._guess_col(df, {"text", "content", "body", "message", "sentence", "comment"})
        if label_col is None:
            label_col = self._guess_col(df, {"label", "labels", "class", "category", "tag", "sentiment"})

        # --- Basic: sample counts ---
        features.n_samples = len(df)

        # --- Basic: label info ---
        if label_col and label_col in df.columns:
            label_series = df[label_col].dropna()
            features.n_labeled = len(label_series)
            counts = label_series.value_counts()
            features.n_classes = len(counts)
            if len(counts) >= 2:
                features.label_balance_ratio = round(float(counts.iloc[0] / counts.iloc[-1]), 2)
        else:
            features.n_labeled = 0
            features.n_classes = user_inputs.get("n_classes", 0)

        # --- Basic: text length ---
        if text_col and text_col in df.columns:
            valid_texts = df[text_col].dropna().astype(str)
            valid_texts = valid_texts[valid_texts.str.strip() != ""]
            if len(valid_texts) > 0:
                token_counts = valid_texts.apply(_count_tokens)
                features.avg_text_length_tokens = int(token_counts.mean())

                # Language detection
                lang, is_multi = _detect_language(valid_texts)
                features.language = lang
                features.is_multilingual = is_multi

        # --- User inputs: task metadata ---
        if "task_type" in user_inputs:
            tt = user_inputs["task_type"]
            features.task_type = tt if tt in TASK_TYPES else "custom"

        if "target_f1" in user_inputs:
            features.target_f1 = float(user_inputs["target_f1"])

        if "budget_level" in user_inputs:
            bl = user_inputs["budget_level"]
            features.budget_level = bl if bl in BUDGET_LEVELS else "medium"

        if "n_classes" in user_inputs and features.n_classes == 0:
            features.n_classes = int(user_inputs["n_classes"])

        # --- Advanced features ---
        if enable_advanced:
            if text_col and text_col in df.columns:
                valid_texts = df[text_col].dropna().astype(str)
                valid_texts = valid_texts[valid_texts.str.strip() != ""]
                if len(valid_texts) > 1:
                    features.text_diversity = _compute_text_diversity(valid_texts)

        # Advanced features from user input (always accepted if provided)
        if "domain_specificity" in user_inputs:
            features.domain_specificity = user_inputs["domain_specificity"]
        if "annotation_agreement" in user_inputs:
            features.annotation_agreement = float(user_inputs["annotation_agreement"])
        if "class_boundary_clarity" in user_inputs:
            features.class_boundary_clarity = user_inputs["class_boundary_clarity"]
        if "downstream_use" in user_inputs:
            features.downstream_use = user_inputs["downstream_use"]

        return features

    @staticmethod
    def _guess_col(df: pd.DataFrame, hints: set[str]) -> str | None:
        for col in df.columns:
            if col.strip().lower() in hints:
                return col
        return None
