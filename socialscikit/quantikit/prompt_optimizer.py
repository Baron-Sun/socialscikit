"""Automatic prompt generation, variant creation, and selection.

Literature:
- Zhou et al. (2022). Large Language Models Are Human-Level Prompt Engineers.
  ICLR 2023. (APE — automatic prompt search framework)
- Gonen et al. (2022). Demystifying Prompts in Language Models via Perplexity
  Estimation. (few-shot example selection: diversity > random sampling)
- Dunivin (2024). Scalable qualitative coding with LLMs. arXiv:2401.15170.
  (exclusion criteria significantly improve coding accuracy)
- Lo (2023). The CLEAR path: A framework for enhancing information literacy
  through prompt engineering. JMC. (social science prompt best practices)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import pandas as pd

from socialscikit.core.llm_client import LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PromptVariant:
    """A single prompt variant with optional evaluation score."""

    prompt_text: str
    style: str = ""  # e.g. "concise", "detailed", "structured"
    f1_score: float | None = None
    accuracy: float | None = None


@dataclass
class PromptEvalResult:
    """Evaluation results comparing multiple prompt variants."""

    best_variant: PromptVariant
    all_variants: list[PromptVariant] = field(default_factory=list)
    eval_summary: pd.DataFrame | None = None  # comparison table


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a text classification assistant for social science research.
Your task is to classify each text into exactly one of the provided categories.
Follow the definitions and examples carefully. Respond with ONLY the label name."""

_PROMPT_TEMPLATE = """\
## 任务
{task_description}

## 类别定义
{label_definitions}

## 正例（应当归入各类别的示例文本）
{inclusion_examples}

## 反例（不应归入各类别的边界案例）
{exclusion_examples}

## 待分类文本
{text_placeholder}

请只输出类别标签，不要输出其他内容："""

_VARIANT_GENERATION_PROMPT = """\
You are an expert in prompt engineering for text classification in social science.

Below is a classification prompt. Generate {n} alternative versions of this prompt.
Each variant should use a DIFFERENT style:
1. "concise" — shorter, more direct instructions, fewer words
2. "detailed" — more elaborate definitions with boundary clarifications and decision rules
3. "structured" — uses numbered steps or explicit decision tree logic

CRITICAL RULES:
- Each variant MUST contain the exact placeholder {{text}} (with curly braces). This is where the actual text will be inserted at runtime. Do NOT rename it to [text], <text>, or anything else.
- Keep the same task, categories, and output format — only rephrase the instructions and framing.
- Each variant should genuinely differ in how it frames the classification task, not just in word choice.

Return a JSON array of objects, each with "style" and "prompt_text" keys.

Original prompt:
---
{original_prompt}
---

Return ONLY valid JSON (no markdown fencing):"""

_EXAMPLE_SELECTION_SYSTEM = """\
You are helping select representative examples for a classification task.
Given a set of texts for a category, pick the {n} most DIVERSE examples that
cover different aspects or edge cases of the category.
Return a JSON array of the selected text strings.
Return ONLY valid JSON (no markdown fencing)."""


# ---------------------------------------------------------------------------
# Prompt Optimizer
# ---------------------------------------------------------------------------


class PromptOptimizer:
    """Automatic prompt generation, variant creation, and evaluation.

    Usage::

        optimizer = PromptOptimizer(llm_client)
        initial = optimizer.generate_initial(
            task_description="Classify Reddit posts by sentiment",
            label_definitions={"positive": "Expresses approval...", ...},
            inclusion_examples={"positive": ["Great product!", ...], ...},
            exclusion_examples={"positive": ["It's okay I guess", ...], ...},
        )
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # 1. Generate initial prompt
    # ------------------------------------------------------------------

    def generate_initial(
        self,
        task_description: str,
        label_definitions: dict[str, str],
        inclusion_examples: dict[str, list[str]] | None = None,
        exclusion_examples: dict[str, list[str]] | None = None,
    ) -> str:
        """Generate a complete classification prompt from components.

        Parameters
        ----------
        task_description : str
            High-level description of the classification task.
        label_definitions : dict[str, str]
            ``{label_name: definition}`` for each category.
        inclusion_examples : dict[str, list[str]] or None
            ``{label_name: [example_texts]}`` — texts that belong to each category.
        exclusion_examples : dict[str, list[str]] or None
            ``{label_name: [example_texts]}`` — boundary cases that do NOT belong.
            Strongly recommended (Dunivin 2024).

        Returns
        -------
        str
            The assembled prompt text.
        """
        # Format label definitions
        label_def_lines = []
        for label, defn in label_definitions.items():
            label_def_lines.append(f"- **{label}**: {defn}")
        label_defs_str = "\n".join(label_def_lines)

        # Format inclusion examples
        incl_str = self._format_examples(inclusion_examples, "inclusion")

        # Format exclusion examples
        # Dunivin (2024): exclusion criteria significantly improve coding accuracy
        excl_str = self._format_examples(exclusion_examples, "exclusion")
        if not exclusion_examples:
            excl_str = "(No exclusion examples provided. Adding exclusion examples is strongly recommended — Dunivin 2024.)"

        prompt = _PROMPT_TEMPLATE.format(
            task_description=task_description,
            label_definitions=label_defs_str,
            inclusion_examples=incl_str,
            exclusion_examples=excl_str,
            text_placeholder="{text}",
        )
        return prompt

    # ------------------------------------------------------------------
    # 2. Generate variants (requires LLM)
    # ------------------------------------------------------------------

    def generate_variants(
        self,
        initial_prompt: str,
        n: int = 3,
    ) -> list[PromptVariant]:
        """Use LLM to generate n prompt variants in different styles.

        References APE framework (Zhou et al. 2022): generating multiple
        prompt candidates and selecting the best one.

        Parameters
        ----------
        initial_prompt : str
            The base prompt to rephrase.
        n : int
            Number of variants to generate.

        Returns
        -------
        list[PromptVariant]
            The original + n generated variants.
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for variant generation. Pass llm_client to PromptOptimizer.")

        gen_prompt = _VARIANT_GENERATION_PROMPT.format(
            n=n, original_prompt=initial_prompt,
        )
        resp = self.llm_client.complete(gen_prompt, system=_SYSTEM_TEMPLATE, max_tokens=4096)

        variants = [PromptVariant(prompt_text=initial_prompt, style="original")]

        try:
            parsed = json.loads(resp.text)
            for item in parsed:
                variants.append(PromptVariant(
                    prompt_text=item["prompt_text"],
                    style=item.get("style", "unknown"),
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Failed to parse LLM variant response: %s", e)
            # Return just the original if parsing fails
            pass

        return variants

    # ------------------------------------------------------------------
    # 3. Select diverse few-shot examples (requires LLM or TF-IDF)
    # ------------------------------------------------------------------

    def select_examples(
        self,
        df: pd.DataFrame,
        label_col: str,
        text_col: str = "text",
        n_per_class: int = 3,
        method: str = "tfidf",
    ) -> dict[str, list[str]]:
        """Select diverse, representative few-shot examples per class.

        Gonen et al. (2022): diversity-based selection outperforms random
        sampling for few-shot example choice.

        Parameters
        ----------
        df : pd.DataFrame
        label_col : str
        text_col : str
        n_per_class : int
            Number of examples to select per class.
        method : str
            ``"tfidf"`` (fast, no LLM) or ``"llm"`` (uses LLM to pick).

        Returns
        -------
        dict[str, list[str]]
            ``{label: [selected_texts]}``
        """
        selected: dict[str, list[str]] = {}

        for label in df[label_col].dropna().unique():
            class_texts = df.loc[df[label_col] == label, text_col].dropna().tolist()
            if len(class_texts) <= n_per_class:
                selected[str(label)] = class_texts
                continue

            if method == "tfidf":
                selected[str(label)] = self._select_by_tfidf(class_texts, n_per_class)
            elif method == "llm" and self.llm_client is not None:
                selected[str(label)] = self._select_by_llm(class_texts, n_per_class)
            else:
                selected[str(label)] = self._select_by_tfidf(class_texts, n_per_class)

        return selected

    # ------------------------------------------------------------------
    # 4. Evaluate and select best variant (requires LLM)
    # ------------------------------------------------------------------

    def evaluate_and_select(
        self,
        variants: list[PromptVariant],
        eval_df: pd.DataFrame,
        label_col: str,
        text_col: str = "text",
    ) -> PromptEvalResult:
        """Evaluate prompt variants on a validation set and pick the best.

        Parameters
        ----------
        variants : list[PromptVariant]
        eval_df : pd.DataFrame
            Validation set (20-50 rows recommended).
        label_col : str
        text_col : str

        Returns
        -------
        PromptEvalResult
        """
        if self.llm_client is None:
            raise RuntimeError("LLM client required for evaluation.")

        texts = eval_df[text_col].tolist()
        true_labels = eval_df[label_col].astype(str).tolist()
        results_rows = []

        for variant in variants:
            preds = self._classify_batch(variant.prompt_text, texts)
            correct = sum(1 for p, t in zip(preds, true_labels) if p.strip().lower() == t.strip().lower())
            acc = correct / len(true_labels) if true_labels else 0.0
            variant.accuracy = round(acc, 4)

            # Simple macro-F1 approximation
            variant.f1_score = self._macro_f1(preds, true_labels)

            results_rows.append({
                "style": variant.style,
                "accuracy": variant.accuracy,
                "f1": variant.f1_score,
            })

        summary_df = pd.DataFrame(results_rows)

        best = max(variants, key=lambda v: v.f1_score or 0.0)

        return PromptEvalResult(
            best_variant=best,
            all_variants=variants,
            eval_summary=summary_df,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_examples(
        examples: dict[str, list[str]] | None, kind: str,
    ) -> str:
        if not examples:
            return f"(No {kind} examples provided.)"
        lines = []
        for label, texts in examples.items():
            lines.append(f"### {label}")
            for t in texts:
                lines.append(f'  - "{t}"')
        return "\n".join(lines)

    @staticmethod
    def _select_by_tfidf(texts: list[str], n: int) -> list[str]:
        """Select n most diverse texts using TF-IDF + max-min distance."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_distances
        except ImportError:
            # Fallback: evenly spaced selection
            step = max(1, len(texts) // n)
            return [texts[i * step] for i in range(min(n, len(texts)))]

        vectorizer = TfidfVectorizer(max_features=3000)
        tfidf = vectorizer.fit_transform(texts)
        dist_matrix = cosine_distances(tfidf)

        # Greedy max-min diversity selection
        selected_idx: list[int] = [0]  # start with first text
        for _ in range(n - 1):
            min_dists = dist_matrix[selected_idx].min(axis=0)
            # Pick the text that is farthest from any already-selected text
            candidate = int(min_dists.argmax())
            if candidate in selected_idx:
                # Fallback if somehow duplicated
                remaining = [i for i in range(len(texts)) if i not in selected_idx]
                if remaining:
                    candidate = remaining[0]
                else:
                    break
            selected_idx.append(candidate)

        return [texts[i] for i in selected_idx]

    def _select_by_llm(self, texts: list[str], n: int) -> list[str]:
        """Use LLM to pick the n most representative and diverse texts."""
        system = _EXAMPLE_SELECTION_SYSTEM.format(n=n)
        prompt = f"Texts:\n" + "\n".join(f"- {t}" for t in texts[:50])  # cap input
        resp = self.llm_client.complete(prompt, system=system, max_tokens=2048)
        try:
            selected = json.loads(resp.text)
            if isinstance(selected, list):
                return [str(s) for s in selected[:n]]
        except (json.JSONDecodeError, TypeError):
            pass
        return self._select_by_tfidf(texts, n)

    def _classify_batch(self, prompt_template: str, texts: list[str]) -> list[str]:
        """Classify each text using the prompt template synchronously."""
        predictions = []
        for text in texts:
            filled = prompt_template.replace("{text}", text)
            # Don't override with _SYSTEM_TEMPLATE — the prompt itself
            # already contains output-format instructions that may conflict.
            resp = self.llm_client.complete(filled, max_tokens=100)
            label = self._extract_label(resp.text)
            predictions.append(label)
        return predictions

    @staticmethod
    def _extract_label(raw: str) -> str:
        """Extract a label from an LLM response.

        Handles JSON (``{"label": "W"}``), quoted strings (``"W"``),
        and plain text.
        """
        raw = raw.strip()
        # 1. Try full JSON parse
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "label" in data:
                return str(data["label"]).strip()
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # 2. Regex for embedded JSON
        m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
        if m:
            return m.group(1).strip()
        # 3. Strip surrounding quotes
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
            return raw[1:-1].strip()
        # 4. If multi-line, take first non-empty line
        for line in raw.splitlines():
            line = line.strip()
            if line:
                return line
        return raw

    @staticmethod
    def _macro_f1(preds: list[str], trues: list[str]) -> float:
        """Compute macro F1 from string predictions and true labels."""
        preds_norm = [p.strip().lower() for p in preds]
        trues_norm = [t.strip().lower() for t in trues]
        labels = set(trues_norm)

        f1_scores = []
        for label in labels:
            tp = sum(1 for p, t in zip(preds_norm, trues_norm) if p == label and t == label)
            fp = sum(1 for p, t in zip(preds_norm, trues_norm) if p == label and t != label)
            fn = sum(1 for p, t in zip(preds_norm, trues_norm) if p != label and t == label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        return round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0.0
