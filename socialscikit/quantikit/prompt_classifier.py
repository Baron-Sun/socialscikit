"""Zero-shot and few-shot text classification via LLM prompts.

Provides synchronous and async batch classification with structured output
parsing, confidence extraction, and automatic result export.
"""

from __future__ import annotations

import asyncio
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
class ClassificationResult:
    """Result for a single text."""

    text: str
    predicted_label: str
    confidence: float | None = None  # 0.0–1.0 if extractable
    raw_response: str = ""


@dataclass
class BatchClassificationReport:
    """Full batch classification output."""

    results: list[ClassificationResult] = field(default_factory=list)
    n_total: int = 0
    n_classified: int = 0
    n_failed: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ---------------------------------------------------------------------------
# Default prompt templates
# ---------------------------------------------------------------------------

_ZERO_SHOT_SYSTEM = """\
You are a precise text classification assistant for social science research.
Classify the given text into exactly one of the provided categories.
Respond with ONLY a JSON object: {{"label": "<category>", "confidence": <0.0-1.0>}}"""

_ZERO_SHOT_PROMPT = """\
Categories: {labels}

{label_definitions}

Text: "{text}"

Respond with ONLY valid JSON: {{"label": "<category>", "confidence": <0.0-1.0>}}"""

_FEW_SHOT_SYSTEM = """\
You are a precise text classification assistant for social science research.
Learn from the examples below and classify the new text into exactly one category.
Respond with ONLY a JSON object: {{"label": "<category>", "confidence": <0.0-1.0>}}"""

_FEW_SHOT_PROMPT = """\
Categories: {labels}

{label_definitions}

## Examples
{examples}

## Classify this text
Text: "{text}"

Respond with ONLY valid JSON: {{"label": "<category>", "confidence": <0.0-1.0>}}"""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class PromptClassifier:
    """Zero-shot and few-shot LLM-based text classifier.

    Usage::

        classifier = PromptClassifier(llm_client)
        # Zero-shot
        report = classifier.classify(
            texts=["I love this!", "Terrible product."],
            labels=["positive", "negative", "neutral"],
        )
        # Few-shot
        report = classifier.classify(
            texts=["New text to classify"],
            labels=["positive", "negative", "neutral"],
            examples={"positive": ["Great!"], "negative": ["Awful."], "neutral": ["OK."]},
        )
    """

    def __init__(
        self,
        llm_client: LLMClient,
        custom_prompt: str | None = None,
        custom_system: str | None = None,
    ):
        """
        Parameters
        ----------
        llm_client : LLMClient
            The LLM backend to use.
        custom_prompt : str or None
            Custom prompt template. Must contain ``{text}`` and ``{labels}`` placeholders.
            If None, uses built-in zero/few-shot templates.
        custom_system : str or None
            Custom system prompt. If None, uses built-in defaults.
        """
        self.llm_client = llm_client
        self.custom_prompt = custom_prompt
        self.custom_system = custom_system

    # ------------------------------------------------------------------
    # Synchronous classification
    # ------------------------------------------------------------------

    def classify(
        self,
        texts: list[str],
        labels: list[str],
        label_definitions: dict[str, str] | None = None,
        examples: dict[str, list[str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 100,
    ) -> BatchClassificationReport:
        """Classify texts synchronously.

        Parameters
        ----------
        texts : list[str]
            Texts to classify.
        labels : list[str]
            Valid category names.
        label_definitions : dict or None
            ``{label: definition}`` for each category.
        examples : dict or None
            ``{label: [example_texts]}``. If provided, uses few-shot mode.
        temperature : float
        max_tokens : int

        Returns
        -------
        BatchClassificationReport
        """
        results: list[ClassificationResult] = []
        label_set = set(labels)

        for text in texts:
            prompt, system = self._build_prompt(
                text, labels, label_definitions, examples,
            )
            try:
                resp = self.llm_client.complete(
                    prompt, system=system,
                    temperature=temperature, max_tokens=max_tokens,
                )
                parsed = self._parse_response(resp.text, label_set)
                results.append(ClassificationResult(
                    text=text,
                    predicted_label=parsed["label"],
                    confidence=parsed.get("confidence"),
                    raw_response=resp.text,
                ))
            except Exception as e:
                logger.warning("Classification failed for text: %s — %s", text[:50], e)
                results.append(ClassificationResult(
                    text=text,
                    predicted_label="UNKNOWN",
                    raw_response=str(e),
                ))

        return self._build_report(results)

    # ------------------------------------------------------------------
    # Async batch classification
    # ------------------------------------------------------------------

    async def classify_async(
        self,
        texts: list[str],
        labels: list[str],
        label_definitions: dict[str, str] | None = None,
        examples: dict[str, list[str]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 100,
        concurrency: int = 10,
    ) -> BatchClassificationReport:
        """Classify texts asynchronously with concurrency control."""
        label_set = set(labels)
        sem = asyncio.Semaphore(concurrency)
        results: list[ClassificationResult] = [None] * len(texts)  # type: ignore

        async def _classify_one(idx: int, text: str) -> None:
            async with sem:
                prompt, system = self._build_prompt(
                    text, labels, label_definitions, examples,
                )
                try:
                    prompts = [prompt]
                    resps = await self.llm_client.batch_complete(
                        prompts, system=system,
                        temperature=temperature, max_tokens=max_tokens,
                        concurrency=1, confirm_cost=False,
                    )
                    resp = resps[0]
                    parsed = self._parse_response(resp.text, label_set)
                    results[idx] = ClassificationResult(
                        text=text,
                        predicted_label=parsed["label"],
                        confidence=parsed.get("confidence"),
                        raw_response=resp.text,
                    )
                except Exception as e:
                    logger.warning("Async classification failed: %s", e)
                    results[idx] = ClassificationResult(
                        text=text,
                        predicted_label="UNKNOWN",
                        raw_response=str(e),
                    )

        await asyncio.gather(*[_classify_one(i, t) for i, t in enumerate(texts)])
        return self._build_report(results)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def to_dataframe(report: BatchClassificationReport) -> pd.DataFrame:
        """Convert a BatchClassificationReport to a DataFrame."""
        rows = []
        for r in report.results:
            rows.append({
                "text": r.text,
                "predicted_label": r.predicted_label,
                "confidence": r.confidence,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        text: str,
        labels: list[str],
        label_definitions: dict[str, str] | None,
        examples: dict[str, list[str]] | None,
    ) -> tuple[str, str]:
        """Build the prompt and system message for one text."""
        labels_str = ", ".join(labels)

        # Format label definitions
        if label_definitions:
            defs_str = "\n".join(f"- {k}: {v}" for k, v in label_definitions.items())
        else:
            defs_str = ""

        # Custom prompt path — use .replace() to avoid crashes on literal
        # JSON braces like {"label": ...} that .format() would choke on.
        if self.custom_prompt:
            prompt = self.custom_prompt
            prompt = prompt.replace("{text}", text)
            prompt = prompt.replace("{labels}", labels_str)
            prompt = prompt.replace("{label_definitions}", defs_str)
            prompt = prompt.replace("{examples}", "")
            system = self.custom_system or _ZERO_SHOT_SYSTEM
            return prompt, system

        # Few-shot mode
        if examples:
            examples_str = self._format_examples(examples)
            prompt = _FEW_SHOT_PROMPT.format(
                labels=labels_str,
                label_definitions=defs_str,
                examples=examples_str,
                text=text,
            )
            system = self.custom_system or _FEW_SHOT_SYSTEM
            return prompt, system

        # Zero-shot mode
        prompt = _ZERO_SHOT_PROMPT.format(
            labels=labels_str,
            label_definitions=defs_str,
            text=text,
        )
        system = self.custom_system or _ZERO_SHOT_SYSTEM
        return prompt, system

    @staticmethod
    def _format_examples(examples: dict[str, list[str]]) -> str:
        lines = []
        for label, texts in examples.items():
            for t in texts:
                lines.append(f'Text: "{t}" → Label: {label}')
        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str, valid_labels: set[str]) -> dict:
        """Parse the LLM response to extract label and confidence.

        Tries JSON parsing first, then falls back to regex / direct matching.
        """
        raw_stripped = raw.strip()

        # 1. Try JSON
        try:
            data = json.loads(raw_stripped)
            if isinstance(data, dict) and "label" in data:
                label = str(data["label"]).strip()
                confidence = data.get("confidence")
                if confidence is not None:
                    confidence = float(confidence)
                return {"label": label, "confidence": confidence}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # 2. Try regex for JSON-like pattern in mixed output
        json_match = re.search(r'\{[^}]*"label"\s*:\s*"([^"]+)"[^}]*\}', raw_stripped)
        if json_match:
            label = json_match.group(1).strip()
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_stripped)
            confidence = float(conf_match.group(1)) if conf_match else None
            return {"label": label, "confidence": confidence}

        # 3. Direct label matching — check if the response IS a valid label
        for label in valid_labels:
            if raw_stripped.lower() == label.lower():
                return {"label": label, "confidence": None}

        # 4. Check if any valid label appears in the response
        for label in valid_labels:
            if label.lower() in raw_stripped.lower():
                return {"label": label, "confidence": None}

        # 5. Fallback
        return {"label": raw_stripped[:50], "confidence": None}

    @staticmethod
    def _build_report(results: list[ClassificationResult]) -> BatchClassificationReport:
        """Aggregate individual results into a report."""
        n_total = len(results)
        n_failed = sum(1 for r in results if r.predicted_label == "UNKNOWN")
        n_classified = n_total - n_failed

        label_dist: dict[str, int] = {}
        for r in results:
            if r.predicted_label != "UNKNOWN":
                label_dist[r.predicted_label] = label_dist.get(r.predicted_label, 0) + 1

        return BatchClassificationReport(
            results=results,
            n_total=n_total,
            n_classified=n_classified,
            n_failed=n_failed,
            label_distribution=label_dist,
        )
