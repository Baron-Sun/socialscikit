"""LLM batch coding — async qualitative coding with confidence scoring.

Sends texts to the LLM in batches for thematic coding. Each text can be
assigned multiple themes (multi-label). Results include confidence scores
and trigger words for coding justification.

Confidence tiers (per dev plan):
- High (>0.85): auto-accept, spot-check
- Medium (0.6-0.85): push to human review (yellow)
- Low (<0.6): must be manually reviewed (red)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field

from socialscikit.core.llm_client import LLMClient
from socialscikit.qualikit.theme_definer import Theme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence tiers
# ---------------------------------------------------------------------------

HIGH_CONFIDENCE = 0.85
MEDIUM_CONFIDENCE = 0.60


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CodingResult:
    """Result of coding a single text."""

    text_id: int
    text: str
    themes: list[str] = field(default_factory=list)
    confidences: dict[str, float] = field(default_factory=dict)
    trigger_words: dict[str, list[str]] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def confidence_tier(self) -> str:
        """Overall confidence tier based on minimum theme confidence."""
        if not self.confidences:
            return "low"
        min_conf = min(self.confidences.values())
        if min_conf >= HIGH_CONFIDENCE:
            return "high"
        elif min_conf >= MEDIUM_CONFIDENCE:
            return "medium"
        return "low"


@dataclass
class CodingReport:
    """Summary of a batch coding run."""

    results: list[CodingResult] = field(default_factory=list)
    n_total: int = 0
    n_coded: int = 0
    n_failed: int = 0
    theme_distribution: dict[str, int] = field(default_factory=dict)

    @property
    def high_confidence_count(self) -> int:
        return sum(1 for r in self.results if r.confidence_tier == "high")

    @property
    def medium_confidence_count(self) -> int:
        return sum(1 for r in self.results if r.confidence_tier == "medium")

    @property
    def low_confidence_count(self) -> int:
        return sum(1 for r in self.results if r.confidence_tier == "low")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CODING_SYSTEM = """\
You are an expert qualitative researcher performing thematic coding.
Assign one or more themes to each text segment. For each assigned theme,
provide a confidence score (0.0-1.0) and key trigger words from the text.
Be conservative: only assign a theme if the text clearly relates to it."""

_CODING_PROMPT = """\
## Themes
{theme_definitions}

## Text to code
{text}

## Instructions
Assign relevant themes to this text. For each theme:
1. Confidence (0.0-1.0): how certain you are this theme applies
2. Trigger words: specific words/phrases from the text that support this coding
3. Brief reasoning

If NO themes apply, return an empty "themes" array.

Return ONLY valid JSON (no markdown fencing):
{{"themes": [{{"name": "theme_name", "confidence": 0.85, "trigger_words": ["word1", "word2"], "reasoning": "brief reason"}}]}}"""


# ---------------------------------------------------------------------------
# Coder
# ---------------------------------------------------------------------------


class Coder:
    """LLM-based qualitative coding engine.

    Usage::

        coder = Coder(llm_client)
        report = coder.code(
            texts=["interview segment 1", ...],
            themes=[Theme(name="...", description="..."), ...],
        )
        # or async:
        report = await coder.code_async(texts, themes, batch_size=50)
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def code(
        self,
        texts: list[str],
        themes: list[Theme],
    ) -> CodingReport:
        """Synchronous coding of all texts."""
        results: list[CodingResult] = []
        n_failed = 0
        theme_dist: dict[str, int] = {t.name: 0 for t in themes}

        theme_defs = self._format_themes(themes)

        for i, text in enumerate(texts):
            try:
                result = self._code_single(i, text, theme_defs, themes)
                results.append(result)
                for t in result.themes:
                    theme_dist[t] = theme_dist.get(t, 0) + 1
            except Exception as e:
                logger.warning("Failed to code text %d: %s", i, e)
                results.append(CodingResult(text_id=i, text=text))
                n_failed += 1

        return CodingReport(
            results=results,
            n_total=len(texts),
            n_coded=len(texts) - n_failed,
            n_failed=n_failed,
            theme_distribution=theme_dist,
        )

    async def code_async(
        self,
        texts: list[str],
        themes: list[Theme],
        batch_size: int = 50,
    ) -> CodingReport:
        """Async batch coding with concurrency control."""
        results: list[CodingResult] = [None] * len(texts)  # type: ignore
        n_failed = 0
        theme_dist: dict[str, int] = {t.name: 0 for t in themes}

        theme_defs = self._format_themes(themes)

        # Process in batches
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            prompts = []
            for text in batch_texts:
                prompt = _CODING_PROMPT.format(
                    theme_definitions=theme_defs, text=text,
                )
                prompts.append(prompt)

            try:
                responses = await self.llm_client.batch_complete(
                    prompts, system=_CODING_SYSTEM, max_tokens=1024,
                )

                for j, resp in enumerate(responses):
                    idx = batch_start + j
                    try:
                        result = self._parse_response(idx, batch_texts[j], resp.text, themes)
                        results[idx] = result
                        for t in result.themes:
                            theme_dist[t] = theme_dist.get(t, 0) + 1
                    except Exception as e:
                        logger.warning("Failed to parse coding for text %d: %s", idx, e)
                        results[idx] = CodingResult(text_id=idx, text=batch_texts[j])
                        n_failed += 1
            except Exception as e:
                logger.error("Batch coding failed: %s", e)
                for j in range(len(batch_texts)):
                    idx = batch_start + j
                    if results[idx] is None:
                        results[idx] = CodingResult(text_id=idx, text=batch_texts[j])
                        n_failed += 1

        return CodingReport(
            results=[r for r in results if r is not None],
            n_total=len(texts),
            n_coded=len(texts) - n_failed,
            n_failed=n_failed,
            theme_distribution=theme_dist,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _code_single(
        self, text_id: int, text: str, theme_defs: str, themes: list[Theme],
    ) -> CodingResult:
        """Code a single text synchronously."""
        prompt = _CODING_PROMPT.format(theme_definitions=theme_defs, text=text)
        resp = self.llm_client.complete(prompt, system=_CODING_SYSTEM, max_tokens=1024)
        return self._parse_response(text_id, text, resp.text, themes)

    def _parse_response(
        self, text_id: int, text: str, response_text: str, themes: list[Theme],
    ) -> CodingResult:
        """Parse LLM JSON response into CodingResult."""
        valid_names = {t.name for t in themes}

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
            else:
                return CodingResult(text_id=text_id, text=text)

        assigned_themes = []
        confidences = {}
        trigger_words = {}
        reasoning_parts = []

        raw_themes = parsed.get("themes", [])
        if not isinstance(raw_themes, list):
            raw_themes = []

        for item in raw_themes:
            name = item.get("name", "")
            # Fuzzy match to valid theme names
            matched = self._match_theme_name(name, valid_names)
            if not matched:
                continue

            conf = float(item.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
            triggers = item.get("trigger_words", [])
            reason = item.get("reasoning", "")

            assigned_themes.append(matched)
            confidences[matched] = round(conf, 3)
            trigger_words[matched] = triggers if isinstance(triggers, list) else []
            if reason:
                reasoning_parts.append(f"{matched}: {reason}")

        return CodingResult(
            text_id=text_id,
            text=text,
            themes=assigned_themes,
            confidences=confidences,
            trigger_words=trigger_words,
            reasoning="; ".join(reasoning_parts),
        )

    @staticmethod
    def _format_themes(themes: list[Theme]) -> str:
        """Format theme definitions for the prompt."""
        lines = []
        for t in themes:
            lines.append(f"### {t.name}")
            lines.append(f"Description: {t.description}")
            if t.inclusion_examples:
                lines.append("Include (examples):")
                for ex in t.inclusion_examples[:3]:
                    lines.append(f'  - "{ex}"')
            if t.exclusion_examples:
                lines.append("Exclude (NOT this theme):")
                for ex in t.exclusion_examples[:3]:
                    lines.append(f'  - "{ex}"')
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _match_theme_name(name: str, valid_names: set[str]) -> str | None:
        """Fuzzy match a theme name to the valid set."""
        # Exact match
        if name in valid_names:
            return name
        # Case-insensitive match
        lower_map = {n.lower(): n for n in valid_names}
        if name.lower() in lower_map:
            return lower_map[name.lower()]
        # Substring match
        for valid in valid_names:
            if name.lower() in valid.lower() or valid.lower() in name.lower():
                return valid
        return None
