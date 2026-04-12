"""Text segmentation with position tracking for qualitative analysis.

Splits raw interview transcripts / field notes into analysable units
while preserving exact character offsets and line numbers so every
extracted segment can be traced back to its position in the original
document.

Supports three segmentation modes:
- **sentence**: split on sentence-ending punctuation (handles Chinese)
- **paragraph**: split on blank lines
- **context_window**: capture a core sentence + N surrounding sentences
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TextPosition:
    """Position of a text span in the original document."""

    line_start: int  # 1-based line number
    line_end: int  # 1-based, inclusive
    char_start: int  # 0-based character offset from document start
    char_end: int  # exclusive
    paragraph_index: int  # 0-based paragraph number


@dataclass
class TextSegment:
    """A single segmented unit of text with position tracking."""

    segment_id: int
    text: str
    position: TextPosition
    # For context-window mode: the "core" sentence within the window
    core_sentence: str | None = None
    core_char_start: int | None = None  # relative to segment text
    core_char_end: int | None = None  # relative to segment text


# ---------------------------------------------------------------------------
# Sentence boundary regex
# ---------------------------------------------------------------------------

# Chinese sentence-ending punctuation + Western equivalents.
# Handles:
#   - Chinese: 。！？  (and full-width variants)
#   - Western: . ! ?
#   - Ellipsis: ……  or  ...  (treated as sentence end)
#   - Avoids splitting on decimal numbers (3.14) or abbreviations (Dr.)
_SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？!?])"  # after Chinese/Western sentence-enders
    r'(?:\s*(?=[^。！？!?\s）\)」』"\'"])|$)'  # followed by non-punct or end
    r"|(?<=\.{3})\s+"  # after Western ellipsis
    r"|(?<=……)\s*"  # after Chinese ellipsis
)

# Simpler fallback: split on sentence-ending punctuation followed by space or newline
_SENTENCE_SPLIT_SIMPLE = re.compile(
    r"(?<=[。！？.!?])\s+(?=\S)"
)


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------


class Segmenter:
    """Split raw text into segments with full position tracking.

    Usage::

        segmenter = Segmenter()
        segments = segmenter.segment(text, mode="paragraph")
        for seg in segments:
            print(f"[{seg.segment_id}] line {seg.position.line_start}: {seg.text[:60]}")
    """

    def segment(
        self,
        text: str,
        mode: str = "paragraph",
        context_window: int = 2,
    ) -> list[TextSegment]:
        """Segment text into analysable units.

        Parameters
        ----------
        text:
            The full document text.
        mode:
            ``"sentence"`` | ``"paragraph"`` | ``"context_window"``
        context_window:
            For context_window mode: number of surrounding sentences
            on each side (e.g. 2 means ±2 sentences).

        Returns
        -------
        list[TextSegment]
            Segments with unique IDs and position info.
        """
        if not text or not text.strip():
            return []

        line_offsets = self._build_line_index(text)
        para_ranges = self._build_paragraph_index(text)

        if mode == "sentence":
            return self._segment_sentences(text, line_offsets, para_ranges)
        elif mode == "context_window":
            return self._segment_context_window(
                text, line_offsets, para_ranges, context_window,
            )
        else:  # default: paragraph
            return self._segment_paragraphs(text, line_offsets, para_ranges)

    # ------------------------------------------------------------------
    # Internal: index builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_line_index(text: str) -> list[int]:
        """Build character-offset-to-line mapping.

        Returns a list where ``offsets[i]`` is the character offset where
        line ``i`` (0-based) begins.  Use ``bisect_right`` to convert a
        character offset to a 1-based line number.
        """
        offsets = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                offsets.append(i + 1)
        return offsets

    @staticmethod
    def _build_paragraph_index(text: str) -> list[tuple[int, int]]:
        """Find (char_start, char_end) for each paragraph.

        Paragraphs are separated by one or more blank lines.
        """
        ranges: list[tuple[int, int]] = []
        for m in re.finditer(r"[^\n](?:[^\n]|\n(?!\s*\n))*", text):
            ranges.append((m.start(), m.end()))
        if not ranges and text.strip():
            ranges.append((0, len(text)))
        return ranges

    def _char_to_line(self, char_offset: int, line_offsets: list[int]) -> int:
        """Convert character offset to 1-based line number."""
        return bisect.bisect_right(line_offsets, char_offset)

    def _char_to_para(
        self, char_start: int, para_ranges: list[tuple[int, int]],
    ) -> int:
        """Find the 0-based paragraph index for a character offset."""
        for i, (ps, pe) in enumerate(para_ranges):
            if ps <= char_start < pe:
                return i
        return max(0, len(para_ranges) - 1)

    def _make_position(
        self,
        char_start: int,
        char_end: int,
        line_offsets: list[int],
        para_ranges: list[tuple[int, int]],
    ) -> TextPosition:
        return TextPosition(
            line_start=self._char_to_line(char_start, line_offsets),
            line_end=self._char_to_line(max(char_start, char_end - 1), line_offsets),
            char_start=char_start,
            char_end=char_end,
            paragraph_index=self._char_to_para(char_start, para_ranges),
        )

    # ------------------------------------------------------------------
    # Sentence segmentation
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences, returning (text, char_start, char_end).

        Uses regex splitting on sentence-ending punctuation.
        Falls back to simpler pattern if the primary produces too few splits.
        """
        # Try primary pattern first
        parts = _SENTENCE_SPLIT.split(text)
        if len(parts) < 2:
            parts = _SENTENCE_SPLIT_SIMPLE.split(text)
        if len(parts) < 2:
            # Last resort: split on newlines
            parts = text.split("\n")

        sentences: list[tuple[str, int, int]] = []
        cursor = 0
        for part in parts:
            part_stripped = part.strip()
            if not part_stripped:
                cursor += len(part)
                continue
            # Find the part's position in the original text
            idx = text.find(part_stripped, cursor)
            if idx == -1:
                idx = cursor
            sentences.append((part_stripped, idx, idx + len(part_stripped)))
            cursor = idx + len(part_stripped)

        return sentences

    def _segment_sentences(
        self,
        text: str,
        line_offsets: list[int],
        para_ranges: list[tuple[int, int]],
    ) -> list[TextSegment]:
        raw = self._split_sentences(text)
        segments: list[TextSegment] = []
        for i, (sent, cs, ce) in enumerate(raw):
            if not sent.strip():
                continue
            segments.append(TextSegment(
                segment_id=i + 1,
                text=sent,
                position=self._make_position(cs, ce, line_offsets, para_ranges),
            ))
        return segments

    # ------------------------------------------------------------------
    # Paragraph segmentation
    # ------------------------------------------------------------------

    def _segment_paragraphs(
        self,
        text: str,
        line_offsets: list[int],
        para_ranges: list[tuple[int, int]],
    ) -> list[TextSegment]:
        segments: list[TextSegment] = []
        for i, (cs, ce) in enumerate(para_ranges):
            para_text = text[cs:ce].strip()
            if not para_text:
                continue
            segments.append(TextSegment(
                segment_id=i + 1,
                text=para_text,
                position=self._make_position(cs, ce, line_offsets, para_ranges),
            ))
        return segments

    # ------------------------------------------------------------------
    # Context window segmentation
    # ------------------------------------------------------------------

    def _segment_context_window(
        self,
        text: str,
        line_offsets: list[int],
        para_ranges: list[tuple[int, int]],
        n: int,
    ) -> list[TextSegment]:
        """For each sentence, capture ±N surrounding sentences as context."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        segments: list[TextSegment] = []
        seen_ranges: set[tuple[int, int]] = set()

        for i, (core_text, core_cs, core_ce) in enumerate(sentences):
            if not core_text.strip():
                continue

            # Window boundaries
            win_start = max(0, i - n)
            win_end = min(len(sentences) - 1, i + n)

            window_cs = sentences[win_start][1]  # char_start of first in window
            window_ce = sentences[win_end][2]  # char_end of last in window

            # Deduplicate identical windows
            key = (window_cs, window_ce)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)

            window_text = text[window_cs:window_ce].strip()
            if not window_text:
                continue

            # Core sentence position relative to window text
            core_rel_start = core_cs - window_cs
            core_rel_end = core_ce - window_cs

            segments.append(TextSegment(
                segment_id=len(segments) + 1,
                text=window_text,
                position=self._make_position(
                    window_cs, window_ce, line_offsets, para_ranges,
                ),
                core_sentence=core_text,
                core_char_start=max(0, core_rel_start),
                core_char_end=min(len(window_text), core_rel_end),
            ))

        return segments
