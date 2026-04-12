"""Built-in annotation interface — session management and data export.

Manages a labeling session: tracks progress, stores annotations, supports
undo, and exports labeled data. Designed to back the Gradio annotation UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class AnnotationStatus(str, Enum):
    PENDING = "pending"
    LABELED = "labeled"
    SKIPPED = "skipped"
    FLAGGED = "flagged"  # ambiguous / needs discussion


@dataclass
class Annotation:
    """A single annotation record."""

    idx: int  # row index in the original DataFrame
    text: str
    label: str | None = None
    status: AnnotationStatus = AnnotationStatus.PENDING
    timestamp: float | None = None
    annotator_note: str = ""


@dataclass
class AnnotationSessionStats:
    """Live session statistics."""

    total: int = 0
    labeled: int = 0
    skipped: int = 0
    flagged: int = 0
    pending: int = 0
    labels_distribution: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    @property
    def progress_pct(self) -> float:
        if self.total == 0:
            return 0.0
        return round((self.labeled + self.skipped + self.flagged) / self.total * 100, 1)


# ---------------------------------------------------------------------------
# Annotation Session
# ---------------------------------------------------------------------------


class AnnotationSession:
    """Manages a labeling session for a set of texts.

    Usage::

        session = AnnotationSession.from_dataframe(df, text_col="text", labels=["pos", "neg"])
        item = session.current()
        session.annotate(label="pos")        # label current and advance
        session.skip()                        # skip current
        session.undo()                        # go back one step
        df = session.export()                 # export labeled data
    """

    def __init__(
        self,
        items: list[Annotation],
        labels: list[str],
        shuffle: bool = False,
    ):
        self.labels = labels
        self._items = items
        self._cursor = 0
        self._history: list[int] = []  # stack of previously visited indices
        self._start_time = time.monotonic()

        if shuffle:
            import random
            random.shuffle(self._items)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        text_col: str = "text",
        labels: list[str] | None = None,
        label_col: str | None = None,
        shuffle: bool = False,
    ) -> AnnotationSession:
        """Create a session from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        text_col : str
            Column containing texts to annotate.
        labels : list[str] or None
            Valid label set. If None and label_col exists, inferred from data.
        label_col : str or None
            If provided, pre-existing labels are loaded (partial annotation resume).
        shuffle : bool
            Randomise presentation order.
        """
        if text_col not in df.columns:
            raise ValueError(f"Text column '{text_col}' not found in DataFrame.")

        # Infer labels
        if labels is None and label_col and label_col in df.columns:
            labels = sorted(df[label_col].dropna().unique().tolist())
        if labels is None:
            labels = []

        items: list[Annotation] = []
        for i, row in df.iterrows():
            text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            existing_label = None
            status = AnnotationStatus.PENDING
            if label_col and label_col in df.columns and pd.notna(row.get(label_col)):
                existing_label = str(row[label_col])
                status = AnnotationStatus.LABELED

            items.append(Annotation(
                idx=int(i),
                text=text,
                label=existing_label,
                status=status,
            ))

        session = cls(items=items, labels=labels, shuffle=shuffle)
        # Skip cursor past already-labeled items to first pending
        session._advance_to_next_pending()
        return session

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def current(self) -> Annotation | None:
        """Return the current item, or None if all done."""
        if self._cursor >= len(self._items):
            return None
        return self._items[self._cursor]

    def annotate(self, label: str, note: str = "") -> Annotation:
        """Label the current item and advance to next pending.

        Parameters
        ----------
        label : str
            The assigned label.
        note : str
            Optional annotator note.

        Returns
        -------
        Annotation
            The annotated item.
        """
        if self._cursor >= len(self._items):
            raise IndexError("No more items to annotate.")
        if self.labels and label not in self.labels:
            raise ValueError(f"Invalid label '{label}'. Valid: {self.labels}")

        item = self._items[self._cursor]
        item.label = label
        item.status = AnnotationStatus.LABELED
        item.timestamp = time.monotonic() - self._start_time
        item.annotator_note = note

        self._history.append(self._cursor)
        self._cursor += 1
        self._advance_to_next_pending()
        return item

    def skip(self) -> Annotation:
        """Skip the current item and advance."""
        if self._cursor >= len(self._items):
            raise IndexError("No more items to skip.")

        item = self._items[self._cursor]
        item.status = AnnotationStatus.SKIPPED
        item.timestamp = time.monotonic() - self._start_time

        self._history.append(self._cursor)
        self._cursor += 1
        self._advance_to_next_pending()
        return item

    def flag(self, note: str = "") -> Annotation:
        """Flag the current item as ambiguous and advance."""
        if self._cursor >= len(self._items):
            raise IndexError("No more items to flag.")

        item = self._items[self._cursor]
        item.status = AnnotationStatus.FLAGGED
        item.annotator_note = note
        item.timestamp = time.monotonic() - self._start_time

        self._history.append(self._cursor)
        self._cursor += 1
        self._advance_to_next_pending()
        return item

    def undo(self) -> Annotation | None:
        """Go back to the previous item and reset its status.

        Returns the item that was undone, or None if nothing to undo.
        """
        if not self._history:
            return None

        prev_idx = self._history.pop()
        item = self._items[prev_idx]
        item.label = None
        item.status = AnnotationStatus.PENDING
        item.timestamp = None
        item.annotator_note = ""
        self._cursor = prev_idx
        return item

    def goto(self, index: int) -> Annotation:
        """Jump to a specific item by its position in the list."""
        if index < 0 or index >= len(self._items):
            raise IndexError(f"Index {index} out of range [0, {len(self._items)}).")
        self._cursor = index
        return self._items[index]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> AnnotationSessionStats:
        """Return current session statistics."""
        counts = {s: 0 for s in AnnotationStatus}
        label_dist: dict[str, int] = {}
        for item in self._items:
            counts[item.status] += 1
            if item.label:
                label_dist[item.label] = label_dist.get(item.label, 0) + 1

        return AnnotationSessionStats(
            total=len(self._items),
            labeled=counts[AnnotationStatus.LABELED],
            skipped=counts[AnnotationStatus.SKIPPED],
            flagged=counts[AnnotationStatus.FLAGGED],
            pending=counts[AnnotationStatus.PENDING],
            labels_distribution=label_dist,
            elapsed_seconds=round(time.monotonic() - self._start_time, 1),
        )

    @property
    def is_complete(self) -> bool:
        """True if no pending items remain."""
        return all(
            item.status != AnnotationStatus.PENDING for item in self._items
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, include_all: bool = False) -> pd.DataFrame:
        """Export annotations as a DataFrame.

        Parameters
        ----------
        include_all : bool
            If True, include skipped and flagged items. Otherwise only labeled.
        """
        rows = []
        for item in self._items:
            if not include_all and item.status != AnnotationStatus.LABELED:
                continue
            rows.append({
                "idx": item.idx,
                "text": item.text,
                "label": item.label,
                "status": item.status.value,
                "annotator_note": item.annotator_note,
            })
        return pd.DataFrame(rows)

    def export_for_training(self) -> pd.DataFrame:
        """Export only labeled items as (text, label) for model training."""
        rows = []
        for item in self._items:
            if item.status == AnnotationStatus.LABELED and item.label:
                rows.append({"text": item.text, "label": item.label})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _advance_to_next_pending(self) -> None:
        """Move cursor to the next pending item (if any)."""
        while self._cursor < len(self._items):
            if self._items[self._cursor].status == AnnotationStatus.PENDING:
                break
            self._cursor += 1
