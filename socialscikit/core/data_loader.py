"""Data loading utilities — supports CSV, Excel, JSON, and plain text files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt"}


class DataLoadError(Exception):
    """Raised when a file cannot be loaded."""


def load_file(path: str | Path) -> pd.DataFrame:
    """Load a data file into a DataFrame.

    Supported formats:
    - CSV (.csv) — auto-detects delimiter and encoding
    - Excel (.xlsx, .xls)
    - JSON (.json) — list of objects or {"data": [...]}
    - JSON Lines (.jsonl) — one JSON object per line
    - Plain text (.txt) — one row per line, single "text" column

    Parameters
    ----------
    path : str or Path
        Path to the data file.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    DataLoadError
        If the file cannot be read or the format is unsupported.
    """
    path = Path(path)

    if not path.exists():
        raise DataLoadError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file format '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        if ext == ".csv":
            return _load_csv(path)
        elif ext in (".xlsx", ".xls"):
            return _load_excel(path)
        elif ext == ".json":
            return _load_json(path)
        elif ext == ".jsonl":
            return _load_jsonl(path)
        elif ext == ".txt":
            return _load_txt(path)
    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Failed to read {path.name}: {e}") from e


def _load_csv(path: Path) -> pd.DataFrame:
    """Load CSV with encoding and delimiter auto-detection."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=encoding)
            if len(df.columns) == 1 and "\t" in df.columns[0]:
                df = pd.read_csv(path, encoding=encoding, sep="\t")
            return df
        except UnicodeDecodeError:
            continue
    raise DataLoadError(
        f"Cannot detect encoding for {path.name}. "
        "Please save the file as UTF-8 and retry."
    )


def _load_excel(path: Path) -> pd.DataFrame:
    """Load first sheet of an Excel workbook."""
    return pd.read_excel(path, engine="openpyxl")


def _load_json(path: Path) -> pd.DataFrame:
    """Load JSON — expects a list of objects or {\"data\": [...]}."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        return pd.DataFrame(raw["data"])
    raise DataLoadError(
        "JSON must be a list of objects or an object with a 'data' key containing a list."
    )


def _load_jsonl(path: Path) -> pd.DataFrame:
    """Load JSON Lines — one JSON object per line."""
    return pd.read_json(path, lines=True)


def _load_txt(path: Path) -> pd.DataFrame:
    """Load plain text — one line per row, single 'text' column."""
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.strip()]
    if not lines:
        raise DataLoadError("Text file is empty.")
    return pd.DataFrame({"text": lines})


def get_template_path(mode: str) -> Path:
    """Return the path to a built-in template CSV.

    Parameters
    ----------
    mode : str
        "quantikit" or "qualikit"
    """
    templates_dir = Path(__file__).parent / "templates"
    if mode == "quantikit":
        return templates_dir / "quantikit_template.csv"
    elif mode == "qualikit":
        return templates_dir / "qualikit_template.csv"
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'quantikit' or 'qualikit'.")
