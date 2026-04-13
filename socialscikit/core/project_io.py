"""Project save & restore — serialize / deserialize all session state to JSON.

Allows users to save their progress and resume later without losing any
annotation, coding, or review state.  All complex types are converted to
plain dicts with a ``__type__`` discriminator for safe, human-readable
round-tripping.

Usage::

    from socialscikit.core.project_io import save_project, load_project

    json_str = save_project({"qt_df": df, "ql_segments": segments, ...})
    states   = load_project(json_str)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import pandas as pd

from socialscikit.quantikit.annotator import (
    Annotation,
    AnnotationSession,
    AnnotationStatus,
)
from socialscikit.qualikit.extraction_reviewer import (
    ExtractionReviewSession,
    ReviewAction,
    ReviewedExtraction,
)
from socialscikit.qualikit.segment_extractor import (
    ExtractionResult,
    ResearchQuestion,
)
from socialscikit.qualikit.segmenter import TextPosition, TextSegment

logger = logging.getLogger(__name__)

PROJECT_VERSION = "1.0"


# ======================================================================
# Serializers  (Python object → JSON-safe dict)
# ======================================================================


def _ser_dataframe(df: pd.DataFrame) -> dict:
    return {
        "__type__": "DataFrame",
        "records": json.loads(df.to_json(orient="records", force_ascii=False)),
        "columns": list(df.columns),
    }


def _ser_text_position(pos: TextPosition) -> dict:
    return {
        "__type__": "TextPosition",
        "line_start": pos.line_start,
        "line_end": pos.line_end,
        "char_start": pos.char_start,
        "char_end": pos.char_end,
        "paragraph_index": pos.paragraph_index,
    }


def _ser_text_segment(seg: TextSegment) -> dict:
    return {
        "__type__": "TextSegment",
        "segment_id": seg.segment_id,
        "text": seg.text,
        "position": _ser_text_position(seg.position),
        "core_sentence": seg.core_sentence,
        "core_char_start": seg.core_char_start,
        "core_char_end": seg.core_char_end,
    }


def _ser_research_question(rq: ResearchQuestion) -> dict:
    return {
        "__type__": "ResearchQuestion",
        "rq_id": rq.rq_id,
        "description": rq.description,
        "sub_themes": list(rq.sub_themes),
    }


def _ser_extraction_result(r: ExtractionResult) -> dict:
    return {
        "__type__": "ExtractionResult",
        "segment_id": r.segment_id,
        "text": r.text,
        "rq_label": r.rq_label,
        "sub_theme": r.sub_theme,
        "confidence": r.confidence,
        "reasoning": r.reasoning,
        "evidence_span": getattr(r, "evidence_span", ""),
        "position": _ser_text_position(r.position) if r.position else None,
    }


def _ser_reviewed_extraction(item: ReviewedExtraction) -> dict:
    return {
        "__type__": "ReviewedExtraction",
        "result": _ser_extraction_result(item.result),
        "action": item.action.value,
        "edited_rq_label": item.edited_rq_label,
        "edited_sub_theme": item.edited_sub_theme,
    }


def _ser_extraction_review_session(sess: ExtractionReviewSession) -> dict:
    return {
        "__type__": "ExtractionReviewSession",
        "items": [_ser_reviewed_extraction(i) for i in sess.items],
        "original_text": sess.original_text,
        "segments": [_ser_text_segment(s) for s in sess.segments],
        "research_questions": [_ser_research_question(rq) for rq in sess.research_questions],
    }


def _ser_annotation(a: Annotation) -> dict:
    return {
        "__type__": "Annotation",
        "idx": a.idx,
        "text": a.text,
        "label": a.label,
        "status": a.status.value,
        "timestamp": a.timestamp,
        "annotator_note": a.annotator_note,
    }


def _ser_annotation_session(sess: AnnotationSession) -> dict:
    elapsed = time.monotonic() - sess._start_time
    return {
        "__type__": "AnnotationSession",
        "labels": list(sess.labels),
        "items": [_ser_annotation(a) for a in sess._items],
        "cursor": sess._cursor,
        "history": list(sess._history),
        "elapsed_seconds": round(elapsed, 1),
    }


# ======================================================================
# Deserializers  (JSON-safe dict → Python object)
# ======================================================================


def _de_dataframe(d: dict) -> pd.DataFrame:
    df = pd.DataFrame(d["records"])
    # Restore column order
    if d.get("columns"):
        cols = [c for c in d["columns"] if c in df.columns]
        df = df[cols]
    return df


def _de_text_position(d: dict) -> TextPosition:
    return TextPosition(
        line_start=d["line_start"],
        line_end=d["line_end"],
        char_start=d["char_start"],
        char_end=d["char_end"],
        paragraph_index=d.get("paragraph_index", 0),
    )


def _de_text_segment(d: dict) -> TextSegment:
    return TextSegment(
        segment_id=d["segment_id"],
        text=d["text"],
        position=_de_text_position(d["position"]),
        core_sentence=d.get("core_sentence"),
        core_char_start=d.get("core_char_start"),
        core_char_end=d.get("core_char_end"),
    )


def _de_research_question(d: dict) -> ResearchQuestion:
    return ResearchQuestion(
        rq_id=d["rq_id"],
        description=d["description"],
        sub_themes=d.get("sub_themes", []),
    )


def _de_extraction_result(d: dict) -> ExtractionResult:
    pos_data = d.get("position")
    return ExtractionResult(
        segment_id=d["segment_id"],
        text=d["text"],
        rq_label=d["rq_label"],
        sub_theme=d["sub_theme"],
        confidence=d["confidence"],
        reasoning=d.get("reasoning", ""),
        evidence_span=d.get("evidence_span", ""),
        position=_de_text_position(pos_data) if pos_data else None,
    )


def _de_reviewed_extraction(d: dict) -> ReviewedExtraction:
    return ReviewedExtraction(
        result=_de_extraction_result(d["result"]),
        action=ReviewAction(d["action"]),
        edited_rq_label=d.get("edited_rq_label"),
        edited_sub_theme=d.get("edited_sub_theme"),
    )


def _de_extraction_review_session(d: dict) -> ExtractionReviewSession:
    return ExtractionReviewSession(
        items=[_de_reviewed_extraction(i) for i in d.get("items", [])],
        original_text=d.get("original_text", ""),
        segments=[_de_text_segment(s) for s in d.get("segments", [])],
        research_questions=[_de_research_question(rq) for rq in d.get("research_questions", [])],
    )


def _de_annotation(d: dict) -> Annotation:
    return Annotation(
        idx=d["idx"],
        text=d["text"],
        label=d.get("label"),
        status=AnnotationStatus(d.get("status", "pending")),
        timestamp=d.get("timestamp"),
        annotator_note=d.get("annotator_note", ""),
    )


def _de_annotation_session(d: dict) -> AnnotationSession:
    items = [_de_annotation(a) for a in d.get("items", [])]
    labels = d.get("labels", [])
    sess = AnnotationSession(items=items, labels=labels, shuffle=False)
    sess._cursor = d.get("cursor", 0)
    sess._history = d.get("history", [])
    elapsed = d.get("elapsed_seconds", 0.0)
    sess._start_time = time.monotonic() - elapsed
    return sess


# ======================================================================
# Top-level dispatch
# ======================================================================

_TYPE_SERIALIZERS = {
    "DataFrame": _ser_dataframe,
    "TextPosition": _ser_text_position,
    "TextSegment": _ser_text_segment,
    "ResearchQuestion": _ser_research_question,
    "ExtractionResult": _ser_extraction_result,
    "ReviewedExtraction": _ser_reviewed_extraction,
    "ExtractionReviewSession": _ser_extraction_review_session,
    "Annotation": _ser_annotation,
    "AnnotationSession": _ser_annotation_session,
}

_TYPE_DESERIALIZERS = {
    "DataFrame": _de_dataframe,
    "TextPosition": _de_text_position,
    "TextSegment": _de_text_segment,
    "ResearchQuestion": _de_research_question,
    "ExtractionResult": _de_extraction_result,
    "ReviewedExtraction": _de_reviewed_extraction,
    "ExtractionReviewSession": _de_extraction_review_session,
    "Annotation": _de_annotation,
    "AnnotationSession": _de_annotation_session,
}

_TYPE_MAP = {
    pd.DataFrame: "DataFrame",
    TextPosition: "TextPosition",
    TextSegment: "TextSegment",
    ResearchQuestion: "ResearchQuestion",
    ExtractionResult: "ExtractionResult",
    ReviewedExtraction: "ReviewedExtraction",
    ExtractionReviewSession: "ExtractionReviewSession",
    Annotation: "Annotation",
    AnnotationSession: "AnnotationSession",
}


def _serialize_value(val: Any) -> Any:
    """Recursively serialize a value to JSON-safe form."""
    if val is None:
        return None
    # Check known types
    type_name = _TYPE_MAP.get(type(val))
    if type_name:
        return _TYPE_SERIALIZERS[type_name](val)
    # Lists
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    # Dicts
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    # Primitives
    if isinstance(val, (str, int, float, bool)):
        return val
    # Fallback: try str
    logger.warning("Cannot serialize type %s, converting to str", type(val).__name__)
    return str(val)


def _deserialize_value(val: Any) -> Any:
    """Recursively deserialize a JSON-safe value back to Python objects."""
    if val is None:
        return None
    if isinstance(val, dict):
        type_tag = val.get("__type__")
        if type_tag and type_tag in _TYPE_DESERIALIZERS:
            return _TYPE_DESERIALIZERS[type_tag](val)
        # Regular dict
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_deserialize_value(v) for v in val]
    return val


# ======================================================================
# Public API
# ======================================================================


def save_project(states: dict[str, Any]) -> str:
    """Serialize all session states to a JSON string.

    Parameters
    ----------
    states : dict
        Mapping of state names to their Python values.  Keys include
        ``qt_df``, ``qt_result_df``, ``qt_ann_session``, ``ql_raw_text``,
        ``ql_segments``, ``ql_rqs``, ``ql_ext_session``, ``ql_lang``.

    Returns
    -------
    str
        JSON string that can be written to a file.
    """
    payload: dict[str, Any] = {
        "__project_version__": PROJECT_VERSION,
        "__toolkit__": "SocialSciKit",
    }
    for key, val in states.items():
        try:
            payload[key] = _serialize_value(val)
        except Exception as e:
            logger.warning("Failed to serialize '%s': %s", key, e)
            payload[key] = None
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_project(json_str: str) -> dict[str, Any]:
    """Deserialize a JSON project file back into session states.

    Parameters
    ----------
    json_str : str
        The JSON string from a previously saved project file.

    Returns
    -------
    dict
        Mapping of state names to reconstructed Python objects.

    Raises
    ------
    ValueError
        If the JSON cannot be parsed or is not a valid project file.
    """
    try:
        raw = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid project file: {e}") from e

    if not isinstance(raw, dict):
        raise ValueError("Project file must be a JSON object.")

    version = raw.pop("__project_version__", "unknown")
    raw.pop("__toolkit__", None)
    logger.info("Loading project file (version %s)", version)

    states: dict[str, Any] = {}
    for key, val in raw.items():
        try:
            states[key] = _deserialize_value(val)
        except Exception as e:
            logger.warning("Failed to deserialize '%s': %s", key, e)
            states[key] = None
    return states
