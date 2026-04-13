"""Tests for project save & restore (project_io.py)."""

import json
import time
from unittest.mock import patch

import pandas as pd
import pytest

from socialscikit.core.project_io import (
    PROJECT_VERSION,
    load_project,
    save_project,
)
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


# ======================================================================
# Helpers
# ======================================================================


def _make_position(start=0, end=100):
    return TextPosition(
        line_start=1, line_end=3,
        char_start=start, char_end=end, paragraph_index=0,
    )


def _make_segment(sid=1, text="Hello world"):
    return TextSegment(
        segment_id=sid, text=text, position=_make_position(),
    )


def _make_extraction_result(sid=1, evidence="key phrase"):
    return ExtractionResult(
        segment_id=sid, text="Some text",
        rq_label="RQ1", sub_theme="Theme A",
        confidence=0.85, reasoning="because",
        evidence_span=evidence,
        position=_make_position(),
    )


def _make_review_session():
    items = [
        ReviewedExtraction(
            result=_make_extraction_result(1, "evidence one"),
            action=ReviewAction.ACCEPTED,
        ),
        ReviewedExtraction(
            result=_make_extraction_result(2, ""),
            action=ReviewAction.PENDING,
            edited_rq_label="RQ2",
        ),
    ]
    segments = [_make_segment(1, "Text A"), _make_segment(2, "Text B")]
    rqs = [ResearchQuestion(rq_id="RQ1", description="Test RQ", sub_themes=["Theme A"])]
    return ExtractionReviewSession(
        items=items,
        original_text="Full document text here.",
        segments=segments,
        research_questions=rqs,
    )


def _make_annotation_session():
    items = [
        Annotation(idx=0, text="First text", label="pos", status=AnnotationStatus.LABELED),
        Annotation(idx=1, text="Second text", label=None, status=AnnotationStatus.PENDING),
        Annotation(idx=2, text="Third text", label=None, status=AnnotationStatus.SKIPPED),
    ]
    sess = AnnotationSession(items=items, labels=["pos", "neg"], shuffle=False)
    sess._cursor = 1
    sess._history = [0]
    return sess


# ======================================================================
# Round-trip tests
# ======================================================================


class TestDataFrameRoundTrip:
    def test_basic_dataframe(self):
        df = pd.DataFrame({"text": ["hello", "world"], "label": ["a", "b"]})
        states = {"qt_df": df}
        json_str = save_project(states)
        loaded = load_project(json_str)
        pd.testing.assert_frame_equal(loaded["qt_df"], df)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        states = {"qt_df": df}
        json_str = save_project(states)
        loaded = load_project(json_str)
        assert loaded["qt_df"].empty

    def test_none_value(self):
        states = {"qt_df": None, "qt_result_df": None}
        json_str = save_project(states)
        loaded = load_project(json_str)
        assert loaded["qt_df"] is None
        assert loaded["qt_result_df"] is None


class TestTextSegmentRoundTrip:
    def test_single_segment(self):
        seg = _make_segment(5, "Test segment text")
        states = {"ql_segments": [seg]}
        json_str = save_project(states)
        loaded = load_project(json_str)
        result = loaded["ql_segments"][0]
        assert isinstance(result, TextSegment)
        assert result.segment_id == 5
        assert result.text == "Test segment text"
        assert result.position.line_start == 1

    def test_segment_with_core(self):
        seg = TextSegment(
            segment_id=1, text="Full text",
            position=_make_position(),
            core_sentence="core", core_char_start=0, core_char_end=4,
        )
        json_str = save_project({"segs": [seg]})
        loaded = load_project(json_str)
        result = loaded["segs"][0]
        assert result.core_sentence == "core"
        assert result.core_char_start == 0


class TestResearchQuestionRoundTrip:
    def test_with_sub_themes(self):
        rq = ResearchQuestion(rq_id="RQ1", description="Desc", sub_themes=["A", "B"])
        json_str = save_project({"rqs": [rq]})
        loaded = load_project(json_str)
        result = loaded["rqs"][0]
        assert isinstance(result, ResearchQuestion)
        assert result.rq_id == "RQ1"
        assert result.sub_themes == ["A", "B"]

    def test_without_sub_themes(self):
        rq = ResearchQuestion(rq_id="RQ2", description="Desc")
        json_str = save_project({"rqs": [rq]})
        loaded = load_project(json_str)
        assert loaded["rqs"][0].sub_themes == []


class TestExtractionResultRoundTrip:
    def test_with_evidence(self):
        r = _make_extraction_result(1, "important phrase")
        json_str = save_project({"results": [r]})
        loaded = load_project(json_str)
        result = loaded["results"][0]
        assert isinstance(result, ExtractionResult)
        assert result.evidence_span == "important phrase"
        assert result.confidence == 0.85

    def test_without_evidence(self):
        r = ExtractionResult(
            segment_id=1, text="text",
            rq_label="RQ1", sub_theme="T",
            confidence=0.5,
        )
        json_str = save_project({"results": [r]})
        loaded = load_project(json_str)
        assert loaded["results"][0].evidence_span == ""

    def test_position_none(self):
        r = ExtractionResult(
            segment_id=1, text="t",
            rq_label="RQ1", sub_theme="T",
            confidence=0.5, position=None,
        )
        json_str = save_project({"results": [r]})
        loaded = load_project(json_str)
        assert loaded["results"][0].position is None


class TestExtractionReviewSessionRoundTrip:
    def test_full_session(self):
        sess = _make_review_session()
        json_str = save_project({"ql_ext_session": sess})
        loaded = load_project(json_str)
        result = loaded["ql_ext_session"]
        assert isinstance(result, ExtractionReviewSession)
        assert len(result.items) == 2
        assert result.items[0].action == ReviewAction.ACCEPTED
        assert result.items[1].edited_rq_label == "RQ2"
        assert result.original_text == "Full document text here."
        assert len(result.segments) == 2
        assert len(result.research_questions) == 1

    def test_empty_session(self):
        sess = ExtractionReviewSession()
        json_str = save_project({"s": sess})
        loaded = load_project(json_str)
        result = loaded["s"]
        assert isinstance(result, ExtractionReviewSession)
        assert len(result.items) == 0


class TestAnnotationSessionRoundTrip:
    def test_full_session(self):
        sess = _make_annotation_session()
        json_str = save_project({"qt_ann_session": sess})
        loaded = load_project(json_str)
        result = loaded["qt_ann_session"]
        assert isinstance(result, AnnotationSession)
        assert len(result._items) == 3
        assert result.labels == ["pos", "neg"]
        assert result._cursor == 1
        assert result._history == [0]
        assert result._items[0].label == "pos"
        assert result._items[0].status == AnnotationStatus.LABELED
        assert result._items[2].status == AnnotationStatus.SKIPPED

    def test_elapsed_time_preserved(self):
        sess = _make_annotation_session()
        json_str = save_project({"s": sess})
        loaded = load_project(json_str)
        result = loaded["s"]
        # Elapsed time should be close to the original
        original_elapsed = time.monotonic() - sess._start_time
        restored_elapsed = time.monotonic() - result._start_time
        assert abs(original_elapsed - restored_elapsed) < 2.0  # within 2 seconds


class TestFullProjectRoundTrip:
    def test_complete_project(self):
        """Round-trip a full project with all state types."""
        df = pd.DataFrame({"text": ["hello"], "label": ["a"]})
        states = {
            "qt_df": df,
            "qt_result_df": None,
            "qt_ann_session": _make_annotation_session(),
            "ql_raw_text": "Document text",
            "ql_segments": [_make_segment(1), _make_segment(2)],
            "ql_rqs": [ResearchQuestion("RQ1", "Desc", ["A"])],
            "ql_ext_session": _make_review_session(),
            "ql_lang": "zh",
        }
        json_str = save_project(states)
        loaded = load_project(json_str)

        pd.testing.assert_frame_equal(loaded["qt_df"], df)
        assert loaded["qt_result_df"] is None
        assert isinstance(loaded["qt_ann_session"], AnnotationSession)
        assert loaded["ql_raw_text"] == "Document text"
        assert len(loaded["ql_segments"]) == 2
        assert isinstance(loaded["ql_segments"][0], TextSegment)
        assert isinstance(loaded["ql_ext_session"], ExtractionReviewSession)
        assert loaded["ql_lang"] == "zh"


class TestVersionAndMetadata:
    def test_version_embedded(self):
        json_str = save_project({"x": 1})
        data = json.loads(json_str)
        assert data["__project_version__"] == PROJECT_VERSION
        assert data["__toolkit__"] == "SocialSciKit"

    def test_unknown_keys_ignored(self):
        """Future project files may have extra keys — they should not crash."""
        data = {
            "__project_version__": "99.0",
            "__toolkit__": "SocialSciKit",
            "ql_lang": "en",
            "future_key": {"__type__": "UnknownType", "data": 42},
        }
        loaded = load_project(json.dumps(data))
        assert loaded["ql_lang"] == "en"
        # Unknown type is kept as a dict
        assert isinstance(loaded["future_key"], dict)


class TestErrorHandling:
    def test_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid project file"):
            load_project("not json at all {{{")

    def test_non_object_json(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_project("[1, 2, 3]")

    def test_primitives_preserved(self):
        states = {"a": 42, "b": "hello", "c": True, "d": 3.14}
        loaded = load_project(save_project(states))
        assert loaded["a"] == 42
        assert loaded["b"] == "hello"
        assert loaded["c"] is True
        assert loaded["d"] == 3.14
