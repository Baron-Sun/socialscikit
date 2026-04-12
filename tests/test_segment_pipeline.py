"""Tests for QualiKit Phase 1: segmenter, segment_extractor, extraction_reviewer."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from socialscikit.qualikit.segmenter import Segmenter, TextPosition, TextSegment
from socialscikit.qualikit.segment_extractor import (
    ExtractionReport,
    ExtractionResult,
    ResearchQuestion,
    SegmentExtractor,
)
from socialscikit.qualikit.extraction_reviewer import (
    ExtractionReviewer,
    ExtractionReviewSession,
    ReviewAction,
    ReviewedExtraction,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_TEXT = """\
第一段：这是关于社区服务的讨论。居民对医疗服务比较满意。

第二段：经济方面压力较大。工厂减了工资，收入不稳定。

第三段：邻里关系还不错，但新老居民之间交流不多。"""

SAMPLE_TEXT_SENTENCES = "社区医疗服务很方便。但是药品种类有限。希望能增加更多服务项目。经济压力比较大。"


@pytest.fixture()
def segmenter():
    return Segmenter()


@pytest.fixture()
def sample_rqs():
    return [
        ResearchQuestion(rq_id="RQ1", description="社区公共服务体验"),
        ResearchQuestion(rq_id="RQ2", description="经济与就业状况"),
    ]


@pytest.fixture()
def sample_segments(segmenter):
    return segmenter.segment(SAMPLE_TEXT, mode="paragraph")


@pytest.fixture()
def sample_report(sample_segments, sample_rqs):
    """Build a mock extraction report."""
    results = [
        ExtractionResult(
            segment_id=1,
            text=sample_segments[0].text,
            rq_label="RQ1",
            sub_theme="医疗满意度",
            confidence=0.90,
            reasoning="提到社区医疗服务",
            position=sample_segments[0].position,
        ),
        ExtractionResult(
            segment_id=2,
            text=sample_segments[1].text,
            rq_label="RQ2",
            sub_theme="收入下降",
            confidence=0.75,
            reasoning="提到工厂减工资",
            position=sample_segments[1].position,
        ),
        ExtractionResult(
            segment_id=3,
            text=sample_segments[2].text,
            rq_label="RQ1",
            sub_theme="邻里交流",
            confidence=0.60,
            reasoning="提到新老居民交流",
            position=sample_segments[2].position,
        ),
    ]
    return ExtractionReport(
        results=results,
        n_segments_total=3,
        n_relevant=3,
        rq_distribution={"RQ1": 2, "RQ2": 1},
    )


# ===========================================================================
# Test: Segmenter
# ===========================================================================


class TestSegmenterParagraph:
    def test_basic_paragraph_split(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT, mode="paragraph")
        assert len(segments) == 3

    def test_segment_ids_sequential(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT, mode="paragraph")
        ids = [s.segment_id for s in segments]
        assert ids == [1, 2, 3]

    def test_position_char_offsets(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT, mode="paragraph")
        for seg in segments:
            # The text at the recorded offset should match the segment text
            extracted = SAMPLE_TEXT[seg.position.char_start:seg.position.char_end].strip()
            assert seg.text == extracted

    def test_position_line_numbers(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT, mode="paragraph")
        # First paragraph starts at line 1
        assert segments[0].position.line_start == 1
        # Second paragraph starts after a blank line
        assert segments[1].position.line_start > segments[0].position.line_end

    def test_paragraph_index(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT, mode="paragraph")
        for i, seg in enumerate(segments):
            assert seg.position.paragraph_index == i

    def test_empty_text(self, segmenter):
        assert segmenter.segment("", mode="paragraph") == []
        assert segmenter.segment("   ", mode="paragraph") == []


class TestSegmenterSentence:
    def test_chinese_sentence_split(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT_SENTENCES, mode="sentence")
        assert len(segments) >= 3  # at least 3 sentences

    def test_sentence_offsets_valid(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT_SENTENCES, mode="sentence")
        for seg in segments:
            extracted = SAMPLE_TEXT_SENTENCES[
                seg.position.char_start:seg.position.char_end
            ]
            assert seg.text in extracted or extracted.strip() == seg.text

    def test_multiline_text_sentences(self, segmenter):
        text = "第一句话。第二句话。\n\n第三句话。第四句话。"
        segments = segmenter.segment(text, mode="sentence")
        assert len(segments) >= 3


class TestSegmenterContextWindow:
    def test_context_window_basic(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT_SENTENCES, mode="context_window", context_window=1)
        assert len(segments) > 0

    def test_context_window_has_core_sentence(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT_SENTENCES, mode="context_window", context_window=1)
        for seg in segments:
            assert seg.core_sentence is not None
            assert seg.core_char_start is not None
            assert seg.core_char_end is not None

    def test_context_window_core_in_text(self, segmenter):
        segments = segmenter.segment(SAMPLE_TEXT_SENTENCES, mode="context_window", context_window=1)
        for seg in segments:
            # Core sentence should appear in the segment text
            assert seg.core_sentence in seg.text

    def test_context_window_larger_than_text(self, segmenter):
        """Window larger than available sentences should still work."""
        segments = segmenter.segment("只有一句话。", mode="context_window", context_window=5)
        assert len(segments) == 1


class TestSegmenterPositionAccuracy:
    """Critical: verify position tracking never drifts."""

    def test_roundtrip_paragraph(self, segmenter):
        """Every paragraph segment must roundtrip through char offsets."""
        text = "段落一的内容。\n\n段落二的内容。\n\n段落三的内容。"
        segments = segmenter.segment(text, mode="paragraph")
        for seg in segments:
            assert text[seg.position.char_start:seg.position.char_end].strip() == seg.text

    def test_roundtrip_with_interview(self, segmenter):
        """Test with realistic interview text."""
        text = (
            "研究员：您觉得服务怎么样？\n\n"
            "受访者：还可以，但有改进空间。医疗服务比较方便，但药品种类有限。\n\n"
            "研究员：有什么建议吗？\n\n"
            "受访者：希望能增加心理咨询服务。\n"
        )
        segments = segmenter.segment(text, mode="paragraph")
        for seg in segments:
            extracted = text[seg.position.char_start:seg.position.char_end].strip()
            assert seg.text == extracted


# ===========================================================================
# Test: SegmentExtractor
# ===========================================================================


class TestSegmentExtractor:
    def _make_mock_llm(self, response_json):
        """Create a mock LLMClient that returns a fixed JSON response."""
        mock = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(response_json)
        mock.complete.return_value = mock_resp
        return mock

    def test_extract_finds_relevant(self, sample_segments, sample_rqs):
        """Single-RQ backward compat: rq_id inferred when missing."""
        llm = self._make_mock_llm({
            "relevant_segments": [
                {"segment_id": 1, "sub_theme": "医疗服务", "confidence": 0.9, "reasoning": "test"},
            ]
        })
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, [sample_rqs[0]])
        assert report.n_relevant == 1
        assert report.results[0].rq_label == "RQ1"
        assert report.results[0].sub_theme == "医疗服务"

    def test_extract_multiple_rqs_one_pass(self, sample_segments, sample_rqs):
        """All RQs processed in one LLM call per batch."""
        call_count = 0

        def mock_complete(prompt, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.text = json.dumps({"matches": [
                {"segment_id": 1, "rq_id": "RQ1", "sub_theme": "服务", "confidence": 0.8, "reasoning": ""},
                {"segment_id": 2, "rq_id": "RQ2", "sub_theme": "收入", "confidence": 0.7, "reasoning": ""},
            ]})
            return resp

        llm = MagicMock()
        llm.complete.side_effect = mock_complete

        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, sample_rqs)
        # All RQs in one call per batch (3 segments fit in one batch)
        assert call_count == 1
        assert report.rq_distribution["RQ1"] == 1
        assert report.rq_distribution["RQ2"] == 1

    def test_extract_multi_label(self, sample_segments, sample_rqs):
        """A segment can match multiple RQs."""
        llm = self._make_mock_llm({
            "matches": [
                {"segment_id": 1, "rq_id": "RQ1", "sub_theme": "医疗", "confidence": 0.9, "reasoning": ""},
                {"segment_id": 1, "rq_id": "RQ2", "sub_theme": "经济", "confidence": 0.7, "reasoning": ""},
            ]
        })
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, sample_rqs)
        assert report.n_relevant == 2
        seg1_rqs = {r.rq_label for r in report.results if r.segment_id == 1}
        assert seg1_rqs == {"RQ1", "RQ2"}

    def test_extract_validates_sub_themes(self, sample_segments):
        """When sub-themes are predefined, only those are accepted."""
        rq = ResearchQuestion(
            rq_id="RQ1", description="test",
            sub_themes=["医疗服务", "教育设施"],
        )
        llm = self._make_mock_llm({
            "matches": [
                {"segment_id": 1, "rq_id": "RQ1", "sub_theme": "医疗服务", "confidence": 0.9, "reasoning": ""},
                {"segment_id": 2, "rq_id": "RQ1", "sub_theme": "无效主题", "confidence": 0.8, "reasoning": ""},
            ]
        })
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, [rq])
        # Only the valid sub-theme should be kept
        assert report.n_relevant == 1
        assert report.results[0].sub_theme == "医疗服务"

    def test_extract_handles_llm_error(self, sample_segments, sample_rqs):
        llm = MagicMock()
        llm.complete.side_effect = Exception("API error")
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, [sample_rqs[0]])
        assert report.n_relevant == 0

    def test_extract_handles_bad_json(self, sample_segments, sample_rqs):
        llm = self._make_mock_llm({"wrong_key": []})
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, [sample_rqs[0]])
        assert report.n_relevant == 0

    def test_extract_clamps_confidence(self, sample_segments, sample_rqs):
        llm = self._make_mock_llm({
            "relevant_segments": [
                {"segment_id": 1, "sub_theme": "test", "confidence": 1.5, "reasoning": ""},
            ]
        })
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, [sample_rqs[0]])
        assert report.results[0].confidence == 1.0

    def test_extract_attaches_position(self, sample_segments, sample_rqs):
        llm = self._make_mock_llm({
            "relevant_segments": [
                {"segment_id": 1, "sub_theme": "test", "confidence": 0.8, "reasoning": ""},
            ]
        })
        extractor = SegmentExtractor(llm)
        report = extractor.extract(sample_segments, [sample_rqs[0]])
        assert report.results[0].position is not None
        assert report.results[0].position.line_start >= 1


# ===========================================================================
# Test: ExtractionReviewer
# ===========================================================================


class TestExtractionReviewer:
    @pytest.fixture()
    def reviewer(self):
        return ExtractionReviewer()

    @pytest.fixture()
    def session(self, reviewer, sample_report, sample_segments, sample_rqs):
        return reviewer.create_session(
            sample_report, SAMPLE_TEXT, sample_segments, sample_rqs,
        )

    def test_create_session(self, session):
        assert len(session.items) == 3
        assert session.original_text == SAMPLE_TEXT

    def test_accept(self, reviewer, session):
        reviewer.accept(session, 0)
        assert session.items[0].action == ReviewAction.ACCEPTED

    def test_reject(self, reviewer, session):
        reviewer.reject(session, 0)
        assert session.items[0].action == ReviewAction.REJECTED

    def test_edit(self, reviewer, session):
        reviewer.edit(session, 0, new_sub_theme="新标签")
        assert session.items[0].action == ReviewAction.EDITED
        assert session.items[0].final_sub_theme == "新标签"
        # Original RQ label preserved if not edited
        assert session.items[0].final_rq_label == session.items[0].result.rq_label

    def test_edit_rq_label(self, reviewer, session):
        reviewer.edit(session, 0, new_rq_label="RQ2")
        assert session.items[0].final_rq_label == "RQ2"

    def test_accept_all_high(self, reviewer, session):
        count = reviewer.accept_all_high(session, threshold=0.85)
        assert count == 1  # only the 0.90 confidence one
        assert session.items[0].action == ReviewAction.ACCEPTED

    def test_add_manual(self, reviewer, session, sample_segments):
        result = reviewer.add_manual(session, segment_id=3, rq_label="RQ2", sub_theme="手动主题")
        assert result is not None
        assert result.action == ReviewAction.ACCEPTED
        assert result.result.confidence == 1.0
        assert len(session.items) == 4  # 3 original + 1 manual

    def test_add_manual_invalid_id(self, reviewer, session):
        result = reviewer.add_manual(session, segment_id=999, rq_label="RQ1", sub_theme="x")
        assert result is None

    def test_stats(self, reviewer, session):
        reviewer.accept(session, 0)
        reviewer.reject(session, 1)
        stats = reviewer.stats(session)
        assert stats["accepted"] == 1
        assert stats["rejected"] == 1
        assert stats["pending"] == 1
        assert stats["total"] == 3

    def test_export_excludes_rejected(self, reviewer, session):
        reviewer.accept(session, 0)
        reviewer.reject(session, 1)
        df = reviewer.export_to_dataframe(session)
        assert len(df) == 2  # 1 accepted + 1 pending (not rejected)
        assert "segment_id" in df.columns
        assert "text" in df.columns
        assert "rq_label" in df.columns
        assert "line_start" in df.columns
        assert "char_start" in df.columns

    def test_export_uses_edited_labels(self, reviewer, session):
        reviewer.edit(session, 0, new_sub_theme="编辑后主题", new_rq_label="RQ3")
        df = reviewer.export_to_dataframe(session)
        edited_row = df[df["segment_id"] == session.items[0].result.segment_id].iloc[0]
        assert edited_row["sub_theme"] == "编辑后主题"
        assert edited_row["rq_label"] == "RQ3"

    def test_export_empty_session(self, reviewer):
        empty_report = ExtractionReport()
        session = reviewer.create_session(empty_report, "", [], [])
        df = reviewer.export_to_dataframe(session)
        assert df.empty
