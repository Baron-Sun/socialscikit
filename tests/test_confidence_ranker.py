"""Tests for socialscikit.qualikit.confidence_ranker."""

from __future__ import annotations

import pytest

from socialscikit.qualikit.coder import CodingResult
from socialscikit.qualikit.confidence_ranker import ConfidenceRanker, RankedResults


@pytest.fixture()
def ranker():
    return ConfidenceRanker()


@pytest.fixture()
def sample_results():
    return [
        CodingResult(text_id=0, text="a", themes=["A"], confidences={"A": 0.95}),
        CodingResult(text_id=1, text="b", themes=["B"], confidences={"B": 0.90}),
        CodingResult(text_id=2, text="c", themes=["A"], confidences={"A": 0.75}),
        CodingResult(text_id=3, text="d", themes=["B"], confidences={"B": 0.65}),
        CodingResult(text_id=4, text="e", themes=["A"], confidences={"A": 0.40}),
        CodingResult(text_id=5, text="f", themes=[], confidences={}),
    ]


class TestRank:
    def test_basic_ranking(self, ranker, sample_results):
        ranked = ranker.rank(sample_results)
        assert len(ranked.high) == 2   # 0.95, 0.90
        assert len(ranked.medium) == 2  # 0.75, 0.65
        assert len(ranked.low) == 2     # 0.40, empty

    def test_total(self, ranker, sample_results):
        ranked = ranker.rank(sample_results)
        assert ranked.total == 6

    def test_empty_input(self, ranker):
        ranked = ranker.rank([])
        assert ranked.total == 0

    def test_all_high(self, ranker):
        results = [
            CodingResult(text_id=i, text="t", themes=["A"], confidences={"A": 0.95})
            for i in range(3)
        ]
        ranked = ranker.rank(results)
        assert len(ranked.high) == 3
        assert len(ranked.medium) == 0
        assert len(ranked.low) == 0

    def test_custom_thresholds(self):
        ranker = ConfidenceRanker(high_threshold=0.9, medium_threshold=0.7)
        results = [
            CodingResult(text_id=0, text="a", themes=["A"], confidences={"A": 0.85}),
        ]
        ranked = ranker.rank(results)
        assert len(ranked.medium) == 1  # 0.85 is below 0.9 but above 0.7


class TestSummary:
    def test_summary_format(self, ranker, sample_results):
        ranked = ranker.rank(sample_results)
        summary = ranker.summary(ranked)
        assert "高置信度" in summary
        assert "中置信度" in summary
        assert "低置信度" in summary
        assert "总计" in summary
