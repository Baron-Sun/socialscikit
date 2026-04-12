"""Tests for the Multi-LLM Consensus Coding module."""

import pytest
from unittest.mock import MagicMock, patch

from socialscikit.core.llm_client import LLMClient, LLMResponse
from socialscikit.qualikit.coder import Coder, CodingReport, CodingResult
from socialscikit.qualikit.consensus import ConsensusCoder, ConsensusReport, SegmentConsensus
from socialscikit.qualikit.theme_definer import Theme


@pytest.fixture
def themes():
    return [
        Theme(name="economy", description="Economic topics"),
        Theme(name="health", description="Health topics"),
        Theme(name="education", description="Education topics"),
    ]


def _make_mock_client(backend="openai", model="test-model"):
    """Create a mock LLMClient."""
    client = MagicMock(spec=LLMClient)
    client.backend = backend
    client.model = model
    client.call_log = []
    return client


def _make_coding_result(text_id, text, themes, confidences=None):
    """Create a CodingResult for testing."""
    if confidences is None:
        confidences = {t: 0.9 for t in themes}
    return CodingResult(
        text_id=text_id,
        text=text,
        themes=themes,
        confidences=confidences,
        trigger_words={t: ["word"] for t in themes},
        reasoning="test",
    )


# =====================================================================
# Initialization
# =====================================================================


class TestConsensusCoder:
    def test_requires_at_least_two_clients(self):
        with pytest.raises(ValueError, match="at least 2"):
            ConsensusCoder([_make_mock_client()])

    def test_default_threshold(self):
        clients = [_make_mock_client(), _make_mock_client()]
        cc = ConsensusCoder(clients)
        assert cc.majority_threshold == 1  # ceil(2/2) = 1

    def test_three_coder_threshold(self):
        clients = [_make_mock_client() for _ in range(3)]
        cc = ConsensusCoder(clients)
        assert cc.majority_threshold == 2  # ceil(3/2) = 2

    def test_custom_threshold(self):
        clients = [_make_mock_client(), _make_mock_client()]
        cc = ConsensusCoder(clients, majority_threshold=2)
        assert cc.majority_threshold == 2


# =====================================================================
# Majority Vote
# =====================================================================


class TestMajorityVote:
    def test_all_agree(self):
        results = [
            _make_coding_result(0, "text", ["economy", "health"]),
            _make_coding_result(0, "text", ["economy", "health"]),
            _make_coding_result(0, "text", ["economy", "health"]),
        ]
        themes, confs, votes = ConsensusCoder._majority_vote(results, threshold=2)
        assert set(themes) == {"economy", "health"}
        assert votes["economy"] == 3
        assert votes["health"] == 3

    def test_none_agree(self):
        results = [
            _make_coding_result(0, "text", ["economy"]),
            _make_coding_result(0, "text", ["health"]),
            _make_coding_result(0, "text", ["education"]),
        ]
        themes, confs, votes = ConsensusCoder._majority_vote(results, threshold=2)
        assert themes == []

    def test_two_of_three_agree(self):
        results = [
            _make_coding_result(0, "text", ["economy", "health"]),
            _make_coding_result(0, "text", ["economy"]),
            _make_coding_result(0, "text", ["education"]),
        ]
        themes, confs, votes = ConsensusCoder._majority_vote(results, threshold=2)
        assert "economy" in themes
        assert "health" not in themes  # only 1/3 voted
        assert "education" not in themes

    def test_confidence_averaging(self):
        results = [
            _make_coding_result(0, "text", ["economy"], {"economy": 0.9}),
            _make_coding_result(0, "text", ["economy"], {"economy": 0.7}),
        ]
        themes, confs, votes = ConsensusCoder._majority_vote(results, threshold=1)
        assert "economy" in themes
        assert abs(confs["economy"] - 0.8) < 0.01  # avg of 0.9 and 0.7

    def test_empty_results(self):
        results = [
            _make_coding_result(0, "text", []),
            _make_coding_result(0, "text", []),
        ]
        themes, confs, votes = ConsensusCoder._majority_vote(results, threshold=1)
        assert themes == []
        assert votes == {}

    def test_multi_theme_partial(self):
        results = [
            _make_coding_result(0, "text", ["a", "b", "c"]),
            _make_coding_result(0, "text", ["a", "b"]),
            _make_coding_result(0, "text", ["a", "d"]),
        ]
        themes, confs, votes = ConsensusCoder._majority_vote(results, threshold=2)
        assert "a" in themes  # 3/3
        assert "b" in themes  # 2/3
        assert "c" not in themes  # 1/3
        assert "d" not in themes  # 1/3


# =====================================================================
# Full Consensus Coding (with mocked Coder)
# =====================================================================


class TestConsensusCoding:
    def test_two_coders_full_agreement(self, themes):
        """Two coders agree on everything."""
        clients = [_make_mock_client(), _make_mock_client()]
        cc = ConsensusCoder(clients)

        # Mock the coders
        report = CodingReport(
            results=[
                _make_coding_result(0, "text1", ["economy"]),
                _make_coding_result(1, "text2", ["health"]),
            ],
            n_total=2, n_coded=2, n_failed=0,
            theme_distribution={"economy": 1, "health": 1},
        )

        with patch.object(Coder, 'code', return_value=report):
            result = cc.code(["text1", "text2"], themes)

        assert isinstance(result, ConsensusReport)
        assert result.n_total == 2
        assert result.n_coders == 2
        assert len(result.segments) == 2
        assert result.segments[0].consensus_themes == ["economy"]
        assert result.segments[1].consensus_themes == ["health"]

    def test_two_coders_no_agreement(self, themes):
        """Two coders disagree on everything (threshold=2 for 2 coders)."""
        clients = [_make_mock_client(), _make_mock_client()]
        cc = ConsensusCoder(clients, majority_threshold=2)

        report1 = CodingReport(
            results=[_make_coding_result(0, "text1", ["economy"])],
            n_total=1, n_coded=1, n_failed=0,
            theme_distribution={"economy": 1},
        )
        report2 = CodingReport(
            results=[_make_coding_result(0, "text1", ["health"])],
            n_total=1, n_coded=1, n_failed=0,
            theme_distribution={"health": 1},
        )

        with patch.object(Coder, 'code', side_effect=[report1, report2]):
            result = cc.code(["text1"], themes)

        assert result.segments[0].consensus_themes == []

    def test_three_coders_partial(self, themes):
        """Three coders, two agree on 'economy'."""
        clients = [_make_mock_client() for _ in range(3)]
        cc = ConsensusCoder(clients)  # threshold = ceil(3/2) = 2

        report1 = CodingReport(
            results=[_make_coding_result(0, "text1", ["economy", "health"])],
            n_total=1, n_coded=1, n_failed=0,
            theme_distribution={"economy": 1, "health": 1},
        )
        report2 = CodingReport(
            results=[_make_coding_result(0, "text1", ["economy"])],
            n_total=1, n_coded=1, n_failed=0,
            theme_distribution={"economy": 1},
        )
        report3 = CodingReport(
            results=[_make_coding_result(0, "text1", ["education"])],
            n_total=1, n_coded=1, n_failed=0,
            theme_distribution={"education": 1},
        )

        with patch.object(Coder, 'code', side_effect=[report1, report2, report3]):
            result = cc.code(["text1"], themes)

        assert "economy" in result.segments[0].consensus_themes  # 2/3
        assert "health" not in result.segments[0].consensus_themes  # 1/3
        assert "education" not in result.segments[0].consensus_themes  # 1/3


# =====================================================================
# SegmentConsensus
# =====================================================================


class TestSegmentConsensus:
    def test_to_coding_result(self):
        seg = SegmentConsensus(
            text_id=0,
            text="sample text",
            consensus_themes=["economy", "health"],
            consensus_confidences={"economy": 0.9, "health": 0.8},
            agreement_rate=0.85,
            individual_results=[
                _make_coding_result(0, "sample text", ["economy", "health"]),
                _make_coding_result(0, "sample text", ["economy"]),
            ],
            vote_counts={"economy": 2, "health": 1},
        )

        cr = seg.to_coding_result()
        assert isinstance(cr, CodingResult)
        assert cr.text_id == 0
        assert set(cr.themes) == {"economy", "health"}
        assert cr.confidences["economy"] == 0.9


# =====================================================================
# ConsensusReport
# =====================================================================


class TestConsensusReport:
    def test_to_coding_report(self):
        seg = SegmentConsensus(
            text_id=0, text="text",
            consensus_themes=["economy"],
            consensus_confidences={"economy": 0.9},
            individual_results=[],
        )
        report = ConsensusReport(
            segments=[seg], n_coders=2, n_total=1, n_coded=1,
            theme_distribution={"economy": 1},
        )
        cr = report.to_coding_report()
        assert isinstance(cr, CodingReport)
        assert len(cr.results) == 1
        assert cr.results[0].themes == ["economy"]

    def test_format_report_zh(self):
        report = ConsensusReport(
            n_coders=2, coder_models=["openai:gpt-4o", "anthropic:claude"],
            n_total=10, n_coded=10, overall_agreement=0.85,
            theme_distribution={"economy": 5, "health": 3},
            total_cost=0.05,
        )
        text = ConsensusCoder.format_report(report, lang="zh")
        assert "共识" in text
        assert "85" in text  # 0.85 formatted
        assert "economy" in text

    def test_format_report_en(self):
        report = ConsensusReport(
            n_coders=2, coder_models=["test:a", "test:b"],
            n_total=5, n_coded=5, overall_agreement=0.9,
            theme_distribution={"a": 3},
            total_cost=0.01,
        )
        text = ConsensusCoder.format_report(report, lang="en")
        assert "Consensus" in text
