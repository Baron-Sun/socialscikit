"""Tests for socialscikit.core.llm_client.

These tests use monkeypatching to avoid real API calls.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from socialscikit.core.llm_client import (
    CallLog,
    CostEstimate,
    CostThresholdExceeded,
    LLMClient,
    LLMResponse,
    RateLimitError,
    TransientError,
    _backoff,
    _count_tokens,
)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_basic(self):
        n = _count_tokens("hello world")
        assert n >= 2  # tiktoken or word-split

    def test_empty(self):
        assert _count_tokens("") == 0


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------


class TestBackoff:
    def test_values(self):
        assert _backoff(1) == 1
        assert _backoff(2) == 2
        assert _backoff(3) == 4

    def test_cap(self):
        assert _backoff(10) == 30


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


class TestCostEstimate:
    def test_known_model(self):
        client = LLMClient(backend="openai", model="gpt-4o-mini")
        est = client.estimate_cost(["hello world"], system="be helpful", max_tokens=100)
        assert isinstance(est, CostEstimate)
        assert est.estimated_input_tokens > 0
        assert est.estimated_output_tokens == 100
        assert est.estimated_cost_usd >= 0
        assert est.model == "gpt-4o-mini"

    def test_unknown_model_zero_cost(self):
        client = LLMClient(backend="ollama", model="llama3")
        est = client.estimate_cost(["test"], max_tokens=50)
        assert est.estimated_cost_usd == 0.0

    def test_multiple_prompts(self):
        client = LLMClient(backend="openai", model="gpt-4o-mini")
        est = client.estimate_cost(["a", "b", "c"], max_tokens=100)
        assert est.estimated_output_tokens == 300


# ---------------------------------------------------------------------------
# CallLog
# ---------------------------------------------------------------------------


class TestCallLog:
    def test_record(self):
        log = CallLog()
        resp = LLMResponse(text="hi", input_tokens=10, output_tokens=5, model="m", cost_usd=0.01)
        log.record(resp)
        assert log.n_calls == 1
        assert log.total_input_tokens == 10
        assert log.total_output_tokens == 5
        assert log.total_cost_usd == 0.01

    def test_multiple_records(self):
        log = CallLog()
        for _ in range(3):
            log.record(LLMResponse(text="", input_tokens=100, output_tokens=50, cost_usd=0.1))
        assert log.n_calls == 3
        assert log.total_input_tokens == 300
        assert log.total_cost_usd == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Synchronous complete — mocked OpenAI
# ---------------------------------------------------------------------------


def _make_openai_response(text: str = "mocked reply", prompt_tokens: int = 10, completion_tokens: int = 5):
    """Create a mock OpenAI ChatCompletion response."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


class TestCompleteOpenAI:
    def test_basic_call(self):
        client = LLMClient(backend="openai", model="gpt-4o-mini", api_key="test-key")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create.return_value = _make_openai_response()
        client._client = mock_openai

        resp = client.complete("hello")
        assert resp.text == "mocked reply"
        assert resp.input_tokens == 10
        assert resp.output_tokens == 5
        assert client.call_log.n_calls == 1

    def test_retry_on_rate_limit(self):
        client = LLMClient(backend="openai", model="gpt-4o-mini", api_key="k", max_retries=2)
        mock_openai = MagicMock()

        # First call raises RateLimitError-like exception, second succeeds
        exc = type("RateLimitError", (Exception,), {})("rate limited")
        mock_openai.chat.completions.create.side_effect = [exc, _make_openai_response()]
        client._client = mock_openai

        # Should succeed on retry (but our _raise_mapped checks class name)
        # We need to simulate the mapped exception
        with patch("socialscikit.core.llm_client.time.sleep"):
            resp = client.complete("hello")
        assert resp.text == "mocked reply"
        assert client.call_log.n_calls == 1


# ---------------------------------------------------------------------------
# Synchronous complete — mocked Anthropic
# ---------------------------------------------------------------------------


def _make_anthropic_response(text: str = "anthropic reply", input_tokens: int = 8, output_tokens: int = 4):
    content_block = MagicMock()
    content_block.text = text
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    resp = MagicMock()
    resp.content = [content_block]
    resp.usage = usage
    return resp


class TestCompleteAnthropic:
    def test_basic_call(self):
        client = LLMClient(backend="anthropic", model="claude-sonnet-4-20250514", api_key="test-key")
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create.return_value = _make_anthropic_response()
        client._client = mock_anthropic

        resp = client.complete("hello", system="be helpful")
        assert resp.text == "anthropic reply"
        assert resp.input_tokens == 8
        assert client.call_log.n_calls == 1


# ---------------------------------------------------------------------------
# Async batch_complete — mocked
# ---------------------------------------------------------------------------


class TestBatchComplete:
    def test_batch_basic(self):
        client = LLMClient(backend="openai", model="gpt-4o-mini", api_key="k")
        mock_async = MagicMock()
        mock_async.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )
        client._async_client = mock_async

        results = asyncio.get_event_loop().run_until_complete(
            client.batch_complete(["a", "b", "c"], confirm_cost=False)
        )
        assert len(results) == 3
        assert all(r.text == "mocked reply" for r in results)
        assert client.call_log.n_calls == 3

    def test_cost_threshold(self):
        client = LLMClient(
            backend="openai", model="gpt-4o", api_key="k",
            cost_confirm_threshold=0.001,
        )
        with pytest.raises(CostThresholdExceeded):
            asyncio.get_event_loop().run_until_complete(
                client.batch_complete(
                    ["long prompt " * 100] * 100,
                    max_tokens=1000,
                    confirm_cost=True,
                )
            )

    def test_cost_threshold_skipped(self):
        client = LLMClient(backend="openai", model="gpt-4o-mini", api_key="k")
        mock_async = MagicMock()
        mock_async.chat.completions.create = AsyncMock(
            return_value=_make_openai_response()
        )
        client._async_client = mock_async

        results = asyncio.get_event_loop().run_until_complete(
            client.batch_complete(["a"], confirm_cost=False)
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_backend(self):
        client = LLMClient(backend="unknown")
        with pytest.raises(ValueError, match="Unknown backend"):
            client.complete("hello")

    def test_cost_exceeded_message(self):
        est = CostEstimate(
            estimated_input_tokens=10000,
            estimated_output_tokens=5000,
            estimated_cost_usd=12.50,
            model="gpt-4o",
        )
        exc = CostThresholdExceeded(est)
        assert "$12.50" in str(exc)
        assert exc.estimate is est
