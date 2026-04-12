"""Unified LLM client — supports OpenAI, Anthropic, and Ollama backends.

Features:
- Unified ``complete()`` and async ``batch_complete()`` interface
- Retry with exponential backoff (max 3 attempts)
- Rate-limit handling (auto-wait on 429)
- Cost estimation before calls; per-call token/cost logging
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

try:
    import tiktoken

    _enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    tiktoken = None
    _enc = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost table (USD per 1M tokens) — updated as of 2025-05
# ---------------------------------------------------------------------------

_COST_TABLE: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (0.80, 4.00),
    "claude-opus-4-20250514": (15.00, 75.00),
}

# Ollama / unknown models default to zero cost
_DEFAULT_COST = (0.0, 0.0)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Single LLM call result."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0
    latency_s: float = 0.0


@dataclass
class CallLog:
    """Accumulated usage across multiple calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    n_calls: int = 0
    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, resp: LLMResponse) -> None:
        self.total_input_tokens += resp.input_tokens
        self.total_output_tokens += resp.output_tokens
        self.total_cost_usd += resp.cost_usd
        self.n_calls += 1
        self.entries.append({
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": resp.cost_usd,
            "latency_s": resp.latency_s,
        })


@dataclass
class CostEstimate:
    """Pre-call cost estimate."""

    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost_usd: float
    model: str


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified interface for OpenAI, Anthropic, and Ollama.

    Parameters
    ----------
    backend : str
        ``"openai"``, ``"anthropic"``, or ``"ollama"``.
    model : str
        Model identifier, e.g. ``"gpt-4o"``, ``"claude-sonnet-4-20250514"``, ``"llama3"``.
    api_key : str or None
        API key. Not needed for Ollama.
    base_url : str or None
        Override base URL (useful for Ollama or proxies).
    max_retries : int
        Maximum retry attempts on transient errors.
    cost_confirm_threshold : float
        If estimated cost exceeds this (USD), ``batch_complete`` will raise
        ``CostThresholdExceeded`` so the caller can ask for user confirmation.
    """

    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        cost_confirm_threshold: float = 5.0,
    ):
        self.backend = backend.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.cost_confirm_threshold = cost_confirm_threshold
        self.call_log = CallLog()

        self._client: Any = None
        self._async_client: Any = None

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def estimate_cost(
        self,
        prompts: list[str],
        system: str = "",
        max_tokens: int = 512,
    ) -> CostEstimate:
        """Estimate token count and cost *before* making API calls."""
        input_tokens = 0
        for p in prompts:
            input_tokens += _count_tokens(system + "\n" + p)
        output_tokens = max_tokens * len(prompts)

        in_rate, out_rate = _COST_TABLE.get(self.model, _DEFAULT_COST)
        cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
        return CostEstimate(
            estimated_input_tokens=input_tokens,
            estimated_output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 4),
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Synchronous single call
    # ------------------------------------------------------------------

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Make a single synchronous LLM call with retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = self._call(prompt, system, temperature, max_tokens)
                resp.latency_s = round(time.monotonic() - t0, 2)
                resp.cost_usd = self._calc_cost(resp.input_tokens, resp.output_tokens)
                self.call_log.record(resp)
                return resp
            except RateLimitError as e:
                wait = _backoff(attempt)
                logger.warning("Rate limited (attempt %d/%d), waiting %.1fs", attempt, self.max_retries, wait)
                time.sleep(wait)
                if attempt == self.max_retries:
                    raise
            except TransientError as e:
                wait = _backoff(attempt)
                logger.warning("Transient error (attempt %d/%d): %s", attempt, self.max_retries, e)
                time.sleep(wait)
                if attempt == self.max_retries:
                    raise

    # ------------------------------------------------------------------
    # Async batch call
    # ------------------------------------------------------------------

    async def batch_complete(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 512,
        concurrency: int = 10,
        confirm_cost: bool = True,
    ) -> list[LLMResponse]:
        """Call the LLM for each prompt concurrently.

        Parameters
        ----------
        confirm_cost : bool
            If True and estimated cost > threshold, raises ``CostThresholdExceeded``.
        """
        if confirm_cost:
            est = self.estimate_cost(prompts, system, max_tokens)
            if est.estimated_cost_usd > self.cost_confirm_threshold:
                raise CostThresholdExceeded(est)

        sem = asyncio.Semaphore(concurrency)

        async def _one(p: str) -> LLMResponse:
            async with sem:
                return await self._acall_with_retry(p, system, temperature, max_tokens)

        results = await asyncio.gather(*[_one(p) for p in prompts])
        return list(results)

    async def _acall_with_retry(
        self, prompt: str, system: str, temperature: float, max_tokens: int
    ) -> LLMResponse:
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = await self._acall(prompt, system, temperature, max_tokens)
                resp.latency_s = round(time.monotonic() - t0, 2)
                resp.cost_usd = self._calc_cost(resp.input_tokens, resp.output_tokens)
                self.call_log.record(resp)
                return resp
            except RateLimitError:
                wait = _backoff(attempt)
                logger.warning("Rate limited (attempt %d/%d), waiting %.1fs", attempt, self.max_retries, wait)
                await asyncio.sleep(wait)
                if attempt == self.max_retries:
                    raise
            except TransientError as e:
                wait = _backoff(attempt)
                logger.warning("Transient error (attempt %d/%d): %s", attempt, self.max_retries, e)
                await asyncio.sleep(wait)
                if attempt == self.max_retries:
                    raise

    # ------------------------------------------------------------------
    # Backend dispatchers
    # ------------------------------------------------------------------

    def _call(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        if self.backend == "openai":
            return self._call_openai(prompt, system, temperature, max_tokens)
        elif self.backend == "anthropic":
            return self._call_anthropic(prompt, system, temperature, max_tokens)
        elif self.backend == "ollama":
            return self._call_ollama(prompt, system, temperature, max_tokens)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    async def _acall(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        if self.backend == "openai":
            return await self._acall_openai(prompt, system, temperature, max_tokens)
        elif self.backend == "anthropic":
            return await self._acall_anthropic(prompt, system, temperature, max_tokens)
        elif self.backend == "ollama":
            return await self._acall_ollama(prompt, system, temperature, max_tokens)
        raise ValueError(f"Unknown backend: {self.backend!r}")

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _get_async_openai_client(self):
        if self._async_client is None:
            from openai import AsyncOpenAI
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._async_client = AsyncOpenAI(**kwargs)
        return self._async_client

    def _call_openai(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        client = self._get_openai_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            _raise_mapped(e)
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )

    async def _acall_openai(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        client = self._get_async_openai_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
        except Exception as e:
            _raise_mapped(e)
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------

    def _get_anthropic_client(self):
        if self._client is None:
            from anthropic import Anthropic
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = Anthropic(**kwargs)
        return self._client

    def _get_async_anthropic_client(self):
        if self._async_client is None:
            from anthropic import AsyncAnthropic
            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._async_client = AsyncAnthropic(**kwargs)
        return self._async_client

    def _call_anthropic(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        client = self._get_anthropic_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        try:
            resp = client.messages.create(**kwargs)
        except Exception as e:
            _raise_mapped(e)
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )

    async def _acall_anthropic(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        client = self._get_async_anthropic_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        try:
            resp = await client.messages.create(**kwargs)
        except Exception as e:
            _raise_mapped(e)
        text = resp.content[0].text if resp.content else ""
        return LLMResponse(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Ollama (via OpenAI-compatible API)
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        import httpx

        base = self.base_url or "http://localhost:11434"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            r = httpx.post(
                f"{base}/v1/chat/completions",
                json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                timeout=120.0,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e)) from e
            raise TransientError(str(e)) from e
        except httpx.HTTPError as e:
            raise TransientError(str(e)) from e
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )

    async def _acall_ollama(self, prompt: str, system: str, temperature: float, max_tokens: int) -> LLMResponse:
        import httpx

        base = self.base_url or "http://localhost:11434"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            async with httpx.AsyncClient() as ac:
                r = await ac.post(
                    f"{base}/v1/chat/completions",
                    json={"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                    timeout=120.0,
                )
                r.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise RateLimitError(str(e)) from e
            raise TransientError(str(e)) from e
        except httpx.HTTPError as e:
            raise TransientError(str(e)) from e
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _calc_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_rate, out_rate = _COST_TABLE.get(self.model, _DEFAULT_COST)
        return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 6)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RateLimitError(Exception):
    """429 / rate limit from the API."""


class TransientError(Exception):
    """Retryable server error (5xx, timeout, etc.)."""


class CostThresholdExceeded(Exception):
    """Estimated cost exceeds the configured threshold."""

    def __init__(self, estimate: CostEstimate):
        self.estimate = estimate
        super().__init__(
            f"Estimated cost ${estimate.estimated_cost_usd:.2f} exceeds threshold. "
            f"Model: {estimate.model}, "
            f"~{estimate.estimated_input_tokens} input + "
            f"~{estimate.estimated_output_tokens} output tokens."
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _count_tokens(text: str) -> int:
    if _enc is not None:
        return len(_enc.encode(text))
    return len(text.split())


def _backoff(attempt: int) -> float:
    """Exponential backoff: 1s, 2s, 4s, ..."""
    return min(2 ** (attempt - 1), 30)


def _raise_mapped(exc: Exception) -> None:
    """Map SDK exceptions to our error types."""
    exc_type = type(exc).__name__
    exc_str = str(exc)

    # OpenAI
    if "RateLimitError" in exc_type or "429" in exc_str:
        raise RateLimitError(exc_str) from exc
    if "APIConnectionError" in exc_type or "APITimeoutError" in exc_type:
        raise TransientError(exc_str) from exc
    if "InternalServerError" in exc_type or "500" in exc_str or "503" in exc_str:
        raise TransientError(exc_str) from exc

    # Anthropic
    if "RateLimitError" in exc_type:
        raise RateLimitError(exc_str) from exc
    if "APIConnectionError" in exc_type or "InternalServerError" in exc_type:
        raise TransientError(exc_str) from exc

    # Not a transient error — re-raise as-is
    raise exc
