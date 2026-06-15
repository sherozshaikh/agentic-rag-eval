from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

from openai import APITimeoutError, OpenAI
from openai import OpenAIError as _OpenAIError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import LLMCallResult, LLMMessage

logger = get_logger(__name__)


class LLMClientError(RuntimeError):
    """Raised when an LLM call fails after all retries are exhausted."""


COST_TABLE: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-pro": (1.25, 10.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
    "qwen/qwen-2.5-7b-instruct": (0.07, 0.07),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_TABLE.get(model)
    if rates is None:
        return 0.0
    prompt_rate, completion_rate = rates
    return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000


@dataclass
class _BackendConfig:
    backend: str
    model: str
    api_base: str
    api_key: str
    timeout: float
    max_retries: int
    retry_base: float
    retry_max: float


class LLMClient:
    """OpenAI-compatible chat-completion client."""

    def __init__(self, cfg: _BackendConfig) -> None:
        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.api_key or "unused",
            base_url=cfg.api_base,
            timeout=cfg.timeout,
            max_retries=0,
        )
        logger.info(
            "llm_client_initialized",
            extra={
                "backend": cfg.backend,
                "model": cfg.model,
                "api_base": cfg.api_base,
                "timeout": cfg.timeout,
            },
        )

    @property
    def model(self) -> str:
        return self._cfg.model

    @property
    def backend(self) -> str:
        return self._cfg.backend

    def complete(
        self,
        messages: Iterable[LLMMessage] | list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMCallResult:
        """Issue a single chat-completion call with retries and usage tracking."""
        payload: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, LLMMessage):
                payload.append({"role": m.role, "content": m.content})
            else:
                payload.append(m)

        return self._complete_with_retry(
            payload=payload,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            response_format=response_format,
        )

    def _complete_with_retry(
        self,
        *,
        payload: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
        stop: list[str] | None,
        response_format: dict[str, Any] | None,
    ) -> LLMCallResult:
        @retry(
            stop=stop_after_attempt(self._cfg.max_retries + 1),
            wait=wait_exponential(
                multiplier=self._cfg.retry_base,
                min=self._cfg.retry_base,
                max=self._cfg.retry_max,
            ),
            retry=retry_if_exception_type((APITimeoutError, _OpenAIError, ConnectionError)),
            reraise=True,
        )
        def _call() -> LLMCallResult:
            start = time.perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self._cfg.model,
                    "messages": payload,
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                if stop:
                    kwargs["stop"] = stop
                if response_format is not None:
                    kwargs["response_format"] = response_format
                if self._cfg.backend == "local":
                    kwargs["extra_body"] = {"keep_alive": -1}

                resp = self._client.chat.completions.create(**kwargs)
            except (APITimeoutError, _OpenAIError, ConnectionError) as e:
                logger.warning(
                    "llm_call_failed_retrying",
                    extra={
                        "backend": self._cfg.backend,
                        "model": self._cfg.model,
                        "error": str(e),
                    },
                )
                raise

            latency_ms = (time.perf_counter() - start) * 1000.0

            choice = resp.choices[0] if resp.choices else None
            content = (choice.message.content if choice else "") or ""
            finish_reason = choice.finish_reason if choice else None

            usage = resp.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

            cost = (
                0.0
                if self._cfg.backend == "local"
                else _estimate_cost(self._cfg.model, prompt_tokens, completion_tokens)
            )

            return LLMCallResult(
                content=content,
                model=self._cfg.model,
                backend=self._cfg.backend,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
                finish_reason=finish_reason,
            )

        try:
            return _call()
        except Exception as e:
            raise LLMClientError(
                f"LLM call failed after {self._cfg.max_retries + 1} attempts: {e}"
            ) from e


def build_llm_client(
    settings: Settings | None = None,
    *,
    role: Literal["agent", "eval"] = "agent",
) -> LLMClient:
    """Build an LLMClient for the agent or evaluator role."""
    settings = settings or get_settings()

    if role == "agent":
        cfg = _BackendConfig(
            backend=settings.llm_backend,
            model=settings.llm_model,
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_base=settings.llm_retry_base_delay,
            retry_max=settings.llm_retry_max_delay,
        )
    else:
        cfg = _BackendConfig(
            backend=settings.eval_llm_backend,
            model=settings.eval_llm_model,
            api_base=settings.eval_llm_api_base,
            api_key=settings.eval_llm_api_key.get_secret_value(),
            timeout=settings.eval_llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            retry_base=settings.llm_retry_base_delay,
            retry_max=settings.llm_retry_max_delay,
        )
        if cfg.backend == settings.llm_backend and cfg.model == settings.llm_model:
            logger.warning(
                "eval_llm_same_as_agent",
                extra={"model": cfg.model, "backend": cfg.backend},
            )

    return LLMClient(cfg)
