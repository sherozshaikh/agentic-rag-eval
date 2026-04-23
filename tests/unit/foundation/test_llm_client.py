from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from openai import APITimeoutError

from agentic_rag_eval.config import Settings
from agentic_rag_eval.llm.client import (
    COST_TABLE,
    LLMClient,
    LLMClientError,
    _BackendConfig,
    _estimate_cost,
    build_llm_client,
)
from agentic_rag_eval.schemas import LLMCallResult, LLMMessage


def _make_chat_response(
    *,
    content: str = "hello",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    """Build a minimal object that looks like an OpenAI ChatCompletion."""
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        finish_reason=finish_reason,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_cfg(**overrides: Any) -> _BackendConfig:
    base = dict(
        backend="api",
        model="openai/gpt-4o-mini",
        api_base="https://api.example.com/v1",
        api_key="sk-test",
        timeout=5.0,
        max_retries=2,
        retry_base=0.01,
        retry_max=0.02,
    )
    base.update(overrides)
    return _BackendConfig(**base)


def test_complete_happy_path_returns_llm_call_result() -> None:
    """A successful call must return a populated LLMCallResult."""
    fake_openai_instance = MagicMock()
    fake_openai_instance.chat.completions.create.return_value = _make_chat_response(
        content="Paris",
        prompt_tokens=12,
        completion_tokens=3,
    )

    with patch("agentic_rag_eval.llm.client.OpenAI", return_value=fake_openai_instance):
        client = LLMClient(_make_cfg())
        result = client.complete([LLMMessage(role="user", content="capital of France?")])

    assert isinstance(result, LLMCallResult)
    assert result.content == "Paris"
    assert result.model == "openai/gpt-4o-mini"
    assert result.backend == "api"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 3
    assert result.total_tokens == 15
    assert result.finish_reason == "stop"
    assert result.latency_ms >= 0.0
    assert result.cost_usd > 0.0

    fake_openai_instance.chat.completions.create.assert_called_once()
    call_kwargs = fake_openai_instance.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert call_kwargs["messages"] == [{"role": "user", "content": "capital of France?"}]


def test_complete_accepts_dict_messages() -> None:
    """Passing plain dicts for messages must also work."""
    fake_openai_instance = MagicMock()
    fake_openai_instance.chat.completions.create.return_value = _make_chat_response()

    with patch("agentic_rag_eval.llm.client.OpenAI", return_value=fake_openai_instance):
        client = LLMClient(_make_cfg())
        result = client.complete([{"role": "user", "content": "hi"}])

    assert result.content == "hello"


def test_retry_exhausts_then_raises_llm_client_error() -> None:
    """Persistent APITimeoutError must bubble up as LLMClientError after retries."""
    fake_openai_instance = MagicMock()
    fake_openai_instance.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    with patch("agentic_rag_eval.llm.client.OpenAI", return_value=fake_openai_instance):
        client = LLMClient(_make_cfg(max_retries=2))
        with pytest.raises(LLMClientError, match="LLM call failed"):
            client.complete([{"role": "user", "content": "q"}])

    assert fake_openai_instance.chat.completions.create.call_count == 3


def test_retry_succeeds_after_transient_failure() -> None:
    """A transient timeout followed by success must return a valid result."""
    fake_openai_instance = MagicMock()
    fake_openai_instance.chat.completions.create.side_effect = [
        APITimeoutError(request=MagicMock()),
        _make_chat_response(content="ok"),
    ]

    with patch("agentic_rag_eval.llm.client.OpenAI", return_value=fake_openai_instance):
        client = LLMClient(_make_cfg(max_retries=2))
        result = client.complete([{"role": "user", "content": "q"}])

    assert result.content == "ok"
    assert fake_openai_instance.chat.completions.create.call_count == 2


def test_estimate_cost_known_model() -> None:
    """Known models must compute cost from COST_TABLE."""
    prompt_rate, completion_rate = COST_TABLE["openai/gpt-4o-mini"]
    cost = _estimate_cost("openai/gpt-4o-mini", prompt_tokens=1_000_000, completion_tokens=0)
    assert cost == pytest.approx(prompt_rate)

    cost = _estimate_cost("openai/gpt-4o-mini", prompt_tokens=0, completion_tokens=1_000_000)
    assert cost == pytest.approx(completion_rate)


def test_estimate_cost_unknown_model_returns_zero() -> None:
    assert _estimate_cost("some/random-model", 1000, 1000) == 0.0


def test_local_backend_always_zero_cost() -> None:
    """When backend=='local', cost_usd must be zero regardless of the model."""
    fake_openai_instance = MagicMock()
    fake_openai_instance.chat.completions.create.return_value = _make_chat_response(
        prompt_tokens=1000,
        completion_tokens=1000,
    )

    with patch("agentic_rag_eval.llm.client.OpenAI", return_value=fake_openai_instance):
        client = LLMClient(_make_cfg(backend="local", model="openai/gpt-4o-mini"))
        result = client.complete([{"role": "user", "content": "q"}])

    assert result.cost_usd == 0.0
    assert result.backend == "local"


def test_build_llm_client_agent_vs_eval_differ() -> None:
    """`role=agent` and `role=eval` must produce distinct client configs."""
    settings = Settings(
        llm_backend="local",
        llm_model="qwen3.5:9b",
        llm_api_base="http://localhost:11434/v1",
        llm_api_key="ollama",
        eval_llm_backend="api",
        eval_llm_model="google/gemini-2.5-flash",
        eval_llm_api_base="https://openrouter.ai/api/v1",
        eval_llm_api_key="sk-eval",
    )

    with patch("agentic_rag_eval.llm.client.OpenAI") as mock_openai_cls:
        agent_client = build_llm_client(settings, role="agent")
        eval_client = build_llm_client(settings, role="eval")

    assert agent_client.model == "qwen3.5:9b"
    assert agent_client.backend == "local"
    assert eval_client.model == "google/gemini-2.5-flash"
    assert eval_client.backend == "api"

    assert agent_client.model != eval_client.model
    assert agent_client.backend != eval_client.backend

    assert mock_openai_cls.call_count == 2
