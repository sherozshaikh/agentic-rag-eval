from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.llm.client import LLMClient
from agentic_rag_eval.schemas import (
    EvalRecord,
    FailureMode,
    LLMCallResult,
    Passage,
    QueryResponse,
    QueryType,
    RetrievalStrategy,
)
from agentic_rag_eval.tracing.logger import TraceLogger

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


# ---------------------------------------------------------------------------
# Settings fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Return a `Settings` instance with test-safe defaults.

    The trace DB and memory storage paths are redirected into `tmp_path`, and
    both LLM backends are set to `local` so no network credentials are
    required. This fixture does NOT mutate environment variables — use
    :func:`mock_settings` for that.
    """
    return Settings(
        llm_backend="local",
        llm_model="qwen2.5:7b-instruct",
        llm_api_base="http://localhost:11434/v1",
        llm_api_key="ollama",
        eval_llm_backend="local",
        eval_llm_model="qwen2.5:7b-instruct",
        eval_llm_api_base="http://localhost:11434/v1",
        eval_llm_api_key="ollama",
        trace_db_path=tmp_path / "traces.duckdb",
        mem0_storage_path=tmp_path / "mem0",
        log_format="json",
        log_level="INFO",
    )


@pytest.fixture
def mock_settings(monkeypatch: MonkeyPatch, tmp_path: Path) -> Settings:
    """Set environment variables to safe defaults and return a fresh `Settings`.

    Unlike :func:`settings`, this fixture mutates the environment via
    `monkeypatch` so that downstream code calling :func:`get_settings` picks
    up the same values. The ``get_settings`` LRU cache is cleared to guarantee
    a fresh read.
    """
    env_vars = {
        "LLM_BACKEND": "local",
        "LLM_MODEL": "qwen2.5:7b-instruct",
        "LLM_API_BASE": "http://localhost:11434/v1",
        "LLM_API_KEY": "ollama",
        "EVAL_LLM_BACKEND": "local",
        "EVAL_LLM_MODEL": "qwen2.5:7b-instruct",
        "EVAL_LLM_API_BASE": "http://localhost:11434/v1",
        "EVAL_LLM_API_KEY": "ollama",
        "TRACE_DB_PATH": str(tmp_path / "traces.duckdb"),
        "MEM0_STORAGE_PATH": str(tmp_path / "mem0"),
        "LOG_FORMAT": "json",
        "LOG_LEVEL": "INFO",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    get_settings.cache_clear()
    return get_settings()


# ---------------------------------------------------------------------------
# TraceLogger fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_trace_db(tmp_path: Path) -> TraceLogger:
    """Return a fresh :class:`TraceLogger` backed by a per-test DuckDB file.

    The DDL is created eagerly on instantiation so tests can immediately
    begin writing/reading rows.
    """
    db_path = tmp_path / "trace_test.duckdb"
    return TraceLogger(db_path=db_path)


# ---------------------------------------------------------------------------
# LLM client mock
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Return a `MagicMock` that quacks like an :class:`LLMClient`.

    The mock's ``.complete()`` method returns a deterministic
    :class:`LLMCallResult` so tests that exercise prompt/response plumbing
    can assert on concrete fields without any network I/O.
    """
    client = MagicMock(spec=LLMClient)
    client.model = "mock-model"
    client.backend = "local"
    client.complete.return_value = LLMCallResult(
        content="mocked answer",
        model="mock-model",
        backend="local",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=1.23,
        cost_usd=0.0,
        finish_reason="stop",
    )
    return client


# ---------------------------------------------------------------------------
# Schema sample fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_passages() -> list[Passage]:
    """Return five deterministic :class:`Passage` objects for retrieval tests."""
    return [
        Passage(
            passage_id=f"p{i}",
            title=f"Doc {i}",
            text=f"This is the text of passage {i}. It contains fact {i}.",
            score=1.0 - (i * 0.1),
            rerank_score=0.9 - (i * 0.1),
            source_strategy=RetrievalStrategy.HYBRID,
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_eval_records() -> list[EvalRecord]:
    """Return three deterministic :class:`EvalRecord` objects for eval tests."""
    return [
        EvalRecord(
            eval_run_id="run-test",
            question_id=f"q{i}",
            question=f"What is fact {i}?",
            gold_answer=f"Fact {i}",
            predicted_answer=f"Fact {i}",
            gold_supporting_facts=[{"title": f"Doc {i}", "sent_id": 0}],
            retrieved_passage_ids=[f"p{i}", f"p{i + 1}"],
            exact_match=1.0,
            f1=1.0,
            sf_precision=1.0,
            sf_recall=1.0,
            sf_f1=1.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            recall_at_20=1.0,
            precision_at_5=0.4,
            precision_at_10=0.2,
            mrr=1.0,
            ndcg=1.0,
            latency_ms=42.0 + i,
            failure_mode=FailureMode.NONE,
            strategy_used=RetrievalStrategy.HYBRID,
        )
        for i in range(3)
    ]


@pytest.fixture
def sample_query_response(sample_passages: list[Passage]) -> QueryResponse:
    """Return a fully populated :class:`QueryResponse` for serving tests."""
    return QueryResponse(
        answer="The answer is 42.",
        confidence="high",
        query_type=QueryType.BRIDGE,
        sub_questions=[],
        reasoning_chain=[],
        evidence=sample_passages[:2],
        evidence_conflict=False,
        reason=None,
        latency_ms=123.4,
        token_usage={"prompt": 100, "completion": 20, "total": 120},
        cost_usd=0.001,
        trace_id="trace-abc",
    )
