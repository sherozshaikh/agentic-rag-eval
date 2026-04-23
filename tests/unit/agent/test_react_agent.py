from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_rag_eval.agent.decomposer import QueryDecomposer
from agentic_rag_eval.agent.react_agent import ReActAgent
from agentic_rag_eval.config import Settings
from agentic_rag_eval.schemas import (
    LLMCallResult,
    Passage,
    QueryType,
    RetrievalStrategy,
    SubQuestion,
)


def _make_llm_result(content: str) -> LLMCallResult:
    return LLMCallResult(
        content=content,
        model="test-model",
        backend="local",
        prompt_tokens=5,
        completion_tokens=7,
        total_tokens=12,
        latency_ms=1.0,
        cost_usd=0.0,
        finish_reason="stop",
    )


def _make_passage(pid: str, text: str = "body", score: float = 0.9) -> Passage:
    return Passage(
        passage_id=pid,
        title=f"Title-{pid}",
        text=text,
        score=score,
        rerank_score=score,
        source_strategy=RetrievalStrategy.HYBRID,
    )


def _fake_decomposer(query_type: QueryType, subs: list[SubQuestion]) -> Any:
    dec = MagicMock(spec=QueryDecomposer)
    dec.decompose.return_value = (query_type, subs)
    return dec


def _fake_registry() -> Any:
    reg = MagicMock()

    def get(name: str) -> Any:
        prompt = MagicMock()
        prompt.render.return_value = f"RENDERED:{name}"
        return prompt

    reg.get.side_effect = get
    return reg


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        trace_db_path=tmp_path / "traces.duckdb",
        max_agent_steps=3,
        context_budget_tokens=2000,
        generation_reserve_tokens=500,
        mem0_storage_path=tmp_path / "mem0",
    )


def _build_agent(
    *,
    llm: Any,
    retriever: Any,
    settings: Settings,
    decomposer: Any,
    memory: Any | None = None,
) -> ReActAgent:
    return ReActAgent(
        llm=llm,
        retriever=retriever,
        memory=memory,
        trace_logger=None,
        settings=settings,
        decomposer=decomposer,
        prompt_registry=_fake_registry(),
    )


def test_react_agent_happy_path(settings: Settings) -> None:
    subs = [
        SubQuestion(text="q1", strategy=RetrievalStrategy.HYBRID),
    ]
    dec = _fake_decomposer(QueryType.SINGLE_HOP, subs)

    retriever = MagicMock()
    retriever.retrieve_and_rerank.return_value = [
        _make_passage("p1", "short body"),
        _make_passage("p2", "short body"),
    ]

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result(
        '{"answer": "Paris", "confidence": "high", '
        '"reasoning": "p1 says Paris", "evidence_conflict": false, "reason": null}'
    )

    agent = _build_agent(llm=llm, retriever=retriever, settings=settings, decomposer=dec)
    resp = agent.answer("What is the capital of France?")

    assert resp.answer == "Paris"
    assert resp.confidence == "high"
    assert resp.query_type == QueryType.SINGLE_HOP
    assert len(resp.evidence) == 2
    assert len(resp.reasoning_chain) >= 1
    assert resp.evidence_conflict is False
    retriever.retrieve_and_rerank.assert_called_once()


def test_react_agent_respects_max_agent_steps(settings: Settings) -> None:
    subs = [SubQuestion(text=f"q{i}", strategy=RetrievalStrategy.HYBRID) for i in range(5)]
    dec = _fake_decomposer(QueryType.COMPARISON, subs)

    retriever = MagicMock()
    retriever.retrieve_and_rerank.return_value = [_make_passage("p1", "x")]

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result(
        '{"answer": "ok", "confidence": "medium", "reasoning": "r",'
        ' "evidence_conflict": false, "reason": null}'
    )

    agent = _build_agent(llm=llm, retriever=retriever, settings=settings, decomposer=dec)
    resp = agent.answer("multi")

    assert retriever.retrieve_and_rerank.call_count == 3
    assert resp.confidence == "medium"


def test_react_agent_context_budget_exhausted(settings: Settings) -> None:
    big = "x" * 8000
    subs = [
        SubQuestion(text="q1", strategy=RetrievalStrategy.HYBRID),
        SubQuestion(text="q2", strategy=RetrievalStrategy.HYBRID),
    ]
    dec = _fake_decomposer(QueryType.BRIDGE, subs)

    retriever = MagicMock()
    retriever.retrieve_and_rerank.return_value = [
        _make_passage("p1", big),
        _make_passage("p2", big),
    ]

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result(
        '{"answer": "partial", "confidence": "high", '
        '"reasoning": "r", "evidence_conflict": false, "reason": null}'
    )

    agent = _build_agent(llm=llm, retriever=retriever, settings=settings, decomposer=dec)
    resp = agent.answer("huge question")

    assert resp.confidence == "low"
    assert resp.reason is not None
    assert "Context budget exhausted at step" in resp.reason


def test_react_agent_propagates_evidence_conflict(settings: Settings) -> None:
    subs = [SubQuestion(text="q1", strategy=RetrievalStrategy.HYBRID)]
    dec = _fake_decomposer(QueryType.SINGLE_HOP, subs)

    retriever = MagicMock()
    retriever.retrieve_and_rerank.return_value = [
        _make_passage("p1", "short"),
        _make_passage("p2", "short"),
    ]

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result(
        '{"answer": "ambiguous", "confidence": "medium", '
        '"reasoning": "p1 disagrees with p2", "evidence_conflict": true, "reason": null}'
    )

    agent = _build_agent(llm=llm, retriever=retriever, settings=settings, decomposer=dec)
    resp = agent.answer("conflicting question")

    assert resp.evidence_conflict is True
    assert resp.answer == "ambiguous"


def test_react_agent_memory_lookup(settings: Settings) -> None:
    subs = [SubQuestion(text="q1", strategy=RetrievalStrategy.HYBRID)]
    dec = _fake_decomposer(QueryType.SINGLE_HOP, subs)

    retriever = MagicMock()
    retriever.retrieve_and_rerank.return_value = [_make_passage("p1", "x")]

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result(
        '{"answer": "ok", "confidence": "high", '
        '"reasoning": "r", "evidence_conflict": false, "reason": null}'
    )

    memory = MagicMock()
    memory.search.return_value = [
        {"memory": "Previously discussed fact A"},
        {"text": "And fact B"},
    ]
    memory.add_from_response = MagicMock()

    agent = _build_agent(
        llm=llm,
        retriever=retriever,
        settings=settings,
        decomposer=dec,
        memory=memory,
    )
    resp = agent.answer("q", user_id="user-1")

    memory.search.assert_called_once()
    memory.add_from_response.assert_called_once()
    assert resp.answer == "ok"


def test_react_agent_fallback_on_reason_json_parse(settings: Settings) -> None:
    subs = [SubQuestion(text="q", strategy=RetrievalStrategy.HYBRID)]
    dec = _fake_decomposer(QueryType.SINGLE_HOP, subs)

    retriever = MagicMock()
    retriever.retrieve_and_rerank.return_value = [_make_passage("p1", "x")]

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result("not-json-at-all")

    agent = _build_agent(llm=llm, retriever=retriever, settings=settings, decomposer=dec)
    resp = agent.answer("q")

    assert resp.confidence == "low"
    assert resp.reason == "Could not parse reasoning JSON"


def test_react_agent_topological_order_respects_depends_on(settings: Settings) -> None:
    subs = [
        SubQuestion(text="first-listed", strategy=RetrievalStrategy.HYBRID, depends_on=[1]),
        SubQuestion(text="second-listed", strategy=RetrievalStrategy.HYBRID, depends_on=[]),
    ]
    dec = _fake_decomposer(QueryType.BRIDGE, subs)

    call_order: list[str] = []

    def fake_retrieve(query: str, top_k: int, rerank_top_k: int) -> list[Passage]:
        call_order.append(query)
        return [_make_passage(f"p-{query}", "x")]

    retriever = MagicMock()
    retriever.retrieve_and_rerank.side_effect = fake_retrieve

    llm = MagicMock()
    llm.complete.return_value = _make_llm_result(
        '{"answer": "ok", "confidence": "high", "reasoning": "r",'
        ' "evidence_conflict": false, "reason": null}'
    )

    agent = _build_agent(llm=llm, retriever=retriever, settings=settings, decomposer=dec)
    agent.answer("q")

    assert call_order == ["second-listed", "first-listed"]
