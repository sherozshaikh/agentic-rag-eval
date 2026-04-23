from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from agentic_rag_eval.agent.decomposer import QueryDecomposer
from agentic_rag_eval.schemas import LLMCallResult, QueryType, RetrievalStrategy


def _make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    llm.complete.return_value = LLMCallResult(
        content=content,
        model="test-model",
        backend="local",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        latency_ms=1.0,
        cost_usd=0.0,
        finish_reason="stop",
    )
    return llm


@pytest.fixture
def registry() -> Any:
    reg = MagicMock()
    prompt = MagicMock()
    prompt.render.return_value = "PROMPT"
    reg.get.return_value = prompt
    return reg


def test_decomposer_parses_bridge_question(registry: Any) -> None:
    payload = (
        '{"query_type": "bridge", '
        '"sub_questions": ['
        '{"text": "Who directed Inception?", "strategy": "sparse", "depends_on": []},'
        '{"text": "What other films did that director make?", "strategy": "hybrid", "depends_on": [0]}'
        "]}"
    )
    llm = _make_llm(payload)
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("What films were made by the director of Inception?")

    assert qt == QueryType.BRIDGE
    assert len(subs) == 2
    assert subs[0].strategy == RetrievalStrategy.SPARSE
    assert subs[1].depends_on == [0]


def test_decomposer_parses_single_hop(registry: Any) -> None:
    payload = (
        '{"query_type": "single_hop", '
        '"sub_questions": [{"text": "What is the capital of France?", "strategy": "dense"}]}'
    )
    llm = _make_llm(payload)
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("What is the capital of France?")
    assert qt == QueryType.SINGLE_HOP
    assert len(subs) == 1
    assert subs[0].strategy == RetrievalStrategy.DENSE
    assert subs[0].depends_on == []


def test_decomposer_extracts_json_from_prose(registry: Any) -> None:
    payload = (
        "Sure — here is the decomposition:\n"
        '```json\n{"query_type": "comparison", '
        '"sub_questions": ['
        '{"text": "Height of Everest?", "strategy": "sparse"},'
        '{"text": "Height of K2?", "strategy": "sparse"}'
        "]}\n```"
    )
    llm = _make_llm(payload)
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("Which is taller, Everest or K2?")
    assert qt == QueryType.COMPARISON
    assert len(subs) == 2


def test_decomposer_falls_back_on_malformed_json(registry: Any) -> None:
    llm = _make_llm("this is not json at all")
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("Some question")
    assert qt == QueryType.SINGLE_HOP
    assert len(subs) == 1
    assert subs[0].text == "Some question"
    assert subs[0].strategy == RetrievalStrategy.HYBRID


def test_decomposer_falls_back_on_llm_error(registry: Any) -> None:
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("boom")
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("Another question")
    assert qt == QueryType.SINGLE_HOP
    assert subs[0].text == "Another question"


def test_decomposer_falls_back_on_empty_sub_questions(registry: Any) -> None:
    payload = '{"query_type": "single_hop", "sub_questions": []}'
    llm = _make_llm(payload)
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("edge case question")
    assert len(subs) == 1
    assert subs[0].text == "edge case question"


def test_decomposer_handles_invalid_strategy(registry: Any) -> None:
    payload = '{"query_type": "single_hop", "sub_questions": [{"text": "q", "strategy": "banana"}]}'
    llm = _make_llm(payload)
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("q")
    assert subs[0].strategy == RetrievalStrategy.HYBRID


def test_decomposer_handles_unknown_query_type(registry: Any) -> None:
    payload = (
        '{"query_type": "something_odd", "sub_questions": [{"text": "q", "strategy": "hybrid"}]}'
    )
    llm = _make_llm(payload)
    decomposer = QueryDecomposer(llm=llm, prompt_registry=registry)

    qt, subs = decomposer.decompose("q")
    assert qt == QueryType.UNKNOWN
    assert len(subs) == 1
