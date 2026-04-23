from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic_rag_eval.schemas import (
    EvalRecord,
    FailureMode,
    Passage,
    QueryRequest,
    QueryResponse,
    QueryType,
    RetrievalStrategy,
)


def test_query_request_rejects_empty_question() -> None:
    """Empty strings must fail pydantic's min_length validator."""
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_rejects_whitespace_only_question() -> None:
    """Whitespace-only questions must be rejected by the field validator."""
    with pytest.raises(ValidationError, match="cannot be empty or whitespace"):
        QueryRequest(question="   \t\n  ")


def test_query_request_strips_whitespace() -> None:
    """Valid questions with leading/trailing whitespace must be stripped."""
    req = QueryRequest(question="  What is the capital of France?  ")
    assert req.question == "What is the capital of France?"


def test_query_request_enforces_length_limit() -> None:
    """Questions longer than 2000 chars must be rejected."""
    with pytest.raises(ValidationError):
        QueryRequest(question="a" * 2001)


def test_query_request_defaults() -> None:
    req = QueryRequest(question="test")
    assert req.user_id is None
    assert req.use_memory is True
    assert req.top_k == 10
    assert req.rerank_top_k == 5


def test_query_request_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=0)
    with pytest.raises(ValidationError):
        QueryRequest(question="test", top_k=101)


def test_query_response_defaults() -> None:
    """QueryResponse must be constructible from just an answer."""
    resp = QueryResponse(answer="42")

    assert resp.answer == "42"
    assert resp.confidence == "high"
    assert resp.query_type == QueryType.UNKNOWN
    assert resp.sub_questions == []
    assert resp.reasoning_chain == []
    assert resp.evidence == []
    assert resp.evidence_conflict is False
    assert resp.reason is None
    assert resp.latency_ms == 0.0
    assert resp.token_usage == {}
    assert resp.cost_usd == 0.0
    assert resp.trace_id is None


@pytest.mark.parametrize(
    "enum_cls,value",
    [
        (QueryType, "single_hop"),
        (QueryType, "bridge"),
        (QueryType, "comparison"),
        (QueryType, "unknown"),
        (RetrievalStrategy, "dense"),
        (RetrievalStrategy, "sparse"),
        (RetrievalStrategy, "hybrid"),
        (FailureMode, "none"),
        (FailureMode, "retrieval_miss"),
        (FailureMode, "reasoning_error"),
        (FailureMode, "decomposition_failure"),
        (FailureMode, "context_overflow"),
        (FailureMode, "memory_stale"),
    ],
)
def test_enum_round_trip(enum_cls: type, value: str) -> None:
    """Each enum value must round-trip via its string representation."""
    member = enum_cls(value)
    assert member.value == value
    assert enum_cls(member.value) is member


def test_eval_record_defaults() -> None:
    """EvalRecord must be constructible with just the required fields."""
    rec = EvalRecord(
        eval_run_id="run-1",
        question_id="q1",
        question="Q?",
        gold_answer="A",
        predicted_answer="A",
    )

    assert rec.gold_supporting_facts == []
    assert rec.retrieved_passage_ids == []
    assert rec.exact_match == 0.0
    assert rec.f1 == 0.0
    assert rec.ragas_faithfulness is None
    assert rec.deepeval_g_eval is None
    assert rec.judge_coherence is None
    assert rec.failure_mode == FailureMode.NONE
    assert rec.strategy_used is None


def test_query_response_dump_validate_round_trip() -> None:
    original = QueryResponse(
        answer="Paris",
        confidence="high",
        query_type=QueryType.SINGLE_HOP,
        evidence=[
            Passage(passage_id="p1", title="Geo", text="Paris is the capital."),
        ],
        token_usage={"prompt": 10, "completion": 3, "total": 13},
        cost_usd=0.0001,
        trace_id="trace-xyz",
    )
    dumped = original.model_dump()
    restored = QueryResponse.model_validate(dumped)
    assert restored == original


def test_eval_record_dump_validate_round_trip() -> None:
    original = EvalRecord(
        eval_run_id="run-1",
        question_id="q-42",
        question="Q?",
        gold_answer="A",
        predicted_answer="A",
        gold_supporting_facts=[{"title": "T", "sent_id": 0}],
        retrieved_passage_ids=["p1", "p2"],
        f1=0.75,
        failure_mode=FailureMode.RETRIEVAL_MISS,
        strategy_used=RetrievalStrategy.HYBRID,
    )
    dumped = original.model_dump()
    restored = EvalRecord.model_validate(dumped)
    assert restored == original


def test_passage_dump_validate_round_trip() -> None:
    p = Passage(
        passage_id="p1",
        title="T",
        text="body",
        score=0.42,
        rerank_score=0.9,
        source_strategy=RetrievalStrategy.DENSE,
    )
    assert Passage.model_validate(p.model_dump()) == p
