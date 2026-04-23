from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryType(str, Enum):
    SINGLE_HOP = "single_hop"
    BRIDGE = "bridge"
    COMPARISON = "comparison"
    UNKNOWN = "unknown"


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class SubQuestion(BaseModel):
    """One sub-question produced by the decomposer."""

    text: str
    strategy: RetrievalStrategy = RetrievalStrategy.HYBRID
    depends_on: list[int] = Field(default_factory=list)


class Passage(BaseModel):
    """A retrieved passage."""

    passage_id: str
    title: str | None = None
    text: str
    score: float = 0.0
    rerank_score: float | None = None
    source_strategy: RetrievalStrategy | None = None


class RetrievalResult(BaseModel):
    """Output of a retrieval call."""

    query: str
    strategy: RetrievalStrategy
    passages: list[Passage] = Field(default_factory=list)
    latency_ms: float = 0.0


class QueryRequest(BaseModel):
    """User query to the /query endpoint."""

    question: str = Field(min_length=1, max_length=2000)
    user_id: str | None = Field(default=None, max_length=128)
    use_memory: bool = True
    top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def _no_empty_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be empty or whitespace")
        return v.strip()


class ReasoningStep(BaseModel):
    """One step of the ReAct loop."""

    step: int
    thought: str
    action: str
    observation: str | None = None
    retrieved: list[Passage] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    answer: str
    confidence: str = Field(default="high", description="high | medium | low")
    query_type: QueryType = QueryType.UNKNOWN
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    reasoning_chain: list[ReasoningStep] = Field(default_factory=list)
    evidence: list[Passage] = Field(default_factory=list)
    evidence_conflict: bool = False
    reason: str | None = None
    latency_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    trace_id: str | None = None


class LLMMessage(BaseModel):
    role: str
    content: str


class LLMCallResult(BaseModel):
    """Result of a single LLM call — returned by LLMClient."""

    content: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    finish_reason: str | None = None


class FailureMode(str, Enum):
    NONE = "none"
    RETRIEVAL_MISS = "retrieval_miss"
    REASONING_ERROR = "reasoning_error"
    DECOMPOSITION_FAILURE = "decomposition_failure"
    CONTEXT_OVERFLOW = "context_overflow"
    MEMORY_STALE = "memory_stale"


class EvalRecord(BaseModel):
    """One row of evaluation output — question, prediction, metrics."""

    eval_run_id: str
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    gold_supporting_facts: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_passage_ids: list[str] = Field(default_factory=list)

    exact_match: float = 0.0
    f1: float = 0.0
    sf_precision: float = 0.0
    sf_recall: float = 0.0
    sf_f1: float = 0.0

    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0

    ragas_faithfulness: float | None = None
    ragas_answer_relevancy: float | None = None
    ragas_context_precision: float | None = None
    ragas_context_recall: float | None = None

    deepeval_g_eval: float | None = None
    deepeval_hallucination: float | None = None
    deepeval_answer_relevancy: float | None = None

    judge_coherence: float | None = None
    judge_completeness: float | None = None

    latency_ms: float = 0.0
    failure_mode: FailureMode = FailureMode.NONE
    strategy_used: RetrievalStrategy | None = None


class EvalRunMetadata(BaseModel):
    """Metadata captured for each evaluation run."""

    eval_run_id: str
    git_sha: str | None = None
    started_at: datetime
    finished_at: datetime | None = None
    pipeline: str
    dataset_split: str
    num_questions: int
    config_snapshot: dict[str, Any]
