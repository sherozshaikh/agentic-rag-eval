from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import EvalRecord, EvalRunMetadata

logger = get_logger(__name__)


_DDL = [
    """
    CREATE TABLE IF NOT EXISTS traces (
        trace_id        VARCHAR PRIMARY KEY,
        ts              TIMESTAMP,
        question        VARCHAR,
        pipeline        VARCHAR,        -- baseline | agentic_* | full
        eval_run_id     VARCHAR,
        total_latency_ms DOUBLE,
        prompt_tokens   INTEGER,
        completion_tokens INTEGER,
        total_tokens    INTEGER,
        cost_usd        DOUBLE,
        strategy_used   VARCHAR,
        confidence      VARCHAR,
        failure_mode    VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS spans (
        span_id         VARCHAR PRIMARY KEY,
        trace_id        VARCHAR,
        ts              TIMESTAMP,
        name            VARCHAR,        -- decompose | retrieve | rerank | memory | llm | synthesize
        latency_ms      DOUBLE,
        metadata        VARCHAR         -- JSON blob
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS llm_calls (
        call_id         VARCHAR PRIMARY KEY,
        trace_id        VARCHAR,
        ts              TIMESTAMP,
        role            VARCHAR,        -- agent | eval | judge | decompose | reason
        backend         VARCHAR,
        model           VARCHAR,
        prompt          VARCHAR,
        response        VARCHAR,
        prompt_tokens   INTEGER,
        completion_tokens INTEGER,
        total_tokens    INTEGER,
        latency_ms      DOUBLE,
        cost_usd        DOUBLE,
        finish_reason   VARCHAR
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        eval_run_id     VARCHAR PRIMARY KEY,
        git_sha         VARCHAR,
        started_at      TIMESTAMP,
        finished_at     TIMESTAMP,
        pipeline        VARCHAR,
        dataset_split   VARCHAR,
        num_questions   INTEGER,
        config_snapshot VARCHAR         -- JSON blob
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_records (
        eval_run_id     VARCHAR,
        question_id     VARCHAR,
        question        VARCHAR,
        gold_answer     VARCHAR,
        predicted_answer VARCHAR,
        exact_match     DOUBLE,
        f1              DOUBLE,
        sf_precision    DOUBLE,
        sf_recall       DOUBLE,
        sf_f1           DOUBLE,
        recall_at_5     DOUBLE,
        recall_at_10    DOUBLE,
        recall_at_20    DOUBLE,
        precision_at_5  DOUBLE,
        precision_at_10 DOUBLE,
        mrr             DOUBLE,
        ndcg            DOUBLE,
        ragas_faithfulness DOUBLE,
        ragas_answer_relevancy DOUBLE,
        ragas_context_precision DOUBLE,
        ragas_context_recall DOUBLE,
        deepeval_g_eval DOUBLE,
        deepeval_hallucination DOUBLE,
        deepeval_answer_relevancy DOUBLE,
        judge_coherence DOUBLE,
        judge_completeness DOUBLE,
        latency_ms      DOUBLE,
        failure_mode    VARCHAR,
        strategy_used   VARCHAR,
        retrieved_passage_ids VARCHAR, -- JSON array
        gold_supporting_facts VARCHAR,  -- JSON array
        PRIMARY KEY (eval_run_id, question_id)
    );
    """,
]


@dataclass
class TraceSpan:
    name: str
    start_ms: float
    end_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        return max(0.0, self.end_ms - self.start_ms)


@dataclass
class TraceContext:
    trace_id: str
    question: str
    pipeline: str
    eval_run_id: str | None
    started_at: float
    spans: list[TraceSpan] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    strategy_used: str | None = None
    confidence: str | None = None
    failure_mode: str | None = None


class TraceLogger:
    """DuckDB-backed trace writer. Thread-safe via an instance-level lock."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self._db_path))

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            for stmt in _DDL:
                conn.execute(stmt)

    @contextmanager
    def trace(
        self,
        question: str,
        *,
        pipeline: str,
        eval_run_id: str | None = None,
    ) -> Iterator[TraceContext]:
        ctx = TraceContext(
            trace_id=uuid.uuid4().hex,
            question=question,
            pipeline=pipeline,
            eval_run_id=eval_run_id,
            started_at=time.perf_counter(),
        )
        try:
            yield ctx
        finally:
            total_latency = (time.perf_counter() - ctx.started_at) * 1000.0
            self._persist_trace(ctx, total_latency)

    @contextmanager
    def span(self, ctx: TraceContext, name: str) -> Iterator[TraceSpan]:
        span = TraceSpan(name=name, start_ms=time.perf_counter() * 1000.0)
        try:
            yield span
        finally:
            span.end_ms = time.perf_counter() * 1000.0
            ctx.spans.append(span)

    def record_llm_call(
        self,
        ctx: TraceContext,
        *,
        role: str,
        call: Any,
        prompt: str,
        response: str,
    ) -> None:
        ctx.prompt_tokens += getattr(call, "prompt_tokens", 0) or 0
        ctx.completion_tokens += getattr(call, "completion_tokens", 0) or 0
        ctx.total_tokens += getattr(call, "total_tokens", 0) or 0
        ctx.cost_usd += float(getattr(call, "cost_usd", 0.0) or 0.0)

        ctx.llm_calls.append(
            {
                "call_id": uuid.uuid4().hex,
                "trace_id": ctx.trace_id,
                "ts": datetime.now(UTC),
                "role": role,
                "backend": getattr(call, "backend", ""),
                "model": getattr(call, "model", ""),
                "prompt": prompt[:50_000],
                "response": response[:50_000],
                "prompt_tokens": getattr(call, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(call, "completion_tokens", 0) or 0,
                "total_tokens": getattr(call, "total_tokens", 0) or 0,
                "latency_ms": getattr(call, "latency_ms", 0.0) or 0.0,
                "cost_usd": float(getattr(call, "cost_usd", 0.0) or 0.0),
                "finish_reason": getattr(call, "finish_reason", None),
            }
        )

    def _persist_trace(self, ctx: TraceContext, total_latency_ms: float) -> None:
        now = datetime.now(UTC)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ctx.trace_id,
                    now,
                    ctx.question[:2000],
                    ctx.pipeline,
                    ctx.eval_run_id,
                    total_latency_ms,
                    ctx.prompt_tokens,
                    ctx.completion_tokens,
                    ctx.total_tokens,
                    ctx.cost_usd,
                    ctx.strategy_used,
                    ctx.confidence,
                    ctx.failure_mode,
                ],
            )
            for span in ctx.spans:
                conn.execute(
                    """
                    INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        uuid.uuid4().hex,
                        ctx.trace_id,
                        now,
                        span.name,
                        span.latency_ms,
                        json.dumps(span.metadata, default=str),
                    ],
                )
            for call in ctx.llm_calls:
                conn.execute(
                    """
                    INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        call["call_id"],
                        call["trace_id"],
                        call["ts"],
                        call["role"],
                        call["backend"],
                        call["model"],
                        call["prompt"],
                        call["response"],
                        call["prompt_tokens"],
                        call["completion_tokens"],
                        call["total_tokens"],
                        call["latency_ms"],
                        call["cost_usd"],
                        call["finish_reason"],
                    ],
                )

    def record_eval_run(self, meta: EvalRunMetadata) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    meta.eval_run_id,
                    meta.git_sha,
                    meta.started_at,
                    meta.finished_at,
                    meta.pipeline,
                    meta.dataset_split,
                    meta.num_questions,
                    json.dumps(meta.config_snapshot, default=str),
                ],
            )

    def finalize_eval_run(self, eval_run_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE eval_runs SET finished_at = ? WHERE eval_run_id = ?",
                [datetime.now(UTC), eval_run_id],
            )

    def record_eval_records(self, records: list[EvalRecord]) -> None:
        if not records:
            return
        with self._lock, self._connect() as conn:
            for r in records:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO eval_records (
                        eval_run_id, question_id, question, gold_answer, predicted_answer,
                        exact_match, f1, sf_precision, sf_recall, sf_f1,
                        recall_at_5, recall_at_10, recall_at_20, precision_at_5, precision_at_10,
                        mrr, ndcg,
                        ragas_faithfulness, ragas_answer_relevancy,
                        ragas_context_precision, ragas_context_recall,
                        deepeval_g_eval, deepeval_hallucination, deepeval_answer_relevancy,
                        judge_coherence, judge_completeness,
                        latency_ms, failure_mode, strategy_used,
                        retrieved_passage_ids, gold_supporting_facts
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?,
                        ?, ?, ?,
                        ?, ?
                    )
                    """,
                    [
                        r.eval_run_id,
                        r.question_id,
                        r.question[:2000],
                        r.gold_answer[:2000],
                        r.predicted_answer[:2000],
                        r.exact_match,
                        r.f1,
                        r.sf_precision,
                        r.sf_recall,
                        r.sf_f1,
                        r.recall_at_5,
                        r.recall_at_10,
                        r.recall_at_20,
                        r.precision_at_5,
                        r.precision_at_10,
                        r.mrr,
                        r.ndcg,
                        r.ragas_faithfulness,
                        r.ragas_answer_relevancy,
                        r.ragas_context_precision,
                        r.ragas_context_recall,
                        r.deepeval_g_eval,
                        r.deepeval_hallucination,
                        r.deepeval_answer_relevancy,
                        r.judge_coherence,
                        r.judge_completeness,
                        r.latency_ms,
                        r.failure_mode.value,
                        r.strategy_used.value if r.strategy_used else None,
                        json.dumps(r.retrieved_passage_ids),
                        json.dumps(r.gold_supporting_facts, default=str),
                    ],
                )


_instance: TraceLogger | None = None
_instance_lock = threading.Lock()


def get_trace_logger(settings: Settings | None = None) -> TraceLogger:
    """Return the cached TraceLogger, rebuilding it if the db path changed."""
    global _instance
    settings = settings or get_settings()
    with _instance_lock:
        if _instance is None or _instance._db_path != settings.trace_db_path:
            _instance = TraceLogger(settings.trace_db_path)
    return _instance
