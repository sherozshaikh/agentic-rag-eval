from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from agentic_rag_eval.schemas import (
    EvalRecord,
    EvalRunMetadata,
    FailureMode,
    LLMCallResult,
    RetrievalStrategy,
)
from agentic_rag_eval.tracing.logger import TraceLogger


def test_ddl_created_on_first_use(tmp_path: Path) -> None:
    """Construction must eagerly create all expected tables."""
    db_path = tmp_path / "trace.duckdb"
    TraceLogger(db_path=db_path)

    with duckdb.connect(str(db_path)) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
        }

    expected = {"traces", "spans", "llm_calls", "eval_runs", "eval_records"}
    assert expected.issubset(tables)


def test_ddl_is_idempotent(tmp_path: Path) -> None:
    """Re-instantiating must not fail even though DDL has already run."""
    db_path = tmp_path / "trace.duckdb"
    TraceLogger(db_path=db_path)
    TraceLogger(db_path=db_path)


def test_trace_context_persists_row(tmp_trace_db: TraceLogger) -> None:
    """The `trace` CM must write exactly one row into the traces table."""
    with tmp_trace_db.trace("What is X?", pipeline="baseline") as ctx:
        ctx.strategy_used = "hybrid"
        ctx.confidence = "high"
        ctx.failure_mode = "none"

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        rows = conn.execute(
            "SELECT question, pipeline, strategy_used, confidence, failure_mode FROM traces"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0] == ("What is X?", "baseline", "hybrid", "high", "none")


def test_span_records_latency(tmp_trace_db: TraceLogger) -> None:
    """`span` CM must record positive latency and persist a row."""
    with tmp_trace_db.trace("Q", pipeline="agentic_phase2") as ctx:
        with tmp_trace_db.span(ctx, "retrieve") as sp:
            time.sleep(0.005)
            sp.metadata["strategy"] = "dense"
        with tmp_trace_db.span(ctx, "rerank"):
            pass

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        rows = conn.execute("SELECT name, latency_ms, metadata FROM spans ORDER BY name").fetchall()

    assert len(rows) == 2
    names = [r[0] for r in rows]
    assert names == ["rerank", "retrieve"]

    retrieve_row = next(r for r in rows if r[0] == "retrieve")
    assert retrieve_row[1] >= 0.0
    metadata = json.loads(retrieve_row[2])
    assert metadata == {"strategy": "dense"}


def test_record_llm_call_accumulates_tokens_and_cost(
    tmp_trace_db: TraceLogger,
) -> None:
    """Calling record_llm_call twice must add tokens/cost onto the context."""
    call1 = LLMCallResult(
        content="a",
        model="m",
        backend="api",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=1.0,
        cost_usd=0.001,
    )
    call2 = LLMCallResult(
        content="b",
        model="m",
        backend="api",
        prompt_tokens=20,
        completion_tokens=4,
        total_tokens=24,
        latency_ms=2.0,
        cost_usd=0.002,
    )

    with tmp_trace_db.trace("Q", pipeline="p") as ctx:
        tmp_trace_db.record_llm_call(ctx, role="agent", call=call1, prompt="p1", response="r1")
        tmp_trace_db.record_llm_call(ctx, role="agent", call=call2, prompt="p2", response="r2")

        assert ctx.prompt_tokens == 30
        assert ctx.completion_tokens == 9
        assert ctx.total_tokens == 39
        assert ctx.cost_usd == pytest.approx(0.003)

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
        trace_row = conn.execute(
            "SELECT prompt_tokens, completion_tokens, total_tokens, cost_usd FROM traces"
        ).fetchone()

    assert count == 2
    assert trace_row == (30, 9, 39, pytest.approx(0.003))


def test_record_and_finalize_eval_run(tmp_trace_db: TraceLogger) -> None:
    """record_eval_run + finalize_eval_run must populate both timestamps."""
    meta = EvalRunMetadata(
        eval_run_id="run-xyz",
        git_sha="abc123",
        started_at=datetime.now(UTC),
        finished_at=None,
        pipeline="baseline",
        dataset_split="validation",
        num_questions=100,
        config_snapshot={"llm_model": "qwen2.5:7b-instruct"},
    )

    tmp_trace_db.record_eval_run(meta)

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        row = conn.execute(
            "SELECT eval_run_id, git_sha, pipeline, dataset_split, num_questions, finished_at, "
            "config_snapshot FROM eval_runs WHERE eval_run_id = ?",
            ["run-xyz"],
        ).fetchone()

    assert row is not None
    assert row[0] == "run-xyz"
    assert row[1] == "abc123"
    assert row[2] == "baseline"
    assert row[3] == "validation"
    assert row[4] == 100
    assert row[5] is None
    assert json.loads(row[6]) == {"llm_model": "qwen2.5:7b-instruct"}

    tmp_trace_db.finalize_eval_run("run-xyz")

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        finished_at = conn.execute(
            "SELECT finished_at FROM eval_runs WHERE eval_run_id = ?",
            ["run-xyz"],
        ).fetchone()[0]

    assert finished_at is not None


def test_record_eval_run_is_idempotent(tmp_trace_db: TraceLogger) -> None:
    """Re-recording the same run id must not duplicate rows (INSERT OR REPLACE)."""
    meta = EvalRunMetadata(
        eval_run_id="run-same",
        started_at=datetime.now(UTC),
        pipeline="full",
        dataset_split="validation",
        num_questions=10,
        config_snapshot={},
    )
    tmp_trace_db.record_eval_run(meta)
    tmp_trace_db.record_eval_run(meta)

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM eval_runs WHERE eval_run_id = ?", ["run-same"]
        ).fetchone()[0]
    assert count == 1


def test_record_eval_records_persists_rows(
    tmp_trace_db: TraceLogger, sample_eval_records: list[EvalRecord]
) -> None:
    """All supplied records must land in the eval_records table."""
    tmp_trace_db.record_eval_records(sample_eval_records)

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        rows = conn.execute(
            "SELECT question_id, exact_match, strategy_used, retrieved_passage_ids, "
            "gold_supporting_facts FROM eval_records ORDER BY question_id"
        ).fetchall()

    assert len(rows) == len(sample_eval_records)
    for row, original in zip(rows, sample_eval_records, strict=False):
        assert row[0] == original.question_id
        assert row[1] == original.exact_match
        assert row[2] == (original.strategy_used.value if original.strategy_used else None)
        assert json.loads(row[3]) == original.retrieved_passage_ids
        assert json.loads(row[4]) == original.gold_supporting_facts


def test_record_eval_records_is_idempotent_for_resume(
    tmp_trace_db: TraceLogger,
) -> None:
    """INSERT OR REPLACE means re-submitting the same (run_id, question_id) updates in place."""
    original = EvalRecord(
        eval_run_id="run-r",
        question_id="q1",
        question="Q?",
        gold_answer="A",
        predicted_answer="wrong",
        f1=0.0,
        failure_mode=FailureMode.REASONING_ERROR,
        strategy_used=RetrievalStrategy.DENSE,
    )
    updated = original.model_copy(update={"predicted_answer": "A", "f1": 1.0})

    tmp_trace_db.record_eval_records([original])
    tmp_trace_db.record_eval_records([updated])

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        rows = conn.execute(
            "SELECT predicted_answer, f1 FROM eval_records WHERE eval_run_id = ? "
            "AND question_id = ?",
            ["run-r", "q1"],
        ).fetchall()

    assert len(rows) == 1
    assert rows[0] == ("A", 1.0)


def test_record_eval_records_empty_list_is_noop(tmp_trace_db: TraceLogger) -> None:
    """Passing an empty list must not raise or write anything."""
    tmp_trace_db.record_eval_records([])

    with duckdb.connect(str(tmp_trace_db._db_path)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM eval_records").fetchone()[0]
    assert count == 0
