from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.serving.auth import verify_api_key
from agentic_rag_eval.serving.deps import get_react_agent
from agentic_rag_eval.serving.validators import ValidationError, validate_question

logger = get_logger(__name__)
router = APIRouter(tags=["evaluate"])


_batch_lock = asyncio.Lock()
_current_batch: dict[str, Any] | None = None
_background_tasks: set[asyncio.Task[None]] = set()


class EvalPair(BaseModel):
    """A single ``(question, gold_answer)`` pair for synchronous eval."""

    question: str = Field(min_length=1, max_length=2000)
    gold_answer: str = Field(min_length=1, max_length=2000)
    question_id: str | None = None


class EvaluateRequest(BaseModel):
    pairs: list[EvalPair] = Field(min_length=1, max_length=50)


class EvalPairResult(BaseModel):
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    exact_match: float
    latency_ms: float


class EvaluateResponse(BaseModel):
    eval_run_id: str
    num_questions: int
    avg_latency_ms: float
    exact_match_mean: float
    results: list[EvalPairResult]


class BatchEvalRequest(BaseModel):
    pipeline: str = Field(default="full", description="baseline | agentic | full")
    dataset_split: str = Field(default="validation")
    num_questions: int | None = Field(default=None, ge=1, le=5000)


class BatchEvalResponse(BaseModel):
    eval_run_id: str
    status: str
    started_at: datetime
    pipeline: str


class EvalRunSummary(BaseModel):
    eval_run_id: str
    pipeline: str
    dataset_split: str
    num_questions: int
    started_at: datetime | None
    finished_at: datetime | None
    em_mean: float | None = None
    f1_mean: float | None = None
    latency_ms_mean: float | None = None


def _connect_read_only(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Open the trace DB read-only."""
    path: Path = settings.trace_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        duckdb.connect(str(path)).close()
    return duckdb.connect(str(path), read_only=True)


def _simple_em(pred: str, gold: str) -> float:
    """Case-insensitive exact-match used by the sync endpoint."""
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Run a synchronous eval on a small list of pairs",
)
async def evaluate_endpoint(
    body: EvaluateRequest,
    agent: Any = Depends(get_react_agent),
) -> EvaluateResponse:
    """Evaluate up to 50 pairs synchronously and return per-pair metrics."""
    eval_run_id = f"sync-{uuid.uuid4().hex[:12]}"
    results: list[EvalPairResult] = []
    total_latency = 0.0
    em_total = 0.0

    for idx, pair in enumerate(body.pairs):
        try:
            question = validate_question(pair.question)
        except ValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"pair[{idx}]: {exc}",
            ) from exc

        try:
            response = agent.answer(question=question, user_id=None, trace_ctx=None)
            predicted = getattr(response, "answer", "") or ""
            latency = float(getattr(response, "latency_ms", 0.0) or 0.0)
        except Exception as exc:
            logger.exception("evaluate.pair_failed", extra={"index": idx, "error": str(exc)})
            predicted = ""
            latency = 0.0

        em = _simple_em(predicted, pair.gold_answer)
        em_total += em
        total_latency += latency
        results.append(
            EvalPairResult(
                question_id=pair.question_id or f"q-{idx}",
                question=question,
                gold_answer=pair.gold_answer,
                predicted_answer=predicted,
                exact_match=em,
                latency_ms=latency,
            )
        )

    n = len(results)
    return EvaluateResponse(
        eval_run_id=eval_run_id,
        num_questions=n,
        avg_latency_ms=total_latency / n if n else 0.0,
        exact_match_mean=em_total / n if n else 0.0,
        results=results,
    )


async def _run_batch_eval(
    eval_run_id: str,
    request: BatchEvalRequest,
    settings: Settings,
) -> None:
    """Background-task wrapper around ``EvalRunner.run``."""
    global _current_batch
    started = datetime.now(UTC)
    logger.info(
        "batch_eval.started",
        extra={"eval_run_id": eval_run_id, "pipeline": request.pipeline},
    )
    try:
        from agentic_rag_eval.evaluation import EvalRunner

        runner = EvalRunner(settings=settings)
        await asyncio.to_thread(
            runner.run,
            eval_run_id=eval_run_id,
            pipeline=request.pipeline,
            dataset_split=request.dataset_split,
            num_questions=request.num_questions,
        )
    except Exception:
        logger.exception("batch_eval.failed", extra={"eval_run_id": eval_run_id})
    finally:
        finished = datetime.now(UTC)
        logger.info(
            "batch_eval.finished",
            extra={
                "eval_run_id": eval_run_id,
                "duration_s": (finished - started).total_seconds(),
            },
        )
        _current_batch = None


@router.post(
    "/batch-eval",
    response_model=BatchEvalResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
    summary="Trigger a full evaluation run in the background",
)
async def batch_eval_endpoint(
    body: BatchEvalRequest,
    settings: Settings = Depends(get_settings),
) -> BatchEvalResponse:
    """Start a background eval run; returns 409 if one is already in flight."""
    global _current_batch

    if _batch_lock.locked() and _current_batch is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "batch_eval_in_progress",
                "running_eval_run_id": _current_batch["eval_run_id"],
                "started_at": _current_batch["started_at"].isoformat(),
            },
        )

    eval_run_id = f"batch-{uuid.uuid4().hex[:12]}"
    started_at = datetime.now(UTC)

    async def _runner() -> None:
        async with _batch_lock:
            await _run_batch_eval(eval_run_id, body, settings)

    _current_batch = {"eval_run_id": eval_run_id, "started_at": started_at}
    task = asyncio.create_task(_runner())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    return BatchEvalResponse(
        eval_run_id=eval_run_id,
        status="accepted",
        started_at=started_at,
        pipeline=body.pipeline,
    )


@router.get(
    "/eval-runs",
    response_model=list[EvalRunSummary],
    summary="List recent eval runs",
)
async def list_eval_runs(
    limit: int = 50,
    settings: Settings = Depends(get_settings),
) -> list[EvalRunSummary]:
    """List the most recent eval runs, newest first."""
    limit = max(1, min(limit, 500))
    with _connect_read_only(settings) as conn:
        rows = conn.execute(
            """
            SELECT r.eval_run_id, r.pipeline, r.dataset_split, r.num_questions,
                   r.started_at, r.finished_at,
                   AVG(er.exact_match) AS em,
                   AVG(er.f1) AS f1,
                   AVG(er.latency_ms) AS lat
            FROM eval_runs r
            LEFT JOIN eval_records er USING (eval_run_id)
            GROUP BY r.eval_run_id, r.pipeline, r.dataset_split, r.num_questions,
                     r.started_at, r.finished_at
            ORDER BY r.started_at DESC NULLS LAST
            LIMIT ?
            """,
            [limit],
        ).fetchall()

    return [
        EvalRunSummary(
            eval_run_id=row[0],
            pipeline=row[1] or "",
            dataset_split=row[2] or "",
            num_questions=row[3] or 0,
            started_at=row[4],
            finished_at=row[5],
            em_mean=row[6],
            f1_mean=row[7],
            latency_ms_mean=row[8],
        )
        for row in rows
    ]


@router.get(
    "/eval-runs/{eval_run_id}",
    summary="Get one eval run's summary + aggregate metrics",
)
async def get_eval_run(
    eval_run_id: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return the metadata row plus aggregate metrics for a run."""
    with _connect_read_only(settings) as conn:
        meta = conn.execute(
            """
            SELECT eval_run_id, git_sha, started_at, finished_at, pipeline,
                   dataset_split, num_questions, config_snapshot
            FROM eval_runs
            WHERE eval_run_id = ?
            """,
            [eval_run_id],
        ).fetchone()
        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"eval_run_id {eval_run_id!r} not found",
            )

        agg = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(exact_match) AS em,
                AVG(f1) AS f1,
                AVG(sf_f1) AS sf_f1,
                AVG(recall_at_5) AS r5,
                AVG(recall_at_10) AS r10,
                AVG(mrr) AS mrr,
                AVG(ndcg) AS ndcg,
                AVG(ragas_faithfulness) AS faith,
                AVG(ragas_answer_relevancy) AS ans_rel,
                AVG(latency_ms) AS lat
            FROM eval_records
            WHERE eval_run_id = ?
            """,
            [eval_run_id],
        ).fetchone()

        failure_rows = conn.execute(
            """
            SELECT failure_mode, COUNT(*)
            FROM eval_records
            WHERE eval_run_id = ?
            GROUP BY failure_mode
            ORDER BY 2 DESC
            """,
            [eval_run_id],
        ).fetchall()

    try:
        config_snapshot = json.loads(meta[7]) if meta[7] else {}
    except json.JSONDecodeError:
        config_snapshot = {}

    return {
        "eval_run_id": meta[0],
        "git_sha": meta[1],
        "started_at": meta[2],
        "finished_at": meta[3],
        "pipeline": meta[4],
        "dataset_split": meta[5],
        "num_questions": meta[6],
        "config_snapshot": config_snapshot,
        "metrics": {
            "num_records": agg[0] or 0,
            "exact_match_mean": agg[1],
            "f1_mean": agg[2],
            "sf_f1_mean": agg[3],
            "recall_at_5_mean": agg[4],
            "recall_at_10_mean": agg[5],
            "mrr_mean": agg[6],
            "ndcg_mean": agg[7],
            "ragas_faithfulness_mean": agg[8],
            "ragas_answer_relevancy_mean": agg[9],
            "latency_ms_mean": agg[10],
        },
        "failure_modes": {row[0] or "none": row[1] for row in failure_rows},
    }


@router.get(
    "/eval-runs/{eval_run_id}/records",
    summary="Get per-question records for an eval run",
)
async def get_eval_run_records(
    eval_run_id: str,
    limit: int = 200,
    offset: int = 0,
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return a page of per-question records for an eval run."""
    limit = max(1, min(limit, 1000))
    offset = max(0, offset)

    with _connect_read_only(settings) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM eval_records WHERE eval_run_id = ?",
            [eval_run_id],
        ).fetchone()[0]
        rows = conn.execute(
            """
            SELECT question_id, question, gold_answer, predicted_answer,
                   exact_match, f1, sf_f1, latency_ms, failure_mode,
                   strategy_used
            FROM eval_records
            WHERE eval_run_id = ?
            ORDER BY question_id
            LIMIT ? OFFSET ?
            """,
            [eval_run_id, limit, offset],
        ).fetchall()

    records = [
        {
            "question_id": r[0],
            "question": r[1],
            "gold_answer": r[2],
            "predicted_answer": r[3],
            "exact_match": r[4],
            "f1": r[5],
            "sf_f1": r[6],
            "latency_ms": r[7],
            "failure_mode": r[8],
            "strategy_used": r[9],
        }
        for r in rows
    ]
    return {
        "eval_run_id": eval_run_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "records": records,
    }
