from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["dashboard"])


def _get_templates(request: Request) -> Jinja2Templates:
    """Return the Jinja2Templates instance stored on ``app.state``."""
    templates: Jinja2Templates | None = getattr(request.app.state, "templates", None)
    if templates is None:
        raise RuntimeError("app.state.templates is not configured")
    return templates


def _connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    path: Path = settings.trace_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        duckdb.connect(str(path)).close()
    return duckdb.connect(str(path), read_only=True)


@router.get("/", include_in_schema=False)
async def root_redirect() -> RedirectResponse:
    """Redirect ``/`` to ``/dashboard``."""
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_home(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Render the overall dashboard: recent runs, comparison matrix, charts."""
    since_24h = datetime.now(UTC) - timedelta(hours=24)

    with _connect(settings) as conn:
        runs = conn.execute(
            """
            SELECT r.eval_run_id, r.pipeline, r.dataset_split, r.num_questions,
                   r.started_at, r.finished_at,
                   AVG(er.exact_match) AS em,
                   AVG(er.f1) AS f1,
                   AVG(er.sf_f1) AS sf_f1,
                   AVG(er.recall_at_5) AS r5,
                   AVG(er.mrr) AS mrr,
                   AVG(er.latency_ms) AS lat
            FROM eval_runs r
            LEFT JOIN eval_records er USING (eval_run_id)
            GROUP BY r.eval_run_id, r.pipeline, r.dataset_split, r.num_questions,
                     r.started_at, r.finished_at
            ORDER BY r.started_at DESC NULLS LAST
            LIMIT 20
            """
        ).fetchall()

        by_pipeline = conn.execute(
            """
            SELECT r.pipeline,
                   AVG(er.exact_match) AS em,
                   AVG(er.f1) AS f1,
                   AVG(er.sf_f1) AS sf_f1,
                   AVG(er.recall_at_5) AS r5,
                   AVG(er.mrr) AS mrr,
                   AVG(er.latency_ms) AS lat,
                   COUNT(DISTINCT r.eval_run_id) AS runs
            FROM eval_runs r
            JOIN eval_records er USING (eval_run_id)
            GROUP BY r.pipeline
            ORDER BY r.pipeline
            """
        ).fetchall()

        traces_24h = conn.execute(
            """
            SELECT COUNT(*), AVG(total_latency_ms),
                   SUM(total_tokens), SUM(cost_usd),
                   SUM(CASE WHEN failure_mode IS NOT NULL AND failure_mode <> 'none'
                            THEN 1 ELSE 0 END)
            FROM traces
            WHERE ts >= ?
            """,
            [since_24h],
        ).fetchone()

        failure_rows = conn.execute(
            """
            SELECT COALESCE(failure_mode, 'none'), COUNT(*)
            FROM eval_records
            GROUP BY 1
            ORDER BY 2 DESC
            """
        ).fetchall()

    runs_data = [
        {
            "eval_run_id": r[0],
            "pipeline": r[1],
            "dataset_split": r[2],
            "num_questions": r[3],
            "started_at": r[4],
            "finished_at": r[5],
            "em": r[6],
            "f1": r[7],
            "sf_f1": r[8],
            "recall_at_5": r[9],
            "mrr": r[10],
            "latency_ms": r[11],
        }
        for r in runs
    ]

    pipeline_matrix = [
        {
            "pipeline": r[0],
            "em": r[1],
            "f1": r[2],
            "sf_f1": r[3],
            "recall_at_5": r[4],
            "mrr": r[5],
            "latency_ms": r[6],
            "runs": r[7],
        }
        for r in by_pipeline
    ]

    traces_summary = {
        "count": traces_24h[0] or 0,
        "avg_latency_ms": traces_24h[1],
        "total_tokens": int(traces_24h[2] or 0),
        "total_cost_usd": float(traces_24h[3] or 0.0),
        "errors": traces_24h[4] or 0,
    }

    failure_modes = {r[0]: r[1] for r in failure_rows}

    chart_data = {
        "labels": [row["pipeline"] for row in pipeline_matrix],
        "em": [row["em"] or 0 for row in pipeline_matrix],
        "f1": [row["f1"] or 0 for row in pipeline_matrix],
        "latency": [row["latency_ms"] or 0 for row in pipeline_matrix],
        "failure_labels": list(failure_modes.keys()),
        "failure_counts": list(failure_modes.values()),
    }

    return _get_templates(request).TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Overview — agentic-rag-eval",
            "runs": runs_data,
            "pipeline_matrix": pipeline_matrix,
            "traces_summary": traces_summary,
            "failure_modes": failure_modes,
            "chart_data_json": json.dumps(chart_data, default=str),
        },
    )


@router.get(
    "/dashboard/runs/{eval_run_id}",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def dashboard_run_detail(
    request: Request,
    eval_run_id: str,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Drill into one eval run and show per-question records."""
    with _connect(settings) as conn:
        meta = conn.execute(
            """
            SELECT eval_run_id, git_sha, started_at, finished_at, pipeline,
                   dataset_split, num_questions, config_snapshot
            FROM eval_runs WHERE eval_run_id = ?
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
                COUNT(*), AVG(exact_match), AVG(f1), AVG(sf_f1),
                AVG(recall_at_5), AVG(recall_at_10),
                AVG(mrr), AVG(ndcg),
                AVG(ragas_faithfulness), AVG(ragas_answer_relevancy),
                AVG(latency_ms)
            FROM eval_records WHERE eval_run_id = ?
            """,
            [eval_run_id],
        ).fetchone()

        records = conn.execute(
            """
            SELECT question_id, question, gold_answer, predicted_answer,
                   exact_match, f1, sf_f1, latency_ms, failure_mode, strategy_used
            FROM eval_records
            WHERE eval_run_id = ?
            ORDER BY question_id
            LIMIT 500
            """,
            [eval_run_id],
        ).fetchall()

        failure_rows = conn.execute(
            """
            SELECT COALESCE(failure_mode, 'none'), COUNT(*)
            FROM eval_records
            WHERE eval_run_id = ?
            GROUP BY 1 ORDER BY 2 DESC
            """,
            [eval_run_id],
        ).fetchall()

    try:
        config_snapshot = json.loads(meta[7]) if meta[7] else {}
    except json.JSONDecodeError:
        config_snapshot = {}

    run = {
        "eval_run_id": meta[0],
        "git_sha": meta[1],
        "started_at": meta[2],
        "finished_at": meta[3],
        "pipeline": meta[4],
        "dataset_split": meta[5],
        "num_questions": meta[6],
        "config_snapshot": config_snapshot,
    }
    metrics = {
        "num_records": agg[0] or 0,
        "em": agg[1],
        "f1": agg[2],
        "sf_f1": agg[3],
        "recall_at_5": agg[4],
        "recall_at_10": agg[5],
        "mrr": agg[6],
        "ndcg": agg[7],
        "ragas_faithfulness": agg[8],
        "ragas_answer_relevancy": agg[9],
        "latency_ms": agg[10],
    }
    record_list = [
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
        for r in records
    ]
    failure_modes = {r[0]: r[1] for r in failure_rows}

    chart_data = {
        "labels": ["EM", "F1", "SF-F1", "R@5", "R@10", "MRR", "NDCG"],
        "values": [
            metrics["em"] or 0,
            metrics["f1"] or 0,
            metrics["sf_f1"] or 0,
            metrics["recall_at_5"] or 0,
            metrics["recall_at_10"] or 0,
            metrics["mrr"] or 0,
            metrics["ndcg"] or 0,
        ],
        "failure_labels": list(failure_modes.keys()),
        "failure_counts": list(failure_modes.values()),
    }

    return _get_templates(request).TemplateResponse(
        "run_detail.html",
        {
            "request": request,
            "title": f"Run {eval_run_id}",
            "run": run,
            "metrics": metrics,
            "records": record_list,
            "failure_modes": failure_modes,
            "chart_data_json": json.dumps(chart_data, default=str),
        },
    )


@router.get(
    "/dashboard/traces",
    response_class=HTMLResponse,
    include_in_schema=False,
)
async def dashboard_traces(
    request: Request,
    q: str | None = None,
    pipeline: str | None = None,
    limit: int = 100,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    """Recent trace viewer with optional search / pipeline filter."""
    limit = max(1, min(limit, 500))
    filters: list[str] = []
    params: list[Any] = []
    if q:
        filters.append("question ILIKE ?")
        params.append(f"%{q}%")
    if pipeline:
        filters.append("pipeline = ?")
        params.append(pipeline)
    where_clause = (" WHERE " + " AND ".join(filters)) if filters else ""

    with _connect(settings) as conn:
        rows = conn.execute(
            f"""
            SELECT trace_id, ts, question, pipeline, total_latency_ms,
                   total_tokens, cost_usd, strategy_used, confidence, failure_mode
            FROM traces
            {where_clause}
            ORDER BY ts DESC
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()

        pipelines = conn.execute(
            "SELECT DISTINCT pipeline FROM traces ORDER BY pipeline"
        ).fetchall()

    traces = [
        {
            "trace_id": r[0],
            "ts": r[1],
            "question": r[2],
            "pipeline": r[3],
            "latency_ms": r[4],
            "tokens": r[5],
            "cost_usd": r[6],
            "strategy_used": r[7],
            "confidence": r[8],
            "failure_mode": r[9],
        }
        for r in rows
    ]

    return _get_templates(request).TemplateResponse(
        "traces.html",
        {
            "request": request,
            "title": "Traces — agentic-rag-eval",
            "traces": traces,
            "search": q or "",
            "pipeline": pipeline or "",
            "available_pipelines": [row[0] for row in pipelines if row[0]],
            "limit": limit,
        },
    )
