from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
from fastapi import APIRouter, Depends

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["metrics"])


def _connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    path: Path = settings.trace_db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        duckdb.connect(str(path)).close()
    return duckdb.connect(str(path), read_only=True)


@router.get(
    "/metrics",
    summary="System counters (queries, latency, errors) as JSON",
)
async def metrics_endpoint(
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Return aggregate counters over all-time and the last 24 hours."""
    since = datetime.now(UTC) - timedelta(hours=24)

    def _aggregate(
        conn: duckdb.DuckDBPyConnection, where: str, params: list[Any]
    ) -> dict[str, Any]:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                AVG(total_latency_ms) AS avg_latency,
                MAX(total_latency_ms) AS max_latency,
                SUM(total_tokens) AS total_tokens,
                SUM(cost_usd) AS total_cost,
                SUM(CASE WHEN failure_mode IS NOT NULL
                         AND failure_mode <> 'none' THEN 1 ELSE 0 END) AS errors
            FROM traces
            {where}
            """,
            params,
        ).fetchone()
        total = row[0] or 0
        errors = row[5] or 0
        return {
            "total_queries": total,
            "avg_latency_ms": row[1],
            "max_latency_ms": row[2],
            "total_tokens": int(row[3] or 0),
            "total_cost_usd": float(row[4] or 0.0),
            "errors": errors,
            "error_rate": (errors / total) if total else 0.0,
        }

    try:
        with _connect(settings) as conn:
            all_time = _aggregate(conn, "", [])
            last_24h = _aggregate(conn, "WHERE ts >= ?", [since])

            pipeline_rows = conn.execute(
                """
                SELECT pipeline, COUNT(*), AVG(total_latency_ms)
                FROM traces
                GROUP BY pipeline
                ORDER BY 2 DESC
                """
            ).fetchall()
            eval_run_count = conn.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    except Exception as exc:
        logger.exception("metrics.query_failed", extra={"error": str(exc)})
        return {
            "status": "error",
            "error": str(exc),
            "all_time": {},
            "last_24h": {},
        }

    return {
        "status": "ok",
        "all_time": all_time,
        "last_24h": last_24h,
        "by_pipeline": [
            {"pipeline": r[0], "count": r[1], "avg_latency_ms": r[2]} for r in pipeline_rows
        ],
        "eval_runs_total": eval_run_count,
    }
