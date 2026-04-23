from __future__ import annotations

from datetime import UTC, datetime, timedelta

import duckdb

from agentic_rag_eval.config import get_settings
from agentic_rag_eval.logging_setup import configure_logging, get_logger


def main() -> None:
    configure_logging()
    logger = get_logger(__name__)
    settings = get_settings()

    if not settings.trace_db_path.exists():
        logger.info("no_trace_db_found", extra={"path": str(settings.trace_db_path)})
        return

    cutoff = datetime.now(UTC) - timedelta(days=settings.trace_retention_days)

    with duckdb.connect(str(settings.trace_db_path)) as conn:
        trace_ids = conn.execute("SELECT trace_id FROM traces WHERE ts < ?", [cutoff]).fetchall()
        ids = [row[0] for row in trace_ids]
        if not ids:
            logger.info("no_traces_to_delete", extra={"cutoff": cutoff.isoformat()})
            return

        placeholders = ",".join(["?"] * len(ids))
        conn.execute(f"DELETE FROM llm_calls WHERE trace_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM spans     WHERE trace_id IN ({placeholders})", ids)
        conn.execute(f"DELETE FROM traces    WHERE trace_id IN ({placeholders})", ids)

    logger.info(
        "traces_cleaned",
        extra={"deleted_count": len(ids), "cutoff": cutoff.isoformat()},
    )


if __name__ == "__main__":
    main()
