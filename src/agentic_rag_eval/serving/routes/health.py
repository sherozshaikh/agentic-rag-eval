from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


MIN_FREE_DISK_BYTES = 100 * 1024 * 1024


class ComponentStatus(BaseModel):
    qdrant: bool
    llm: bool
    duckdb: bool
    disk: bool


class HealthResponse(BaseModel):
    status: str
    components: ComponentStatus
    details: dict[str, Any] = {}


async def _check_qdrant(settings: Settings) -> tuple[bool, str]:
    """Probe Qdrant connectivity."""
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:
        return False, f"qdrant-client not installed: {exc}"

    def _probe() -> tuple[bool, str]:
        try:
            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=3.0,
                check_compatibility=False,
            )
            client.get_collections()
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    return await asyncio.to_thread(_probe)


async def _check_llm(settings: Settings) -> tuple[bool, str]:
    """Probe the LLM backend."""

    def _probe() -> tuple[bool, str]:
        try:
            from agentic_rag_eval.llm import build_llm_client

            client = build_llm_client(settings)
            for name in ("ping", "health_check", "generate", "complete"):
                fn = getattr(client, name, None)
                if callable(fn):
                    if name in ("ping", "health_check"):
                        fn()
                    else:
                        fn("ping", max_tokens=1)
                    return True, f"ok ({name})"
            return True, "llm client built (no probe method)"
        except Exception as exc:
            return False, str(exc)

    return await asyncio.to_thread(_probe)


async def _check_duckdb(settings: Settings) -> tuple[bool, str]:
    """Probe DuckDB with a trivial write."""

    def _probe() -> tuple[bool, str]:
        try:
            import duckdb

            path: Path = settings.trace_db_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(path)) as conn:
                conn.execute("CREATE TEMP TABLE IF NOT EXISTS _health_probe (x INTEGER)")
                conn.execute("INSERT INTO _health_probe VALUES (1)")
                conn.execute("DROP TABLE _health_probe")
            return True, "ok"
        except Exception as exc:
            return False, str(exc)

    return await asyncio.to_thread(_probe)


def _check_disk(settings: Settings) -> tuple[bool, str]:
    """Probe free space on the trace DB partition."""
    try:
        target = settings.trace_db_path.parent
        target.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(target)
        ok = usage.free >= MIN_FREE_DISK_BYTES
        return ok, f"free={usage.free} bytes"
    except Exception as exc:
        return False, str(exc)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Deep health check (Qdrant, LLM, DuckDB, disk)",
)
async def health_endpoint(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """Run all probes and return 200 if healthy, 503 if degraded."""
    qdrant_task = asyncio.create_task(_check_qdrant(settings))
    llm_task = asyncio.create_task(_check_llm(settings))
    duckdb_task = asyncio.create_task(_check_duckdb(settings))

    qdrant_ok, qdrant_msg = await qdrant_task
    llm_ok, llm_msg = await llm_task
    duckdb_ok, duckdb_msg = await duckdb_task
    disk_ok, disk_msg = _check_disk(settings)

    components = ComponentStatus(
        qdrant=qdrant_ok,
        llm=llm_ok,
        duckdb=duckdb_ok,
        disk=disk_ok,
    )
    all_ok = qdrant_ok and llm_ok and duckdb_ok and disk_ok
    payload = HealthResponse(
        status="ok" if all_ok else "degraded",
        components=components,
        details={
            "qdrant": qdrant_msg,
            "llm": llm_msg,
            "duckdb": duckdb_msg,
            "disk": disk_msg,
        },
    )
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning("health.degraded", extra=payload.model_dump())
    return payload
