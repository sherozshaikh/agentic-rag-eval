from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import QueryRequest, QueryResponse
from agentic_rag_eval.serving.auth import verify_api_key
from agentic_rag_eval.serving.deps import get_react_agent, get_trace_logger_dep
from agentic_rag_eval.serving.validators import ValidationError, validate_question
from agentic_rag_eval.tracing import TraceLogger

logger = get_logger(__name__)
router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Answer a question using the agentic RAG pipeline",
)
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    agent: Any = Depends(get_react_agent),
    tracer: TraceLogger = Depends(get_trace_logger_dep),
) -> QueryResponse:
    """Run a single question through the ReAct agent inside a trace."""
    try:
        cleaned = validate_question(body.question)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    started = time.perf_counter()
    try:
        with tracer.trace(cleaned, pipeline="agentic_serving") as trace_ctx:
            response: QueryResponse = agent.answer(
                question=cleaned,
                user_id=body.user_id,
                trace_ctx=trace_ctx,
            )
            if not response.trace_id:
                response.trace_id = trace_ctx.trace_id
            if not response.latency_ms:
                response.latency_ms = (time.perf_counter() - started) * 1000.0
            return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "query.agent_failure",
            extra={
                "client": request.client.host if request.client else None,
                "question_preview": cleaned[:200],
                "error": str(exc),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "agent_failure",
                "message": "The agent failed to produce an answer.",
                "type": exc.__class__.__name__,
            },
        ) from exc
