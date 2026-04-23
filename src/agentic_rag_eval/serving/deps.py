from __future__ import annotations

import threading
from functools import lru_cache
from typing import Any

from fastapi import HTTPException, status

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.tracing import TraceLogger, get_trace_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_retriever: Any | None = None
_adaptive_retriever: Any | None = None
_react_agent: Any | None = None
_baseline_rag: Any | None = None


def _service_unavailable(component: str, exc: Exception) -> HTTPException:
    """Log and return a 503 HTTPException for an unavailable component."""
    logger.error(
        "deps.component_unavailable",
        extra={"component": component, "error": str(exc)},
    )
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"{component} is not available: {exc}",
    )


@lru_cache(maxsize=1)
def get_settings_cached() -> Settings:
    """Cached settings provider for FastAPI ``Depends``."""
    return get_settings()


def get_trace_logger_dep() -> TraceLogger:
    """Return the process-wide TraceLogger singleton."""
    return get_trace_logger()


def get_retriever() -> Any:
    """Return the cached Retriever singleton."""
    global _retriever
    with _lock:
        if _retriever is not None:
            return _retriever
        try:
            from agentic_rag_eval.retrieval import Retriever
        except Exception as exc:
            raise _service_unavailable("retriever", exc) from exc
        try:
            _retriever = Retriever(settings=get_settings())
        except Exception as exc:
            raise _service_unavailable("retriever", exc) from exc
        return _retriever


def get_adaptive_retriever() -> Any:
    """Return the cached AdaptiveRetriever singleton."""
    global _adaptive_retriever
    with _lock:
        if _adaptive_retriever is not None:
            return _adaptive_retriever
        try:
            from agentic_rag_eval.retrieval import AdaptiveRetriever
        except Exception as exc:
            raise _service_unavailable("adaptive_retriever", exc) from exc
        try:
            _adaptive_retriever = AdaptiveRetriever(
                retriever=get_retriever(),
                settings=get_settings(),
            )
        except Exception as exc:
            raise _service_unavailable("adaptive_retriever", exc) from exc
        return _adaptive_retriever


def get_react_agent() -> Any:
    """Return the cached ReActAgent singleton."""
    global _react_agent
    with _lock:
        if _react_agent is not None:
            return _react_agent
        try:
            from agentic_rag_eval.agent import ReActAgent
            from agentic_rag_eval.llm import build_llm_client
        except Exception as exc:
            raise _service_unavailable("react_agent", exc) from exc
        try:
            _react_agent = ReActAgent(
                retriever=get_adaptive_retriever(),
                llm_client=build_llm_client(get_settings()),
                settings=get_settings(),
            )
        except Exception as exc:
            raise _service_unavailable("react_agent", exc) from exc
        return _react_agent


def get_baseline_rag() -> Any:
    """Return the cached BaselineRAG singleton."""
    global _baseline_rag
    with _lock:
        if _baseline_rag is not None:
            return _baseline_rag
        try:
            from agentic_rag_eval.baseline import BaselineRAG
            from agentic_rag_eval.llm import build_llm_client
        except Exception as exc:
            raise _service_unavailable("baseline_rag", exc) from exc
        try:
            _baseline_rag = BaselineRAG(
                retriever=get_retriever(),
                llm_client=build_llm_client(get_settings()),
                settings=get_settings(),
            )
        except Exception as exc:
            raise _service_unavailable("baseline_rag", exc) from exc
        return _baseline_rag


def reset_singletons() -> None:
    """Clear all cached singletons (test helper)."""
    global _retriever, _adaptive_retriever, _react_agent, _baseline_rag
    with _lock:
        _retriever = None
        _adaptive_retriever = None
        _react_agent = None
        _baseline_rag = None
    get_settings_cached.cache_clear()
