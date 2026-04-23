from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from agentic_rag_eval.config import Settings, get_settings
from agentic_rag_eval.logging_setup import configure_logging, get_logger
from agentic_rag_eval.serving.routes import dashboard, evaluate, health, metrics, query

logger = get_logger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TEMPLATES_DIR = _REPO_ROOT / "templates"
_DEFAULT_STATIC_DIR = _REPO_ROOT / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a fully-configured FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Best-effort warmup of logging, DuckDB, embeddings, and Qdrant."""
        logger.info("serving.startup", extra={"cors_origins": settings.cors_origin_list})

        try:
            from agentic_rag_eval.tracing import get_trace_logger

            get_trace_logger(settings)
        except Exception as exc:
            logger.warning("startup.tracing_init_failed", extra={"error": str(exc)})

        try:
            from qdrant_client import QdrantClient

            QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=3.0,
                check_compatibility=False,
            ).get_collections()
            logger.info("startup.qdrant_ready")
        except Exception as exc:
            logger.warning("startup.qdrant_not_ready", extra={"error": str(exc)})

        try:
            logger.info("startup.embeddings_module_available")
        except Exception:
            logger.debug("startup.embeddings_module_missing")

        yield

        logger.info("serving.shutdown")

    app = FastAPI(
        title="agentic-rag-eval",
        version="0.1.0",
        description=(
            "Multi-Hop Agentic RAG with Systematic Evaluation — "
            "HotpotQA benchmark with Qdrant, LangGraph, RAGAS, DeepEval."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit],
    )
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        logger.warning(
            "rate_limit.exceeded",
            extra={
                "client": request.client.host if request.client else None,
                "path": request.url.path,
            },
        )
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": "rate_limit_exceeded",
                "message": f"Rate limit exceeded: {exc.detail}",
            },
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    templates_dir = _DEFAULT_TEMPLATES_DIR
    static_dir = _DEFAULT_STATIC_DIR
    templates_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)

    app.state.templates = Jinja2Templates(directory=str(templates_dir))
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "app.unhandled_exception",
            extra={"path": request.url.path, "error": str(exc)},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred.",
                "type": exc.__class__.__name__,
            },
        )

    app.include_router(query.router)
    app.include_router(evaluate.router)
    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(dashboard.router)

    return app


app: FastAPI = create_app(get_settings())
