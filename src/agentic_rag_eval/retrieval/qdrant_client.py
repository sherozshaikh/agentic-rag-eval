from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from agentic_rag_eval.logging_setup import get_logger

if TYPE_CHECKING:
    from agentic_rag_eval.config import Settings

logger = get_logger(__name__)


def get_qdrant_client(settings: Settings) -> QdrantClient:
    """Build a `QdrantClient` from `Settings`."""
    logger.info(
        "initializing qdrant client",
        extra={
            "qdrant_host": settings.qdrant_host,
            "qdrant_port": settings.qdrant_port,
            "qdrant_grpc_port": settings.qdrant_grpc_port,
            "qdrant_use_grpc": settings.qdrant_use_grpc,
        },
    )

    if settings.qdrant_use_grpc:
        return QdrantClient(
            host=settings.qdrant_host,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=True,
            check_compatibility=False,
        )
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        check_compatibility=False,
    )


def collection_exists(client: QdrantClient, collection_name: str) -> bool:
    """Return True if `collection_name` exists on the Qdrant server."""
    try:
        return bool(client.collection_exists(collection_name))
    except (UnexpectedResponse, ValueError) as exc:
        logger.debug(
            "collection existence check failed",
            extra={"collection": collection_name, "error": str(exc)},
        )
        return False
    except Exception as exc:
        logger.warning(
            "unexpected error checking collection",
            extra={"collection": collection_name, "error": str(exc)},
        )
        return False
