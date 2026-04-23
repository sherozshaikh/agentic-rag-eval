from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.retrieval.embeddings import (
    DenseEmbedder,
    SparseEmbedder,
    SparseVector,
)
from agentic_rag_eval.schemas import Passage, RetrievalResult, RetrievalStrategy

if TYPE_CHECKING:
    from agentic_rag_eval.config import Settings

logger = get_logger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class Retriever:
    """Unified retriever over a Qdrant collection with named dense and sparse vectors."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        settings: Settings,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: SparseEmbedder | None = None,
    ) -> None:
        self._client = client
        self._collection = collection_name
        self._settings = settings
        self._dense = dense_embedder or DenseEmbedder(model_name=settings.dense_model)
        self._sparse = sparse_embedder or SparseEmbedder(model_name=settings.sparse_model)

    def retrieve(
        self,
        query: str,
        strategy: RetrievalStrategy,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve passages for `query` using the given strategy."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        start = time.perf_counter()
        try:
            if strategy is RetrievalStrategy.DENSE:
                points = self._dense_query(query, top_k)
            elif strategy is RetrievalStrategy.SPARSE:
                points = self._sparse_query(query, top_k)
            elif strategy is RetrievalStrategy.HYBRID:
                points = self._hybrid_query(query, top_k)
            else:
                raise ValueError(f"unknown retrieval strategy: {strategy!r}")
        except UnexpectedResponse as exc:
            logger.error(
                "qdrant retrieval failed",
                extra={
                    "collection": self._collection,
                    "strategy": strategy.value,
                    "error": str(exc),
                },
            )
            latency_ms = (time.perf_counter() - start) * 1000.0
            return RetrievalResult(
                query=query,
                strategy=strategy,
                passages=[],
                latency_ms=latency_ms,
            )

        passages = [self._point_to_passage(p, strategy) for p in points]
        latency_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "retrieval complete",
            extra={
                "collection": self._collection,
                "strategy": strategy.value,
                "top_k": top_k,
                "returned": len(passages),
                "latency_ms": round(latency_ms, 2),
            },
        )

        return RetrievalResult(
            query=query,
            strategy=strategy,
            passages=passages,
            latency_ms=latency_ms,
        )

    def _dense_query(self, query: str, top_k: int) -> list[Any]:
        """Run a dense vector query."""
        vec = self._dense.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return list(response.points)

    def _sparse_query(self, query: str, top_k: int) -> list[Any]:
        """Run a sparse vector query."""
        sv = self._sparse.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection,
            query=self._to_qdrant_sparse(sv),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )
        return list(response.points)

    def _hybrid_query(self, query: str, top_k: int) -> list[Any]:
        """Run a hybrid query fused server-side via Qdrant RRF."""
        dense_vec = self._dense.embed_query(query)
        sparse_vec = self._sparse.embed_query(query)

        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using=DENSE_VECTOR_NAME,
                limit=top_k,
            ),
            models.Prefetch(
                query=self._to_qdrant_sparse(sparse_vec),
                using=SPARSE_VECTOR_NAME,
                limit=top_k,
            ),
        ]

        response = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return list(response.points)

    @staticmethod
    def _to_qdrant_sparse(sv: SparseVector) -> models.SparseVector:
        """Convert the transport `SparseVector` to the Qdrant model type."""
        return models.SparseVector(indices=sv.indices, values=sv.values)

    @staticmethod
    def _point_to_passage(point: Any, strategy: RetrievalStrategy) -> Passage:
        """Convert a Qdrant `ScoredPoint` into a `Passage`."""
        payload: dict[str, Any] = dict(getattr(point, "payload", None) or {})
        passage_id = str(payload.get("passage_id", getattr(point, "id", "")))
        title = payload.get("title")
        text = payload.get("text", "")
        score = float(getattr(point, "score", 0.0) or 0.0)

        return Passage(
            passage_id=passage_id,
            title=title,
            text=text,
            score=score,
            source_strategy=strategy,
        )
