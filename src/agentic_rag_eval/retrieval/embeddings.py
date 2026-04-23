from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agentic_rag_eval.logging_setup import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


@dataclass
class SparseVector:
    """Transport object for a sparse vector."""

    indices: list[int]
    values: list[float]


class DenseEmbedder:
    """Lazy wrapper around FastEmbed `TextEmbedding`."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            logger.info("loading dense embedder", extra={"model": self.model_name})
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
            logger.info("dense embedder loaded", extra={"model": self.model_name})
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string into a dense vector."""
        model = self._ensure_loaded()
        vec = next(iter(model.query_embed([text])))
        return [float(x) for x in vec]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passage texts into dense vectors."""
        if not texts:
            return []
        model = self._ensure_loaded()
        out: list[list[float]] = []
        for vec in model.embed(list(texts)):
            out.append([float(x) for x in vec])
        return out


class SparseEmbedder:
    """Lazy wrapper around FastEmbed `SparseTextEmbedding`."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            logger.info("loading sparse embedder", extra={"model": self.model_name})
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(model_name=self.model_name)
            logger.info("sparse embedder loaded", extra={"model": self.model_name})
        return self._model

    @staticmethod
    def _to_sparse_vector(raw: Any) -> SparseVector:
        """Convert a FastEmbed `SparseEmbedding` to the transport type."""
        indices = [int(i) for i in raw.indices]
        values = [float(v) for v in raw.values]
        return SparseVector(indices=indices, values=values)

    def embed_query(self, text: str) -> SparseVector:
        """Embed a query string into a sparse vector."""
        model = self._ensure_loaded()
        raw = next(iter(model.query_embed([text])))
        return self._to_sparse_vector(raw)

    def embed_passages(self, texts: Sequence[str]) -> list[SparseVector]:
        """Embed a batch of passage texts into sparse vectors."""
        if not texts:
            return []
        model = self._ensure_loaded()
        return [self._to_sparse_vector(raw) for raw in model.embed(list(texts))]
