from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentic_rag_eval.config import get_settings
from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.retrieval.reranker import Reranker
from agentic_rag_eval.retrieval.retriever import Retriever
from agentic_rag_eval.retrieval.strategy_selector import StrategySelector
from agentic_rag_eval.schemas import RetrievalStrategy

if TYPE_CHECKING:
    from agentic_rag_eval.schemas import Passage

logger = get_logger(__name__)

_FORCE_STRATEGY_MAP: dict[str, RetrievalStrategy] = {
    "dense": RetrievalStrategy.DENSE,
    "sparse": RetrievalStrategy.SPARSE,
    "hybrid": RetrievalStrategy.HYBRID,
}


class AdaptiveRetriever:
    """Strategy-aware retrieve-and-rerank pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        selector: StrategySelector | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._selector = selector or StrategySelector()
        self._reranker = reranker or Reranker()

    def retrieve_and_rerank(
        self,
        query: str,
        top_k: int = 20,
        rerank_top_k: int = 5,
    ) -> list[Passage]:
        """Run the adaptive pipeline and return re-ranked passages."""
        if rerank_top_k > top_k:
            logger.warning(
                "rerank_top_k exceeds top_k, clamping",
                extra={"top_k": top_k, "rerank_top_k": rerank_top_k},
            )
            rerank_top_k = top_k

        settings = get_settings()
        start = time.perf_counter()

        # Ablation: force a fixed retrieval strategy instead of adaptive selection
        forced = _FORCE_STRATEGY_MAP.get(settings.ablation_force_strategy.lower())
        strategy = forced if forced is not None else self._selector.select(query)

        retrieval = self._retriever.retrieve(query, strategy=strategy, top_k=top_k)

        if not retrieval.passages:
            logger.info(
                "adaptive pipeline: no candidates",
                extra={"strategy": strategy.value, "query": query[:80]},
            )
            return []

        # Ablation: skip cross-encoder reranker, return raw retrieval top-k
        if settings.ablation_no_reranker:
            result = retrieval.passages[:rerank_top_k]
        else:
            result = self._reranker.rerank(
                query=query,
                passages=retrieval.passages,
                top_k=rerank_top_k,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "adaptive pipeline complete",
            extra={
                "strategy": strategy.value,
                "first_stage": len(retrieval.passages),
                "final": len(result),
                "latency_ms": round(latency_ms, 2),
                "no_reranker": settings.ablation_no_reranker,
                "force_strategy": settings.ablation_force_strategy or "adaptive",
            },
        )
        return result
