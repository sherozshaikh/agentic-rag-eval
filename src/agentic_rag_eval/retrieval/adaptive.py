from __future__ import annotations

import time
from typing import TYPE_CHECKING

from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.retrieval.reranker import Reranker
from agentic_rag_eval.retrieval.retriever import Retriever
from agentic_rag_eval.retrieval.strategy_selector import StrategySelector

if TYPE_CHECKING:
    from agentic_rag_eval.schemas import Passage

logger = get_logger(__name__)


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

        start = time.perf_counter()

        strategy = self._selector.select(query)
        retrieval = self._retriever.retrieve(query, strategy=strategy, top_k=top_k)

        if not retrieval.passages:
            logger.info(
                "adaptive pipeline: no candidates",
                extra={"strategy": strategy.value, "query": query[:80]},
            )
            return []

        reranked = self._reranker.rerank(
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
                "final": len(reranked),
                "latency_ms": round(latency_ms, 2),
            },
        )
        return reranked
