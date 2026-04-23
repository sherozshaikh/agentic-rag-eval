from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import Passage

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder re-ranker over `Passage` objects."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _ensure_loaded(self) -> Any:
        if self._model is None:
            logger.info("loading cross-encoder reranker", extra={"model": self.model_name})
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            logger.info("cross-encoder reranker loaded", extra={"model": self.model_name})
        return self._model

    def rerank(
        self,
        query: str,
        passages: Sequence[Passage],
        top_k: int = 5,
    ) -> list[Passage]:
        """Re-rank `passages` against `query` and return the top-k.

        Mutates each passage's `rerank_score` in place.
        """
        if not passages:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        start = time.perf_counter()
        model = self._ensure_loaded()

        pairs = [[query, p.text] for p in passages]
        raw_scores = model.predict(pairs)
        scores = [float(s) for s in raw_scores]

        for passage, score in zip(passages, scores, strict=True):
            passage.rerank_score = score

        ranked = sorted(
            passages,
            key=lambda p: (p.rerank_score if p.rerank_score is not None else float("-inf")),
            reverse=True,
        )[:top_k]

        latency_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "rerank complete",
            extra={
                "input_count": len(passages),
                "top_k": top_k,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return ranked

    @staticmethod
    def lift_metric(
        pre: Sequence[Passage],
        post: Sequence[Passage],
        gold_ids: Sequence[str] | None = None,
    ) -> float:
        """Return a re-ranker lift metric; positive means the re-ranker helped.

        With `gold_ids`, returns `hit_rate(post) - hit_rate(pre)`. Without,
        returns the delta in mean score as a rough proxy.
        """
        if gold_ids is not None:
            gold_set = set(gold_ids)
            pre_hit = 1.0 if any(p.passage_id in gold_set for p in pre) else 0.0
            post_hit = 1.0 if any(p.passage_id in gold_set for p in post) else 0.0
            return post_hit - pre_hit

        if not pre or not post:
            return 0.0
        pre_mean = sum(p.score for p in pre) / len(pre)
        post_mean = sum(
            (p.rerank_score if p.rerank_score is not None else p.score) for p in post
        ) / len(post)
        return post_mean - pre_mean
