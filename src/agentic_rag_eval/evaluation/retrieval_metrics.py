from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)


def _to_set(items: Iterable[str]) -> set[str]:
    return {str(x) for x in items if x is not None and str(x) != ""}


def recall_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of gold items present in the top-k retrieved."""
    if k <= 0:
        raise ValueError("k must be > 0")
    gold = _to_set(gold_ids)
    if not gold:
        return 0.0
    topk = _to_set(retrieved_ids[:k])
    return len(gold & topk) / len(gold)


def precision_at_k(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of the top-k retrieved that are in the gold set."""
    if k <= 0:
        raise ValueError("k must be > 0")
    gold = _to_set(gold_ids)
    topk = [str(r) for r in retrieved_ids[:k] if r is not None]
    if not topk:
        return 0.0
    hits = sum(1 for r in topk if r in gold)
    return hits / len(topk)


def mrr(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
) -> float:
    """Reciprocal rank of the first relevant item (0.0 if none)."""
    gold = _to_set(gold_ids)
    if not gold:
        return 0.0
    for idx, rid in enumerate(retrieved_ids, start=1):
        if str(rid) in gold:
            return 1.0 / idx
    return 0.0


def ndcg(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> float:
    """NDCG@k with binary relevance."""
    if k <= 0:
        raise ValueError("k must be > 0")
    gold = _to_set(gold_ids)
    if not gold:
        return 0.0

    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if str(rid) in gold:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(gold), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def rerank_lift(
    pre: Sequence[str],
    post: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> float:
    """Absolute lift in Recall@k from re-ranking (post - pre)."""
    return recall_at_k(post, gold_ids, k) - recall_at_k(pre, gold_ids, k)
