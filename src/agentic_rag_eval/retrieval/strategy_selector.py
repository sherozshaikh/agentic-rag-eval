from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agentic_rag_eval.logging_setup import get_logger
from agentic_rag_eval.schemas import RetrievalStrategy

if TYPE_CHECKING:
    from agentic_rag_eval.schemas import RetrievalResult

logger = get_logger(__name__)


_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+(?:of|de|la|von|van)\s+|\s+)[A-Z][a-zA-Z]+\b")

_DATE_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")

_CONCEPTUAL_PREFIXES = (
    "what is",
    "what are",
    "why",
    "how does",
    "how do",
    "explain",
    "describe",
    "define",
)


class StrategySelector:
    """Rule-based retrieval strategy selector. Stateless and thread-safe."""

    def __init__(self, short_token_threshold: int = 5) -> None:
        self.short_token_threshold = short_token_threshold

    def select(self, sub_question: str) -> RetrievalStrategy:
        """Pick a retrieval strategy for `sub_question`."""
        if not sub_question or not sub_question.strip():
            return RetrievalStrategy.HYBRID

        text = sub_question.strip()
        lowered = text.lower()
        tokens = text.split()

        has_entity = bool(_ENTITY_RE.search(text))
        has_date = bool(_DATE_RE.search(text))
        has_number = bool(_NUMBER_RE.search(text))
        is_short = len(tokens) < self.short_token_threshold
        is_conceptual = any(lowered.startswith(p) for p in _CONCEPTUAL_PREFIXES)

        if has_entity or has_date or has_number:
            strategy = RetrievalStrategy.SPARSE
        elif is_short or is_conceptual:
            strategy = RetrievalStrategy.DENSE
        else:
            strategy = RetrievalStrategy.HYBRID

        logger.debug(
            "strategy selected",
            extra={
                "sub_question": text,
                "strategy": strategy.value,
                "has_entity": has_entity,
                "has_date": has_date,
                "has_number": has_number,
                "is_short": is_short,
                "is_conceptual": is_conceptual,
            },
        )
        return strategy

    def oracle_compare(
        self,
        sub_question: str,
        all_results: dict[RetrievalStrategy, RetrievalResult],
        gold_ids: list[str],
    ) -> dict[str, object]:
        """Compare the selector's pick to the oracle-best strategy."""
        gold_set = set(gold_ids)
        hits: dict[str, int] = {}
        for strat, result in all_results.items():
            hits[strat.value] = sum(1 for p in result.passages if p.passage_id in gold_set)

        if hits:
            order = [
                RetrievalStrategy.DENSE,
                RetrievalStrategy.SPARSE,
                RetrievalStrategy.HYBRID,
            ]
            best = max(
                (s for s in order if s in all_results),
                key=lambda s: hits.get(s.value, 0),
            )
        else:
            best = RetrievalStrategy.HYBRID

        selected = self.select(sub_question)
        return {
            "selected": selected,
            "best": best,
            "match": selected == best,
            "hits": hits,
        }
