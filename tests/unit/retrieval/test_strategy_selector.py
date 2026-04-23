from __future__ import annotations

import pytest

from agentic_rag_eval.retrieval.strategy_selector import StrategySelector
from agentic_rag_eval.schemas import Passage, RetrievalResult, RetrievalStrategy


@pytest.fixture
def selector() -> StrategySelector:
    return StrategySelector()


class TestSelectSparse:
    """Questions with entities / dates / numbers should pick SPARSE."""

    def test_named_entity_two_words(self, selector: StrategySelector) -> None:
        assert selector.select("Who directed Pulp Fiction") == RetrievalStrategy.SPARSE

    def test_named_entity_with_filler(self, selector: StrategySelector) -> None:
        assert (
            selector.select("What is the address of Bank of America headquarters")
            == RetrievalStrategy.SPARSE
        )

    def test_year_triggers_sparse(self, selector: StrategySelector) -> None:
        assert (
            selector.select("which team won the championship in 1998") == RetrievalStrategy.SPARSE
        )

    def test_number_triggers_sparse(self, selector: StrategySelector) -> None:
        assert selector.select("how tall is a person at 180 cm") == RetrievalStrategy.SPARSE

    def test_date_triggers_sparse(self, selector: StrategySelector) -> None:
        assert selector.select("what happened on 07/04/1776") == RetrievalStrategy.SPARSE


class TestSelectDense:
    """Short or obviously conceptual questions should pick DENSE."""

    def test_short_question(self, selector: StrategySelector) -> None:
        assert selector.select("meaning of life") == RetrievalStrategy.DENSE

    def test_conceptual_what_is(self, selector: StrategySelector) -> None:
        assert (
            selector.select("what is the meaning of epistemology in philosophy")
            == RetrievalStrategy.DENSE
        )

    def test_conceptual_why(self, selector: StrategySelector) -> None:
        assert selector.select("why do people dream during sleep") == RetrievalStrategy.DENSE

    def test_conceptual_how_does(self, selector: StrategySelector) -> None:
        assert selector.select("how does photosynthesis work in plants") == RetrievalStrategy.DENSE


class TestSelectHybrid:
    """Everything else falls through to HYBRID."""

    def test_generic_question(self, selector: StrategySelector) -> None:
        assert (
            selector.select("tell me about the best places to visit for a holiday")
            == RetrievalStrategy.HYBRID
        )

    def test_empty_string_defaults_to_hybrid(self, selector: StrategySelector) -> None:
        assert selector.select("") == RetrievalStrategy.HYBRID

    def test_whitespace_defaults_to_hybrid(self, selector: StrategySelector) -> None:
        assert selector.select("   ") == RetrievalStrategy.HYBRID


class TestOracleCompare:
    """Oracle comparison must identify the best strategy and match flag."""

    @staticmethod
    def _result(strategy: RetrievalStrategy, ids: list[str]) -> RetrievalResult:
        return RetrievalResult(
            query="q",
            strategy=strategy,
            passages=[
                Passage(passage_id=pid, text=f"text-{pid}", source_strategy=strategy) for pid in ids
            ],
            latency_ms=1.0,
        )

    def test_sparse_wins_and_selector_matches(self, selector: StrategySelector) -> None:
        sub_q = "Who founded Microsoft Corporation"
        gold = ["g1", "g2"]
        all_results = {
            RetrievalStrategy.DENSE: self._result(RetrievalStrategy.DENSE, ["x"]),
            RetrievalStrategy.SPARSE: self._result(RetrievalStrategy.SPARSE, ["g1", "g2"]),
            RetrievalStrategy.HYBRID: self._result(RetrievalStrategy.HYBRID, ["g1"]),
        }
        report = selector.oracle_compare(sub_q, all_results, gold)
        assert report["best"] == RetrievalStrategy.SPARSE
        assert report["selected"] == RetrievalStrategy.SPARSE
        assert report["match"] is True
        assert report["hits"] == {"dense": 0, "sparse": 2, "hybrid": 1}

    def test_selector_mismatch_reported(self, selector: StrategySelector) -> None:
        sub_q = "why photosynthesis"
        gold = ["g1"]
        all_results = {
            RetrievalStrategy.DENSE: self._result(RetrievalStrategy.DENSE, ["x"]),
            RetrievalStrategy.SPARSE: self._result(RetrievalStrategy.SPARSE, ["g1"]),
            RetrievalStrategy.HYBRID: self._result(RetrievalStrategy.HYBRID, ["y"]),
        }
        report = selector.oracle_compare(sub_q, all_results, gold)
        assert report["selected"] == RetrievalStrategy.DENSE
        assert report["best"] == RetrievalStrategy.SPARSE
        assert report["match"] is False
