from __future__ import annotations

import math

import pytest

from agentic_rag_eval.evaluation.retrieval_metrics import (
    mrr,
    ndcg,
    precision_at_k,
    recall_at_k,
    rerank_lift,
)


class TestRecallAtK:
    def test_all_gold_in_topk(self) -> None:
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_some_gold_missing(self) -> None:
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5

    def test_cutoff_exclusion(self) -> None:
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == 0.5

    def test_no_gold(self) -> None:
        assert recall_at_k(["a"], [], k=5) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            recall_at_k(["a"], ["a"], k=0)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b"], ["a", "b"], k=2) == 1.0

    def test_half_relevant(self) -> None:
        assert precision_at_k(["a", "x"], ["a"], k=2) == 0.5

    def test_cutoff_smaller_than_retrieved(self) -> None:
        assert precision_at_k(["a", "b", "c"], ["a"], k=1) == 1.0

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], ["a"], k=5) == 0.0


class TestMRR:
    def test_first_hit_is_one(self) -> None:
        assert mrr(["a", "b"], ["a"]) == 1.0

    def test_second_hit(self) -> None:
        assert mrr(["x", "a"], ["a"]) == 0.5

    def test_third_hit(self) -> None:
        assert math.isclose(mrr(["x", "y", "a"], ["a"]), 1 / 3)

    def test_no_hit(self) -> None:
        assert mrr(["x", "y"], ["a"]) == 0.0

    def test_no_gold(self) -> None:
        assert mrr(["a"], []) == 0.0


class TestNDCG:
    def test_perfect_ranking(self) -> None:
        val = ndcg(["a", "b", "c"], ["a", "b"], k=3)
        assert math.isclose(val, 1.0, abs_tol=1e-9)

    def test_reversed_still_same_ideal(self) -> None:
        val = ndcg(["b", "a", "c"], ["a", "b"], k=3)
        assert math.isclose(val, 1.0, abs_tol=1e-9)

    def test_penalized_for_late_relevance(self) -> None:
        val = ndcg(["x", "y", "a"], ["a"], k=3)
        assert math.isclose(val, 0.5, abs_tol=1e-9)

    def test_no_gold(self) -> None:
        assert ndcg(["a"], [], k=3) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            ndcg(["a"], ["a"], k=0)


class TestRerankLift:
    def test_positive_lift(self) -> None:
        pre = ["x", "y", "a"]
        post = ["a", "x", "y"]
        assert rerank_lift(pre, post, ["a"], k=2) == 1.0

    def test_no_lift(self) -> None:
        assert rerank_lift(["a"], ["a"], ["a"], k=1) == 0.0

    def test_negative_lift(self) -> None:
        pre = ["a", "b"]
        post = ["b", "a"]
        assert rerank_lift(pre, post, ["a"], k=1) == -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
