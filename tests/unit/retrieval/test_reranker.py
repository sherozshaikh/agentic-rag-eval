from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest

from agentic_rag_eval.retrieval.reranker import Reranker
from agentic_rag_eval.schemas import Passage, RetrievalStrategy


class _FakeCrossEncoder:
    """Deterministic stand-in for `sentence_transformers.CrossEncoder`.

    Returns a caller-provided score list on `.predict()`.
    """

    def __init__(self, scores: Sequence[float]) -> None:
        self._scores = list(scores)
        self.predict_called_with: list[list[str]] | None = None

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.predict_called_with = pairs
        return list(self._scores)


def _make_passages(n: int) -> list[Passage]:
    return [
        Passage(
            passage_id=f"p{i}",
            title=f"Title {i}",
            text=f"content {i}",
            score=float(i) * 0.1,
            source_strategy=RetrievalStrategy.HYBRID,
        )
        for i in range(n)
    ]


class TestRerank:
    def test_returns_empty_for_empty_input(self) -> None:
        reranker = Reranker()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_rejects_non_positive_top_k(self) -> None:
        reranker = Reranker()
        with pytest.raises(ValueError):
            reranker.rerank("query", _make_passages(3), top_k=0)

    def test_scores_assigned_and_sorted(self) -> None:
        passages = _make_passages(4)
        fake = _FakeCrossEncoder([0.1, 0.9, 0.5, 0.7])
        reranker = Reranker()

        with patch.object(reranker, "_ensure_loaded", return_value=fake):
            result = reranker.rerank("q", passages, top_k=3)

        assert [p.passage_id for p in result] == ["p1", "p3", "p2"]
        assert result[0].rerank_score == pytest.approx(0.9)
        assert result[1].rerank_score == pytest.approx(0.7)
        assert result[2].rerank_score == pytest.approx(0.5)

        assert all(p.rerank_score is not None for p in passages)

    def test_top_k_larger_than_input_returns_all(self) -> None:
        passages = _make_passages(2)
        fake = _FakeCrossEncoder([0.2, 0.8])
        reranker = Reranker()

        with patch.object(reranker, "_ensure_loaded", return_value=fake):
            result = reranker.rerank("q", passages, top_k=10)

        assert [p.passage_id for p in result] == ["p1", "p0"]

    def test_predict_receives_query_and_text_pairs(self) -> None:
        passages = _make_passages(2)
        fake = _FakeCrossEncoder([0.1, 0.2])
        reranker = Reranker()

        with patch.object(reranker, "_ensure_loaded", return_value=fake):
            reranker.rerank("my query", passages, top_k=2)

        assert fake.predict_called_with == [
            ["my query", "content 0"],
            ["my query", "content 1"],
        ]

    def test_lazy_model_load(self) -> None:
        """`_ensure_loaded` should only be hit the first time `.rerank` runs."""
        reranker = Reranker()
        fake = _FakeCrossEncoder([0.5])
        with patch(
            "agentic_rag_eval.retrieval.reranker.Reranker._ensure_loaded",
            return_value=fake,
        ) as m:
            reranker.rerank("q", _make_passages(1), top_k=1)
            reranker.rerank("q", _make_passages(1), top_k=1)
            assert m.call_count == 2

        r2 = Reranker()
        assert r2._model is None


class TestLiftMetric:
    def test_lift_with_gold_post_improves(self) -> None:
        pre = [
            Passage(passage_id="a", text="a"),
            Passage(passage_id="b", text="b"),
        ]
        post = [
            Passage(passage_id="gold1", text="g", rerank_score=0.9),
            Passage(passage_id="a", text="a", rerank_score=0.1),
        ]
        lift = Reranker.lift_metric(pre, post, gold_ids=["gold1"])
        assert lift == pytest.approx(1.0)

    def test_lift_with_gold_no_change(self) -> None:
        pre = [Passage(passage_id="gold1", text="g")]
        post = [Passage(passage_id="gold1", text="g", rerank_score=0.9)]
        assert Reranker.lift_metric(pre, post, gold_ids=["gold1"]) == 0.0

    def test_lift_with_gold_degradation(self) -> None:
        pre = [Passage(passage_id="gold1", text="g")]
        post = [Passage(passage_id="x", text="x", rerank_score=0.1)]
        assert Reranker.lift_metric(pre, post, gold_ids=["gold1"]) == -1.0

    def test_lift_without_gold_uses_score_mean(self) -> None:
        pre = [
            Passage(passage_id="a", text="a", score=0.1),
            Passage(passage_id="b", text="b", score=0.2),
        ]
        post = [
            Passage(passage_id="a", text="a", score=0.1, rerank_score=0.5),
            Passage(passage_id="b", text="b", score=0.2, rerank_score=0.9),
        ]
        lift = Reranker.lift_metric(pre, post)

        assert lift == pytest.approx(0.55)

    def test_lift_empty_inputs_zero(self) -> None:
        assert Reranker.lift_metric([], []) == 0.0
