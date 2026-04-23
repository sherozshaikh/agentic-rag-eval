from __future__ import annotations

import math

import pytest

from agentic_rag_eval.evaluation.hotpotqa_metrics import (
    exact_match,
    f1_score,
    normalize_answer,
    supporting_fact_f1,
)


class TestNormalizeAnswer:
    def test_lowercases(self) -> None:
        assert normalize_answer("Hello World") == "hello world"

    def test_strips_articles(self) -> None:
        assert normalize_answer("The cat") == "cat"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("An apple") == "apple"

    def test_strips_punctuation(self) -> None:
        assert normalize_answer("Hello, world!") == "hello world"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    def test_combined(self) -> None:
        assert normalize_answer("The Quick, Brown Fox!!") == "quick brown fox"

    def test_none(self) -> None:
        assert normalize_answer(None) == ""

    def test_article_in_word_not_stripped(self) -> None:
        assert normalize_answer("theatre") == "theatre"


class TestExactMatch:
    def test_identical(self) -> None:
        assert exact_match("Paris", "Paris") == 1.0

    def test_normalization_differences(self) -> None:
        assert exact_match("The Paris", "paris") == 1.0
        assert exact_match("Paris.", "paris") == 1.0

    def test_different(self) -> None:
        assert exact_match("Paris", "London") == 0.0

    def test_empty(self) -> None:
        assert exact_match("", "") == 1.0


class TestF1Score:
    def test_identical(self) -> None:
        assert f1_score("Barack Obama", "Barack Obama") == 1.0

    def test_partial_overlap(self) -> None:
        f1 = f1_score("Barack Obama", "Barack H Obama")
        assert math.isclose(f1, 0.8, abs_tol=1e-6)

    def test_no_overlap(self) -> None:
        assert f1_score("Paris", "London") == 0.0

    def test_yes_no_mismatch_is_zero(self) -> None:
        assert f1_score("yes", "no") == 0.0
        assert f1_score("no", "maybe") == 0.0

    def test_yes_yes_is_one(self) -> None:
        assert f1_score("yes", "yes") == 1.0

    def test_articles_ignored(self) -> None:
        assert f1_score("the Eiffel Tower", "Eiffel Tower") == 1.0


class TestSupportingFactF1:
    def test_perfect(self) -> None:
        gold = [["Alice", 0], ["Bob", 1]]
        pred = ["Alice", "Bob"]
        p, r, f1 = supporting_fact_f1(pred, gold)
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_recall_miss(self) -> None:
        gold = [["Alice", 0], ["Bob", 1]]
        pred = ["Alice"]
        p, r, f1 = supporting_fact_f1(pred, gold)
        assert p == 1.0
        assert r == 0.5
        assert math.isclose(f1, 2 / 3, abs_tol=1e-6)

    def test_precision_miss(self) -> None:
        gold = [["Alice", 0]]
        pred = ["Alice", "Carol"]
        p, r, f1 = supporting_fact_f1(pred, gold)
        assert p == 0.5
        assert r == 1.0
        assert math.isclose(f1, 2 / 3, abs_tol=1e-6)

    def test_no_overlap(self) -> None:
        p, r, f1 = supporting_fact_f1(["X"], [["Y", 0]])
        assert (p, r, f1) == (0.0, 0.0, 0.0)

    def test_both_empty(self) -> None:
        p, r, f1 = supporting_fact_f1([], [])
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_dict_facts(self) -> None:
        gold = [{"title": "Alice", "sent_id": 0}, {"title": "Bob", "sent_id": 1}]
        p, r, f1 = supporting_fact_f1(["Alice", "Bob"], gold)
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_normalization_matters(self) -> None:
        p, r, f1 = supporting_fact_f1(["The Alice"], [["alice", 0]])
        assert (p, r, f1) == (1.0, 1.0, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
