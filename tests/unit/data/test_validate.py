from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_rag_eval.data.subset import stratified_subset
from agentic_rag_eval.data.validate import ChiSquaredReport, chi_squared_validate


def _make_dataset(
    n: int,
    type_probs: tuple[float, float] = (0.8, 0.2),
    level_probs: tuple[float, float, float] = (0.3, 0.4, 0.3),
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["bridge", "comparison"], size=n, p=list(type_probs))
    levels = rng.choice(["easy", "medium", "hard"], size=n, p=list(level_probs))
    length_choices = np.arange(1, 10)
    length_probs = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.08, 0.06, 0.04, 0.02])
    length_probs = length_probs / length_probs.sum()
    token_counts = rng.choice(length_choices, size=n, p=length_probs)
    answers = [" ".join(["t"] * int(tc)) for tc in token_counts]
    return pd.DataFrame(
        {
            "_id": [f"q{i:05d}" for i in range(n)],
            "question": [f"q{i}" for i in range(n)],
            "answer": answers,
            "type": types,
            "level": levels,
            "supporting_facts": [{"title": [], "sent_id": []}] * n,
            "context": [{"title": [], "sentences": []}] * n,
        }
    )


def test_chi_squared_validate_passes_for_stratified_subset() -> None:
    full = _make_dataset(n=4000, seed=1)
    result = stratified_subset(full, size=800, seed=1)
    report = chi_squared_validate(result.subset, full)
    assert isinstance(report, ChiSquaredReport)
    assert report.passed is True
    assert report.type_p_value > 0.05
    assert report.level_p_value > 0.05
    assert report.length_bucket_p_value > 0.05


def test_chi_squared_validate_fails_for_biased_subset() -> None:
    full = _make_dataset(n=4000, seed=2)
    biased = full[(full["type"] == "bridge") & (full["level"] == "easy")].head(500)
    assert len(biased) > 0
    with pytest.raises(AssertionError):
        chi_squared_validate(biased, full, assert_pass=True)


def test_chi_squared_validate_report_contains_all_marginals() -> None:
    full = _make_dataset(n=2000, seed=3)
    result = stratified_subset(full, size=400, seed=3)
    report = chi_squared_validate(result.subset, full, assert_pass=False)
    assert "type" in report.details
    assert "level" in report.details
    assert "length_bucket" in report.details
    assert report.details["subset_size"] == 400
    assert report.details["full_size"] == 2000


def test_chi_squared_validate_assert_false_never_raises() -> None:
    full = _make_dataset(n=2000, seed=4)
    biased = full[full["type"] == "comparison"].head(200)
    report = chi_squared_validate(biased, full, assert_pass=False)
    assert report.passed is False


def test_chi_squared_validate_rejects_empty_subset() -> None:
    full = _make_dataset(n=100, seed=0)
    empty = full.iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        chi_squared_validate(empty, full)


def test_chi_squared_validate_rejects_empty_full() -> None:
    full = _make_dataset(n=100, seed=0)
    empty = full.iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        chi_squared_validate(full, empty)


def test_chi_squared_validate_rejects_missing_columns() -> None:
    full = _make_dataset(n=100, seed=0)
    bad = full.drop(columns=["level"])
    with pytest.raises(KeyError, match="level"):
        chi_squared_validate(bad, full)


def test_chi_squared_report_serializable() -> None:
    import json

    full = _make_dataset(n=1000, seed=5)
    result = stratified_subset(full, size=200, seed=5)
    report = chi_squared_validate(result.subset, full, assert_pass=False)
    payload = json.dumps(report.as_dict())
    decoded = json.loads(payload)
    assert "type_p_value" in decoded
    assert "level_p_value" in decoded
    assert "length_bucket_p_value" in decoded
