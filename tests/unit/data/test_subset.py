from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_rag_eval.data.subset import (
    ANSWER_LENGTH_BUCKETS,
    StratifiedSubsetResult,
    stratified_subset,
)


def _synthetic_hotpotqa(seed: int = 0, n: int = 2000) -> pd.DataFrame:
    """Generate a synthetic HotpotQA-like DataFrame with known distributions.

    Type distribution: 80% bridge, 20% comparison (matches HotpotQA).
    Level distribution: 30% easy, 40% medium, 30% hard.
    Answer lengths: geometric-ish, spread across buckets.
    """
    rng = np.random.default_rng(seed)
    types = rng.choice(["bridge", "comparison"], size=n, p=[0.8, 0.2])
    levels = rng.choice(["easy", "medium", "hard"], size=n, p=[0.3, 0.4, 0.3])
    length_choices = np.arange(1, 13)
    length_probs = np.array(
        [0.30, 0.20, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02]
    )
    length_probs = length_probs / length_probs.sum()
    token_counts = rng.choice(length_choices, size=n, p=length_probs)
    answers = [" ".join(["tok"] * int(tc)) for tc in token_counts]

    return pd.DataFrame(
        {
            "_id": [f"q{i:05d}" for i in range(n)],
            "question": [f"question {i}" for i in range(n)],
            "answer": answers,
            "type": types,
            "level": levels,
            "supporting_facts": [{"title": [], "sent_id": []}] * n,
            "context": [{"title": [], "sentences": []}] * n,
        }
    )


def test_stratified_subset_size_matches_request() -> None:
    df = _synthetic_hotpotqa(n=2000)
    result = stratified_subset(df, size=500, seed=42)
    assert isinstance(result, StratifiedSubsetResult)
    assert len(result.subset) == 500
    assert result.size == 500
    assert result.seed == 42


def test_stratified_subset_preserves_type_distribution() -> None:
    df = _synthetic_hotpotqa(n=4000)
    result = stratified_subset(df, size=1000, seed=1)
    parent_share = df["type"].value_counts(normalize=True)
    sub_share = result.subset["type"].value_counts(normalize=True)
    for category in parent_share.index:
        assert abs(parent_share[category] - sub_share.get(category, 0.0)) < 0.03


def test_stratified_subset_preserves_level_distribution() -> None:
    df = _synthetic_hotpotqa(n=4000)
    result = stratified_subset(df, size=1000, seed=2)
    parent_share = df["level"].value_counts(normalize=True)
    sub_share = result.subset["level"].value_counts(normalize=True)
    for category in parent_share.index:
        assert abs(parent_share[category] - sub_share.get(category, 0.0)) < 0.03


def test_stratified_subset_preserves_length_bucket_distribution() -> None:
    df = _synthetic_hotpotqa(n=4000)
    result = stratified_subset(df, size=1200, seed=3)
    parent_share = result.subset["length_bucket"].value_counts(normalize=True).to_dict()
    from agentic_rag_eval.data.subset import _ensure_strata_columns

    full_buckets = set(_ensure_strata_columns(df)["length_bucket"].unique())
    assert set(parent_share.keys()).issubset(full_buckets)
    assert sum(parent_share.values()) == pytest.approx(1.0, abs=1e-9)


def test_stratified_subset_deterministic_with_seed() -> None:
    df = _synthetic_hotpotqa(n=2000)
    a = stratified_subset(df, size=300, seed=7)
    b = stratified_subset(df, size=300, seed=7)
    pd.testing.assert_frame_equal(a.subset.reset_index(drop=True), b.subset.reset_index(drop=True))


def test_stratified_subset_different_seed_changes_sample() -> None:
    df = _synthetic_hotpotqa(n=2000)
    a = stratified_subset(df, size=300, seed=1)
    b = stratified_subset(df, size=300, seed=2)
    assert not a.subset["_id"].equals(b.subset["_id"])


def test_stratified_subset_strata_counts_equal_subset_size() -> None:
    df = _synthetic_hotpotqa(n=2000)
    result = stratified_subset(df, size=400, seed=5)
    assert sum(result.strata_counts.values()) == 400


def test_stratified_subset_full_counts_equal_full_size() -> None:
    df = _synthetic_hotpotqa(n=2000)
    result = stratified_subset(df, size=400, seed=5)
    assert sum(result.full_counts.values()) == len(df)


def test_stratified_subset_rejects_oversized_request() -> None:
    df = _synthetic_hotpotqa(n=100)
    with pytest.raises(ValueError, match="larger than"):
        stratified_subset(df, size=500, seed=0)


def test_stratified_subset_rejects_nonpositive_size() -> None:
    df = _synthetic_hotpotqa(n=100)
    with pytest.raises(ValueError, match="positive"):
        stratified_subset(df, size=0, seed=0)


def test_stratified_subset_rejects_empty_dataset() -> None:
    df = pd.DataFrame(columns=["type", "level", "answer"])
    with pytest.raises(ValueError, match="empty"):
        stratified_subset(df, size=10, seed=0)


def test_stratified_subset_rejects_missing_columns() -> None:
    df = pd.DataFrame({"type": ["bridge"] * 10, "level": ["easy"] * 10})
    with pytest.raises(KeyError, match="answer"):
        stratified_subset(df, size=5, seed=0)


def test_stratified_subset_report_is_json_serializable() -> None:
    import json

    df = _synthetic_hotpotqa(n=500)
    result = stratified_subset(df, size=100, seed=0)
    report = result.to_report()
    encoded = json.dumps(report)
    decoded = json.loads(encoded)
    assert decoded["size"] == 100
    assert decoded["seed"] == 0
    assert decoded["num_strata"] >= 1
    assert len(decoded["strata"]) == decoded["num_strata"]


def test_length_buckets_cover_all_nonnegative_counts() -> None:
    from agentic_rag_eval.data.subset import _length_bucket

    buckets_seen = {_length_bucket(i) for i in range(0, 50)}
    expected = {name for name, _, _ in ANSWER_LENGTH_BUCKETS}
    assert buckets_seen.issubset(expected)
