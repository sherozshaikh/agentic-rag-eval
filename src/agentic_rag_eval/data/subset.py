from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)

ANSWER_LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("short", 0, 1),
    ("medium", 2, 3),
    ("long", 4, 7),
    ("xlong", 8, 10_000),
)


@dataclass(frozen=True)
class StratifiedSubsetResult:
    """Sampled subset plus per-stratum counts for the source and sample."""

    subset: pd.DataFrame
    strata_counts: dict[tuple[str, str, str], int]
    full_counts: dict[tuple[str, str, str], int]
    size: int
    seed: int

    def to_report(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the subset strata."""
        return {
            "size": self.size,
            "seed": self.seed,
            "num_strata": len(self.strata_counts),
            "strata": [
                {
                    "type": key[0],
                    "level": key[1],
                    "length_bucket": key[2],
                    "subset_count": self.strata_counts[key],
                    "full_count": self.full_counts.get(key, 0),
                }
                for key in sorted(self.strata_counts)
            ],
        }


def _answer_token_count(answer: Any) -> int:
    if answer is None:
        return 0
    if not isinstance(answer, str):
        answer = str(answer)
    return len(answer.split())


def _length_bucket(token_count: int) -> str:
    for name, lo, hi in ANSWER_LENGTH_BUCKETS:
        if lo <= token_count <= hi:
            return name
    return ANSWER_LENGTH_BUCKETS[-1][0]


def _ensure_strata_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the answer_token_count and length_bucket helper columns."""
    out = df.copy()
    if "answer_token_count" not in out.columns:
        out["answer_token_count"] = out["answer"].map(_answer_token_count)
    if "length_bucket" not in out.columns:
        out["length_bucket"] = out["answer_token_count"].map(_length_bucket)
    return out


def stratified_subset(
    dataset: pd.DataFrame,
    size: int,
    seed: int = 42,
) -> StratifiedSubsetResult:
    """Draw a stratified sample preserving joint type/level/length_bucket mass."""
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if len(dataset) == 0:
        raise ValueError("stratified_subset called on an empty dataset")
    if size > len(dataset):
        raise ValueError(f"size={size} is larger than the source dataset ({len(dataset)} rows)")

    for required in ("type", "level", "answer"):
        if required not in dataset.columns:
            raise KeyError(f"dataset is missing required column {required!r}")

    df = _ensure_strata_columns(dataset)
    total = len(df)

    grouped = df.groupby(["type", "level", "length_bucket"], sort=True)

    full_counts: dict[tuple[str, str, str], int] = {key: len(g) for key, g in grouped}

    raw_allocations: dict[tuple[str, str, str], float] = {
        key: size * (count / total) for key, count in full_counts.items()
    }
    floor_allocations: dict[tuple[str, str, str], int] = {
        key: int(np.floor(v)) for key, v in raw_allocations.items()
    }
    remainder = size - sum(floor_allocations.values())
    remainder_order = sorted(
        raw_allocations.keys(),
        key=lambda k: (raw_allocations[k] - floor_allocations[k], full_counts[k]),
        reverse=True,
    )
    allocations = dict(floor_allocations)
    for key in remainder_order:
        if remainder <= 0:
            break
        allocations[key] += 1
        remainder -= 1

    shortfall = 0
    for key, alloc in list(allocations.items()):
        available = full_counts[key]
        if alloc > available:
            shortfall += alloc - available
            allocations[key] = available

    if shortfall > 0:
        redistribute_order = sorted(
            allocations.keys(),
            key=lambda k: full_counts[k] - allocations[k],
            reverse=True,
        )
        for key in redistribute_order:
            if shortfall <= 0:
                break
            room = full_counts[key] - allocations[key]
            if room <= 0:
                continue
            take = min(room, shortfall)
            allocations[key] += take
            shortfall -= take

    if shortfall > 0:
        raise RuntimeError(
            f"Could not satisfy size={size}: shortfall of {shortfall} rows remaining."
        )

    rng = np.random.default_rng(seed)
    sampled_frames: list[pd.DataFrame] = []
    strata_counts: dict[tuple[str, str, str], int] = {}

    for key, group in grouped:
        n = allocations.get(key, 0)
        if n <= 0:
            strata_counts[key] = 0
            continue
        if n >= len(group):
            sampled = group
        else:
            idx = rng.choice(len(group), size=n, replace=False)
            sampled = group.iloc[np.sort(idx)]
        sampled_frames.append(sampled)
        strata_counts[key] = len(sampled)

    subset = pd.concat(sampled_frames, axis=0).reset_index(drop=True)

    logger.info(
        "Built stratified subset",
        extra={
            "requested_size": size,
            "actual_size": len(subset),
            "num_strata": len(strata_counts),
            "seed": seed,
        },
    )

    return StratifiedSubsetResult(
        subset=subset,
        strata_counts=strata_counts,
        full_counts=full_counts,
        size=size,
        seed=seed,
    )
