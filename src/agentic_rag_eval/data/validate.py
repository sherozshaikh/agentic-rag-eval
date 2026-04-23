from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from agentic_rag_eval.data.subset import _ensure_strata_columns
from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_ALPHA = 0.05


@dataclass
class ChiSquaredReport:
    """Per-marginal p-values and pass/fail result of a chi-squared check."""

    type_p_value: float
    level_p_value: float
    length_bucket_p_value: float
    alpha: float = DEFAULT_ALPHA
    passed: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _observed_expected(
    subset: pd.Series,
    full: pd.Series,
    subset_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align observed and expected counts by category for chi-squared."""
    subset_counts = subset.value_counts()
    full_counts = full.value_counts()

    categories = sorted(set(subset_counts.index).union(full_counts.index))
    total_full = int(full_counts.sum())

    observed = np.array([float(subset_counts.get(cat, 0)) for cat in categories], dtype=float)
    if total_full == 0:
        raise ValueError("Full dataset is empty — cannot compute expected counts.")

    expected = np.array(
        [(float(full_counts.get(cat, 0)) / total_full) * subset_size for cat in categories],
        dtype=float,
    )

    mask = expected > 0
    observed = observed[mask]
    expected = expected[mask]
    kept = [cat for cat, keep in zip(categories, mask, strict=True) if keep]

    obs_sum = observed.sum()
    exp_sum = expected.sum()
    if exp_sum > 0 and not np.isclose(exp_sum, obs_sum):
        expected = expected * (obs_sum / exp_sum)

    return observed, expected, kept


def _chisquare_safe(observed: np.ndarray, expected: np.ndarray) -> float:
    """Run chi-squared, returning 1.0 for degenerate inputs."""
    if len(observed) <= 1:
        return 1.0
    if observed.sum() == 0:
        return 1.0
    result = chisquare(f_obs=observed, f_exp=expected)
    p_value = float(result.pvalue)
    if np.isnan(p_value):
        return 1.0
    return p_value


def chi_squared_validate(
    subset: pd.DataFrame,
    full: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    assert_pass: bool = True,
) -> ChiSquaredReport:
    """Validate a subset's type/level/length distributions against the source.

    Raises ``AssertionError`` on failure when ``assert_pass`` is True.
    """
    if len(subset) == 0:
        raise ValueError("subset is empty — nothing to validate")
    if len(full) == 0:
        raise ValueError("full dataset is empty — cannot validate")
    for col in ("type", "level", "answer"):
        if col not in subset.columns:
            raise KeyError(f"subset missing column {col!r}")
        if col not in full.columns:
            raise KeyError(f"full dataset missing column {col!r}")

    subset_aug = _ensure_strata_columns(subset)
    full_aug = _ensure_strata_columns(full)

    subset_size = len(subset_aug)

    obs_t, exp_t, cats_t = _observed_expected(subset_aug["type"], full_aug["type"], subset_size)
    type_p = _chisquare_safe(obs_t, exp_t)

    obs_l, exp_l, cats_l = _observed_expected(subset_aug["level"], full_aug["level"], subset_size)
    level_p = _chisquare_safe(obs_l, exp_l)

    obs_b, exp_b, cats_b = _observed_expected(
        subset_aug["length_bucket"], full_aug["length_bucket"], subset_size
    )
    bucket_p = _chisquare_safe(obs_b, exp_b)

    details = {
        "subset_size": int(subset_size),
        "full_size": len(full_aug),
        "type": {
            "categories": cats_t,
            "observed": obs_t.tolist(),
            "expected": exp_t.tolist(),
        },
        "level": {
            "categories": cats_l,
            "observed": obs_l.tolist(),
            "expected": exp_l.tolist(),
        },
        "length_bucket": {
            "categories": cats_b,
            "observed": obs_b.tolist(),
            "expected": exp_b.tolist(),
        },
    }

    passed = all(p > alpha for p in (type_p, level_p, bucket_p))

    report = ChiSquaredReport(
        type_p_value=type_p,
        level_p_value=level_p,
        length_bucket_p_value=bucket_p,
        alpha=alpha,
        passed=passed,
        details=details,
    )

    logger.info(
        "Chi-squared validation complete",
        extra={
            "type_p_value": type_p,
            "level_p_value": level_p,
            "length_bucket_p_value": bucket_p,
            "alpha": alpha,
            "passed": passed,
        },
    )

    if assert_pass and not passed:
        raise AssertionError(
            "Stratified subset failed chi-squared validation "
            f"(type={type_p:.4f}, level={level_p:.4f}, "
            f"length_bucket={bucket_p:.4f}, alpha={alpha})"
        )

    return report
