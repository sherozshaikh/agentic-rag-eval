"""
Bootstrap confidence intervals for all ablation variants.
Outputs: results/bootstrap_ci.csv + console table.
"""

import duckdb
import numpy as np
from pathlib import Path
import csv

RESULTS = Path(__file__).parent.parent / "results"
RNG     = np.random.default_rng(42)
N_BOOT  = 10_000
ALPHA   = 0.05   # 95% CI

# Map filename → display label (order matters for table)
DB_MAP = {
    "traces_baseline_5k.duckdb":     "baseline",
    "traces_agentic_5k.duckdb":      "agentic_full",
    "ablation_no_decomp.duckdb":     "no-decomp",
    "ablation_no_reranker.duckdb":   "no-reranker",
    "ablation_steps_1.duckdb":       "steps-1",
    "ablation_steps_2.duckdb":       "steps-2",
    "ablation_steps_3.duckdb":       "steps-3",
    "ablation_dense_only.duckdb":    "dense-only",
    "ablation_sparse_only.duckdb":   "sparse-only",
    "ablation_hybrid_only.duckdb":   "hybrid-only",
}


def bootstrap_ci(arr: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) using percentile bootstrap."""
    arr = arr[~np.isnan(arr)]
    means = np.array([RNG.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)])
    return float(arr.mean()), float(np.percentile(means, 100 * ALPHA / 2)), float(np.percentile(means, 100 * (1 - ALPHA / 2)))


def pvalue(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT) -> float:
    """One-sided bootstrap p-value: P(mean(b) > mean(a)) via permutation."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    obs_diff = b.mean() - a.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    diffs = []
    for _ in range(n_boot):
        perm = RNG.permutation(combined)
        diffs.append(perm[na:].mean() - perm[:na].mean())
    return float(np.mean(np.array(diffs) >= obs_diff))


# NOTE ON SIGNIFICANCE TESTING
# ----------------------------
# All conditions are evaluated on the SAME fixed 5,000 questions, so every
# pairwise comparison is paired per question. Pairwise p-values are therefore
# computed with paired tests in scripts/paired_significance.py (McNemar's exact
# test for EM, Wilcoxon signed-rank for F1). The unpaired `pvalue` helper above
# is retained only for reference; it is NOT used for any number reported in the
# paper, because an unpaired test on paired data is the wrong and underpowered
# choice. This script reports per-system bootstrap confidence intervals only.

rows = []
arrays = {}

for fname, label in DB_MAP.items():
    db_path = RESULTS / fname
    if not db_path.exists():
        print(f"  MISSING: {fname}")
        continue
    con = duckdb.connect(str(db_path), read_only=True)
    df  = con.execute("SELECT exact_match, f1 FROM eval_records").fetchdf()
    con.close()

    em_arr = df["exact_match"].to_numpy() * 100.0
    f1_arr = df["f1"].to_numpy() * 100.0
    arrays[label] = {"em": em_arr, "f1": f1_arr}

    em_mean, em_lo, em_hi = bootstrap_ci(em_arr)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1_arr)

    rows.append({
        "variant": label,
        "em_mean": em_mean, "em_ci_lo": em_lo, "em_ci_hi": em_hi,
        "f1_mean": f1_mean, "f1_ci_lo": f1_lo, "f1_ci_hi": f1_hi,
    })

print("\n" + "="*80)
print("  Bootstrap Confidence Intervals (95%, n_boot=10,000)")
print("="*80)
print(f"  {'Variant':<16} {'EM':>6}  {'95% CI':^16}  {'F1':>6}  {'95% CI':^16}")
print("-"*80)
for r in rows:
    print(f"  {r['variant']:<16} {r['em_mean']:>6.1f}  "
          f"[{r['em_ci_lo']:5.1f}, {r['em_ci_hi']:5.1f}]  "
          f"{r['f1_mean']:>6.1f}  "
          f"[{r['f1_ci_lo']:5.1f}, {r['f1_ci_hi']:5.1f}]")
print("="*80)
print("\n  Pairwise p-values: see scripts/paired_significance.py (paired tests).")

# Save CSV
out_csv = RESULTS / "bootstrap_ci.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved: {out_csv}")
