"""
Paired significance tests for all key ablation comparisons.

The eight ablation conditions, the full pipeline, and the baseline are all
evaluated on the SAME fixed 5,000-question HotpotQA sample. Comparisons between
two systems are therefore paired per question, and a paired test is the correct
choice. We use:

  - EM (binary per question): McNemar's exact test (deterministic, no RNG).
  - F1 (continuous per question): Wilcoxon signed-rank test (deterministic).

Both are one-sided in the direction stated by H1 (the better system wins).
This replaces the earlier unpaired permutation test in bootstrap_ci.py, which
ignored the pairing and was underpowered. Outputs results/paired_significance.csv
and a console table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import duckdb
import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

RESULTS = Path(__file__).parent.parent / "results"

DB = {
    "baseline": "traces_baseline_5k.duckdb",
    "agentic_full": "traces_agentic_5k.duckdb",
    "no_decomp": "ablation_no_decomp.duckdb",
    "no_reranker": "ablation_no_reranker.duckdb",
    "steps_1": "ablation_steps_1.duckdb",
    "steps_2": "ablation_steps_2.duckdb",
    "hybrid_only": "ablation_hybrid_only.duckdb",
}

# (worse, better): H1 is mean(better) > mean(worse)
COMPARISONS = [
    ("agentic_full vs baseline", "baseline", "agentic_full"),
    ("hybrid_only vs agentic_full", "agentic_full", "hybrid_only"),
    ("steps_2 vs steps_1", "steps_1", "steps_2"),
    ("agentic_full vs no_decomp", "no_decomp", "agentic_full"),
    ("agentic_full vs no_reranker", "no_reranker", "agentic_full"),
]


def load(name: str):
    con = duckdb.connect(str(RESULTS / DB[name]), read_only=True)
    df = con.execute(
        "SELECT question_id, exact_match, f1 FROM eval_records ORDER BY question_id"
    ).fetchdf()
    con.close()
    return df


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def mcnemar_one_sided(worse_em: np.ndarray, better_em: np.ndarray) -> tuple[int, int, float]:
    """One-sided McNemar exact test, H1: better has more wins than worse.

    Returns (b01, b10, p) where b10 = better-right/worse-wrong (discordant in
    favour of H1) and b01 = better-wrong/worse-right.
    """
    better_right_worse_wrong = int(np.sum((better_em == 1) & (worse_em == 0)))  # b10
    better_wrong_worse_right = int(np.sum((better_em == 0) & (worse_em == 1)))  # b01
    n = better_right_worse_wrong + better_wrong_worse_right
    # One-sided exact binomial tail: P(X >= b10) with X~Binom(n, 0.5).
    from scipy.stats import binomtest

    p = binomtest(better_right_worse_wrong, n, 0.5, alternative="greater").pvalue
    return better_wrong_worse_right, better_right_worse_wrong, float(p)


def main() -> None:
    data = {k: load(k) for k in DB}
    rows = []
    print("\n" + "=" * 96)
    print("  Paired significance tests (EM: McNemar exact; F1: Wilcoxon signed-rank). One-sided.")
    print("=" * 96)
    print(f"  {'comparison':30} {'dEM':>6} {'EM p':>10} {'dF1':>6} {'F1 p':>12}  discordant(b10/b01)")
    print("-" * 96)
    for desc, worse, better in COMPARISONS:
        w = data[worse].merge(data[better], on="question_id", suffixes=("_w", "_b"))
        n = len(w)
        w_em = w["exact_match_w"].to_numpy()
        b_em = w["exact_match_b"].to_numpy()
        w_f1 = w["f1_w"].to_numpy()
        b_f1 = w["f1_b"].to_numpy()

        d_em = (b_em.mean() - w_em.mean()) * 100
        d_f1 = (b_f1.mean() - w_f1.mean()) * 100

        b01, b10, p_em = mcnemar_one_sided(w_em, b_em)

        # Wilcoxon signed-rank on per-question F1 differences, one-sided (better > worse).
        diff = b_f1 - w_f1
        if np.any(diff != 0):
            p_f1 = wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue
        else:
            p_f1 = 1.0

        print(
            f"  {desc:30} {d_em:>+6.2f} {p_em:>10.2e}{stars(p_em):>4} "
            f"{d_f1:>+6.2f} {p_f1:>10.2e}{stars(p_f1):>4}  {b10}/{b01}  (n={n})"
        )
        rows.append(
            {
                "comparison": desc,
                "n": n,
                "delta_em": round(d_em, 3),
                "em_p_mcnemar": p_em,
                "em_sig": stars(p_em),
                "delta_f1": round(d_f1, 3),
                "f1_p_wilcoxon": p_f1,
                "f1_sig": stars(p_f1),
                "em_b10_betterwins": b10,
                "em_b01_worsewins": b01,
            }
        )
    print("=" * 96)

    out = RESULTS / "paired_significance.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
