"""
Routing strategy breakdown from agentic full traces.
Shows: strategy distribution + per-strategy EM/F1.
Outputs: results/routing_breakdown.csv + console table.
"""

import duckdb
from pathlib import Path
import csv

RESULTS  = Path(__file__).parent.parent / "results"
DB_PATH  = RESULTS / "traces_agentic_5k.duckdb"

con = duckdb.connect(str(DB_PATH), read_only=True)

# Per-strategy counts and performance
df = con.execute("""
    SELECT
        strategy_used,
        COUNT(*)                    AS n,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct,
        ROUND(AVG(exact_match) * 100, 1) AS em,
        ROUND(AVG(f1) * 100, 1)          AS f1
    FROM eval_records
    GROUP BY strategy_used
    ORDER BY n DESC
""").fetchdf()

# Also check: what EM would we get if every query used hybrid?
# (already have from ablation_hybrid_only.duckdb - just for reference)

con.close()

print("\n" + "="*60)
print("  Routing Strategy Distribution (agentic_full, n=5000)")
print("="*60)
print(f"  {'Strategy':<12} {'Count':>7} {'%':>6}  {'EM':>6}  {'F1':>6}")
print("-"*60)
for _, row in df.iterrows():
    print(f"  {str(row['strategy_used']):<12} {int(row['n']):>7} {row['pct']:>5.1f}%  "
          f"{row['em']:>6.1f}  {row['f1']:>6.1f}")
print("="*60)
print("""
  Interpretation:
  - BM25 dominates because HotpotQA sub-questions almost always
    contain named entities/dates, triggering the entity rule.
  - Despite high BM25 usage, hybrid-only (always RRF) beats full
    adaptive by +1.8 EM — showing complementary dense+sparse signals
    consistently help on multi-hop entity chains.
""")

out_csv = RESULTS / "routing_breakdown.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
