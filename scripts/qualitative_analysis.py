"""
Qualitative error analysis:
1. Cases where hybrid-only succeeds but agentic-full fails (EM)
2. Cases where steps-2 succeeds but steps-1 fails (EM)
Outputs 3 examples each to console + results/qualitative_examples.txt
"""

import duckdb
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"

def load(fname: str) -> dict:
    """Load question_id → row dict from eval_records."""
    path = RESULTS / fname
    con  = duckdb.connect(str(path), read_only=True)
    df   = con.execute("""
        SELECT question_id, question, gold_answer, predicted_answer,
               exact_match, f1
        FROM eval_records
    """).fetchdf()
    con.close()
    return df.set_index("question_id").to_dict("index")


def find_wins(winner: dict, loser: dict, n: int = 3) -> list:
    """Return n cases where winner EM=1 and loser EM=0."""
    wins = []
    for qid, w in winner.items():
        if qid in loser:
            l = loser[qid]
            if w["exact_match"] == 1.0 and l["exact_match"] == 0.0:
                wins.append({
                    "question_id": qid,
                    "question":    w["question"],
                    "gold":        w["gold_answer"],
                    "winner_pred": w["predicted_answer"],
                    "loser_pred":  l["predicted_answer"],
                    "loser_f1":    l["f1"],
                })
                if len(wins) >= n:
                    break
    return wins


def print_examples(title: str, winner_label: str, loser_label: str, examples: list, f):
    header = f"\n{'='*70}\n  {title}\n{'='*70}"
    print(header); f.write(header + "\n")
    for i, ex in enumerate(examples, 1):
        block = (
            f"\n  Example {i}:\n"
            f"  Q:  {ex['question']}\n"
            f"  Gold:          {ex['gold']}\n"
            f"  {winner_label} (correct): {ex['winner_pred']}\n"
            f"  {loser_label} (wrong):   {ex['loser_pred']}  (F1={ex['loser_f1']:.2f})\n"
        )
        print(block); f.write(block + "\n")


print("Loading DuckDB files...")
hybrid  = load("ablation_hybrid_only.duckdb")
agentic = load("traces_agentic_5k.duckdb")
steps1  = load("ablation_steps_1.duckdb")
steps2  = load("ablation_steps_2.duckdb")

wins_hybrid = find_wins(hybrid,  agentic, n=3)
wins_steps2 = find_wins(steps2,  steps1,  n=3)

out_path = RESULTS / "qualitative_examples.txt"
with open(out_path, "w") as f:
    print_examples(
        "Hybrid-only CORRECT, Agentic-full WRONG",
        "hybrid-only", "agentic-full",
        wins_hybrid, f
    )
    print_examples(
        "Steps-2 CORRECT, Steps-1 WRONG",
        "steps-2", "steps-1",
        wins_steps2, f
    )

print(f"\nSaved: {out_path}")
