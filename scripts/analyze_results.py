from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb

AGENTIC_DB = Path("results/traces_agentic_5k.duckdb")
BASELINE_DB = Path("results/traces_baseline_5k.duckdb")
EXPORT_DIR = Path("results")


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    if not path.exists():
        raise FileNotFoundError(f"DuckDB not found: {path}")
    return duckdb.connect(str(path), read_only=True)


def _summary(conn: duckdb.DuckDBPyConnection) -> dict:
    r = conn.execute(
        """
        SELECT
            COUNT(*)                                        AS n,
            AVG(exact_match)                               AS em,
            AVG(f1)                                        AS f1,
            AVG(latency_ms)                                AS latency_ms,
            AVG(sf_precision)                              AS sf_precision,
            AVG(sf_recall)                                 AS sf_recall,
            AVG(sf_f1)                                     AS sf_f1,
            SUM(CASE WHEN exact_match = 1 THEN 1 ELSE 0 END) AS exact_hits
        FROM eval_records
    """
    ).fetchone()
    return dict(
        zip(
            [
                "n",
                "em",
                "f1",
                "latency_ms",
                "sf_precision",
                "sf_recall",
                "sf_f1",
                "exact_hits",
            ],
            r,
            strict=False,
        )
    )


def _by_strategy(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            COALESCE(strategy_used, 'unknown') AS strategy,
            COUNT(*)        AS n,
            AVG(exact_match) AS em,
            AVG(f1)          AS f1
        FROM eval_records
        GROUP BY strategy_used
        ORDER BY n DESC
    """
    ).fetchall()
    return [dict(zip(["strategy", "n", "em", "f1"], r, strict=False)) for r in rows]


def _by_failure(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT
            COALESCE(failure_mode, 'none') AS failure_mode,
            COUNT(*) AS n
        FROM eval_records
        GROUP BY failure_mode
        ORDER BY n DESC
    """
    ).fetchall()
    return [dict(zip(["failure_mode", "n"], r, strict=False)) for r in rows]


def _latency_percentiles(conn: duckdb.DuckDBPyConnection) -> dict:
    r = conn.execute(
        """
        SELECT
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY latency_ms) AS p90,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99,
            MAX(latency_ms) AS max_ms
        FROM eval_records
    """
    ).fetchone()
    return dict(zip(["p50", "p90", "p99", "max"], r, strict=False))


W = 64


def _hr():
    print("─" * W)


def _header(title: str):
    _hr()
    print(f"  {title}")
    _hr()


def _row(label: str, val: str):
    print(f"  {label:<30} {val}")


def _delta(a: float, b: float) -> str:
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _print_summary(label: str, s: dict):
    _header(label)
    _row("Questions", f"{s['n']:,}")
    _row("Exact Match (EM)", f"{s['em']:.3f}  ({s['em'] * 100:.1f}%)")
    _row("F1", f"{s['f1']:.3f}  ({s['f1'] * 100:.1f}%)")
    _row("Exact hits", f"{int(s['exact_hits']):,} / {s['n']:,}")
    _row("Avg latency", f"{s['latency_ms']:.0f} ms")
    _row("SF Precision", f"{s['sf_precision']:.3f}")
    _row("SF Recall", f"{s['sf_recall']:.3f}")
    _row("SF F1", f"{s['sf_f1']:.3f}")


def _print_delta(ag: dict, bl: dict):
    _header("Delta  (Agentic - Baseline)")
    _row("EM delta", _delta(ag["em"], bl["em"]))
    _row("F1 delta", _delta(ag["f1"], bl["f1"]))
    _row("Latency delta", f"{ag['latency_ms'] - bl['latency_ms']:+.0f} ms")


def _print_strategy(rows: list[dict]):
    _header("Agentic — Retrieval strategy breakdown")
    print(f"  {'Strategy':<14} {'n':>6}  {'EM':>6}  {'F1':>6}")
    _hr()
    for r in rows:
        print(f"  {r['strategy']:<14} {r['n']:>6}  {r['em']:>6.3f}  {r['f1']:>6.3f}")


def _print_latency(ag: dict, bl: dict):
    _header("Latency percentiles (ms)")
    print(f"  {'Percentile':<10} {'Agentic':>10} {'Baseline':>10}")
    _hr()
    for k in ["p50", "p90", "p99", "max"]:
        print(f"  {k:<10} {ag[k]:>10.0f} {bl[k]:>10.0f}")


def _export(agentic: dict, baseline: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics_comparison.csv"
    rows = [
        {"pipeline": "agentic", **agentic},
        {"pipeline": "baseline", **baseline},
        {
            "pipeline": "delta",
            "n": "",
            "em": round(agentic["em"] - baseline["em"], 4),
            "f1": round(agentic["f1"] - baseline["f1"], 4),
            "latency_ms": round(agentic["latency_ms"] - baseline["latency_ms"], 1),
            "sf_precision": round(agentic["sf_precision"] - baseline["sf_precision"], 4),
            "sf_recall": round(agentic["sf_recall"] - baseline["sf_recall"], 4),
            "sf_f1": round(agentic["sf_f1"] - baseline["sf_f1"], 4),
            "exact_hits": "",
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Exported → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentic", default=str(AGENTIC_DB))
    parser.add_argument("--baseline", default=str(BASELINE_DB))
    parser.add_argument(
        "--export", action="store_true", help="Write results/metrics_comparison.csv"
    )
    args = parser.parse_args()

    ag_conn = _connect(Path(args.agentic))
    bl_conn = _connect(Path(args.baseline))

    ag_sum = _summary(ag_conn)
    bl_sum = _summary(bl_conn)
    ag_strat = _by_strategy(ag_conn)
    ag_fail = _by_failure(ag_conn)
    ag_lat = _latency_percentiles(ag_conn)
    bl_lat = _latency_percentiles(bl_conn)

    print()
    _print_summary("Agentic RAG  (qwen2.5:7b-instruct, 5K questions)", ag_sum)
    print()
    _print_summary("Baseline RAG (single-shot dense, 5K questions)", bl_sum)
    print()
    _print_delta(ag_sum, bl_sum)
    print()
    _print_strategy(ag_strat)
    print()
    _print_latency(ag_lat, bl_lat)

    print()
    _header("Agentic — Failure mode breakdown")
    print(f"  {'Failure mode':<25} {'n':>6}")
    _hr()
    for r in ag_fail:
        print(f"  {r['failure_mode']:<25} {r['n']:>6}")

    _hr()
    print()

    if args.export:
        _export(ag_sum, bl_sum, EXPORT_DIR)


if __name__ == "__main__":
    main()
