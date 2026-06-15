"""Inspect DuckDB schema for all result files."""
import duckdb
from pathlib import Path

results_dir = Path(__file__).parent.parent / "results"
db_files = sorted(results_dir.glob("*.duckdb"))

for db_path in db_files:
    print(f"\n{'='*60}")
    print(f"FILE: {db_path.name}")
    print('='*60)
    con = duckdb.connect(str(db_path), read_only=True)
    tables = con.execute("SHOW TABLES").fetchall()
    for (table,) in tables:
        print(f"\n  TABLE: {table}")
        desc = con.execute(f"DESCRIBE {table}").fetchall()
        for row in desc:
            print(f"    {row[0]:30s} {row[1]}")
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"    --- row count: {count}")
        print(f"\n  SAMPLE ROW:")
        sample = con.execute(f"SELECT * FROM {table} LIMIT 1").fetchdf()
        for col in sample.columns:
            print(f"    {col}: {sample[col].iloc[0]!r}")
    con.close()
