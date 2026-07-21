#!/usr/bin/env python3
"""
Insert and query throughput, and the transaction/WAL cost of a commit.

Three numbers, each isolating a different part of the engine:
  1. Insert throughput -- batched INSERTs (many rows per statement, one
     transaction) vs. one-row-per-transaction INSERTs, to show the cost
     of the fsync() on every commit (see wal.py's log_commit).
  2. Full-table scan throughput once the table is built.
  3. Indexed point-query throughput.

Usage: python3 benchmarks/bench_throughput.py [--rows N]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.database import Database

DEFAULT_ROWS = 5_000


def bench_batched_insert(db: Database, n: int, batch_size: int = 200) -> float:
    start = time.perf_counter()
    for batch_start in range(0, n, batch_size):
        batch = range(batch_start, min(batch_start + batch_size, n))
        values = ", ".join(f"({i}, 'row-{i}')" for i in batch)
        db.execute(f"INSERT INTO batched VALUES {values}")
    return n / (time.perf_counter() - start)


def bench_one_row_per_commit(db: Database, n: int) -> float:
    start = time.perf_counter()
    for i in range(n):
        db.execute(f"INSERT INTO single VALUES ({i}, 'row-{i}')")
    return n / (time.perf_counter() - start)


def bench_full_scan(db: Database, trials: int = 20) -> float:
    start = time.perf_counter()
    for _ in range(trials):
        db.execute("SELECT * FROM batched")
    return trials / (time.perf_counter() - start)


def bench_indexed_point_query(db: Database, n: int, trials: int = 500) -> float:
    start = time.perf_counter()
    for i in range(trials):
        db.execute(f"SELECT name FROM batched WHERE id = {i % n}")
    return trials / (time.perf_counter() - start)


def run(n: int) -> None:
    tmp_dir = tempfile.mkdtemp()
    db = Database(os.path.join(tmp_dir, "bench.db"))
    db.execute("CREATE TABLE batched (id INT, name TEXT)")
    db.execute("CREATE UNIQUE INDEX batched_id_idx ON batched (id)")
    db.execute("CREATE TABLE single (id INT, name TEXT)")

    batched_rate = bench_batched_insert(db, n)
    print(f"batched insert   ({n} rows, {n // 200} commits): {batched_rate:>10.0f} rows/sec")

    single_n = min(n, 500)  # one fsync per row is slow -- keep this one bounded
    single_rate = bench_one_row_per_commit(db, single_n)
    print(f"one-row commits  ({single_n} rows, {single_n} commits): {single_rate:>10.0f} rows/sec")
    print("  -> the gap between these two is almost entirely the WAL's per-commit fsync")

    scan_rate = bench_full_scan(db)
    print(f"full table scan  ({n} rows/scan): {scan_rate:>10.1f} scans/sec")

    query_rate = bench_indexed_point_query(db, n)
    print(f"indexed point query: {query_rate:>10.0f} queries/sec")

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    args = parser.parse_args()
    run(args.rows)
