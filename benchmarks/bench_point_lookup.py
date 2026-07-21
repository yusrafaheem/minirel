#!/usr/bin/env python3
"""
Point-lookup latency: B+-tree index scan vs. full sequential scan.

This is the benchmark that justifies the B+-tree's existence: it
measures `SELECT ... WHERE id = <x>` latency against a table of
increasing size, once using an index and once forcing a full heap scan
(a non-indexed column with an equally selective predicate), and reports
the resulting speedup plus the tree's height at each size -- the point
being that height (and therefore lookup cost) grows logarithmically
while the table grows linearly.

Usage: python3 benchmarks/bench_point_lookup.py [--rows N ...]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.database import Database
from minirel.index.btree import BPlusTree

DEFAULT_SIZES = [1_000, 5_000, 10_000]  # pass --rows for larger tables; growth is what matters
TRIALS = 200


def build_database(path: str, n: int) -> Database:
    db = Database(path)
    db.execute("CREATE TABLE items (id INT, tag INT, payload TEXT)")
    db.execute("CREATE UNIQUE INDEX items_id_idx ON items (id)")

    batch = []
    for i in range(n):
        batch.append((i, i % 997, f"row-payload-{i}"))
        if len(batch) == 500:
            _flush_batch(db, batch)
            batch = []
    if batch:
        _flush_batch(db, batch)
    return db


def _flush_batch(db: Database, batch: list[tuple]) -> None:
    values = ", ".join(f"({i}, {tag}, '{payload}')" for i, tag, payload in batch)
    db.execute(f"INSERT INTO items VALUES {values}")


def time_lookup(db: Database, sql_template: str, keys: list[int]) -> float:
    start = time.perf_counter()
    for key in keys:
        db.execute(sql_template.format(key=key))
    elapsed = time.perf_counter() - start
    return elapsed / len(keys)


def run(sizes: list[int]) -> None:
    rng = random.Random(42)
    header = f"{'rows':>8} | {'tree height':>11} | {'index scan (us)':>16} | {'seq scan (us)':>14}"
    print(f"{header} | {'speedup':>8}")
    print("-" * 70)

    for n in sizes:
        tmp_dir = tempfile.mkdtemp()
        db_path = os.path.join(tmp_dir, "bench.db")
        db = build_database(db_path, n)

        idx_meta = db.catalog.get_table("items").indexes["items_id_idx"]
        tree = BPlusTree(
            db.buffer_pool, key_type=idx_meta.key_type, root_page_id=idx_meta.root_page_id
        )
        height = tree.height()

        keys = [rng.randrange(n) for _ in range(TRIALS)]
        index_secs = time_lookup(db, "SELECT payload FROM items WHERE id = {key}", keys)
        seq_secs = time_lookup(db, "SELECT payload FROM items WHERE tag = -1 OR id = {key}", keys)

        db.close()

        speedup = seq_secs / index_secs if index_secs > 0 else float("inf")
        row = f"{n:>8} | {height:>11} | {index_secs * 1e6:>16.1f} | {seq_secs * 1e6:>14.1f}"
        print(f"{row} | {speedup:>7.1f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=DEFAULT_SIZES)
    args = parser.parse_args()
    run(args.rows)
