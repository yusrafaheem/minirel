# minirel

A relational database engine built from scratch: paged disk storage, a
real B+-tree index, write-ahead logging with crash recovery,
snapshot-isolation MVCC transactions, and a hand-written SQL front end
(lexer, recursive-descent parser, query planner, Volcano-style
executor) -- all in pure Python with no database, ORM, or parser-
generator dependency.

```sql
minirel> CREATE TABLE widgets (id INT, name TEXT, price FLOAT);
minirel> CREATE UNIQUE INDEX widgets_pk ON widgets (id);
minirel> INSERT INTO widgets VALUES (1, 'bolt', 0.50), (2, 'nail', 0.10);
minirel> SELECT name FROM widgets WHERE id = 1;
name
-----
bolt
(1 row)
```

## Why this exists

Most application code treats a database as a black box behind an ORM.
This project builds the box: a page-based storage engine, the index
structure that makes point lookups fast, the logging that makes commits
durable, the concurrency control that makes concurrent readers and
writers not corrupt each other, and the query engine that turns SQL
text into a stream of rows -- each piece implemented and stress-tested
against a reference model, not just "made to pass the happy path."

## Architecture

```
SQL text
  |  lexer.py (regex tokenizer) -> parser.py (recursive descent) -> ast.py
  v
Statement (CreateTable | CreateIndex | Insert | Select | Update | Delete)
  |
  |  database.py: DDL and INSERT/UPDATE/DELETE handled directly;
  |  SELECT (and UPDATE/DELETE row-selection) goes through:
  v
planner.py  -->  builds a tree of executor.py operators
  |                 SeqScan / IndexScan -> Filter -> Join -> Aggregate
  |                 -> Sort -> Limit -> Project
  v
storage layer: HeapFile (rows) + BPlusTree (indexes), both built on
                BufferPoolManager (LRU page cache) over DiskManager
                (fixed-size page I/O against a real file)
  |
  +-- transaction.py: snapshot-isolation MVCC visibility, on every
  |                    tuple's (xmin, xmax) header
  +-- wal.py: write-ahead log; crash recovery replays committed ops
```

Every layer is real, not mocked out: `.db` files are actual paged
binary files you can inspect with a hex editor, the B+-tree is a real
disk-backed tree (not a Python dict pretending), and the WAL/MVCC
implementation is tested by an actual "kill the process mid-transaction
and reopen" integration test, not just unit tests of the visibility
math in isolation.

## What's simplified (read this before assuming a limitation is a bug)

Every non-trivial system makes scope cuts. Here's the full list, so
none of them are a surprise:

- **No page/space reclamation.** Deleted heap slots and B+-tree pages
  are never returned to a free list for reuse; `DELETE`/aborted
  transactions leave dead space behind (documented in
  `heap_file.py`/`database.py`). There's no `VACUUM`.
- **Index entries for old row versions are never removed.** `UPDATE`
  and `DELETE` leave old index entries in place, relying on
  `IndexScanOperator` re-checking MVCC visibility per entry rather than
  physically cleaning up. Indexes only ever grow.
- **The WAL is logical redo, not ARIES-style physical redo.** Log
  records describe operations ("insert this row", "stamp this xmax"),
  not raw page byte diffs, and there's no undo logging -- an aborted
  transaction's already-applied heap writes are never physically
  unwound, only hidden by MVCC visibility forever. See `wal.py` and
  `transaction.py`'s docstrings for exactly how this is still made
  correct despite the buffer pool being allowed to flush a dirty,
  uncommitted page to disk before a crash (`Database.checkpoint()`'s
  requirement that no transaction be active when it runs is the key
  invariant that makes this sound).
- **DDL isn't transactional.** `CREATE TABLE`/`CREATE INDEX` persist
  immediately to a JSON catalog sidecar file (fsynced), independent of
  the WAL -- the same simplification SQLite makes.
- **No query optimizer beyond one predicate pushdown.** The planner
  recognizes a top-level `col = <literal>` conjunct on an indexed
  column and uses `IndexScanOperator` instead of a full scan; there's
  no cost-based join ordering, no statistics, no multi-column indexes.
- **Nested-loop join only.** O(N\*M), no hash join or sort-merge join.
- **No subqueries, CTEs, `HAVING`, or general arithmetic expressions**
  in the SQL grammar (unary minus on numeric literals is supported;
  `price * 1.1` is not).
- **Single-threaded.** `TransactionManager` does no internal locking;
  concurrent access from multiple threads/processes isn't supported or
  tested. The MVCC/write-conflict logic is still real and tested by
  interleaving multiple `Transaction` objects sequentially within one
  process (which is exactly how snapshot isolation is supposed to
  behave regardless of thread scheduling).

None of these are things I didn't know how to do -- they're places
where building the "textbook-correct, well-tested, honestly scoped"
version was a better use of time than building the
"handles-every-production-edge-case" version. The parts that *are*
implemented (B+-tree splits/merges/duplicate-key handling, MVCC
snapshot visibility, WAL crash recovery) are implemented and tested for
real, not stubbed.

## Bugs the tests actually caught

Kept here because finding these is the point of the stress tests, not
an embarrassment to hide:

- **B+-tree leaf splits could separate a run of duplicate keys** across
  the two resulting leaves. Since search routes a key equal to a
  separator to the *right* child, half the duplicates silently became
  unreachable. Fixed by nudging the split point to a key boundary
  (`_split_point_avoiding_duplicates` in `index/btree.py`).
- **The same duplicate-run problem showed up again in borrow/merge on
  delete**, plus a separate off-by-one in the min/max fill-factor math
  (`ceil` instead of `floor`, which could make a merge of two
  "minimally full" leaves overflow the destination page by one entry).
  Both found by a 4,000-operation randomized insert/delete stress test
  checked against a plain Python `dict`/sorted-list reference model
  (`tests/test_btree.py`).
- **A rare case where neither borrowing nor a plain merge could
  rebalance a delete-underflowed leaf** (the "spare" sibling's excess
  entries were all one duplicate run, blocking a partial borrow, while
  a full merge would overflow the page by exactly one entry). Fixed by
  falling back to *redistributing* across both pages at a fresh
  duplicate-safe split point instead of collapsing to one, when a
  same-page merge would overflow.
- **`CREATE TABLE`/`CREATE INDEX` allocated a page in the buffer pool
  but never flushed it**, so a crash before the next eviction or
  checkpoint left the catalog pointing at a page that was still all
  zeros on disk -- which a page-chain walk misread as a
  self-referencing "next page" pointer and looped forever. Caught by
  `tests/test_wal_recovery.py`'s crash-mid-transaction test actually
  hanging instead of failing. Fixed by making DDL flush its freshly
  allocated page immediately (plus a cycle-detection guard in
  `heap_file.py` as a second line of defense).
- **`UPDATE` on a uniquely-indexed column spuriously conflicted with
  its own old, now-dead index entry**, because the B+-tree's built-in
  uniqueness check isn't MVCC-aware. Fixed by moving uniqueness
  enforcement to the database layer, where visibility can actually be
  checked (`Database._check_unique_constraint`).
- **The SQL lexer had no unary minus** -- `WHERE tag = -1` failed to
  tokenize at all. Found while writing the benchmark scripts, not the
  unit tests, which is its own small lesson about coverage.

## Benchmarks

`benchmarks/bench_point_lookup.py` -- B+-tree index scan vs. full
sequential scan, at increasing table sizes (run with `--rows` for
larger tables; growth, not the absolute numbers on any one machine, is
the point):

```
    rows | tree height |  index scan (us) |  seq scan (us) |  speedup
----------------------------------------------------------------------
    1000 |           2 |            622.2 |         3552.9 |     5.7x
    5000 |           2 |            575.4 |        17394.4 |    30.2x
   10000 |           2 |            728.2 |        35349.8 |    48.5x
```

The tree stays 2 levels deep as the table grows 10x while the
sequential scan's cost grows linearly with it -- exactly the property a
B+-tree's high fanout (hundreds of keys per 4KB page) is supposed to
give you.

`benchmarks/bench_throughput.py` -- batched vs. one-row-per-transaction
inserts (isolating the WAL's per-commit `fsync` cost), full-table-scan
throughput, and indexed point-query throughput.

## Testing

`tests/` is one `unittest` file per component (storage, B+-tree,
catalog, WAL, MVCC, SQL front end, planner/executor), plus two kinds of
integration test that don't fit a single-component file:

- `test_wal_recovery.py` actually simulates a crash -- it closes the
  underlying file handles without calling `Database.close()` (which
  would flush everything and hide exactly the bugs this is supposed to
  catch), then reopens a fresh `Database` over the same files and checks
  what survived.
- `test_btree.py` includes a randomized stress test that interleaves
  thousands of inserts/deletes against a real B+-tree and checks every
  intermediate state against a plain Python `dict`/sorted-list reference
  model, which is what actually caught the duplicate-key split/merge
  bugs listed above.

## Continuous integration

Every push to `main` and every pull request runs, on Python 3.10/3.11/3.12:

- `ruff check` and `black --check` (lint + formatting)
- the full `unittest` suite
- a smoke run of both benchmark scripts at small `--rows`, so a bench
  script bit-rotting silently is caught even though nobody runs
  benchmarks by hand on every change
- a scripted end-to-end session through the installed `minirel` CLI
  entry point, distinct from the in-process unit tests

See `.github/workflows/ci.yml`.

## Running it

```bash
pip install -e ".[dev]"          # dev extras: ruff, black
python -m unittest discover -s tests -v
python benchmarks/bench_point_lookup.py
python benchmarks/bench_throughput.py
minirel path/to/some.db          # interactive SQL REPL
```

Or from Python:

```python
from minirel import Database

db = Database("example.db")
db.execute("CREATE TABLE widgets (id INT, name TEXT)")
db.execute("INSERT INTO widgets VALUES (1, 'bolt')")
result = db.execute("SELECT * FROM widgets")
print(result.rows)  # [{'id': 1, 'name': 'bolt'}]

# Explicit multi-statement transactions:
txn = db.begin()
db.execute("UPDATE widgets SET name = 'big bolt' WHERE id = 1", txn=txn)
txn.commit()  # or txn.abort()

db.close()
```

## Project layout

```
src/minirel/
  storage/          disk_manager.py, buffer_pool.py, heap_file.py, page.py
  index/            btree.py
  sql/              lexer.py, ast.py, parser.py
  wal.py            write-ahead log
  transaction.py    MVCC transaction manager
  catalog.py        table/index metadata
  types.py          column types, row (de)serialization, MVCC tuple header
  expr.py           WHERE/ON/SELECT-item expression evaluation
  executor.py       Volcano-model query operators
  planner.py        AST -> operator tree, with index predicate pushdown
  database.py       top-level Database class (SQL entry point)
  repl.py           interactive shell (`minirel` console script)
tests/              one file per component, plus end-to-end tests and
                    the WAL crash-recovery integration tests
benchmarks/         bench_point_lookup.py, bench_throughput.py
```
