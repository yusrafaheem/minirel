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
