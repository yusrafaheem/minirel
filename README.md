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
