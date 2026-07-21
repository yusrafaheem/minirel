"""
minirel
=======

A relational database engine built from scratch: fixed-size paged storage
backed by real files on disk, an LRU-managed buffer pool, a paged B+-tree
index, write-ahead logging with redo recovery, snapshot-isolation MVCC
transactions, and a hand-written SQL front end (lexer, recursive-descent
parser, query planner, and a Volcano-style iterator executor).

See README.md for the architecture and what's verified where.
"""

from .database import Database

__all__ = ["Database"]

__version__ = "0.1.0"
