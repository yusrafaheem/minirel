"""
minirel.catalog
=================

Table and index metadata: which columns a table has, which page its heap
starts on, and which B+-trees index it.

This is the one piece of "everything real is paged" that deliberately
isn't: the catalog is small (a handful of tables, a handful of indexes
each), so rather than build a bootstrap page-0 system catalog the way a
production database does, it's persisted as a single JSON sidecar file
next to the `.db` data file, rewritten and fsynced on every DDL change
(CREATE TABLE / CREATE INDEX). DDL is therefore *not* part of the
WAL/MVCC transaction story -- it takes effect immediately, the same
simplification SQLite and many teaching databases make. Row-level data
(everything through INSERT/UPDATE/DELETE/SELECT) goes through the real
paged storage engine and is what the WAL and MVCC layers cover.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .index.btree import BPlusTree
from .storage.buffer_pool import BufferPoolManager
from .storage.heap_file import HeapFile
from .types import Column, ColumnType, Schema


@dataclass
class IndexMeta:
    name: str
    column: str
    root_page_id: int
    unique: bool
    key_type: ColumnType


@dataclass
class TableMeta:
    name: str
    schema: Schema
    heap_first_page_id: int
    indexes: dict[str, IndexMeta] = field(default_factory=dict)


class TableAlreadyExistsError(Exception):
    pass


class TableNotFoundError(Exception):
    pass


class Catalog:
    def __init__(self, buffer_pool: BufferPoolManager, path: str):
        self.buffer_pool = buffer_pool
        self.path = path
        self.tables: dict[str, TableMeta] = {}
        if os.path.exists(path):
            self._load()

    # -- DDL -------------------------------------------------------------

    def create_table(self, name: str, schema: Schema) -> TableMeta:
        if name in self.tables:
            raise TableAlreadyExistsError(name)
        heap = HeapFile(self.buffer_pool)
        # DDL is immediately durable (that's the whole point of persisting
        # the catalog outside the WAL/transaction story -- see this
        # module's docstring), so the heap page the catalog is about to
        # point to must actually exist on disk right now too, not just in
        # the buffer pool. Without this, a crash before the next eviction
        # or checkpoint would leave the catalog pointing at a page that's
        # still all zeros on disk -- which a page-chain walk misreads as a
        # self-referencing "next page" pointer and loops forever on.
        self.buffer_pool.flush_page(heap.first_page_id)
        meta = TableMeta(name=name, schema=schema, heap_first_page_id=heap.first_page_id)
        self.tables[name] = meta
        self._save()
        return meta

    def create_index(
        self, table_name: str, index_name: str, column: str, unique: bool = False
    ) -> IndexMeta:
        table = self.get_table(table_name)
        if index_name in table.indexes:
            raise ValueError(f"index already exists: {index_name}")
        col = table.schema.column(column)
        if col.type not in (ColumnType.INT, ColumnType.TEXT):
            raise ValueError(f"cannot index a {col.type} column: {column}")
        tree = BPlusTree(self.buffer_pool, key_type=col.type, unique=unique)
        self.buffer_pool.flush_page(tree.root_page_id)  # see create_table's comment on why
        meta = IndexMeta(
            name=index_name,
            column=column,
            root_page_id=tree.root_page_id,
            unique=unique,
            key_type=col.type,
        )
        table.indexes[index_name] = meta
        self._save()
        return meta

    def get_table(self, name: str) -> TableMeta:
        try:
            return self.tables[name]
        except KeyError:
            raise TableNotFoundError(name) from None

    def has_table(self, name: str) -> bool:
        return name in self.tables

    def update_index_root(self, table_name: str, index_name: str, new_root_page_id: int) -> None:
        """A B+-tree's root page id can change (splits growing the tree,
        deletes shrinking it) -- the catalog's copy needs to stay in sync so
        the index can be found again after a reopen. This is called after
        essentially every indexed row insert/update, but a root split is
        rare (root fanout is in the hundreds, so most tables never split
        their root at all) -- so this only pays for a full fsynced JSON
        rewrite of the catalog on the actual rare occasions the value
        changes, not on every call. When it doesn't change here, the
        in-memory root is already correct and the on-disk copy stays valid
        because WAL replay recomputes it from the `btree_insert` log
        (which is what durability for this value actually rests on, the
        same as any other DML effect -- see database.py's docstring).
        """
        idx = self.tables[table_name].indexes[index_name]
        if idx.root_page_id == new_root_page_id:
            return
        idx.root_page_id = new_root_page_id
        self._save()

    # -- persistence -------------------------------------------------------

    def _save(self) -> None:
        payload = {
            "tables": {
                name: {
                    "schema": [[c.name, c.type.value] for c in meta.schema.columns],
                    "heap_first_page_id": meta.heap_first_page_id,
                    "indexes": {
                        idx_name: {
                            "column": idx.column,
                            "root_page_id": idx.root_page_id,
                            "unique": idx.unique,
                            "key_type": idx.key_type.value,
                        }
                        for idx_name, idx in meta.indexes.items()
                    },
                }
                for name, meta in self.tables.items()
            }
        }
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.path)  # atomic on POSIX: never leaves a half-written catalog

    def _load(self) -> None:
        with open(self.path) as f:
            payload = json.load(f)
        for name, table_payload in payload.get("tables", {}).items():
            schema = Schema(
                columns=tuple(
                    Column(col_name, ColumnType(col_type))
                    for col_name, col_type in table_payload["schema"]
                )
            )
            meta = TableMeta(
                name=name,
                schema=schema,
                heap_first_page_id=table_payload["heap_first_page_id"],
            )
            for idx_name, idx_payload in table_payload.get("indexes", {}).items():
                meta.indexes[idx_name] = IndexMeta(
                    name=idx_name,
                    column=idx_payload["column"],
                    root_page_id=idx_payload["root_page_id"],
                    unique=idx_payload["unique"],
                    key_type=ColumnType(idx_payload["key_type"]),
                )
            self.tables[name] = meta
