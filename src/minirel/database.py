"""
minirel.database
==================

The top-level `Database` class: parses SQL, drives the planner/executor
for SELECT, and directly implements INSERT/UPDATE/DELETE/CREATE (there's
no benefit to routing single-row mutations through the Volcano operator
tree -- those exist to let SELECT compose scan/filter/join/aggregate
freely, not because every statement needs them).

How INSERT/UPDATE/DELETE interact with MVCC + the WAL + indexes:

- INSERT stamps the new tuple with `xmin=txn_id, xmax=INFINITY`, appends
  it to the heap, logs a `heap_insert` WAL record (base64 tuple bytes),
  then inserts `(key, rid)` into every index on the table and logs a
  `btree_insert` record per index.
- DELETE doesn't physically remove anything -- it stamps the current
  version's `xmax = txn_id` (a `heap_set_xmax` WAL record) after a
  first-committer-wins conflict check. The row becomes invisible to
  future snapshots without ever being unlinked from its indexes.
- UPDATE is "delete the old version, insert a new one": stamp the old
  version's xmax, then insert a brand new tuple with the updated values
  and a fresh RID, then insert `(new_key, new_rid)` into every index.

Because deletes/updates never remove index entries, every index
accumulates "dead" entries for old row versions over time -- the trade
that comes with skipping a VACUUM implementation. `IndexScanOperator`
re-checks MVCC visibility against the *current* heap tuple at each dead
entry, so results are still correct; the entries just aren't reclaimed.
That's the same trade-off already made in heap_file.py (no page
reclamation) and transaction.py (no undo logging for aborts) -- see
README.md's "what's simplified" section for the full list in one place.

Recovery (`_recover`, run once at construction) replays every logged
operation belonging to a transaction that committed since the last
checkpoint, in original order -- see transaction.py's
`committed_operations_since_checkpoint` and the docstring on
`uncommitted_txn_ids_since_checkpoint` for how uncommitted transactions'
possibly-already-flushed pages are kept invisible.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from .catalog import Catalog
from .executor import Operator
from .expr import evaluate
from .index.btree import BPlusTree, DuplicateKeyError
from .planner import build_row_source_plan, build_select_plan
from .sql.ast import CreateIndex, CreateTable, Delete, Insert, Select, Statement, Update
from .sql.parser import parse
from .storage.buffer_pool import BufferPoolManager
from .storage.disk_manager import DiskManager
from .storage.heap_file import HeapFile
from .storage.page import RID
from .transaction import (
    Transaction,
    TransactionManager,
    committed_operations_since_checkpoint,
    next_txn_id_after_recovery,
    uncommitted_txn_ids_since_checkpoint,
)
from .types import (
    INFINITY_TXN,
    Column,
    ColumnType,
    Schema,
    decode_row,
    encode_row,
    pack_tuple,
    tuple_payload,
    unpack_tuple_header,
    with_xmax,
)
from .wal import WriteAheadLog


@dataclass
class ExecuteResult:
    rows: list[dict] | None = None
    row_count: int = 0
    message: str = ""

    def __iter__(self):
        return iter(self.rows or [])

    def __len__(self):
        return len(self.rows or [])


class Database:
    def __init__(self, path: str, buffer_pool_size: int = 128):
        self.path = path
        self.disk_manager = DiskManager(path)
        self.buffer_pool = BufferPoolManager(self.disk_manager, pool_size=buffer_pool_size)
        self.catalog = Catalog(self.buffer_pool, path + ".catalog.json")
        self.wal = WriteAheadLog(path + ".wal")

        next_id = next_txn_id_after_recovery(self.wal)
        aborted = uncommitted_txn_ids_since_checkpoint(self.wal)
        self.txn_mgr = TransactionManager(
            self.wal, next_txn_id=next_id, initial_aborted_xids=aborted
        )
        self._recover()

    # -- lifecycle -----------------------------------------------------------

    def _recover(self) -> None:
        for record in committed_operations_since_checkpoint(self.wal):
            self._apply_recovery_op(record.payload)
        if self.buffer_pool.disk_writes or self.buffer_pool.disk_reads:
            self.buffer_pool.flush_all()

    def _apply_recovery_op(self, op: dict) -> None:
        table = self.catalog.get_table(op["table"])
        if op["kind"] == "heap_insert":
            heap = HeapFile(self.buffer_pool, first_page_id=table.heap_first_page_id)
            heap.insert(base64.b64decode(op["tuple"]))
        elif op["kind"] == "heap_set_xmax":
            heap = HeapFile(self.buffer_pool, first_page_id=table.heap_first_page_id)
            rid = RID(op["rid"][0], op["rid"][1])
            raw = heap.get(rid)
            if raw is not None:
                heap.update_in_place(rid, with_xmax(raw, op["xmax"]))
        elif op["kind"] == "btree_insert":
            idx_meta = table.indexes[op["index"]]
            tree = BPlusTree(
                self.buffer_pool,
                key_type=idx_meta.key_type,
                root_page_id=idx_meta.root_page_id,
                unique=False,  # replaying a log that was valid the first time; don't re-validate
            )
            tree.insert(op["key"], RID(op["rid"][0], op["rid"][1]))
            self.catalog.update_index_root(table.name, op["index"], tree.root_page_id)
        else:
            raise ValueError(f"unknown recovery op kind: {op['kind']!r}")

    def checkpoint(self) -> None:
        """Flush every dirty page and mark the WAL so recovery never has to
        replay past this point. Deliberately refuses to run with any
        transaction active -- see uncommitted_txn_ids_since_checkpoint's
        docstring for why that's what makes recovery's visibility handling
        for crashed/uncommitted transactions sound.
        """
        if self.txn_mgr.has_active_transactions:
            raise RuntimeError("cannot checkpoint while a transaction is active")
        self.buffer_pool.flush_all()
        self.wal.log_checkpoint()

    def close(self) -> None:
        self.buffer_pool.flush_all()
        self.disk_manager.close()
        self.wal.close()

    # -- transactions ----------------------------------------------------

    def begin(self) -> Transaction:
        return self.txn_mgr.begin()

    # -- SQL entry point -------------------------------------------------

    def execute(self, sql: str, txn: Transaction | None = None) -> ExecuteResult:
        stmt = parse(sql)
        owns_txn = txn is None and not isinstance(stmt, (CreateTable, CreateIndex))
        if owns_txn:
            txn = self.txn_mgr.begin()
        try:
            result = self._dispatch(stmt, txn)
        except Exception:
            if owns_txn and txn is not None:
                txn.abort()
            raise
        else:
            if owns_txn:
                txn.commit()
            return result

    def _dispatch(self, stmt: Statement, txn: Transaction | None) -> ExecuteResult:
        if isinstance(stmt, CreateTable):
            return self._execute_create_table(stmt)
        if isinstance(stmt, CreateIndex):
            return self._execute_create_index(stmt)
        if isinstance(stmt, Insert):
            return self._execute_insert(stmt, txn)
        if isinstance(stmt, Select):
            return self._execute_select(stmt, txn)
        if isinstance(stmt, Update):
            return self._execute_update(stmt, txn)
        if isinstance(stmt, Delete):
            return self._execute_delete(stmt, txn)
        raise ValueError(f"unsupported statement: {stmt!r}")  # pragma: no cover

    # -- DDL -----------------------------------------------------------------

    def _execute_create_table(self, stmt: CreateTable) -> ExecuteResult:
        schema = Schema(
            columns=tuple(Column(c.name, ColumnType(c.type)) for c in stmt.columns)
        )
        self.catalog.create_table(stmt.table, schema)
        return ExecuteResult(message=f"CREATE TABLE {stmt.table}")

    def _execute_create_index(self, stmt: CreateIndex) -> ExecuteResult:
        table = self.catalog.get_table(stmt.table)
        meta = self.catalog.create_index(
            stmt.table, stmt.index_name, stmt.column, unique=stmt.unique
        )
        heap = HeapFile(self.buffer_pool, first_page_id=table.heap_first_page_id)
        tree = BPlusTree(
            self.buffer_pool, key_type=meta.key_type, root_page_id=meta.root_page_id, unique=False
        )
        col_index = table.schema.index_of(stmt.column)
        for rid, raw in heap.scan():
            values = decode_row(tuple_payload(raw), table.schema)
            tree.insert(values[col_index], rid)
        self.catalog.update_index_root(stmt.table, stmt.index_name, tree.root_page_id)
        return ExecuteResult(message=f"CREATE INDEX {stmt.index_name}")

    # -- DML -----------------------------------------------------------------

    def _resolve_insert_values(self, stmt: Insert, table) -> list[list]:
        columns = stmt.columns or tuple(table.schema.names)
        rows = []
        for row_exprs in stmt.rows:
            if len(row_exprs) != len(columns):
                raise ValueError(
                    f"INSERT has {len(row_exprs)} values but {len(columns)} columns were named"
                )
            value_map = {name: evaluate(e, {}) for name, e in zip(columns, row_exprs)}
            missing = [c for c in table.schema.names if c not in value_map]
            if missing:
                raise ValueError(f"INSERT is missing values for columns: {missing}")
            rows.append([value_map[name] for name in table.schema.names])
        return rows

    def _check_unique_constraint(
        self, table, idx_meta, key, txn: Transaction, heap: HeapFile, exclude_rid: RID | None
    ) -> None:
        """Enforce uniqueness at the row-visibility level rather than inside
        BPlusTree itself: because UPDATE leaves the old (now-dead) index
        entry in place (see module docstring -- no vacuum), a plain
        "does this key already exist anywhere in the tree" check would
        reject a row being updated back through the *same* key it already
        had. Instead, walk every entry for `key` and only object if some
        *other*, currently-visible row already holds it.
        """
        if not idx_meta.unique:
            return
        tree = BPlusTree(
            self.buffer_pool,
            key_type=idx_meta.key_type,
            root_page_id=idx_meta.root_page_id,
            unique=False,
        )
        for rid in tree.search(key):
            if rid == exclude_rid:
                continue
            raw = heap.get(rid)
            if raw is None:
                continue
            xmin, xmax = unpack_tuple_header(raw)
            if self.txn_mgr.is_visible(xmin, xmax, txn.snapshot):
                raise DuplicateKeyError(f"duplicate key in unique index {idx_meta.name!r}: {key!r}")

    def _insert_into_indexes(
        self,
        table,
        txn: Transaction,
        value_map: dict,
        rid: RID,
        heap: HeapFile,
        exclude_rid: RID | None = None,
    ) -> None:
        for idx_name, idx_meta in table.indexes.items():
            key = value_map[idx_meta.column]
            self._check_unique_constraint(table, idx_meta, key, txn, heap, exclude_rid)
            tree = BPlusTree(
                self.buffer_pool,
                key_type=idx_meta.key_type,
                root_page_id=idx_meta.root_page_id,
                unique=False,  # already enforced above, with MVCC visibility taken into account
            )
            tree.insert(key, rid)
            self.catalog.update_index_root(table.name, idx_name, tree.root_page_id)
            txn.log_operation(
                "btree_insert",
                table=table.name,
                index=idx_name,
                key=key,
                rid=[rid.page_id, rid.slot_id],
            )

    def _execute_insert(self, stmt: Insert, txn: Transaction) -> ExecuteResult:
        table = self.catalog.get_table(stmt.table)
        heap = HeapFile(self.buffer_pool, first_page_id=table.heap_first_page_id)
        rows = self._resolve_insert_values(stmt, table)

        count = 0
        for values in rows:
            payload = encode_row(values, table.schema)
            tup = pack_tuple(txn.txn_id, INFINITY_TXN, payload)
            rid = heap.insert(tup)
            txn.log_operation(
                "heap_insert",
                table=table.name,
                tuple=base64.b64encode(tup).decode("ascii"),
                rid=[rid.page_id, rid.slot_id],
            )
            value_map = dict(zip(table.schema.names, values))
            self._insert_into_indexes(table, txn, value_map, rid, heap)
            count += 1
        return ExecuteResult(row_count=count, message=f"INSERT {count}")

    def _execute_select(self, stmt: Select, txn: Transaction) -> ExecuteResult:
        plan: Operator = build_select_plan(stmt, self.catalog, self.buffer_pool, txn, self.txn_mgr)
        rows = list(plan)
        return ExecuteResult(rows=rows, row_count=len(rows), message=f"SELECT {len(rows)}")

    def _execute_update(self, stmt: Update, txn: Transaction) -> ExecuteResult:
        table = self.catalog.get_table(stmt.table)
        heap = HeapFile(self.buffer_pool, first_page_id=table.heap_first_page_id)
        plan = build_row_source_plan(
            stmt.table, stmt.table, stmt.where, self.catalog, self.buffer_pool, txn, self.txn_mgr
        )
        matches = list(plan)  # materialize: we're about to mutate the heap we're scanning

        count = 0
        for row in matches:
            rid: RID = row["__rid__"]
            raw = heap.get(rid)
            if raw is None:
                continue
            xmin, xmax = unpack_tuple_header(raw)
            self.txn_mgr.check_write_conflict(txn, xmax)

            old_values = decode_row(tuple_payload(raw), table.schema)
            value_map = dict(zip(table.schema.names, old_values))
            for col, expr in stmt.assignments:
                value_map[col] = evaluate(expr, {})
            new_values = [value_map[name] for name in table.schema.names]

            heap.update_in_place(rid, with_xmax(raw, txn.txn_id))
            txn.log_operation(
                "heap_set_xmax", table=table.name, rid=[rid.page_id, rid.slot_id], xmax=txn.txn_id
            )

            new_payload = encode_row(new_values, table.schema)
            new_tuple = pack_tuple(txn.txn_id, INFINITY_TXN, new_payload)
            new_rid = heap.insert(new_tuple)
            txn.log_operation(
                "heap_insert",
                table=table.name,
                tuple=base64.b64encode(new_tuple).decode("ascii"),
                rid=[new_rid.page_id, new_rid.slot_id],
            )
            self._insert_into_indexes(table, txn, value_map, new_rid, heap, exclude_rid=rid)
            count += 1
        return ExecuteResult(row_count=count, message=f"UPDATE {count}")

    def _execute_delete(self, stmt: Delete, txn: Transaction) -> ExecuteResult:
        table = self.catalog.get_table(stmt.table)
        heap = HeapFile(self.buffer_pool, first_page_id=table.heap_first_page_id)
        plan = build_row_source_plan(
            stmt.table, stmt.table, stmt.where, self.catalog, self.buffer_pool, txn, self.txn_mgr
        )
        matches = list(plan)

        count = 0
        for row in matches:
            rid: RID = row["__rid__"]
            raw = heap.get(rid)
            if raw is None:
                continue
            xmin, xmax = unpack_tuple_header(raw)
            self.txn_mgr.check_write_conflict(txn, xmax)
            heap.update_in_place(rid, with_xmax(raw, txn.txn_id))
            txn.log_operation(
                "heap_set_xmax", table=table.name, rid=[rid.page_id, rid.slot_id], xmax=txn.txn_id
            )
            count += 1
        return ExecuteResult(row_count=count, message=f"DELETE {count}")
