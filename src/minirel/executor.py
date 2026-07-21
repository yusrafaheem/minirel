"""
minirel.executor
==================

The Volcano/iterator-model query executor: every operator exposes the
same three-method interface -- `open()`, `next()` (returns a row dict or
None at exhaustion), `close()` -- and operators are composed into a tree
by wrapping one another, so a full query plan is executed by pulling
rows through `root.next()` one at a time. This is the same execution
model used by Postgres, MySQL, and pretty much every non-vectorized SQL
engine; the planner (planner.py) is what decides *which* tree of these
operators to build for a given query.

Each row flowing through the tree is a plain dict (see expr.py's
docstring for the key-naming convention); base scan operators also
attach a `"__rid__"` entry so UPDATE/DELETE plans know which physical
row to mutate.
"""

from __future__ import annotations

from collections.abc import Iterator

from .expr import evaluate
from .index.btree import BPlusTree
from .sql.ast import Expr, FunctionCall, OrderByItem, SelectItem, Star
from .storage.heap_file import HeapFile
from .storage.page import RID
from .transaction import Transaction, TransactionManager
from .types import Schema, decode_row, tuple_payload, unpack_tuple_header


class Operator:
    def open(self) -> None:
        raise NotImplementedError

    def next(self) -> dict | None:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __iter__(self) -> Iterator[dict]:
        self.open()
        try:
            while (row := self.next()) is not None:
                yield row
        finally:
            self.close()


def _row_from_tuple(alias: str, schema: Schema, rid: RID, raw: bytes) -> dict:
    values = decode_row(tuple_payload(raw), schema)
    row = {f"{alias}.{col.name}": val for col, val in zip(schema.columns, values)}
    row.update({col.name: val for col, val in zip(schema.columns, values)})
    row["__rid__"] = rid
    return row


class SeqScanOperator(Operator):
    """Full scan of every live tuple in a table's heap, filtered down to
    what's visible to this transaction's MVCC snapshot.
    """

    def __init__(
        self,
        heap: HeapFile,
        schema: Schema,
        alias: str,
        txn: Transaction,
        txn_mgr: TransactionManager,
    ):
        self.heap = heap
        self.schema = schema
        self.alias = alias
        self.txn = txn
        self.txn_mgr = txn_mgr
        self._iter = None

    def open(self) -> None:
        self._iter = self.heap.scan()

    def next(self) -> dict | None:
        for rid, raw in self._iter:
            xmin, xmax = unpack_tuple_header(raw)
            if self.txn_mgr.is_visible(xmin, xmax, self.txn.snapshot):
                return _row_from_tuple(self.alias, self.schema, rid, raw)
        return None


class IndexScanOperator(Operator):
    """Point/range lookup through a B+-tree instead of a full heap scan.
    Still has to re-check MVCC visibility per tuple -- the index only
    tells you *where* a row physically is, not whether your snapshot can
    see the version currently stored there.
    """

    def __init__(
        self,
        tree: BPlusTree,
        heap: HeapFile,
        schema: Schema,
        alias: str,
        txn: Transaction,
        txn_mgr: TransactionManager,
        start=None,
        end=None,
    ):
        self.tree = tree
        self.heap = heap
        self.schema = schema
        self.alias = alias
        self.txn = txn
        self.txn_mgr = txn_mgr
        self.start = start
        self.end = end
        self._iter = None

    def open(self) -> None:
        self._iter = self.tree.range_scan(self.start, self.end)

    def next(self) -> dict | None:
        for _key, rid in self._iter:
            raw = self.heap.get(rid)
            if raw is None:
                continue  # physically reclaimed already (shouldn't happen w/o vacuum, be safe)
            xmin, xmax = unpack_tuple_header(raw)
            if self.txn_mgr.is_visible(xmin, xmax, self.txn.snapshot):
                return _row_from_tuple(self.alias, self.schema, rid, raw)
        return None


class FilterOperator(Operator):
    def __init__(self, child: Operator, predicate: Expr):
        self.child = child
        self.predicate = predicate

    def open(self) -> None:
        self.child.open()

    def next(self) -> dict | None:
        while (row := self.child.next()) is not None:
            if evaluate(self.predicate, row):
                return row
        return None

    def close(self) -> None:
        self.child.close()


class NestedLoopJoinOperator(Operator):
    """The simplest join algorithm: for every left row, rescan the right
    side from scratch and keep the pairs satisfying `on`. O(N*M) -- fine
    for the table sizes this engine is meant to demonstrate correctness
    on, and simple enough to read in one sitting, unlike a hash join.
    """

    def __init__(self, left: Operator, right_factory, on: Expr):
        self.left = left
        self.right_factory = right_factory  # zero-arg callable -> fresh Operator
        self.on = on
        self._left_row = None
        self._right = None

    def open(self) -> None:
        self.left.open()
        self._left_row = self.left.next()
        self._right = self.right_factory() if self._left_row is not None else None
        if self._right is not None:
            self._right.open()

    def next(self) -> dict | None:
        while self._left_row is not None:
            right_row = self._right.next()
            if right_row is None:
                self._right.close()
                self._left_row = self.left.next()
                if self._left_row is None:
                    return None
                self._right = self.right_factory()
                self._right.open()
                continue
            merged = {**self._left_row, **right_row}
            if evaluate(self.on, merged):
                return merged
        return None

    def close(self) -> None:
        self.left.close()
        if self._right is not None:
            self._right.close()


class ProjectOperator(Operator):
    def __init__(self, child: Operator, items: tuple[SelectItem, ...]):
        self.child = child
        self.items = items

    def open(self) -> None:
        self.child.open()

    def next(self) -> dict | None:
        row = self.child.next()
        if row is None:
            return None
        if len(self.items) == 1 and isinstance(self.items[0].expr, Star):
            # Prefer bare "column" over qualified "table.column" for a
            # `SELECT *` display, since _row_from_tuple always stores both.
            # Caveat (documented, not fixed -- a narrow scope cut): a join
            # across two tables that share a bare column name will collide
            # here, with whichever side was merged in last winning; use
            # explicit `table.column` in the SELECT list to disambiguate
            # that case instead of `*`.
            return {k: v for k, v in row.items() if not k.startswith("__") and "." not in k}
        out = {}
        for item in self.items:
            name = item.alias or _default_column_name(item)
            out[name] = evaluate(item.expr, row)
        return out

    def close(self) -> None:
        self.child.close()


def _default_column_name(item: SelectItem) -> str:
    from .sql.ast import ColumnRef

    if isinstance(item.expr, ColumnRef):
        return item.expr.name
    if isinstance(item.expr, FunctionCall):
        return item.expr.name.lower()
    return "expr"


class SortOperator(Operator):
    def __init__(self, child: Operator, order_by: tuple[OrderByItem, ...]):
        self.child = child
        self.order_by = order_by
        self._rows: list[dict] | None = None
        self._pos = 0

    def open(self) -> None:
        self.child.open()
        rows = []
        while (row := self.child.next()) is not None:
            rows.append(row)
        # Stable-sort once per key, from the least-significant key to the
        # most-significant: Python's sort is stable, so this standard trick
        # gives correct multi-key ordering with independent per-key
        # ascending/descending direction.
        for item in reversed(self.order_by):
            rows.sort(key=lambda r, e=item.expr: evaluate(e, r), reverse=item.descending)
        self._rows = rows
        self._pos = 0

    def next(self) -> dict | None:
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def close(self) -> None:
        self.child.close()


class LimitOperator(Operator):
    def __init__(self, child: Operator, limit: int):
        self.child = child
        self.limit = limit
        self._count = 0

    def open(self) -> None:
        self.child.open()
        self._count = 0

    def next(self) -> dict | None:
        if self._count >= self.limit:
            return None
        row = self.child.next()
        if row is None:
            return None
        self._count += 1
        return row

    def close(self) -> None:
        self.child.close()


_AGGREGATE_INITIAL = {"COUNT": 0, "SUM": 0, "AVG": None, "MIN": None, "MAX": None}


class _AggState:
    __slots__ = ("kind", "count", "total", "best")

    def __init__(self, kind: str):
        self.kind = kind
        self.count = 0
        self.total = 0
        self.best = None

    def add(self, value) -> None:
        if self.kind == "COUNT":
            if value is not None:
                self.count += 1
        elif value is None:
            return
        elif self.kind == "SUM":
            self.total += value
        elif self.kind == "AVG":
            self.total += value
            self.count += 1
        elif self.kind == "MIN":
            self.best = value if self.best is None else min(self.best, value)
        elif self.kind == "MAX":
            self.best = value if self.best is None else max(self.best, value)

    def result(self):
        if self.kind == "COUNT":
            return self.count
        if self.kind == "SUM":
            return self.total
        if self.kind == "AVG":
            return self.total / self.count if self.count else None
        return self.best


class HashAggregateOperator(Operator):
    """Groups rows by `group_by` column names (an empty tuple means "one
    implicit group over the whole input", e.g. bare `SELECT COUNT(*)`)
    and computes the requested aggregate SelectItems per group. Rows are
    fully materialized and bucketed in a dict keyed by the group-by
    values -- a hash aggregate, as opposed to the alternative of sorting
    by the group-by columns first.
    """

    def __init__(self, child: Operator, group_by: tuple[str, ...], items: tuple[SelectItem, ...]):
        self.child = child
        self.group_by = group_by
        self.items = items
        self._result_rows: list[dict] | None = None
        self._pos = 0

    def open(self) -> None:
        self.child.open()
        groups: dict[tuple, dict[str, _AggState]] = {}
        group_values: dict[tuple, dict[str, object]] = {}

        while (row := self.child.next()) is not None:
            key = tuple(row.get(col) for col in self.group_by)
            if key not in groups:
                groups[key] = {}
                for i, item in enumerate(self.items):
                    if isinstance(item.expr, FunctionCall):
                        groups[key][i] = _AggState(item.expr.name)
                group_values[key] = {col: row.get(col) for col in self.group_by}
            for i, item in enumerate(self.items):
                if isinstance(item.expr, FunctionCall):
                    if isinstance(item.expr.arg, Star):
                        groups[key][i].add(1)  # COUNT(*): count rows, not a real column value
                    else:
                        groups[key][i].add(evaluate(item.expr.arg, row))

        self._result_rows = []
        for key, agg_states in groups.items():
            out = {}
            for i, item in enumerate(self.items):
                name = item.alias or _default_column_name(item)
                if isinstance(item.expr, FunctionCall):
                    out[name] = agg_states[i].result()
                else:
                    src_name = item.expr.name if hasattr(item.expr, "name") else name
                    out[name] = group_values[key].get(src_name)
            self._result_rows.append(out)

        if not groups and not self.group_by:
            # No input rows at all, but an aggregate over the whole table
            # (e.g. SELECT COUNT(*) FROM empty_table) must still produce
            # one row -- COUNT(*) of nothing is 0, not "no rows".
            out = {}
            for item in self.items:
                name = item.alias or _default_column_name(item)
                if isinstance(item.expr, FunctionCall):
                    out[name] = _AggState(item.expr.name).result()
            self._result_rows.append(out)

        self._pos = 0

    def next(self) -> dict | None:
        if self._pos >= len(self._result_rows):
            return None
        row = self._result_rows[self._pos]
        self._pos += 1
        return row

    def close(self) -> None:
        self.child.close()
