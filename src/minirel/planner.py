"""
minirel.planner
=================

Turns a parsed `Select` (or a bare table + WHERE, for UPDATE/DELETE row
selection) into a tree of executor.py operators.

The one real "optimization" here is predicate pushdown onto an index:
if the WHERE clause has a top-level equality conjunct on an indexed
column (`col = <literal>`, found anywhere in a chain of ANDs -- not
inside an OR, which can't be pushed down this way), the base scan uses
`IndexScanOperator` instead of `SeqScanOperator` and that one conjunct
is dropped from the residual filter applied on top. Everything else
(joins, aggregates, sort, limit) is a fixed, un-optimized pipeline shape
-- there's no cost-based join ordering or predicate reordering, which
real optimizers spend most of their complexity on. `benchmarks/
bench_point_lookup.py` is what demonstrates *why* this one pushdown
matters (index point lookups vs. scanning the whole table).

Evaluation order matches standard SQL semantics: FROM/JOIN -> WHERE ->
(GROUP BY + aggregates) -> ORDER BY -> LIMIT -> SELECT projection --
with one deliberate simplification: for a non-aggregate query, ORDER BY
is evaluated against the *pre-projection* row (so it can reference any
source column, not just ones in the SELECT list) rather than against
SELECT-list aliases the way some real engines additionally allow.
"""

from __future__ import annotations

from .catalog import Catalog
from .executor import (
    FilterOperator,
    HashAggregateOperator,
    IndexScanOperator,
    LimitOperator,
    NestedLoopJoinOperator,
    Operator,
    ProjectOperator,
    SeqScanOperator,
    SortOperator,
)
from .index.btree import BPlusTree
from .sql.ast import BinaryOp, ColumnRef, Expr, FunctionCall, Literal, Select
from .storage.buffer_pool import BufferPoolManager
from .storage.heap_file import HeapFile
from .transaction import Transaction, TransactionManager


def _flatten_and(expr: Expr | None) -> list[Expr]:
    if expr is None:
        return []
    if isinstance(expr, BinaryOp) and expr.op == "AND":
        return _flatten_and(expr.left) + _flatten_and(expr.right)
    return [expr]


def _rebuild_and(conjuncts: list[Expr]) -> Expr | None:
    if not conjuncts:
        return None
    expr = conjuncts[0]
    for c in conjuncts[1:]:
        expr = BinaryOp("AND", expr, c)
    return expr


def _matching_equality(conjunct: Expr, alias: str, indexed_columns: dict[str, str]):
    """If `conjunct` is `col = literal` (in either order) for a column that
    has an index, return (index_name, literal_value); else None.
    """
    if not (isinstance(conjunct, BinaryOp) and conjunct.op == "="):
        return None
    pairs = [(conjunct.left, conjunct.right), (conjunct.right, conjunct.left)]
    for maybe_col, maybe_lit in pairs:
        if isinstance(maybe_col, ColumnRef) and isinstance(maybe_lit, Literal):
            if maybe_col.table is not None and maybe_col.table != alias:
                continue
            if maybe_col.name in indexed_columns:
                return indexed_columns[maybe_col.name], maybe_lit.value
    return None


def build_base_scan(
    table_name: str,
    alias: str,
    where: Expr | None,
    catalog: Catalog,
    buffer_pool: BufferPoolManager,
    txn: Transaction,
    txn_mgr: TransactionManager,
) -> tuple[Operator, Expr | None]:
    """Return (scan_operator, residual_where) -- residual_where is what
    still needs to be applied with a FilterOperator on top (None if the
    index scan already covers the whole predicate).
    """
    meta = catalog.get_table(table_name)
    heap = HeapFile(buffer_pool, first_page_id=meta.heap_first_page_id)
    indexed_columns = {idx.column: name for name, idx in meta.indexes.items()}

    conjuncts = _flatten_and(where)
    for i, conjunct in enumerate(conjuncts):
        match = _matching_equality(conjunct, alias, indexed_columns)
        if match is None:
            continue
        index_name, value = match
        idx_meta = meta.indexes[index_name]
        tree = BPlusTree(
            buffer_pool,
            key_type=idx_meta.key_type,
            root_page_id=idx_meta.root_page_id,
            unique=idx_meta.unique,
        )
        scan = IndexScanOperator(
            tree, heap, meta.schema, alias, txn, txn_mgr, start=value, end=value
        )
        residual = _rebuild_and(conjuncts[:i] + conjuncts[i + 1 :])
        return scan, residual

    scan = SeqScanOperator(heap, meta.schema, alias, txn, txn_mgr)
    return scan, where


def build_row_source_plan(
    table_name: str,
    alias: str,
    where: Expr | None,
    catalog: Catalog,
    buffer_pool: BufferPoolManager,
    txn: Transaction,
    txn_mgr: TransactionManager,
) -> Operator:
    """Scan + filter, yielding full (unprojected) rows with `__rid__` --
    what UPDATE and DELETE use to find which physical rows to mutate.
    """
    scan, residual = build_base_scan(table_name, alias, where, catalog, buffer_pool, txn, txn_mgr)
    if residual is not None:
        return FilterOperator(scan, residual)
    return scan


def build_select_plan(
    stmt: Select,
    catalog: Catalog,
    buffer_pool: BufferPoolManager,
    txn: Transaction,
    txn_mgr: TransactionManager,
) -> Operator:
    left_alias = stmt.table_alias or stmt.table
    plan = build_row_source_plan(
        stmt.table, left_alias, stmt.where, catalog, buffer_pool, txn, txn_mgr
    )

    if stmt.join is not None:
        right_alias = stmt.join.alias or stmt.join.table
        right_table = stmt.join.table

        def right_factory(table=right_table, alias=right_alias):
            meta = catalog.get_table(table)
            heap = HeapFile(buffer_pool, first_page_id=meta.heap_first_page_id)
            return SeqScanOperator(heap, meta.schema, alias, txn, txn_mgr)

        plan = NestedLoopJoinOperator(plan, right_factory, stmt.join.on)

    has_aggregates = any(isinstance(item.expr, FunctionCall) for item in stmt.items)

    if has_aggregates or stmt.group_by:
        plan = HashAggregateOperator(plan, stmt.group_by, stmt.items)
        if stmt.order_by:
            plan = SortOperator(plan, stmt.order_by)
        if stmt.limit is not None:
            plan = LimitOperator(plan, stmt.limit)
        return plan

    if stmt.order_by:
        plan = SortOperator(plan, stmt.order_by)
    if stmt.limit is not None:
        plan = LimitOperator(plan, stmt.limit)
    return ProjectOperator(plan, stmt.items)
