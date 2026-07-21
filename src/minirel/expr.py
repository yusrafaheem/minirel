"""
minirel.expr
=============

Evaluates a parsed WHERE/ON/SELECT-item expression against a row.

Rows are plain dicts. A single-table scan produces bare `{"id": 1, ...}`
keys; a join's output additionally carries `{"orders.id": 1, ...}`
qualified keys so a predicate like `orders.customer_id = customers.id`
resolves unambiguously, while unqualified references still work as long
as they aren't ambiguous across the joined tables (checked at resolve
time, not guessed).
"""

from __future__ import annotations

from .sql.ast import BinaryOp, ColumnRef, Expr, FunctionCall, Literal, Star, UnaryOp


class ColumnNotFoundError(Exception):
    pass


class AmbiguousColumnError(Exception):
    pass


def resolve_column(row: dict, col: ColumnRef):
    if col.table is not None:
        key = f"{col.table}.{col.name}"
        if key in row:
            return row[key]
        raise ColumnNotFoundError(f"no such column: {key}")

    if col.name in row:
        return row[col.name]
    matches = [k for k in row if k.endswith(f".{col.name}")]
    if len(matches) == 1:
        return row[matches[0]]
    if len(matches) > 1:
        raise AmbiguousColumnError(f"column reference {col.name!r} is ambiguous: {matches}")
    raise ColumnNotFoundError(f"no such column: {col.name}")


_COMPARATORS = {
    "=": lambda a, b: a == b,
    "<>": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def evaluate(expr: Expr, row: dict):
    if isinstance(expr, Literal):
        return expr.value
    if isinstance(expr, ColumnRef):
        return resolve_column(row, expr)
    if isinstance(expr, Star):
        raise ValueError("'*' cannot be evaluated as a scalar expression")
    if isinstance(expr, UnaryOp):
        if expr.op == "NOT":
            return not _truthy(evaluate(expr.operand, row))
        raise ValueError(f"unknown unary operator: {expr.op}")
    if isinstance(expr, BinaryOp):
        if expr.op == "AND":
            return _truthy(evaluate(expr.left, row)) and _truthy(evaluate(expr.right, row))
        if expr.op == "OR":
            return _truthy(evaluate(expr.left, row)) or _truthy(evaluate(expr.right, row))
        left = evaluate(expr.left, row)
        right = evaluate(expr.right, row)
        if left is None or right is None:
            return False  # SQL NULL-comparison semantics: unknown, treated as not-matching here
        return _COMPARATORS[expr.op](left, right)
    if isinstance(expr, FunctionCall):
        raise ValueError(
            f"aggregate function {expr.name}(...) is only valid as a top-level SELECT item, "
            "not inside WHERE/ON (minirel doesn't support HAVING or nested aggregates)"
        )
    raise TypeError(f"unknown expression node: {expr!r}")


def _truthy(value) -> bool:
    return bool(value)
