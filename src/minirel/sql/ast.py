"""
minirel.sql.ast
==================

Plain dataclasses for every node the parser can produce. Expressions
(WHERE/ON clauses, SELECT items) are a small tree of Literal /
ColumnRef / UnaryOp / BinaryOp / FunctionCall; statements are one node
per SQL statement kind. The planner (planner.py) consumes these
directly -- there's no separate "resolved"/"bound" AST, which is a
scope cut appropriate for a single-schema-lookup engine with no
subqueries or CTEs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# -- expressions -------------------------------------------------------------


class Expr:
    """Marker base class for all expression nodes."""


@dataclass(frozen=True, slots=True)
class Literal(Expr):
    value: object  # int | float | str | bool | None


@dataclass(frozen=True, slots=True)
class ColumnRef(Expr):
    name: str
    table: str | None = None


@dataclass(frozen=True, slots=True)
class Star(Expr):
    pass


@dataclass(frozen=True, slots=True)
class UnaryOp(Expr):
    op: str  # "NOT"
    operand: Expr


@dataclass(frozen=True, slots=True)
class BinaryOp(Expr):
    op: str  # "=", "<>", "<", "<=", ">", ">=", "AND", "OR"
    left: Expr
    right: Expr


@dataclass(frozen=True, slots=True)
class FunctionCall(Expr):
    name: str  # "COUNT" | "SUM" | "AVG" | "MIN" | "MAX"
    arg: Expr  # Star() for COUNT(*)


# -- statements ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColumnDef:
    name: str
    type: str  # "INT" | "FLOAT" | "TEXT" | "BOOL"


@dataclass(frozen=True, slots=True)
class CreateTable:
    table: str
    columns: tuple[ColumnDef, ...]


@dataclass(frozen=True, slots=True)
class CreateIndex:
    index_name: str
    table: str
    column: str
    unique: bool = False


@dataclass(frozen=True, slots=True)
class Insert:
    table: str
    columns: tuple[str, ...] | None  # None means "all columns, in schema order"
    rows: tuple[tuple[Expr, ...], ...]


@dataclass(frozen=True, slots=True)
class SelectItem:
    expr: Expr
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class JoinClause:
    table: str
    on: Expr
    alias: str | None = None


@dataclass(frozen=True, slots=True)
class OrderByItem:
    expr: Expr
    descending: bool = False


@dataclass(frozen=True, slots=True)
class Select:
    items: tuple[SelectItem, ...]
    table: str
    table_alias: str | None = None
    join: JoinClause | None = None
    where: Expr | None = None
    group_by: tuple[str, ...] = field(default_factory=tuple)
    order_by: tuple[OrderByItem, ...] = field(default_factory=tuple)
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class Update:
    table: str
    assignments: tuple[tuple[str, Expr], ...]
    where: Expr | None = None


@dataclass(frozen=True, slots=True)
class Delete:
    table: str
    where: Expr | None = None


Statement = CreateTable | CreateIndex | Insert | Select | Update | Delete
