"""
minirel.sql.parser
=====================

A textbook recursive-descent parser: one method per grammar production,
each consuming tokens from a flat list and returning an AST node. No
parser-generator, no grammar DSL -- the call graph *is* the grammar.

Expression parsing implements operator precedence by nesting: `or_expr`
calls `and_expr` calls `not_expr` calls `comparison` calls `primary`,
so `AND` always binds tighter than `OR`, `NOT` tighter than `AND`, and
comparisons bind tighter than all three -- the standard way to encode
precedence in a hand-written descent parser without a precedence-climbing
table.
"""

from __future__ import annotations

from .ast import (
    BinaryOp,
    ColumnDef,
    ColumnRef,
    CreateIndex,
    CreateTable,
    Delete,
    Expr,
    FunctionCall,
    Insert,
    JoinClause,
    Literal,
    OrderByItem,
    Select,
    SelectItem,
    Star,
    Statement,
    UnaryOp,
    Update,
)
from .lexer import Token, tokenize

_COMPARISON_OPS = {"=", "<>", "!=", "<", "<=", ">", ">="}
_AGGREGATE_NAMES = {"COUNT", "SUM", "AVG", "MIN", "MAX"}


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    # -- token stream helpers ------------------------------------------------

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _check(self, kind: str, value: str | None = None) -> bool:
        tok = self._peek()
        if tok.kind != kind:
            return False
        return value is None or tok.value == value

    def _match(self, kind: str, value: str | None = None) -> Token | None:
        if self._check(kind, value):
            return self._advance()
        return None

    def _expect(self, kind: str, value: str | None = None) -> Token:
        tok = self._peek()
        if not self._check(kind, value):
            expected = value or kind
            raise ParseError(f"expected {expected!r} but found {tok.value!r} at position {tok.pos}")
        return self._advance()

    # -- entry point ------------------------------------------------------

    def parse_statement(self) -> Statement:
        if self._check("KEYWORD", "CREATE"):
            stmt = self._parse_create()
        elif self._check("KEYWORD", "INSERT"):
            stmt = self._parse_insert()
        elif self._check("KEYWORD", "SELECT"):
            stmt = self._parse_select()
        elif self._check("KEYWORD", "UPDATE"):
            stmt = self._parse_update()
        elif self._check("KEYWORD", "DELETE"):
            stmt = self._parse_delete()
        else:
            tok = self._peek()
            raise ParseError(f"unexpected token {tok.value!r} at position {tok.pos}")
        self._match("OP", ";")
        self._expect("EOF")
        return stmt

    # -- DDL -----------------------------------------------------------------

    def _parse_create(self) -> CreateTable | CreateIndex:
        self._expect("KEYWORD", "CREATE")
        unique = bool(self._match("KEYWORD", "UNIQUE"))
        if self._match("KEYWORD", "TABLE"):
            table = self._expect("IDENT").value
            self._expect("OP", "(")
            columns = [self._parse_column_def()]
            while self._match("OP", ","):
                columns.append(self._parse_column_def())
            self._expect("OP", ")")
            return CreateTable(table=table, columns=tuple(columns))

        self._expect("KEYWORD", "INDEX")
        index_name = self._expect("IDENT").value
        self._expect("KEYWORD", "ON")
        table = self._expect("IDENT").value
        self._expect("OP", "(")
        column = self._expect("IDENT").value
        self._expect("OP", ")")
        return CreateIndex(index_name=index_name, table=table, column=column, unique=unique)

    def _parse_column_def(self) -> ColumnDef:
        name = self._expect("IDENT").value
        type_tok = self._advance()
        if type_tok.value.upper() not in ("INT", "FLOAT", "TEXT", "BOOL"):
            raise ParseError(f"unknown column type {type_tok.value!r} at position {type_tok.pos}")
        return ColumnDef(name=name, type=type_tok.value.upper())

    # -- INSERT --------------------------------------------------------------

    def _parse_insert(self) -> Insert:
        self._expect("KEYWORD", "INSERT")
        self._expect("KEYWORD", "INTO")
        table = self._expect("IDENT").value

        columns = None
        if self._match("OP", "("):
            names = [self._expect("IDENT").value]
            while self._match("OP", ","):
                names.append(self._expect("IDENT").value)
            self._expect("OP", ")")
            columns = tuple(names)

        self._expect("KEYWORD", "VALUES")
        rows = [self._parse_value_row()]
        while self._match("OP", ","):
            rows.append(self._parse_value_row())
        return Insert(table=table, columns=columns, rows=tuple(rows))

    def _parse_value_row(self) -> tuple[Expr, ...]:
        self._expect("OP", "(")
        values = [self._parse_expr()]
        while self._match("OP", ","):
            values.append(self._parse_expr())
        self._expect("OP", ")")
        return tuple(values)

    # -- SELECT --------------------------------------------------------------

    def _parse_select(self) -> Select:
        self._expect("KEYWORD", "SELECT")
        items = self._parse_select_items()
        self._expect("KEYWORD", "FROM")
        table = self._expect("IDENT").value
        table_alias = self._parse_optional_alias()

        join = None
        if self._match("KEYWORD", "JOIN"):
            join_table = self._expect("IDENT").value
            join_alias = self._parse_optional_alias()
            self._expect("KEYWORD", "ON")
            on = self._parse_expr()
            join = JoinClause(table=join_table, on=on, alias=join_alias)

        where = None
        if self._match("KEYWORD", "WHERE"):
            where = self._parse_expr()

        group_by: tuple[str, ...] = ()
        if self._match("KEYWORD", "GROUP"):
            self._expect("KEYWORD", "BY")
            cols = [self._expect("IDENT").value]
            while self._match("OP", ","):
                cols.append(self._expect("IDENT").value)
            group_by = tuple(cols)

        order_by: tuple[OrderByItem, ...] = ()
        if self._match("KEYWORD", "ORDER"):
            self._expect("KEYWORD", "BY")
            items_ob = [self._parse_order_item()]
            while self._match("OP", ","):
                items_ob.append(self._parse_order_item())
            order_by = tuple(items_ob)

        limit = None
        if self._match("KEYWORD", "LIMIT"):
            limit = int(self._expect("NUMBER").value)

        return Select(
            items=items,
            table=table,
            table_alias=table_alias,
            join=join,
            where=where,
            group_by=group_by,
            order_by=order_by,
            limit=limit,
        )

    def _parse_optional_alias(self) -> str | None:
        if self._match("KEYWORD", "AS"):
            return self._expect("IDENT").value
        if self._check("IDENT"):
            return self._advance().value
        return None

    def _parse_select_items(self) -> tuple[SelectItem, ...]:
        if self._match("OP", "*"):
            return (SelectItem(expr=Star()),)
        items = [self._parse_select_item()]
        while self._match("OP", ","):
            items.append(self._parse_select_item())
        return tuple(items)

    def _parse_select_item(self) -> SelectItem:
        expr = self._parse_expr()
        alias = None
        if self._match("KEYWORD", "AS"):
            alias = self._expect("IDENT").value
        return SelectItem(expr=expr, alias=alias)

    def _parse_order_item(self) -> OrderByItem:
        expr = self._parse_expr()
        descending = False
        if self._match("KEYWORD", "DESC"):
            descending = True
        else:
            self._match("KEYWORD", "ASC")
        return OrderByItem(expr=expr, descending=descending)

    # -- UPDATE / DELETE -------------------------------------------------------

    def _parse_update(self) -> Update:
        self._expect("KEYWORD", "UPDATE")
        table = self._expect("IDENT").value
        self._expect("KEYWORD", "SET")
        assignments = [self._parse_assignment()]
        while self._match("OP", ","):
            assignments.append(self._parse_assignment())
        where = None
        if self._match("KEYWORD", "WHERE"):
            where = self._parse_expr()
        return Update(table=table, assignments=tuple(assignments), where=where)

    def _parse_assignment(self) -> tuple[str, Expr]:
        name = self._expect("IDENT").value
        self._expect("OP", "=")
        return name, self._parse_expr()

    def _parse_delete(self) -> Delete:
        self._expect("KEYWORD", "DELETE")
        self._expect("KEYWORD", "FROM")
        table = self._expect("IDENT").value
        where = None
        if self._match("KEYWORD", "WHERE"):
            where = self._parse_expr()
        return Delete(table=table, where=where)

    # -- expressions (precedence: OR < AND < NOT < comparison < primary) -------

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self._match("KEYWORD", "OR"):
            left = BinaryOp("OR", left, self._parse_and())
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self._match("KEYWORD", "AND"):
            left = BinaryOp("AND", left, self._parse_not())
        return left

    def _parse_not(self) -> Expr:
        if self._match("KEYWORD", "NOT"):
            return UnaryOp("NOT", self._parse_not())
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        left = self._parse_primary()
        if self._check("OP") and self._peek().value in _COMPARISON_OPS:
            op = self._advance().value
            op = "<>" if op == "!=" else op
            right = self._parse_primary()
            return BinaryOp(op, left, right)
        return left

    def _parse_primary(self) -> Expr:
        tok = self._peek()

        if tok.kind == "NUMBER":
            self._advance()
            return Literal(float(tok.value) if "." in tok.value else int(tok.value))
        if tok.kind == "STRING":
            self._advance()
            return Literal(tok.value)
        if self._check("KEYWORD", "TRUE"):
            self._advance()
            return Literal(True)
        if self._check("KEYWORD", "FALSE"):
            self._advance()
            return Literal(False)
        if self._check("KEYWORD", "NULL"):
            self._advance()
            return Literal(None)
        if self._match("OP", "("):
            inner = self._parse_expr()
            self._expect("OP", ")")
            return inner
        if self._check("OP", "*"):
            self._advance()
            return Star()
        if tok.kind == "IDENT":
            self._advance()
            if self._check("OP", "("):
                # function call, e.g. COUNT(*), SUM(price)
                self._advance()
                arg = Star() if self._check("OP", "*") else self._parse_expr()
                if isinstance(arg, Star):
                    self._match("OP", "*")
                self._expect("OP", ")")
                return FunctionCall(name=tok.value.upper(), arg=arg)
            if self._match("OP", "."):
                column = self._expect("IDENT").value
                return ColumnRef(name=column, table=tok.value)
            return ColumnRef(name=tok.value)

        raise ParseError(f"unexpected token {tok.value!r} at position {tok.pos}")


def parse(sql: str) -> Statement:
    return Parser(tokenize(sql)).parse_statement()
