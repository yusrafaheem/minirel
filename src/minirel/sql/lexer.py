"""
minirel.sql.lexer
====================

Tokenizes SQL text. One regex with named groups, tried in order at each
position (Python's `re` picks the first alternative that matches at that
exact spot, so ordering encodes priority -- e.g. NUMBER before a bare
DOT, multi-char operators like `<=` before the single-char `<`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KEYWORDS = {
    "CREATE", "TABLE", "INDEX", "ON", "UNIQUE", "INSERT", "INTO", "VALUES",
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "ORDER", "BY", "ASC",
    "DESC", "LIMIT", "GROUP", "JOIN", "UPDATE", "SET", "DELETE", "AS",
    "TRUE", "FALSE", "NULL", "INT", "FLOAT", "TEXT", "BOOL",
}  # fmt: skip

_TOKEN_SPEC = [
    ("WHITESPACE", r"\s+"),
    ("COMMENT", r"--[^\n]*"),
    ("NUMBER", r"\d+\.\d+|\d+"),
    ("STRING", r"'(?:[^']|'')*'"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"<=|>=|<>|!=|[=<>(),;.*-]"),
]
_MASTER_RE = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC))


@dataclass(frozen=True, slots=True)
class Token:
    kind: str  # "KEYWORD" | "IDENT" | "NUMBER" | "STRING" | "OP" | "EOF"
    value: str
    pos: int


class LexError(Exception):
    pass


def tokenize(sql: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    n = len(sql)
    while pos < n:
        match = _MASTER_RE.match(sql, pos)
        if match is None:
            raise LexError(f"unexpected character {sql[pos]!r} at position {pos}")
        kind = match.lastgroup
        text = match.group()
        if kind in ("WHITESPACE", "COMMENT"):
            pos = match.end()
            continue
        if kind == "IDENT" and text.upper() in KEYWORDS:
            tokens.append(Token("KEYWORD", text.upper(), pos))
        elif kind == "STRING":
            # Strip the surrounding quotes and turn doubled '' into a
            # single literal quote (the standard SQL string-escaping rule).
            tokens.append(Token("STRING", text[1:-1].replace("''", "'"), pos))
        else:
            tokens.append(Token(kind, text, pos))
        pos = match.end()
    tokens.append(Token("EOF", "", pos))
    return tokens
