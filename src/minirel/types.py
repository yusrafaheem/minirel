"""
minirel.types
==============

Column types and schema-driven row (de)serialization, plus the MVCC
tuple header every stored row carries.

A "row" as it lives in a heap page is::

    [xmin: u32][xmax: u32][column_0 bytes][column_1 bytes]...

`xmin`/`xmax` are the transaction-visibility fields MVCC uses (see
transaction.py): the row is visible to a snapshot that started after
`xmin` committed and before `xmax` committed (or `xmax` is unset).
Everything after the 8-byte header is the row payload, encoded
column-by-column according to the table's schema in `catalog.py`.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum

INFINITY_TXN = 0xFFFFFFFF  # xmax sentinel meaning "not deleted"

_MVCC_HEADER = struct.Struct("<II")  # xmin, xmax


class ColumnType(str, Enum):
    INT = "INT"
    FLOAT = "FLOAT"
    TEXT = "TEXT"
    BOOL = "BOOL"


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    type: ColumnType


@dataclass(frozen=True, slots=True)
class Schema:
    columns: tuple[Column, ...]

    def index_of(self, name: str) -> int:
        for i, col in enumerate(self.columns):
            if col.name == name:
                return i
        raise KeyError(f"no such column: {name}")

    def column(self, name: str) -> Column:
        return self.columns[self.index_of(name)]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.columns]


# -- value <-> bytes, one column at a time -----------------------------------
#
# TEXT is length-prefixed (2-byte unsigned length + UTF-8 bytes) since rows
# are variable length; every other type is a fixed-width struct field, which
# is what lets column values be read without first scanning past prior
# variable-length columns... except TEXT columns still require a linear scan
# through the row to find where they start. That's a deliberate scope cut
# documented in the README (a "real" system would keep a per-row offset
# array for O(1) column access); minirel decodes a whole row at once instead
# of projecting single columns off raw bytes.

_INT = struct.Struct("<q")
_FLOAT = struct.Struct("<d")
_BOOL = struct.Struct("<B")


def encode_value(value, col_type: ColumnType) -> bytes:
    if col_type == ColumnType.INT:
        return _INT.pack(int(value))
    if col_type == ColumnType.FLOAT:
        return _FLOAT.pack(float(value))
    if col_type == ColumnType.BOOL:
        return _BOOL.pack(1 if value else 0)
    if col_type == ColumnType.TEXT:
        raw = str(value).encode("utf-8")
        if len(raw) > 0xFFFF:
            raise ValueError("TEXT values are capped at 65535 bytes")
        return struct.pack("<H", len(raw)) + raw
    raise ValueError(f"unknown column type: {col_type}")


def decode_value(data: bytes, offset: int, col_type: ColumnType) -> tuple[object, int]:
    """Decode one value starting at `offset`; return (value, bytes_consumed)."""
    if col_type == ColumnType.INT:
        return _INT.unpack_from(data, offset)[0], _INT.size
    if col_type == ColumnType.FLOAT:
        return _FLOAT.unpack_from(data, offset)[0], _FLOAT.size
    if col_type == ColumnType.BOOL:
        return bool(_BOOL.unpack_from(data, offset)[0]), _BOOL.size
    if col_type == ColumnType.TEXT:
        (length,) = struct.unpack_from("<H", data, offset)
        start = offset + 2
        text = data[start : start + length].decode("utf-8")
        return text, 2 + length
    raise ValueError(f"unknown column type: {col_type}")


def encode_row(values: list, schema: Schema) -> bytes:
    if len(values) != len(schema.columns):
        raise ValueError(f"expected {len(schema.columns)} values, got {len(values)}")
    return b"".join(encode_value(v, col.type) for v, col in zip(values, schema.columns))


def decode_row(data: bytes, schema: Schema) -> list:
    values = []
    offset = 0
    for col in schema.columns:
        value, consumed = decode_value(data, offset, col.type)
        values.append(value)
        offset += consumed
    return values


def pack_tuple(xmin: int, xmax: int, row_payload: bytes) -> bytes:
    return _MVCC_HEADER.pack(xmin, xmax) + row_payload


def unpack_tuple_header(data: bytes) -> tuple[int, int]:
    return _MVCC_HEADER.unpack_from(data, 0)


def tuple_payload(data: bytes) -> bytes:
    return data[_MVCC_HEADER.size :]


def with_xmax(data: bytes, xmax: int) -> bytes:
    """Return a copy of `data` with xmax overwritten -- same total length,
    so it's safe to write back via HeapFile.update_in_place.
    """
    xmin, _ = unpack_tuple_header(data)
    return pack_tuple(xmin, xmax, tuple_payload(data))


MVCC_HEADER_SIZE = _MVCC_HEADER.size
