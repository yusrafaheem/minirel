import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.types import (
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

SCHEMA = Schema(
    columns=(
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
        Column("price", ColumnType.FLOAT),
        Column("active", ColumnType.BOOL),
    )
)


class TestRowSerde(unittest.TestCase):
    def test_round_trips_all_types(self):
        row = [42, "widget", 19.99, True]
        encoded = encode_row(row, SCHEMA)
        decoded = decode_row(encoded, SCHEMA)
        self.assertEqual(decoded, row)

    def test_negative_int_round_trips(self):
        row = [-100, "", 0.0, False]
        decoded = decode_row(encode_row(row, SCHEMA), SCHEMA)
        self.assertEqual(decoded, row)

    def test_text_with_unicode_round_trips(self):
        row = [1, "héllo wörld 🎉", 1.5, True]
        decoded = decode_row(encode_row(row, SCHEMA), SCHEMA)
        self.assertEqual(decoded, row)

    def test_wrong_arity_raises(self):
        with self.assertRaises(ValueError):
            encode_row([1, "x"], SCHEMA)


class TestMvccTupleHeader(unittest.TestCase):
    def test_pack_unpack_round_trips(self):
        payload = encode_row([1, "a", 1.0, True], SCHEMA)
        packed = pack_tuple(xmin=5, xmax=INFINITY_TXN, row_payload=payload)
        xmin, xmax = unpack_tuple_header(packed)
        self.assertEqual((xmin, xmax), (5, INFINITY_TXN))
        self.assertEqual(tuple_payload(packed), payload)

    def test_with_xmax_preserves_length_and_payload(self):
        payload = encode_row([1, "a", 1.0, True], SCHEMA)
        packed = pack_tuple(xmin=5, xmax=INFINITY_TXN, row_payload=payload)
        deleted = with_xmax(packed, 9)

        self.assertEqual(len(deleted), len(packed))
        xmin, xmax = unpack_tuple_header(deleted)
        self.assertEqual((xmin, xmax), (5, 9))
        self.assertEqual(tuple_payload(deleted), payload)


if __name__ == "__main__":
    unittest.main()
