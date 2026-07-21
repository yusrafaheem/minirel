import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.sql import ast
from minirel.sql.lexer import LexError, tokenize
from minirel.sql.parser import ParseError, parse


class TestLexer(unittest.TestCase):
    def test_tokenizes_keywords_case_insensitively(self):
        tokens = tokenize("select FROM Where")
        self.assertEqual([t.kind for t in tokens], ["KEYWORD", "KEYWORD", "KEYWORD", "EOF"])
        self.assertEqual([t.value for t in tokens[:3]], ["SELECT", "FROM", "WHERE"])

    def test_tokenizes_numbers_int_and_float(self):
        tokens = tokenize("42 3.14")
        kinds_values = [(t.kind, t.value) for t in tokens[:2]]
        self.assertEqual(kinds_values, [("NUMBER", "42"), ("NUMBER", "3.14")])

    def test_tokenizes_strings_with_escaped_quote(self):
        tokens = tokenize("'it''s a test'")
        self.assertEqual(tokens[0].kind, "STRING")
        self.assertEqual(tokens[0].value, "it's a test")

    def test_tokenizes_multichar_operators_before_single_char(self):
        tokens = tokenize("<= >= <> !=")
        self.assertEqual([t.value for t in tokens[:4]], ["<=", ">=", "<>", "!="])

    def test_comments_are_skipped(self):
        tokens = tokenize("SELECT 1 -- this is a comment\nFROM t")
        self.assertEqual([t.kind for t in tokens], ["KEYWORD", "NUMBER", "KEYWORD", "IDENT", "EOF"])

    def test_unknown_character_raises(self):
        with self.assertRaises(LexError):
            tokenize("SELECT $ FROM t")

    def test_unclosed_string_raises(self):
        with self.assertRaises(LexError):
            tokenize("SELECT 'abc FROM t")


class TestParseDDL(unittest.TestCase):
    def test_create_table(self):
        stmt = parse("CREATE TABLE widgets (id INT, name TEXT, price FLOAT, active BOOL)")
        self.assertIsInstance(stmt, ast.CreateTable)
        self.assertEqual(stmt.table, "widgets")
        self.assertEqual(
            [(c.name, c.type) for c in stmt.columns],
            [("id", "INT"), ("name", "TEXT"), ("price", "FLOAT"), ("active", "BOOL")],
        )

    def test_create_index(self):
        stmt = parse("CREATE INDEX widgets_id_idx ON widgets (id)")
        self.assertIsInstance(stmt, ast.CreateIndex)
        self.assertEqual(stmt.index_name, "widgets_id_idx")
        self.assertEqual(stmt.table, "widgets")
        self.assertEqual(stmt.column, "id")
        self.assertFalse(stmt.unique)

    def test_create_unique_index(self):
        stmt = parse("CREATE UNIQUE INDEX widgets_pk ON widgets (id)")
        self.assertTrue(stmt.unique)


class TestParseInsert(unittest.TestCase):
    def test_insert_all_columns(self):
        stmt = parse("INSERT INTO widgets VALUES (1, 'widget', 9.99, TRUE)")
        self.assertIsInstance(stmt, ast.Insert)
        self.assertIsNone(stmt.columns)
        self.assertEqual(len(stmt.rows), 1)
        self.assertEqual(
            [lit.value for lit in stmt.rows[0]],
            [1, "widget", 9.99, True],
        )

    def test_insert_explicit_columns(self):
        stmt = parse("INSERT INTO widgets (id, name) VALUES (1, 'widget')")
        self.assertEqual(stmt.columns, ("id", "name"))

    def test_insert_false_literal(self):
        stmt = parse("INSERT INTO widgets VALUES (1, FALSE)")
        self.assertEqual([lit.value for lit in stmt.rows[0]], [1, False])

    def test_insert_multiple_rows(self):
        stmt = parse("INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
        self.assertEqual(len(stmt.rows), 3)
        self.assertEqual(stmt.rows[2][0].value, 3)


class TestParseSelect(unittest.TestCase):
    def test_select_star(self):
        stmt = parse("SELECT * FROM widgets")
        self.assertIsInstance(stmt.items[0].expr, ast.Star)
        self.assertEqual(stmt.table, "widgets")

    def test_select_specific_columns_with_alias(self):
        stmt = parse("SELECT id, name AS n FROM widgets")
        self.assertEqual(stmt.items[0].expr, ast.ColumnRef(name="id"))
        self.assertEqual(stmt.items[1].alias, "n")

    def test_where_with_and_or_precedence(self):
        # AND binds tighter than OR: "a OR b AND c" parses as "a OR (b AND c)"
        stmt = parse("SELECT * FROM t WHERE a = 1 OR b = 2 AND c = 3")
        where = stmt.where
        self.assertIsInstance(where, ast.BinaryOp)
        self.assertEqual(where.op, "OR")
        self.assertEqual(where.right.op, "AND")

    def test_where_with_not_and_parens(self):
        stmt = parse("SELECT * FROM t WHERE NOT (a = 1 AND b = 2)")
        self.assertIsInstance(stmt.where, ast.UnaryOp)
        self.assertEqual(stmt.where.op, "NOT")

    def test_negative_number_literal(self):
        stmt = parse("SELECT * FROM t WHERE balance = -100")
        self.assertEqual(stmt.where.right, ast.Literal(-100))

    def test_negative_float_literal(self):
        stmt = parse("SELECT * FROM t WHERE delta = -3.5")
        self.assertEqual(stmt.where.right, ast.Literal(-3.5))

    def test_where_comparison_operators(self):
        cases = [
            ("=", "="),
            ("<>", "<>"),
            ("!=", "<>"),
            ("<", "<"),
            ("<=", "<="),
            (">", ">"),
            (">=", ">="),
        ]
        for text, op in cases:
            stmt = parse(f"SELECT * FROM t WHERE a {text} 1")
            self.assertEqual(stmt.where.op, op, text)

    def test_implicit_table_alias(self):
        stmt = parse("SELECT * FROM widgets w")
        self.assertEqual(stmt.table_alias, "w")

    def test_join_on(self):
        stmt = parse("SELECT * FROM orders JOIN customers ON orders.customer_id = customers.id")
        self.assertIsNotNone(stmt.join)
        self.assertEqual(stmt.join.table, "customers")
        self.assertEqual(stmt.join.on.left, ast.ColumnRef(name="customer_id", table="orders"))

    def test_group_by_and_aggregate_function(self):
        stmt = parse("SELECT category, COUNT(*), SUM(price) FROM widgets GROUP BY category")
        self.assertEqual(stmt.group_by, ("category",))
        self.assertIsInstance(stmt.items[1].expr, ast.FunctionCall)
        self.assertEqual(stmt.items[1].expr.name, "COUNT")
        self.assertIsInstance(stmt.items[1].expr.arg, ast.Star)
        self.assertEqual(stmt.items[2].expr.name, "SUM")

    def test_order_by_desc_and_limit(self):
        stmt = parse("SELECT * FROM widgets ORDER BY price DESC LIMIT 5")
        self.assertEqual(len(stmt.order_by), 1)
        self.assertTrue(stmt.order_by[0].descending)
        self.assertEqual(stmt.limit, 5)

    def test_order_by_multiple_columns(self):
        stmt = parse("SELECT * FROM widgets ORDER BY category, price DESC")
        self.assertEqual(len(stmt.order_by), 2)
        self.assertFalse(stmt.order_by[0].descending)
        self.assertTrue(stmt.order_by[1].descending)

    def test_order_by_default_ascending(self):
        stmt = parse("SELECT * FROM widgets ORDER BY price")
        self.assertFalse(stmt.order_by[0].descending)

    def test_trailing_semicolon_allowed(self):
        stmt = parse("SELECT * FROM widgets;")
        self.assertEqual(stmt.table, "widgets")


class TestParseUpdateDelete(unittest.TestCase):
    def test_update_multiple_assignments_and_where(self):
        stmt = parse("UPDATE widgets SET price = 9.99, active = TRUE WHERE id = 1")
        self.assertEqual(stmt.assignments[0][0], "price")
        self.assertEqual(stmt.assignments[1][0], "active")
        self.assertEqual(stmt.where.op, "=")

    def test_delete_with_where(self):
        stmt = parse("DELETE FROM widgets WHERE id = 1")
        self.assertIsInstance(stmt, ast.Delete)
        self.assertEqual(stmt.table, "widgets")

    def test_delete_without_where(self):
        stmt = parse("DELETE FROM widgets")
        self.assertIsNone(stmt.where)


class TestParseErrors(unittest.TestCase):
    def test_missing_from_raises(self):
        with self.assertRaises(ParseError):
            parse("SELECT *")

    def test_unclosed_paren_raises(self):
        with self.assertRaises(ParseError):
            parse("CREATE TABLE t (id INT")

    def test_garbage_after_statement_raises(self):
        # A bare trailing IDENT is actually valid (implicit table alias, as
        # in real SQL's `FROM orders o`) -- but a stray operator token
        # after a complete statement is not valid anywhere.
        with self.assertRaises(ParseError):
            parse("SELECT * FROM t )")


if __name__ == "__main__":
    unittest.main()
