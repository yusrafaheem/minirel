import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.database import Database
from minirel.repl import repl


class TestRepl(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self._dir, "test.db"))

    def tearDown(self):
        self.db.close()

    def _run(self, script: str) -> str:
        out = io.StringIO()
        repl(self.db, input_stream=io.StringIO(script), output_stream=out)
        return out.getvalue()

    def test_create_insert_select_prints_a_table(self):
        output = self._run(
            "CREATE TABLE t (id INT, name TEXT);\n"
            "INSERT INTO t VALUES (1, 'a');\n"
            "SELECT * FROM t;\n"
            ".exit\n"
        )
        self.assertIn("CREATE TABLE t", output)
        self.assertIn("INSERT 1", output)
        self.assertIn("id", output)
        self.assertIn("name", output)
        self.assertIn("(1 row)", output)

    def test_multiline_statement_across_reads(self):
        output = self._run("CREATE TABLE t (\nid INT\n);\n.exit\n")
        self.assertIn("CREATE TABLE t", output)

    def test_dot_tables_lists_created_tables(self):
        script = "CREATE TABLE alpha (id INT);\nCREATE TABLE beta (id INT);\n.tables\n.exit\n"
        output = self._run(script)
        self.assertIn("alpha", output)
        self.assertIn("beta", output)

    def test_error_is_reported_not_raised(self):
        output = self._run("SELECT * FROM nonexistent;\n.exit\n")
        self.assertIn("error:", output)

    def test_delete_reports_row_count(self):
        script = "CREATE TABLE t (id INT);\nINSERT INTO t VALUES (1), (2);\n"
        script += "DELETE FROM t WHERE id = 1;\n.exit\n"
        output = self._run(script)
        self.assertIn("DELETE 1", output)

    def test_empty_result_set_message(self):
        output = self._run("CREATE TABLE t (id INT);\nSELECT * FROM t WHERE id = 1;\n.exit\n")
        self.assertIn("(0 rows)", output)


if __name__ == "__main__":
    unittest.main()
