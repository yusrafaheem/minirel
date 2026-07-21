import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.catalog import TableAlreadyExistsError, TableNotFoundError
from minirel.database import Database
from minirel.index.btree import DuplicateKeyError
from minirel.transaction import WriteConflictError


class DatabaseTestBase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.db = Database(os.path.join(self._dir, "test.db"))

    def tearDown(self):
        self.db.close()


class TestDDL(DatabaseTestBase):
    def test_create_table_and_insert_and_select_round_trip(self):
        self.db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        self.db.execute("INSERT INTO widgets VALUES (1, 'bolt')")
        rows = self.db.execute("SELECT * FROM widgets").rows
        self.assertEqual(rows, [{"id": 1, "name": "bolt"}])

    def test_duplicate_table_raises(self):
        self.db.execute("CREATE TABLE t (id INT)")
        with self.assertRaises(TableAlreadyExistsError):
            self.db.execute("CREATE TABLE t (id INT)")

    def test_select_from_missing_table_raises(self):
        with self.assertRaises(TableNotFoundError):
            self.db.execute("SELECT * FROM nope")

    def test_create_index_backfills_existing_rows(self):
        self.db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        self.db.execute("INSERT INTO widgets VALUES (1, 'bolt'), (2, 'nail'), (3, 'screw')")
        self.db.execute("CREATE INDEX widgets_id_idx ON widgets (id)")
        rows = self.db.execute("SELECT name FROM widgets WHERE id = 2").rows
        self.assertEqual(rows, [{"name": "nail"}])


class TestInsert(DatabaseTestBase):
    def test_insert_with_explicit_column_list(self):
        self.db.execute("CREATE TABLE widgets (id INT, name TEXT, price FLOAT)")
        self.db.execute("INSERT INTO widgets (name, id, price) VALUES ('bolt', 1, 0.5)")
        rows = self.db.execute("SELECT * FROM widgets").rows
        self.assertEqual(rows, [{"id": 1, "name": "bolt", "price": 0.5}])

    def test_insert_missing_column_raises(self):
        self.db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        with self.assertRaises(ValueError):
            self.db.execute("INSERT INTO widgets (id) VALUES (1)")

    def test_insert_multi_row(self):
        self.db.execute("CREATE TABLE t (id INT)")
        result = self.db.execute("INSERT INTO t VALUES (1), (2), (3)")
        self.assertEqual(result.row_count, 3)
        self.assertEqual(len(self.db.execute("SELECT * FROM t").rows), 3)

    def test_unique_index_rejects_duplicate_on_insert(self):
        self.db.execute("CREATE TABLE t (id INT)")
        self.db.execute("CREATE UNIQUE INDEX t_pk ON t (id)")
        self.db.execute("INSERT INTO t VALUES (1)")
        with self.assertRaises(DuplicateKeyError):
            self.db.execute("INSERT INTO t VALUES (1)")


class TestUpdateDelete(DatabaseTestBase):
    def setUp(self):
        super().setUp()
        self.db.execute("CREATE TABLE widgets (id INT, name TEXT, price FLOAT)")
        self.db.execute("CREATE UNIQUE INDEX widgets_pk ON widgets (id)")
        self.db.execute("INSERT INTO widgets VALUES (1, 'bolt', 0.5), (2, 'nail', 0.1)")

    def test_update_changes_value_and_keeps_row_count(self):
        result = self.db.execute("UPDATE widgets SET price = 0.75 WHERE id = 1")
        self.assertEqual(result.row_count, 1)
        rows = self.db.execute("SELECT price FROM widgets WHERE id = 1").rows
        self.assertEqual(rows, [{"price": 0.75}])
        self.assertEqual(len(self.db.execute("SELECT * FROM widgets").rows), 2)

    def test_update_on_unique_indexed_column_still_works(self):
        # Regression test: updating a row's own unique-indexed column value
        # must not spuriously conflict with its own now-dead index entry.
        self.db.execute("UPDATE widgets SET id = 10 WHERE id = 1")
        rows = self.db.execute("SELECT name FROM widgets WHERE id = 10").rows
        self.assertEqual(rows, [{"name": "bolt"}])

    def test_update_to_a_value_already_used_by_another_row_conflicts(self):
        with self.assertRaises(DuplicateKeyError):
            self.db.execute("UPDATE widgets SET id = 2 WHERE id = 1")

    def test_delete_removes_row(self):
        result = self.db.execute("DELETE FROM widgets WHERE id = 1")
        self.assertEqual(result.row_count, 1)
        rows = self.db.execute("SELECT id FROM widgets").rows
        self.assertEqual(rows, [{"id": 2}])

    def test_delete_without_where_removes_everything(self):
        self.db.execute("DELETE FROM widgets")
        self.assertEqual(self.db.execute("SELECT * FROM widgets").rows, [])

    def test_deleted_id_can_be_reinserted(self):
        self.db.execute("DELETE FROM widgets WHERE id = 1")
        self.db.execute("INSERT INTO widgets VALUES (1, 'new-bolt', 1.5)")
        rows = self.db.execute("SELECT name FROM widgets WHERE id = 1").rows
        self.assertEqual(rows, [{"name": "new-bolt"}])


class TestExplicitTransactions(DatabaseTestBase):
    def setUp(self):
        super().setUp()
        self.db.execute("CREATE TABLE t (id INT, val INT)")
        self.db.execute("INSERT INTO t VALUES (1, 100)")

    def test_uncommitted_write_invisible_to_other_transaction(self):
        txn_a = self.db.begin()
        self.db.execute("UPDATE t SET val = 999 WHERE id = 1", txn=txn_a)

        txn_b = self.db.begin()
        rows = self.db.execute("SELECT val FROM t WHERE id = 1", txn=txn_b).rows
        txn_b.commit()
        self.assertEqual(rows, [{"val": 100}])

        txn_a.commit()
        rows_after = self.db.execute("SELECT val FROM t WHERE id = 1").rows
        self.assertEqual(rows_after, [{"val": 999}])

    def test_rolled_back_transaction_has_no_effect(self):
        txn = self.db.begin()
        self.db.execute("UPDATE t SET val = 999 WHERE id = 1", txn=txn)
        txn.abort()
        rows = self.db.execute("SELECT val FROM t WHERE id = 1").rows
        self.assertEqual(rows, [{"val": 100}])

    def test_concurrent_updates_to_same_row_conflict(self):
        txn_a = self.db.begin()
        txn_b = self.db.begin()
        self.db.execute("UPDATE t SET val = 1 WHERE id = 1", txn=txn_a)
        with self.assertRaises(WriteConflictError):
            self.db.execute("UPDATE t SET val = 2 WHERE id = 1", txn=txn_b)
        txn_a.commit()
        txn_b.abort()

    def test_multi_statement_transaction_commits_atomically(self):
        txn = self.db.begin()
        self.db.execute("INSERT INTO t VALUES (2, 200)", txn=txn)
        self.db.execute("INSERT INTO t VALUES (3, 300)", txn=txn)
        # Not yet visible to a fresh snapshot.
        outside_before = self.db.execute("SELECT * FROM t").rows
        self.assertEqual(len(outside_before), 1)
        txn.commit()
        outside_after = self.db.execute("SELECT * FROM t").rows
        self.assertEqual(len(outside_after), 3)


class TestQueryVariety(DatabaseTestBase):
    def setUp(self):
        super().setUp()
        self.db.execute("CREATE TABLE sales (id INT, region TEXT, amount FLOAT)")
        self.db.execute(
            "INSERT INTO sales VALUES "
            "(1, 'west', 100.0), (2, 'east', 50.0), (3, 'west', 75.0), (4, 'east', 200.0)"
        )

    def test_group_by_aggregate(self):
        rows = self.db.execute(
            "SELECT region, COUNT(*), SUM(amount) FROM sales GROUP BY region"
        ).rows
        by_region = {r["region"]: r for r in rows}
        self.assertEqual(by_region["west"]["count"], 2)
        self.assertEqual(by_region["west"]["sum"], 175.0)
        self.assertEqual(by_region["east"]["sum"], 250.0)

    def test_order_by_and_limit(self):
        rows = self.db.execute("SELECT id FROM sales ORDER BY amount DESC LIMIT 1").rows
        self.assertEqual(rows, [{"id": 4}])


if __name__ == "__main__":
    unittest.main()
