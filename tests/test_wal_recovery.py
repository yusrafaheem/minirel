"""
End-to-end crash-recovery tests: these are the ones that actually earn
the WAL its keep. Each test builds a Database, does some work, then
*simulates a crash* by tearing down the underlying file handles without
calling Database.close() (which would flush everything and hide exactly
the bugs this is supposed to catch), then opens a brand new Database
over the same files and checks what survived.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel import Database


class WalRecoveryTestBase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.path = os.path.join(self._dir, "test.db")

    def _crash(self, db: Database) -> None:
        """Kill the process's view of the database without an orderly
        flush -- whatever wasn't already on disk (via an explicit flush or
        a buffer-pool eviction) is gone, exactly like a real crash.
        """
        db.disk_manager._file.close()
        db.wal._append_file.close()

    def _reopen(self) -> Database:
        return Database(self.path)


class TestBasicRecovery(WalRecoveryTestBase):
    def test_committed_data_survives_a_crash(self):
        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        db.execute("INSERT INTO widgets VALUES (1, 'bolt'), (2, 'nail')")
        db.checkpoint()  # ensure this part is durable regardless of the recovery test below
        db.execute("INSERT INTO widgets VALUES (3, 'screw')")  # after checkpoint, WAL only
        self._crash(db)

        recovered = self._reopen()
        rows = recovered.execute("SELECT id, name FROM widgets").rows
        recovered.close()
        self.assertEqual(
            sorted(rows, key=lambda r: r["id"]),
            [
                {"id": 1, "name": "bolt"},
                {"id": 2, "name": "nail"},
                {"id": 3, "name": "screw"},
            ],
        )

    def test_uncommitted_insert_does_not_survive_a_crash(self):
        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        db.execute("INSERT INTO widgets VALUES (1, 'bolt')")
        db.checkpoint()

        txn = db.begin()
        db.execute("INSERT INTO widgets VALUES (2, 'nail')", txn=txn)
        # Simulate the "steal" case explicitly: force the uncommitted
        # insert's dirty page to disk before crashing, the way LRU eviction
        # legitimately could under memory pressure. If recovery's handling
        # of in-flight transactions is correct, this makes no difference to
        # the outcome -- the row must still come back invisible.
        db.buffer_pool.flush_all()
        # (no commit)
        self._crash(db)

        recovered = self._reopen()
        rows = recovered.execute("SELECT id, name FROM widgets").rows
        recovered.close()
        self.assertEqual(rows, [{"id": 1, "name": "bolt"}])

    def test_explicitly_aborted_insert_does_not_survive_a_crash(self):
        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        db.checkpoint()

        txn = db.begin()
        db.execute("INSERT INTO widgets VALUES (1, 'bolt')", txn=txn)
        txn.abort()
        self._crash(db)

        recovered = self._reopen()
        rows = recovered.execute("SELECT id, name FROM widgets").rows
        recovered.close()
        self.assertEqual(rows, [])

    def test_new_transactions_after_reopen_do_not_reuse_old_txn_ids(self):
        db = Database(self.path)
        db.execute("CREATE TABLE t (id INT)")
        db.execute("INSERT INTO t VALUES (1)")
        first_txn_id = db.begin().txn_id
        self._crash(db)

        recovered = self._reopen()
        new_txn = recovered.begin()
        recovered.close()
        self.assertGreater(new_txn.txn_id, first_txn_id)


class TestIndexRecovery(WalRecoveryTestBase):
    def test_index_lookups_work_correctly_after_recovery(self):
        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        db.execute("CREATE UNIQUE INDEX widgets_pk ON widgets (id)")
        db.execute("INSERT INTO widgets VALUES (1, 'bolt'), (2, 'nail'), (3, 'screw')")
        self._crash(db)

        recovered = self._reopen()
        rows = recovered.execute("SELECT name FROM widgets WHERE id = 2").rows
        recovered.close()
        self.assertEqual(rows, [{"name": "nail"}])

    def test_unique_constraint_still_enforced_after_recovery(self):
        from minirel.index.btree import DuplicateKeyError

        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT, name TEXT)")
        db.execute("CREATE UNIQUE INDEX widgets_pk ON widgets (id)")
        db.execute("INSERT INTO widgets VALUES (1, 'bolt')")
        self._crash(db)

        recovered = self._reopen()
        with self.assertRaises(DuplicateKeyError):
            recovered.execute("INSERT INTO widgets VALUES (1, 'dup')")
        recovered.close()


class TestUpdateDeleteRecovery(WalRecoveryTestBase):
    def test_update_survives_crash(self):
        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT, price FLOAT)")
        db.execute("INSERT INTO widgets VALUES (1, 1.0)")
        db.checkpoint()
        db.execute("UPDATE widgets SET price = 2.5 WHERE id = 1")
        self._crash(db)

        recovered = self._reopen()
        rows = recovered.execute("SELECT price FROM widgets WHERE id = 1").rows
        recovered.close()
        self.assertEqual(rows, [{"price": 2.5}])

    def test_delete_survives_crash(self):
        db = Database(self.path)
        db.execute("CREATE TABLE widgets (id INT)")
        db.execute("INSERT INTO widgets VALUES (1), (2)")
        db.checkpoint()
        db.execute("DELETE FROM widgets WHERE id = 1")
        self._crash(db)

        recovered = self._reopen()
        rows = recovered.execute("SELECT id FROM widgets").rows
        recovered.close()
        self.assertEqual(rows, [{"id": 2}])


class TestCheckpoint(WalRecoveryTestBase):
    def test_checkpoint_refuses_with_active_transaction(self):
        db = Database(self.path)
        db.execute("CREATE TABLE t (id INT)")
        db.begin()
        with self.assertRaises(RuntimeError):
            db.checkpoint()
        db.close()

    def test_recovery_after_checkpoint_only_replays_newer_records(self):
        db = Database(self.path)
        db.execute("CREATE TABLE t (id INT)")
        db.execute("INSERT INTO t VALUES (1)")
        db.checkpoint()
        wal_size_at_checkpoint = os.path.getsize(self.path + ".wal")
        db.execute("INSERT INTO t VALUES (2)")
        self._crash(db)

        # Sanity check on the setup itself: there really is post-checkpoint
        # log content to replay, not just an empty tail.
        self.assertGreater(os.path.getsize(self.path + ".wal"), wal_size_at_checkpoint)

        recovered = self._reopen()
        rows = recovered.execute("SELECT id FROM t").rows
        recovered.close()
        self.assertEqual(sorted(r["id"] for r in rows), [1, 2])


if __name__ == "__main__":
    unittest.main()
