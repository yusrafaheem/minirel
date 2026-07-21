import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.transaction import TransactionManager, WriteConflictError
from minirel.types import INFINITY_TXN
from minirel.wal import WriteAheadLog


class MvccTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.wal = WriteAheadLog(self._tmp.name)
        self.mgr = TransactionManager(self.wal)

    def tearDown(self):
        self.wal.close()
        os.unlink(self._tmp.name)


class TestSnapshotIsolationVisibility(MvccTestBase):
    def test_own_uncommitted_insert_is_visible_to_self(self):
        txn = self.mgr.begin()
        visible = self.mgr.is_visible(xmin=txn.txn_id, xmax=INFINITY_TXN, snapshot=txn.snapshot)
        self.assertTrue(visible)

    def test_uncommitted_insert_is_invisible_to_other_transaction(self):
        writer = self.mgr.begin()
        reader = self.mgr.begin()
        self.assertFalse(
            self.mgr.is_visible(xmin=writer.txn_id, xmax=INFINITY_TXN, snapshot=reader.snapshot)
        )

    def test_committed_insert_is_visible_to_transaction_started_after(self):
        writer = self.mgr.begin()
        writer.commit()
        reader = self.mgr.begin()
        self.assertTrue(
            self.mgr.is_visible(xmin=writer.txn_id, xmax=INFINITY_TXN, snapshot=reader.snapshot)
        )

    def test_snapshot_isolation_reader_does_not_see_writes_committed_after_it_started(self):
        # This is the defining property of snapshot isolation (as opposed to
        # read-committed): once a transaction's snapshot is taken, later
        # commits from other transactions never become visible to it, even
        # if the reader is still active when they land.
        reader = self.mgr.begin()
        writer = self.mgr.begin()
        writer.commit()
        self.assertFalse(
            self.mgr.is_visible(xmin=writer.txn_id, xmax=INFINITY_TXN, snapshot=reader.snapshot)
        )

    def test_aborted_insert_is_never_visible_even_to_later_transactions(self):
        writer = self.mgr.begin()
        writer.abort()
        reader = self.mgr.begin()
        self.assertFalse(
            self.mgr.is_visible(xmin=writer.txn_id, xmax=INFINITY_TXN, snapshot=reader.snapshot)
        )

    def test_deleted_row_invisible_once_deleting_txn_commits_before_reader_starts(self):
        inserter = self.mgr.begin()
        inserter.commit()
        deleter = self.mgr.begin()
        deleter.commit()
        reader = self.mgr.begin()
        self.assertFalse(
            self.mgr.is_visible(xmin=inserter.txn_id, xmax=deleter.txn_id, snapshot=reader.snapshot)
        )

    def test_deleted_row_still_visible_to_a_snapshot_taken_before_the_delete_committed(self):
        inserter = self.mgr.begin()
        inserter.commit()
        reader = self.mgr.begin()  # snapshot taken here
        deleter = self.mgr.begin()
        deleter.commit()  # commits after reader's snapshot
        self.assertTrue(
            self.mgr.is_visible(xmin=inserter.txn_id, xmax=deleter.txn_id, snapshot=reader.snapshot)
        )

    def test_aborted_delete_leaves_row_visible(self):
        inserter = self.mgr.begin()
        inserter.commit()
        deleter = self.mgr.begin()
        deleter.abort()
        reader = self.mgr.begin()
        self.assertTrue(
            self.mgr.is_visible(xmin=inserter.txn_id, xmax=deleter.txn_id, snapshot=reader.snapshot)
        )

    def test_own_uncommitted_delete_is_invisible_to_self(self):
        inserter = self.mgr.begin()
        inserter.commit()
        txn = self.mgr.begin()
        # txn deletes the row (stamps its own xmax) -- it should no longer
        # see the row itself, even mid-transaction.
        self.assertFalse(
            self.mgr.is_visible(xmin=inserter.txn_id, xmax=txn.txn_id, snapshot=txn.snapshot)
        )


class TestTransactionStateTransitions(MvccTestBase):
    def test_committing_twice_raises(self):
        txn = self.mgr.begin()
        txn.commit()
        with self.assertRaises(ValueError):
            txn.commit()

    def test_committing_an_aborted_transaction_raises(self):
        txn = self.mgr.begin()
        txn.abort()
        with self.assertRaises(ValueError):
            txn.commit()

    def test_has_active_transactions_reflects_current_state(self):
        self.assertFalse(self.mgr.has_active_transactions)
        txn = self.mgr.begin()
        self.assertTrue(self.mgr.has_active_transactions)
        txn.commit()
        self.assertFalse(self.mgr.has_active_transactions)


class TestWriteConflicts(MvccTestBase):
    def test_second_writer_conflicts_on_already_claimed_row(self):
        a = self.mgr.begin()
        b = self.mgr.begin()
        self.mgr.check_write_conflict(a, current_xmax=INFINITY_TXN)  # fine, unclaimed
        # a claims it by stamping its own xid as xmax (simulated by caller)
        with self.assertRaises(WriteConflictError):
            self.mgr.check_write_conflict(b, current_xmax=a.txn_id)

    def test_reclaiming_your_own_prior_stamp_is_not_a_conflict(self):
        a = self.mgr.begin()
        self.mgr.check_write_conflict(a, current_xmax=a.txn_id)  # no-op, doesn't raise

    def test_row_claimed_by_an_aborted_transaction_is_not_a_conflict(self):
        a = self.mgr.begin()
        a.abort()
        b = self.mgr.begin()
        self.mgr.check_write_conflict(b, current_xmax=a.txn_id)  # doesn't raise


if __name__ == "__main__":
    unittest.main()
