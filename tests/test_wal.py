import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.wal import RecordType, WriteAheadLog


class TestWriteAheadLog(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.path = self._tmp.name

    def tearDown(self):
        os.unlink(self.path)

    def test_records_round_trip_in_order(self):
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_operation(1, "heap_insert", table="widgets", rid=[0, 0])
        wal.log_commit(1)
        wal.close()

        reader = WriteAheadLog(self.path)
        records = reader.read_all()
        reader.close()
        expected = [RecordType.BEGIN, RecordType.OP, RecordType.COMMIT]
        self.assertEqual([r.type for r in records], expected)
        self.assertEqual(records[1].payload["kind"], "heap_insert")
        self.assertEqual(records[1].payload["table"], "widgets")

    def test_read_since_last_checkpoint_skips_earlier_records(self):
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_operation(1, "heap_insert", table="t", rid=[0, 0])
        wal.log_commit(1)
        wal.log_checkpoint()
        wal.log_begin(2)
        wal.log_operation(2, "heap_insert", table="t", rid=[0, 1])
        wal.log_commit(2)

        records = wal.read_since_last_checkpoint()
        wal.close()
        self.assertEqual([r.txn_id for r in records], [2, 2, 2])

    def test_no_checkpoint_returns_everything(self):
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_commit(1)
        records = wal.read_since_last_checkpoint()
        wal.close()
        self.assertEqual(len(records), 2)

    def test_torn_final_record_is_silently_dropped(self):
        # Simulates a crash mid-write of the last log record: the process
        # died after writing the fixed header but before the payload bytes
        # (or the header itself) finished landing on disk. Recovery should
        # treat this as "that operation never really happened" rather than
        # crash trying to parse a truncated record.
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_operation(1, "heap_insert", table="t", rid=[0, 0])
        wal.log_commit(1)
        wal.close()

        with open(self.path, "ab") as f:
            f.write(bytes([RecordType.OP]) + (5).to_bytes(4, "little"))  # header only, no payload

        reader = WriteAheadLog(self.path)
        records = reader.read_all()
        reader.close()
        self.assertEqual(len(records), 3)  # the torn 4th record is dropped, not raised

    def test_empty_log_read_all_returns_empty_list(self):
        wal = WriteAheadLog(self.path)
        records = wal.read_all()
        wal.close()
        self.assertEqual(records, [])

    def test_only_records_after_most_recent_of_two_checkpoints_are_replayed(self):
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_commit(1)
        wal.log_checkpoint()
        wal.log_begin(2)
        wal.log_commit(2)
        wal.log_checkpoint()
        wal.log_begin(3)
        wal.log_commit(3)

        records = wal.read_since_last_checkpoint()
        wal.close()
        self.assertEqual([r.txn_id for r in records], [3, 3])

    def test_abort_record_round_trips(self):
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_abort(1)
        wal.close()

        reader = WriteAheadLog(self.path)
        records = reader.read_all()
        reader.close()
        self.assertEqual([r.type for r in records], [RecordType.BEGIN, RecordType.ABORT])

    def test_uncommitted_transaction_has_no_commit_record(self):
        wal = WriteAheadLog(self.path)
        wal.log_begin(1)
        wal.log_operation(1, "heap_insert", table="t", rid=[0, 0])
        # no commit -- simulates a crash before the transaction finished
        records = wal.read_all()
        wal.close()
        self.assertNotIn(RecordType.COMMIT, [r.type for r in records])


if __name__ == "__main__":
    unittest.main()
