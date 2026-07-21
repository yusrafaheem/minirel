import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.storage.buffer_pool import BufferPoolManager
from minirel.storage.disk_manager import DiskManager
from minirel.storage.heap_file import HeapFile
from minirel.storage.page import PAGE_SIZE


class TestHeapFile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.path = self._tmp.name
        self.dm = DiskManager(self.path)
        self.bpm = BufferPoolManager(self.dm, pool_size=16)
        self.heap = HeapFile(self.bpm)

    def tearDown(self):
        self.dm.close()
        os.unlink(self.path)

    def test_insert_then_get_round_trips(self):
        rid = self.heap.insert(b"hello world")
        self.assertEqual(self.heap.get(rid), b"hello world")

    def test_multiple_inserts_get_distinct_rids(self):
        rid1 = self.heap.insert(b"row one")
        rid2 = self.heap.insert(b"row two")
        self.assertNotEqual(rid1, rid2)
        self.assertEqual(self.heap.get(rid1), b"row one")
        self.assertEqual(self.heap.get(rid2), b"row two")

    def test_delete_makes_tuple_unreadable(self):
        rid = self.heap.insert(b"to be deleted")
        self.assertTrue(self.heap.delete(rid))
        self.assertIsNone(self.heap.get(rid))

    def test_delete_twice_returns_false(self):
        rid = self.heap.insert(b"row")
        self.assertTrue(self.heap.delete(rid))
        self.assertFalse(self.heap.delete(rid))

    def test_deleted_slot_is_reused_by_next_insert(self):
        rid1 = self.heap.insert(b"first")
        self.heap.delete(rid1)
        rid2 = self.heap.insert(b"second")
        self.assertEqual(rid2.page_id, rid1.page_id)
        self.assertEqual(rid2.slot_id, rid1.slot_id)
        self.assertEqual(self.heap.get(rid2), b"second")

    def test_update_in_place_same_length(self):
        rid = self.heap.insert(b"abcde")
        self.assertTrue(self.heap.update_in_place(rid, b"xyzab"))
        self.assertEqual(self.heap.get(rid), b"xyzab")

    def test_update_in_place_rejects_different_length(self):
        rid = self.heap.insert(b"abcde")
        self.assertFalse(self.heap.update_in_place(rid, b"short"[:4]))
        self.assertEqual(self.heap.get(rid), b"abcde")

    def test_page_overflow_spills_to_a_new_linked_page(self):
        # Each row is big enough that only a handful fit per 4KB page, so
        # inserting many of them forces at least one page-chain hop.
        big_row = b"x" * 500
        rids = [self.heap.insert(big_row) for _ in range(50)]
        page_ids = {rid.page_id for rid in rids}
        self.assertGreater(len(page_ids), 1, "expected inserts to span multiple pages")
        for rid in rids:
            self.assertEqual(self.heap.get(rid), big_row)

    def test_scan_yields_all_live_tuples_across_pages(self):
        big_row = b"y" * 500
        inserted = [self.heap.insert(big_row) for _ in range(30)]
        self.heap.delete(inserted[5])
        self.heap.delete(inserted[15])

        scanned_rids = {rid for rid, _ in self.heap.scan()}
        expected = set(inserted) - {inserted[5], inserted[15]}
        self.assertEqual(scanned_rids, expected)

    def test_row_larger_than_page_is_rejected(self):
        with self.assertRaises(ValueError):
            self.heap.insert(b"z" * (PAGE_SIZE + 1))

    def test_reopen_existing_heap_from_first_page_id(self):
        rid = self.heap.insert(b"persisted row")
        self.bpm.flush_all()

        reopened = HeapFile(self.bpm, first_page_id=self.heap.first_page_id)
        self.assertEqual(reopened.get(rid), b"persisted row")


if __name__ == "__main__":
    unittest.main()
