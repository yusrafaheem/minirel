import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.storage.disk_manager import DiskManager
from minirel.storage.page import PAGE_SIZE


class TestDiskManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.path = self._tmp.name
        self.dm = DiskManager(self.path)

    def tearDown(self):
        self.dm.close()
        os.unlink(self.path)

    def test_starts_empty(self):
        self.assertEqual(self.dm.num_pages, 0)

    def test_allocate_page_returns_sequential_ids(self):
        self.assertEqual(self.dm.allocate_page(), 0)
        self.assertEqual(self.dm.allocate_page(), 1)
        self.assertEqual(self.dm.allocate_page(), 2)
        self.assertEqual(self.dm.num_pages, 3)

    def test_write_then_read_round_trips(self):
        page_id = self.dm.allocate_page()
        payload = bytes([7]) * PAGE_SIZE
        self.dm.write_page(page_id, payload)
        self.assertEqual(bytes(self.dm.read_page(page_id)), payload)

    def test_reading_unallocated_page_raises(self):
        with self.assertRaises(ValueError):
            self.dm.read_page(0)

    def test_write_rejects_wrong_size(self):
        page_id = self.dm.allocate_page()
        with self.assertRaises(ValueError):
            self.dm.write_page(page_id, b"too short")

    def test_persists_across_reopen(self):
        page_id = self.dm.allocate_page()
        payload = bytes(range(256)) * (PAGE_SIZE // 256)
        self.dm.write_page(page_id, payload)
        self.dm.fsync()
        self.dm.close()

        dm2 = DiskManager(self.path)
        self.assertEqual(dm2.num_pages, 1)
        self.assertEqual(bytes(dm2.read_page(page_id)), payload)
        dm2.close()


if __name__ == "__main__":
    unittest.main()
