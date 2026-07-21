import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.storage.buffer_pool import BufferPoolFullError, BufferPoolManager
from minirel.storage.disk_manager import DiskManager
from minirel.storage.page import PAGE_SIZE


class TestBufferPoolManager(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.path = self._tmp.name
        self.dm = DiskManager(self.path)

    def tearDown(self):
        self.dm.close()
        os.unlink(self.path)

    def test_new_page_is_immediately_readable_and_writable(self):
        bpm = BufferPoolManager(self.dm, pool_size=4)
        page_id, data = bpm.new_page()
        data[0:5] = b"hello"
        bpm.unpin_page(page_id, dirty=True)

        fetched = bpm.fetch_page(page_id)
        self.assertEqual(bytes(fetched[0:5]), b"hello")

    def test_repeated_fetch_of_resident_page_is_a_cache_hit_not_a_disk_read(self):
        bpm = BufferPoolManager(self.dm, pool_size=4)
        page_id, _ = bpm.new_page()
        bpm.unpin_page(page_id)

        reads_before = bpm.disk_reads
        bpm.fetch_page(page_id)
        bpm.unpin_page(page_id)
        bpm.fetch_page(page_id)
        bpm.unpin_page(page_id)
        self.assertEqual(bpm.disk_reads, reads_before)
        self.assertGreaterEqual(bpm.cache_hits, 2)

    def test_dirty_page_is_written_back_on_eviction(self):
        bpm = BufferPoolManager(self.dm, pool_size=1)
        page_id, data = bpm.new_page()
        data[0:4] = b"abcd"
        bpm.unpin_page(page_id, dirty=True)

        # Force eviction of page_id by requesting a second page in a
        # pool that only has room for one frame.
        other_id, _ = bpm.new_page()
        bpm.unpin_page(other_id)

        on_disk = self.dm.read_page(page_id)
        self.assertEqual(bytes(on_disk[0:4]), b"abcd")

    def test_clean_page_is_not_rewritten_on_eviction(self):
        bpm = BufferPoolManager(self.dm, pool_size=1)
        page_id, _ = bpm.new_page()
        bpm.unpin_page(page_id, dirty=False)
        writes_before = bpm.disk_writes

        other_id, _ = bpm.new_page()
        bpm.unpin_page(other_id)
        self.assertEqual(bpm.disk_writes, writes_before)

    def test_lru_evicts_least_recently_used_unpinned_frame_first(self):
        bpm = BufferPoolManager(self.dm, pool_size=2)
        p0, _ = bpm.new_page()
        bpm.unpin_page(p0)
        p1, _ = bpm.new_page()
        bpm.unpin_page(p1)

        # touch p0 again so p1 becomes the least-recently-used frame
        bpm.fetch_page(p0)
        bpm.unpin_page(p0)

        p2, _ = bpm.new_page()  # pool is full -> must evict p1, not p0
        bpm.unpin_page(p2)

        reads_before = bpm.disk_reads
        bpm.fetch_page(p0)  # should still be resident -> no disk read
        self.assertEqual(bpm.disk_reads, reads_before)

        bpm.fetch_page(p1)  # was evicted -> disk read required
        self.assertEqual(bpm.disk_reads, reads_before + 1)

    def test_pinned_pages_are_never_evicted(self):
        bpm = BufferPoolManager(self.dm, pool_size=1)
        page_id, _ = bpm.new_page()
        # page_id stays pinned (never unpinned) -> the only frame is unavailable
        with self.assertRaises(BufferPoolFullError):
            bpm.new_page()

    def test_flush_all_persists_every_dirty_page(self):
        bpm = BufferPoolManager(self.dm, pool_size=8)
        ids = []
        for i in range(5):
            page_id, data = bpm.new_page()
            data[0] = i
            bpm.unpin_page(page_id, dirty=True)
            ids.append(page_id)

        bpm.flush_all()

        for i, page_id in enumerate(ids):
            self.assertEqual(self.dm.read_page(page_id)[0], i)

    def test_page_bytes_are_full_page_size(self):
        bpm = BufferPoolManager(self.dm, pool_size=2)
        page_id, data = bpm.new_page()
        self.assertEqual(len(data), PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
