"""
minirel.storage.disk_manager
=============================

The bottom of the storage stack: reads and writes fixed-size pages to a
real file on disk. Nothing above this layer (buffer pool, heap file,
B+-tree) ever calls Python's `open()` directly -- everything goes through
here, which is what lets the buffer pool cache pages in memory and only
hit disk on a miss or a flush.

Page ids are dense integers starting at 0; page id `i` lives at byte
offset `i * PAGE_SIZE` in the file. `allocate_page` grows the file by one
page and returns its id; there is no page-reclamation/free-list here (a
deliberate scope cut -- see README's "what's simplified" section), so
deleted heap pages are never returned to a free pool for reuse.
"""

from __future__ import annotations

import os
import threading

from .page import PAGE_SIZE


class DiskManager:
    """Fixed-size page I/O against a single backing file."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        # "r+b" requires the file to already exist; create it first if not.
        if not os.path.exists(path):
            open(path, "wb").close()
        self._file = open(path, "r+b")
        self.num_pages = os.path.getsize(path) // PAGE_SIZE

    def allocate_page(self) -> int:
        """Grow the file by one zero-filled page and return its page id."""
        with self._lock:
            page_id = self.num_pages
            self._file.seek(page_id * PAGE_SIZE)
            self._file.write(bytes(PAGE_SIZE))
            self._file.flush()
            self.num_pages += 1
            return page_id

    def read_page(self, page_id: int) -> bytearray:
        with self._lock:
            if page_id >= self.num_pages:
                raise ValueError(f"page {page_id} does not exist (num_pages={self.num_pages})")
            self._file.seek(page_id * PAGE_SIZE)
            data = self._file.read(PAGE_SIZE)
            if len(data) < PAGE_SIZE:
                data = data + bytes(PAGE_SIZE - len(data))
            return bytearray(data)

    def write_page(self, page_id: int, data: bytes) -> None:
        if len(data) != PAGE_SIZE:
            raise ValueError(f"page write must be exactly {PAGE_SIZE} bytes, got {len(data)}")
        with self._lock:
            self._file.seek(page_id * PAGE_SIZE)
            self._file.write(data)

    def fsync(self) -> None:
        """Force the OS to persist all writes so far to physical storage.

        Called after WAL writes and at commit/checkpoint boundaries -- the
        durability guarantee (a committed transaction survives a crash)
        only holds if the log record actually reached disk, not just the
        page cache.
        """
        with self._lock:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        with self._lock:
            self._file.close()
