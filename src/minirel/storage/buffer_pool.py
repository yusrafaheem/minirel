"""
minirel.storage.buffer_pool
=============================

A fixed-size in-memory cache of pages, sitting between the disk manager
and everything that reads/writes page bytes (heap file, B+-tree).

Standard buffer-pool-manager design (same shape as the one taught in
CMU 15-445 / built in most textbook database courses):

- A `page_table` maps page_id -> frame index for O(1) lookup of pages
  already resident in memory.
- Each frame tracks a `pin_count` (how many callers currently hold the
  page) and a `dirty` flag (has it been modified since it was last
  flushed). A pinned page is never evicted.
- Eviction uses strict LRU among *unpinned* frames: `_lru` is an
  OrderedDict acting as a queue of unpinned frame indices, most-recently
  unpinned at the end. `move_to_end` and `popitem(last=False)` are both
  O(1), so touch/evict never degrade to a linear scan even under heavy
  churn.
- A page fetched while already resident is served straight from memory
  (no disk I/O) -- this is the whole point of the cache, and is exactly
  what test_buffer_pool.py asserts on via a disk-read counter.
"""

from __future__ import annotations

from collections import OrderedDict

from .disk_manager import DiskManager
from .page import PAGE_SIZE


class BufferPoolFullError(RuntimeError):
    """Raised when every frame is pinned and a new page still needs one."""


class _Frame:
    __slots__ = ("page_id", "data", "pin_count", "dirty")

    def __init__(self) -> None:
        self.page_id: int | None = None
        self.data: bytearray = bytearray(PAGE_SIZE)
        self.pin_count = 0
        self.dirty = False


class BufferPoolManager:
    def __init__(self, disk_manager: DiskManager, pool_size: int = 64):
        self.disk_manager = disk_manager
        self.pool_size = pool_size
        self._frames: list[_Frame] = [_Frame() for _ in range(pool_size)]
        self._page_table: dict[int, int] = {}
        self._free_frames: list[int] = list(range(pool_size))
        self._lru: OrderedDict[int, None] = OrderedDict()  # frame_idx -> None, unpinned only

        # Instrumentation used by tests/benchmarks to prove caching is
        # actually happening (hits should dominate once a working set
        # fits in the pool).
        self.disk_reads = 0
        self.disk_writes = 0
        self.cache_hits = 0

    # -- internal helpers ----------------------------------------------

    def _evict_one(self) -> int:
        if self._free_frames:
            return self._free_frames.pop()
        if not self._lru:
            raise BufferPoolFullError("all buffer pool frames are pinned; cannot evict")
        frame_idx, _ = self._lru.popitem(last=False)
        frame = self._frames[frame_idx]
        if frame.dirty:
            self.disk_manager.write_page(frame.page_id, bytes(frame.data))
            self.disk_writes += 1
        del self._page_table[frame.page_id]
        return frame_idx

    def _pin(self, frame_idx: int) -> None:
        frame = self._frames[frame_idx]
        frame.pin_count += 1
        self._lru.pop(frame_idx, None)  # a pinned frame is never a candidate for eviction

    # -- public API -------------------------------------------------------

    def fetch_page(self, page_id: int) -> bytearray:
        """Pin `page_id` and return its bytes (mutable -- callers write in place)."""
        if page_id in self._page_table:
            frame_idx = self._page_table[page_id]
            self.cache_hits += 1
            self._pin(frame_idx)
            return self._frames[frame_idx].data

        frame_idx = self._evict_one()
        frame = self._frames[frame_idx]
        frame.data = self.disk_manager.read_page(page_id)
        self.disk_reads += 1
        frame.page_id = page_id
        frame.dirty = False
        frame.pin_count = 0
        self._page_table[page_id] = frame_idx
        self._pin(frame_idx)
        return frame.data

    def new_page(self) -> tuple[int, bytearray]:
        """Allocate a brand new page on disk, pin it, and return (page_id, bytes)."""
        page_id = self.disk_manager.allocate_page()
        frame_idx = self._evict_one()
        frame = self._frames[frame_idx]
        frame.data = bytearray(PAGE_SIZE)
        frame.page_id = page_id
        # Freshly allocated pages are already zero-filled on disk (the disk
        # manager wrote the zeros at allocation time), so the in-memory copy
        # matches disk until a caller actually mutates it and unpins with
        # dirty=True -- no need to force a write-back of an all-zero page.
        frame.dirty = False
        frame.pin_count = 0
        self._page_table[page_id] = frame_idx
        self._pin(frame_idx)
        return page_id, frame.data

    def unpin_page(self, page_id: int, dirty: bool = False) -> None:
        """Release one hold on `page_id`. `dirty` is sticky (once True for a
        frame, it stays True until the next flush) so an earlier caller's
        write is never lost just because a later unpin passes dirty=False.
        Only once pin_count drops to zero does the frame become eligible
        for LRU eviction.
        """
        frame_idx = self._page_table.get(page_id)
        if frame_idx is None:
            return
        frame = self._frames[frame_idx]
        frame.dirty = frame.dirty or dirty
        if frame.pin_count > 0:
            frame.pin_count -= 1
        if frame.pin_count == 0:
            self._lru[frame_idx] = None  # becomes eligible for eviction, most-recently-used end

    def flush_page(self, page_id: int) -> None:
        frame_idx = self._page_table.get(page_id)
        if frame_idx is None:
            return
        frame = self._frames[frame_idx]
        if frame.dirty:
            self.disk_manager.write_page(page_id, bytes(frame.data))
            self.disk_writes += 1
            frame.dirty = False

    def flush_all(self) -> None:
        for page_id in list(self._page_table):
            self.flush_page(page_id)
        self.disk_manager.fsync()
