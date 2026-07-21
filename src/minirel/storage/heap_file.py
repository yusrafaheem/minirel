"""
minirel.storage.heap_file
============================

Row storage: a singly-linked chain of slotted pages, the same layout
used by Postgres and most textbook storage engines.

Slotted page layout (PAGE_SIZE bytes total)::

    +-------------------------------------------------------------+
    | type(1) | num_slots(2) | free_space_ptr(2) | next_page(4)   |  <- 9-byte header
    +-------------------------------------------------------------+
    | slot[0]: offset(2) length(2) | slot[1] | slot[2] | ...      |  <- slot directory, grows down
    |                                                               |
    |                      ... free space ...                      |
    |                                                               |
    | ... tuple data ... | tuple[1] | tuple[0]                     |  <- tuple bytes, grow up
    +-------------------------------------------------------------+

The slot directory and the tuple data grow toward each other from
opposite ends of the page; a page is full once they'd collide. Storing
tuples by indirection through a slot (rather than a raw byte offset)
means a row's RID -- (page_id, slot_id) -- never changes even if the
tuple's *bytes* move around within the page, which matters once
in-place updates or compaction enter the picture.

A deleted slot is marked with the sentinel length `0xFFFF` and its slot
index is eligible for reuse by a later insert on that page (bounding
slot-directory growth instead of leaking a slot index per delete).
"""

from __future__ import annotations

import struct
from collections.abc import Iterator

from .buffer_pool import BufferPoolManager
from .page import INVALID_PAGE_ID, PAGE_SIZE, PageType, RID

_HEADER = struct.Struct("<BHHI")  # type, num_slots, free_space_ptr, next_page_id
_HEADER_SIZE = _HEADER.size  # 9
_SLOT = struct.Struct("<HH")  # offset, length
_SLOT_SIZE = _SLOT.size  # 4
_DELETED = 0xFFFF


def _read_header(data: bytearray) -> tuple[int, int, int, int]:
    return _HEADER.unpack_from(data, 0)


def _write_header(
    data: bytearray, page_type: int, num_slots: int, free_ptr: int, next_page: int
) -> None:
    _HEADER.pack_into(data, 0, page_type, num_slots, free_ptr, next_page)


def _slot_offset(index: int) -> int:
    return _HEADER_SIZE + index * _SLOT_SIZE


def _read_slot(data: bytearray, index: int) -> tuple[int, int]:
    return _SLOT.unpack_from(data, _slot_offset(index))


def _write_slot(data: bytearray, index: int, offset: int, length: int) -> None:
    _SLOT.pack_into(data, _slot_offset(index), offset, length)


def init_heap_page(data: bytearray) -> None:
    """Format a freshly allocated page as an empty heap page, in place."""
    _write_header(data, PageType.HEAP, 0, PAGE_SIZE, INVALID_PAGE_ID)


class HeapFile:
    """A chain of heap pages holding one table's rows."""

    def __init__(self, buffer_pool: BufferPoolManager, first_page_id: int | None = None):
        self.buffer_pool = buffer_pool
        if first_page_id is None:
            page_id, data = buffer_pool.new_page()
            init_heap_page(data)
            buffer_pool.unpin_page(page_id, dirty=True)
            first_page_id = page_id
        self.first_page_id = first_page_id
        self.last_page_id = self._find_last_page()

    def _find_last_page(self) -> int:
        page_id = self.first_page_id
        seen = {page_id}
        while True:
            data = self.buffer_pool.fetch_page(page_id)
            _, _, _, next_page = _read_header(data)
            self.buffer_pool.unpin_page(page_id)
            if next_page == INVALID_PAGE_ID:
                return page_id
            if next_page in seen:
                # A cycle means this page was never properly initialized as
                # a heap page (e.g. still all-zero bytes on disk, which
                # decodes to page_type=FREE and next_page=0 -- see
                # Catalog.create_table's comment on why DDL flushes its
                # freshly allocated page immediately to prevent exactly
                # this). Fail loudly instead of looping forever.
                raise ValueError(
                    f"heap page chain has a cycle at page {next_page} "
                    "(the data file may be corrupt, or a page was never initialized)"
                )
            seen.add(next_page)
            page_id = next_page

    def insert(self, row_bytes: bytes) -> RID:
        """Insert `row_bytes` (already MVCC-header-prefixed and serialized by
        the caller) and return the RID it was assigned.
        """
        length = len(row_bytes)
        if _HEADER_SIZE + _SLOT_SIZE + length > PAGE_SIZE:
            raise ValueError(
                f"row of {length} bytes cannot fit in a {PAGE_SIZE}-byte page "
                "(minirel does not support overflow/TOAST-style tuples)"
            )

        page_id = self.last_page_id
        data = self.buffer_pool.fetch_page(page_id)
        try:
            _, num_slots, free_ptr, next_page = _read_header(data)

            reuse_index = None
            for i in range(num_slots):
                _, slot_len = _read_slot(data, i)
                if slot_len == _DELETED:
                    reuse_index = i
                    break

            needed_slot_bytes = 0 if reuse_index is not None else _SLOT_SIZE
            slot_dir_end = _slot_offset(num_slots if reuse_index is None else num_slots)
            available = free_ptr - slot_dir_end - needed_slot_bytes

            if available < length:
                # This page is full: allocate a new one and link it in.
                new_page_id, new_data = self.buffer_pool.new_page()
                init_heap_page(new_data)
                self.buffer_pool.unpin_page(new_page_id, dirty=True)

                _write_header(data, PageType.HEAP, num_slots, free_ptr, new_page_id)
                self.buffer_pool.unpin_page(page_id, dirty=True)

                self.last_page_id = new_page_id
                return self.insert(row_bytes)

            new_free_ptr = free_ptr - length
            data[new_free_ptr : new_free_ptr + length] = row_bytes
            index = reuse_index if reuse_index is not None else num_slots
            _write_slot(data, index, new_free_ptr, length)
            new_num_slots = num_slots if reuse_index is not None else num_slots + 1
            _write_header(data, PageType.HEAP, new_num_slots, new_free_ptr, next_page)
            return RID(page_id, index)
        finally:
            self.buffer_pool.unpin_page(page_id, dirty=True)

    def get(self, rid: RID) -> bytes | None:
        data = self.buffer_pool.fetch_page(rid.page_id)
        try:
            _, num_slots, _, _ = _read_header(data)
            if rid.slot_id >= num_slots:
                return None
            offset, length = _read_slot(data, rid.slot_id)
            if length == _DELETED:
                return None
            return bytes(data[offset : offset + length])
        finally:
            self.buffer_pool.unpin_page(rid.page_id)

    def update_in_place(self, rid: RID, new_bytes: bytes) -> bool:
        """Overwrite a tuple's bytes without changing its length or RID.

        Used for MVCC header stamping (xmax) where only a fixed-size field
        inside an existing tuple changes. Returns False (and leaves the
        tuple untouched) if `new_bytes` isn't exactly the original length.
        """
        data = self.buffer_pool.fetch_page(rid.page_id)
        try:
            _, num_slots, _, _ = _read_header(data)
            if rid.slot_id >= num_slots:
                return False
            offset, length = _read_slot(data, rid.slot_id)
            if length == _DELETED or length != len(new_bytes):
                return False
            data[offset : offset + length] = new_bytes
            return True
        finally:
            self.buffer_pool.unpin_page(rid.page_id, dirty=True)

    def delete(self, rid: RID) -> bool:
        """Physically free a slot (its index becomes reusable). Callers that
        need MVCC-visible deletes should stamp `xmax` via `update_in_place`
        instead -- this is for real space reclamation, e.g. after a
        transaction that inserted-then-aborted, or compaction.
        """
        data = self.buffer_pool.fetch_page(rid.page_id)
        try:
            _, num_slots, _, _ = _read_header(data)
            if rid.slot_id >= num_slots:
                return False
            _, length = _read_slot(data, rid.slot_id)
            if length == _DELETED:
                return False
            _write_slot(data, rid.slot_id, 0, _DELETED)
            return True
        finally:
            self.buffer_pool.unpin_page(rid.page_id, dirty=True)

    def scan(self) -> Iterator[tuple[RID, bytes]]:
        """Yield every live (non-deleted) tuple across the whole page chain,
        in RID order. Higher layers (MVCC, the executor) filter these down
        to what's visible to a given transaction's snapshot.
        """
        page_id = self.first_page_id
        while page_id != INVALID_PAGE_ID:
            data = self.buffer_pool.fetch_page(page_id)
            _, num_slots, _, next_page = _read_header(data)
            for i in range(num_slots):
                offset, length = _read_slot(data, i)
                if length != _DELETED:
                    yield RID(page_id, i), bytes(data[offset : offset + length])
            self.buffer_pool.unpin_page(page_id)
            page_id = next_page
