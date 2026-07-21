"""
Shared page-format constants and a couple of tiny value types used
throughout the storage engine.

Every page in a minirel data file is exactly PAGE_SIZE bytes, which is the
unit the disk manager reads/writes and the buffer pool caches. The first
byte of every page is a type tag so a page can be identified generically
(heap page, B+-tree internal node, B+-tree leaf node, or free) without
external bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

PAGE_SIZE = 4096

INVALID_PAGE_ID = 0xFFFFFFFF  # sentinel: "no page" (fits in the 4-byte page-id fields on disk)


class PageType(IntEnum):
    FREE = 0
    HEAP = 1
    BTREE_INTERNAL = 2
    BTREE_LEAF = 3


@dataclass(frozen=True, slots=True)
class RID:
    """A row identifier: which page a tuple lives on, and which slot within it."""

    page_id: int
    slot_id: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RID({self.page_id}, {self.slot_id})"
