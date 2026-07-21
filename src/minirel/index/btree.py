"""
minirel.index.btree
=====================

A disk-backed B+-tree, keyed by an INT or fixed-width TEXT column,
mapping key -> RID. Node splits/merges operate on real pages through the
buffer pool, exactly like the heap file -- this is why a B+-tree (rather
than an in-memory sorted structure) is the right data structure for an
index that has to survive being larger than RAM: each node is sized to
fill a page, giving a very high fanout (hundreds of keys per node for an
8-byte INT key) and therefore a very shallow tree -- a point lookup over
millions of rows costs only a few page fetches, most of which are cache
hits after the top levels warm up. `benchmarks/bench_point_lookup.py`
measures this against a full sequential scan.

Node layouts (see also storage/heap_file.py for the analogous heap-page
layout)::

    Leaf:      [type(1)][num_keys(2)][next_leaf(4)] [key,rid]*
    Internal:  [type(1)][num_keys(2)] [child(4)] ([key][child(4)])*

Keys are decoded to native Python values (int or str) the moment a node
is read off a page, and only re-encoded to fixed-width bytes when a node
is written back -- so all the tree logic (bisect-based search, key
comparisons during splits/merges) works with ordinary comparable Python
values instead of raw bytes, which sidesteps needing a custom byte-order-
preserving encoding for signed integers.

Duplicate keys are supported (an index doesn't have to be unique): each
(key, RID) pair is a distinct leaf entry, and `delete` removes one
specific (key, RID) pair rather than every entry for a key.
"""

from __future__ import annotations

import struct
from bisect import bisect_left, bisect_right
from collections.abc import Iterator

from ..storage.buffer_pool import BufferPoolManager
from ..storage.page import INVALID_PAGE_ID, PAGE_SIZE, PageType, RID
from ..types import ColumnType

_LEAF_HEADER = 7  # type(1) + num_keys(2) + next_leaf(4)
_INTERNAL_HEADER = 3  # type(1) + num_keys(2)
_CHILD_SIZE = 4
_RID_SIZE = 6  # page_id(4) + slot_id(2)


class DuplicateKeyError(Exception):
    pass


class BPlusTree:
    def __init__(
        self,
        buffer_pool: BufferPoolManager,
        key_type: ColumnType = ColumnType.INT,
        key_size: int = 8,
        root_page_id: int | None = None,
        unique: bool = False,
    ):
        if key_type not in (ColumnType.INT, ColumnType.TEXT):
            raise ValueError("BPlusTree only supports INT or TEXT keys")
        self.buffer_pool = buffer_pool
        self.key_type = key_type
        self.key_size = 8 if key_type == ColumnType.INT else key_size
        self.unique = unique

        # min_* must satisfy "two minimally-full siblings can always be
        # merged back into one page" or a delete-triggered merge could
        # overflow the destination page:
        #   leaf merge:     2 * min_leaf_keys                 <= max_leaf_keys
        #   internal merge: 2 * min_internal_keys + 1 (separator) <= max_internal_keys
        # so these use floor division, not ceil -- ceil-ing here was an
        # earlier bug that overflowed a page during a merge of two
        # odd-max-sized minimally-full leaves (see test_btree.py's
        # interleaved insert/delete stress test, which caught it).
        entry = self.key_size + _RID_SIZE
        self.max_leaf_keys = max(4, (PAGE_SIZE - _LEAF_HEADER) // entry)
        self.min_leaf_keys = max(1, self.max_leaf_keys // 2)

        internal_entry = self.key_size + _CHILD_SIZE
        self.max_internal_keys = max(
            4, (PAGE_SIZE - _INTERNAL_HEADER - _CHILD_SIZE) // internal_entry
        )
        self.min_internal_keys = max(1, (self.max_internal_keys - 1) // 2)

        if root_page_id is None:
            page_id, data = buffer_pool.new_page()
            self._write_leaf(data, [], [], INVALID_PAGE_ID)
            buffer_pool.unpin_page(page_id, dirty=True)
            root_page_id = page_id
        self.root_page_id = root_page_id

    # -- key encoding ----------------------------------------------------

    def _encode_key(self, key) -> bytes:
        if self.key_type == ColumnType.INT:
            return struct.pack("<q", int(key))
        raw = str(key).encode("utf-8")[: self.key_size]
        return raw + b"\x00" * (self.key_size - len(raw))

    def _decode_key(self, raw: bytes):
        if self.key_type == ColumnType.INT:
            return struct.unpack("<q", raw)[0]
        return raw.rstrip(b"\x00").decode("utf-8")

    # -- node (de)serialization -------------------------------------------

    def _node_type(self, data: bytearray) -> PageType:
        return PageType(data[0])

    def _read_leaf(self, data: bytearray) -> tuple[list, list[RID], int]:
        (num_keys,) = struct.unpack_from("<H", data, 1)
        (next_leaf,) = struct.unpack_from("<I", data, 3)
        keys: list = []
        rids: list[RID] = []
        offset = _LEAF_HEADER
        for _ in range(num_keys):
            key = self._decode_key(bytes(data[offset : offset + self.key_size]))
            offset += self.key_size
            page_id, slot_id = struct.unpack_from("<IH", data, offset)
            offset += _RID_SIZE
            keys.append(key)
            rids.append(RID(page_id, slot_id))
        return keys, rids, next_leaf

    def _write_leaf(self, data: bytearray, keys: list, rids: list[RID], next_leaf: int) -> None:
        data[0] = PageType.BTREE_LEAF
        struct.pack_into("<H", data, 1, len(keys))
        struct.pack_into("<I", data, 3, next_leaf)
        offset = _LEAF_HEADER
        for key, rid in zip(keys, rids):
            data[offset : offset + self.key_size] = self._encode_key(key)
            offset += self.key_size
            struct.pack_into("<IH", data, offset, rid.page_id, rid.slot_id)
            offset += _RID_SIZE

    def _read_internal(self, data: bytearray) -> tuple[list[int], list]:
        (num_keys,) = struct.unpack_from("<H", data, 1)
        offset = _INTERNAL_HEADER
        (child,) = struct.unpack_from("<I", data, offset)
        offset += _CHILD_SIZE
        children = [child]
        keys: list = []
        for _ in range(num_keys):
            key = self._decode_key(bytes(data[offset : offset + self.key_size]))
            offset += self.key_size
            keys.append(key)
            (child,) = struct.unpack_from("<I", data, offset)
            offset += _CHILD_SIZE
            children.append(child)
        return children, keys

    def _write_internal(self, data: bytearray, children: list[int], keys: list) -> None:
        data[0] = PageType.BTREE_INTERNAL
        struct.pack_into("<H", data, 1, len(keys))
        offset = _INTERNAL_HEADER
        struct.pack_into("<I", data, offset, children[0])
        offset += _CHILD_SIZE
        for key, child in zip(keys, children[1:]):
            data[offset : offset + self.key_size] = self._encode_key(key)
            offset += self.key_size
            struct.pack_into("<I", data, offset, child)
            offset += _CHILD_SIZE

    # -- search ------------------------------------------------------------

    def _find_leaf_path(self, key) -> list[int]:
        """Descend from the root to the leaf that would contain `key`,
        returning the full page-id path (root ... leaf).
        """
        path = [self.root_page_id]
        page_id = self.root_page_id
        while True:
            data = self.buffer_pool.fetch_page(page_id)
            node_type = self._node_type(data)
            if node_type == PageType.BTREE_LEAF:
                self.buffer_pool.unpin_page(page_id)
                return path
            children, keys = self._read_internal(data)
            self.buffer_pool.unpin_page(page_id)
            idx = bisect_right(keys, key)
            page_id = children[idx]
            path.append(page_id)

    def search(self, key) -> list[RID]:
        """Return every RID stored under `key` (empty list if none)."""
        leaf_id = self._find_leaf_path(key)[-1]
        data = self.buffer_pool.fetch_page(leaf_id)
        try:
            keys, rids, _ = self._read_leaf(data)
            lo = bisect_left(keys, key)
            hi = bisect_right(keys, key)
            return rids[lo:hi]
        finally:
            self.buffer_pool.unpin_page(leaf_id)

    def range_scan(self, start=None, end=None) -> Iterator[tuple[object, RID]]:
        """Yield (key, RID) pairs with start <= key <= end, in ascending key
        order, following leaf `next_leaf` pointers -- the whole reason a
        B+-tree (unlike a plain B-tree) threads its leaves together.
        """
        leaf_id = self._find_leaf_path(start)[-1] if start is not None else self._leftmost_leaf()
        while leaf_id != INVALID_PAGE_ID:
            data = self.buffer_pool.fetch_page(leaf_id)
            keys, rids, next_leaf = self._read_leaf(data)
            self.buffer_pool.unpin_page(leaf_id)

            lo = 0 if start is None else bisect_left(keys, start)
            for i in range(lo, len(keys)):
                if end is not None and keys[i] > end:
                    return
                yield keys[i], rids[i]
            leaf_id = next_leaf

    def _leftmost_leaf(self) -> int:
        page_id = self.root_page_id
        while True:
            data = self.buffer_pool.fetch_page(page_id)
            node_type = self._node_type(data)
            if node_type == PageType.BTREE_LEAF:
                self.buffer_pool.unpin_page(page_id)
                return page_id
            children, _ = self._read_internal(data)
            self.buffer_pool.unpin_page(page_id)
            page_id = children[0]

    def __iter__(self) -> Iterator[tuple[object, RID]]:
        yield from self.range_scan()

    # -- insert --------------------------------------------------------------

    def _split_point_avoiding_duplicates(self, keys: list) -> int:
        n = len(keys)
        mid = n // 2
        if keys[mid] != keys[mid - 1]:
            return mid
        # Search forward first for the end of this duplicate run...
        i = mid
        while i < n and keys[i] == keys[mid - 1]:
            i += 1
        if i < n:
            return i
        # ...the whole tail is one duplicate run; try splitting before it
        # starts instead.
        j = mid - 1
        while j > 0 and keys[j - 1] == keys[mid - 1]:
            j -= 1
        if j > 0:
            return j
        # The entire leaf is a single repeated key (more duplicates of one
        # key than fit in a page) -- there is no split point that avoids
        # separating them. This is a documented limit: a single key can
        # have at most roughly `max_leaf_keys` duplicate entries.
        raise ValueError(
            f"cannot split a leaf containing only duplicates of one key "
            f"(more than {self.max_leaf_keys} duplicate entries for the same key "
            "are not supported)"
        )

    def insert(self, key, rid: RID) -> None:
        if self.unique and self.search(key):
            raise DuplicateKeyError(f"duplicate key in unique index: {key!r}")

        path = self._find_leaf_path(key)
        leaf_id = path[-1]
        data = self.buffer_pool.fetch_page(leaf_id)
        keys, rids, next_leaf = self._read_leaf(data)
        pos = bisect_right(keys, key)
        keys.insert(pos, key)
        rids.insert(pos, rid)

        if len(keys) <= self.max_leaf_keys:
            self._write_leaf(data, keys, rids, next_leaf)
            self.buffer_pool.unpin_page(leaf_id, dirty=True)
            return

        # Overflow: split into two leaves, threading next_leaf pointers, and
        # push the right half's first key up as the new separator.
        #
        # Search routes to a child by bisect_right on separator keys, so a
        # key equal to the separator always lands in the *right* leaf. If a
        # naive midpoint split cut a run of duplicate keys in half, the
        # duplicates left behind in the *left* leaf would become permanently
        # unreachable by search. Nudge the split point to a key boundary
        # instead, keeping every duplicate of a given key in one leaf.
        mid = self._split_point_avoiding_duplicates(keys)
        left_keys, right_keys = keys[:mid], keys[mid:]
        left_rids, right_rids = rids[:mid], rids[mid:]

        new_leaf_id, new_data = self.buffer_pool.new_page()
        self._write_leaf(new_data, right_keys, right_rids, next_leaf)
        self.buffer_pool.unpin_page(new_leaf_id, dirty=True)

        self._write_leaf(data, left_keys, left_rids, new_leaf_id)
        self.buffer_pool.unpin_page(leaf_id, dirty=True)

        self._insert_into_parent(path[:-1], leaf_id, right_keys[0], new_leaf_id)

    def _insert_into_parent(
        self, ancestor_path: list[int], left_child: int, separator, right_child: int
    ) -> None:
        if not ancestor_path:
            # left_child was the root; grow the tree by one level.
            new_root_id, new_root_data = self.buffer_pool.new_page()
            self._write_internal(new_root_data, [left_child, right_child], [separator])
            self.buffer_pool.unpin_page(new_root_id, dirty=True)
            self.root_page_id = new_root_id
            return

        parent_id = ancestor_path[-1]
        data = self.buffer_pool.fetch_page(parent_id)
        children, keys = self._read_internal(data)
        idx = children.index(left_child)
        children.insert(idx + 1, right_child)
        keys.insert(idx, separator)

        if len(keys) <= self.max_internal_keys:
            self._write_internal(data, children, keys)
            self.buffer_pool.unpin_page(parent_id, dirty=True)
            return

        mid = len(keys) // 2
        up_key = keys[mid]
        left_keys, right_keys = keys[:mid], keys[mid + 1 :]
        left_children, right_children = children[: mid + 1], children[mid + 1 :]

        new_internal_id, new_data = self.buffer_pool.new_page()
        self._write_internal(new_data, right_children, right_keys)
        self.buffer_pool.unpin_page(new_internal_id, dirty=True)

        self._write_internal(data, left_children, left_keys)
        self.buffer_pool.unpin_page(parent_id, dirty=True)

        self._insert_into_parent(ancestor_path[:-1], parent_id, up_key, new_internal_id)

    # -- delete --------------------------------------------------------------

    def delete(self, key, rid: RID) -> bool:
        path = self._find_leaf_path(key)
        leaf_id = path[-1]
        data = self.buffer_pool.fetch_page(leaf_id)
        keys, rids, next_leaf = self._read_leaf(data)

        target = None
        lo, hi = bisect_left(keys, key), bisect_right(keys, key)
        for i in range(lo, hi):
            if rids[i] == rid:
                target = i
                break
        if target is None:
            self.buffer_pool.unpin_page(leaf_id)
            return False

        del keys[target]
        del rids[target]
        self._write_leaf(data, keys, rids, next_leaf)
        self.buffer_pool.unpin_page(leaf_id, dirty=True)

        self._fixup_after_delete(path)
        return True

    def _fixup_after_delete(self, path: list[int]) -> None:
        """Walk from the modified leaf back up to the root, borrowing from
        or merging with a sibling wherever a node has fallen below its
        minimum occupancy, and updating separator keys in ancestors as
        we go. Standard bottom-up B+-tree deletion.
        """
        for level in range(len(path) - 1, 0, -1):
            node_id = path[level]
            parent_id = path[level - 1]

            data = self.buffer_pool.fetch_page(node_id)
            is_leaf = self._node_type(data) == PageType.BTREE_LEAF
            if is_leaf:
                keys, rids, next_leaf = self._read_leaf(data)
                count, min_keys = len(keys), self.min_leaf_keys
            else:
                children, keys = self._read_internal(data)
                count, min_keys = len(keys), self.min_internal_keys
            self.buffer_pool.unpin_page(node_id)

            if count >= min_keys:
                return  # this node (and everything above it) is fine

            parent_data = self.buffer_pool.fetch_page(parent_id)
            p_children, p_keys = self._read_internal(parent_data)
            idx = p_children.index(node_id)
            self.buffer_pool.unpin_page(parent_id)

            if is_leaf:
                self._fixup_leaf(parent_id, p_children, p_keys, idx)
            else:
                self._fixup_internal(parent_id, p_children, p_keys, idx)

        # If we fell out of the loop, we fixed everything up through level 1;
        # now check whether the root itself needs to shrink.
        self._maybe_shrink_root()

    def _fixup_leaf(self, parent_id: int, p_children: list[int], p_keys: list, idx: int) -> None:
        node_id = p_children[idx]
        node_data = self.buffer_pool.fetch_page(node_id)
        keys, rids, next_leaf = self._read_leaf(node_data)
        self.buffer_pool.unpin_page(node_id)

        right_id = p_children[idx + 1] if idx + 1 < len(p_children) else None
        left_id = p_children[idx - 1] if idx > 0 else None

        if right_id is not None:
            r_data = self.buffer_pool.fetch_page(right_id)
            r_keys, r_rids, r_next = self._read_leaf(r_data)
            # Borrowing moves entries from the front of the right sibling and
            # makes the new r_keys[0] the updated separator. Moving a single
            # entry is only safe if r_keys[0] != r_keys[1]: otherwise the
            # borrowed key is still duplicated in the right sibling, the
            # separator stays equal to it, and the copy we just moved left
            # becomes unreachable (search always routes a key equal to the
            # separator to the right). So instead of always moving exactly
            # one entry, move the *whole* leading run of duplicates -- still
            # a valid borrow as long as it doesn't underflow the sibling or
            # overflow this node.
            k = 1
            while k < len(r_keys) and r_keys[k] == r_keys[0]:
                k += 1
            if (
                k < len(r_keys)  # a distinct key remains to serve as separator
                and len(r_keys) - k >= self.min_leaf_keys
                and len(keys) + k <= self.max_leaf_keys
            ):
                keys.extend(r_keys[:k])
                rids.extend(r_rids[:k])
                del r_keys[:k]
                del r_rids[:k]
                self._write_leaf(r_data, r_keys, r_rids, r_next)
                self.buffer_pool.unpin_page(right_id, dirty=True)
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_leaf(node_data2, keys, rids, next_leaf)
                self.buffer_pool.unpin_page(node_id, dirty=True)
                p_keys[idx] = r_keys[0]
                self._commit_internal(parent_id, p_children, p_keys)
                return
            self.buffer_pool.unpin_page(right_id)

        if left_id is not None:
            l_data = self.buffer_pool.fetch_page(left_id)
            l_keys, l_rids, l_next = self._read_leaf(l_data)
            # Symmetric to the right-borrow case: move the whole trailing
            # duplicate run off the end of the left sibling.
            k = 1
            while k < len(l_keys) and l_keys[-1 - k] == l_keys[-1]:
                k += 1
            if (
                k < len(l_keys)
                and len(l_keys) - k >= self.min_leaf_keys
                and len(keys) + k <= self.max_leaf_keys
            ):
                keys[0:0] = l_keys[-k:]
                rids[0:0] = l_rids[-k:]
                del l_keys[-k:]
                del l_rids[-k:]
                self._write_leaf(l_data, l_keys, l_rids, l_next)
                self.buffer_pool.unpin_page(left_id, dirty=True)
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_leaf(node_data2, keys, rids, next_leaf)
                self.buffer_pool.unpin_page(node_id, dirty=True)
                p_keys[idx - 1] = keys[0]
                self._commit_internal(parent_id, p_children, p_keys)
                return
            self.buffer_pool.unpin_page(left_id)

        # Neither sibling can spare a *safe* single-run borrow (the borrow
        # checks above already tried that). Prefer combining with the right
        # sibling; fall back to the left one otherwise.
        #
        # Normally the combined entries fit in one page and we merge into a
        # single leaf, dropping the separator + orphaned child from the
        # parent. But the same duplicate-run obstruction that can block a
        # borrow can also -- rarely -- leave a "spare" sibling sitting well
        # above min_leaf_keys with no valid partial-borrow boundary (its
        # excess entries are all one trailing/leading duplicate run), so a
        # plain concatenation can occasionally exceed one page. When that
        # happens, redistribute the combined entries across *both* pages at
        # a fresh duplicate-safe split point instead of collapsing to one --
        # this always fits (the combined size is at most 2*max_leaf_keys)
        # and keeps the parent's child count and separator key valid.
        if right_id is not None:
            r_data = self.buffer_pool.fetch_page(right_id)
            r_keys, r_rids, r_next = self._read_leaf(r_data)
            self.buffer_pool.unpin_page(right_id)
            combined_keys = keys + r_keys
            combined_rids = rids + r_rids

            if len(combined_keys) <= self.max_leaf_keys:
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_leaf(node_data2, combined_keys, combined_rids, r_next)
                self.buffer_pool.unpin_page(node_id, dirty=True)
                del p_children[idx + 1]
                del p_keys[idx]
            else:
                split = self._split_point_avoiding_duplicates(combined_keys)
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_leaf(node_data2, combined_keys[:split], combined_rids[:split], right_id)
                self.buffer_pool.unpin_page(node_id, dirty=True)
                r_data2 = self.buffer_pool.fetch_page(right_id)
                self._write_leaf(r_data2, combined_keys[split:], combined_rids[split:], r_next)
                self.buffer_pool.unpin_page(right_id, dirty=True)
                p_keys[idx] = combined_keys[split]  # both children kept; refresh the separator
        else:
            l_data = self.buffer_pool.fetch_page(left_id)
            l_keys, l_rids, l_next = self._read_leaf(l_data)
            self.buffer_pool.unpin_page(left_id)
            combined_keys = l_keys + keys
            combined_rids = l_rids + rids

            if len(combined_keys) <= self.max_leaf_keys:
                l_data2 = self.buffer_pool.fetch_page(left_id)
                self._write_leaf(l_data2, combined_keys, combined_rids, next_leaf)
                self.buffer_pool.unpin_page(left_id, dirty=True)
                del p_children[idx]
                del p_keys[idx - 1]
            else:
                split = self._split_point_avoiding_duplicates(combined_keys)
                l_data2 = self.buffer_pool.fetch_page(left_id)
                self._write_leaf(l_data2, combined_keys[:split], combined_rids[:split], node_id)
                self.buffer_pool.unpin_page(left_id, dirty=True)
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_leaf(
                    node_data2, combined_keys[split:], combined_rids[split:], next_leaf
                )
                self.buffer_pool.unpin_page(node_id, dirty=True)
                p_keys[idx - 1] = combined_keys[split]

        self._commit_internal(parent_id, p_children, p_keys)

    def _fixup_internal(
        self, parent_id: int, p_children: list[int], p_keys: list, idx: int
    ) -> None:
        node_id = p_children[idx]
        node_data = self.buffer_pool.fetch_page(node_id)
        children, keys = self._read_internal(node_data)
        self.buffer_pool.unpin_page(node_id)

        right_id = p_children[idx + 1] if idx + 1 < len(p_children) else None
        left_id = p_children[idx - 1] if idx > 0 else None

        if right_id is not None:
            r_data = self.buffer_pool.fetch_page(right_id)
            r_children, r_keys = self._read_internal(r_data)
            if len(r_keys) > self.min_internal_keys:
                # Rotate through the parent separator.
                keys.append(p_keys[idx])
                children.append(r_children.pop(0))
                new_sep = r_keys.pop(0)
                self._write_internal(r_data, r_children, r_keys)
                self.buffer_pool.unpin_page(right_id, dirty=True)
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_internal(node_data2, children, keys)
                self.buffer_pool.unpin_page(node_id, dirty=True)
                p_keys[idx] = new_sep
                self._commit_internal(parent_id, p_children, p_keys)
                return
            self.buffer_pool.unpin_page(right_id)

        if left_id is not None:
            l_data = self.buffer_pool.fetch_page(left_id)
            l_children, l_keys = self._read_internal(l_data)
            if len(l_keys) > self.min_internal_keys:
                keys.insert(0, p_keys[idx - 1])
                children.insert(0, l_children.pop())
                new_sep = l_keys.pop()
                self._write_internal(l_data, l_children, l_keys)
                self.buffer_pool.unpin_page(left_id, dirty=True)
                node_data2 = self.buffer_pool.fetch_page(node_id)
                self._write_internal(node_data2, children, keys)
                self.buffer_pool.unpin_page(node_id, dirty=True)
                p_keys[idx - 1] = new_sep
                self._commit_internal(parent_id, p_children, p_keys)
                return
            self.buffer_pool.unpin_page(left_id)

        if right_id is not None:
            r_data = self.buffer_pool.fetch_page(right_id)
            r_children, r_keys = self._read_internal(r_data)
            self.buffer_pool.unpin_page(right_id)
            merged_keys = keys + [p_keys[idx]] + r_keys
            merged_children = children + r_children
            node_data2 = self.buffer_pool.fetch_page(node_id)
            self._write_internal(node_data2, merged_children, merged_keys)
            self.buffer_pool.unpin_page(node_id, dirty=True)
            del p_children[idx + 1]
            del p_keys[idx]
        else:
            l_data = self.buffer_pool.fetch_page(left_id)
            l_children, l_keys = self._read_internal(l_data)
            self.buffer_pool.unpin_page(left_id)
            merged_keys = l_keys + [p_keys[idx - 1]] + keys
            merged_children = l_children + children
            l_data2 = self.buffer_pool.fetch_page(left_id)
            self._write_internal(l_data2, merged_children, merged_keys)
            self.buffer_pool.unpin_page(left_id, dirty=True)
            del p_children[idx]
            del p_keys[idx - 1]

        self._commit_internal(parent_id, p_children, p_keys)

    def _commit_internal(self, page_id: int, children: list[int], keys: list) -> None:
        data = self.buffer_pool.fetch_page(page_id)
        self._write_internal(data, children, keys)
        self.buffer_pool.unpin_page(page_id, dirty=True)

    def _maybe_shrink_root(self) -> None:
        data = self.buffer_pool.fetch_page(self.root_page_id)
        node_type = self._node_type(data)
        if node_type == PageType.BTREE_INTERNAL:
            children, keys = self._read_internal(data)
            self.buffer_pool.unpin_page(self.root_page_id)
            if len(keys) == 0:
                # The root is an internal node with a single child left --
                # that child becomes the new root, and the tree gets one
                # level shorter.
                self.root_page_id = children[0]
        else:
            self.buffer_pool.unpin_page(self.root_page_id)

    # -- introspection (used by tests/benchmarks) ---------------------------

    def height(self) -> int:
        h = 1
        page_id = self.root_page_id
        while True:
            data = self.buffer_pool.fetch_page(page_id)
            node_type = self._node_type(data)
            if node_type == PageType.BTREE_LEAF:
                self.buffer_pool.unpin_page(page_id)
                return h
            children, _ = self._read_internal(data)
            self.buffer_pool.unpin_page(page_id)
            page_id = children[0]
            h += 1
