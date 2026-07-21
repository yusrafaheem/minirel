import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.index.btree import BPlusTree, DuplicateKeyError
from minirel.storage.buffer_pool import BufferPoolManager
from minirel.storage.disk_manager import DiskManager
from minirel.storage.page import RID
from minirel.types import ColumnType


class BTreeTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(delete=False)
        self._tmp.close()
        self.path = self._tmp.name
        self.dm = DiskManager(self.path)
        self.bpm = BufferPoolManager(self.dm, pool_size=256)

    def tearDown(self):
        self.dm.close()
        os.unlink(self.path)


class TestBasicOps(BTreeTestBase):
    def test_search_on_empty_tree_returns_empty(self):
        tree = BPlusTree(self.bpm)
        self.assertEqual(tree.search(42), [])

    def test_insert_then_search_single_key(self):
        tree = BPlusTree(self.bpm)
        tree.insert(5, RID(1, 0))
        self.assertEqual(tree.search(5), [RID(1, 0)])
        self.assertEqual(tree.search(6), [])

    def test_duplicate_keys_all_retrievable(self):
        tree = BPlusTree(self.bpm)
        tree.insert(5, RID(1, 0))
        tree.insert(5, RID(2, 0))
        tree.insert(5, RID(3, 0))
        self.assertEqual(set(tree.search(5)), {RID(1, 0), RID(2, 0), RID(3, 0)})

    def test_unique_index_rejects_duplicate(self):
        tree = BPlusTree(self.bpm, unique=True)
        tree.insert(5, RID(1, 0))
        with self.assertRaises(DuplicateKeyError):
            tree.insert(5, RID(2, 0))

    def test_delete_removes_only_matching_rid(self):
        tree = BPlusTree(self.bpm)
        tree.insert(5, RID(1, 0))
        tree.insert(5, RID(2, 0))
        self.assertTrue(tree.delete(5, RID(1, 0)))
        self.assertEqual(tree.search(5), [RID(2, 0)])

    def test_delete_missing_key_returns_false(self):
        tree = BPlusTree(self.bpm)
        self.assertFalse(tree.delete(99, RID(1, 0)))

    def test_delete_existing_key_with_wrong_rid_returns_false_and_leaves_entry(self):
        tree = BPlusTree(self.bpm)
        tree.insert(5, RID(1, 0))
        self.assertFalse(tree.delete(5, RID(2, 0)))  # right key, RID that was never inserted
        self.assertEqual(tree.search(5), [RID(1, 0)])

    def test_range_scan_ascending_order(self):
        tree = BPlusTree(self.bpm)
        for k in [7, 2, 9, 4, 1, 5]:
            tree.insert(k, RID(k, 0))
        result = [k for k, _ in tree.range_scan()]
        self.assertEqual(result, sorted([7, 2, 9, 4, 1, 5]))

    def test_range_scan_bounds_are_inclusive(self):
        tree = BPlusTree(self.bpm)
        for k in range(10):
            tree.insert(k, RID(k, 0))
        result = [k for k, _ in tree.range_scan(start=3, end=6)]
        self.assertEqual(result, [3, 4, 5, 6])

    def test_text_keys_round_trip_and_sort_lexicographically(self):
        tree = BPlusTree(self.bpm, key_type=ColumnType.TEXT, key_size=16)
        for word in ["banana", "apple", "cherry"]:
            tree.insert(word, RID(0, 0))
        self.assertEqual(tree.search("apple"), [RID(0, 0)])
        self.assertEqual([k for k, _ in tree.range_scan()], ["apple", "banana", "cherry"])


class TestSplitsAndFanout(BTreeTestBase):
    def test_many_inserts_force_leaf_and_internal_splits_and_stay_shallow(self):
        tree = BPlusTree(self.bpm)
        n = 5000
        keys = list(range(n))
        random.Random(42).shuffle(keys)
        for k in keys:
            tree.insert(k, RID(k, 0))

        for k in range(n):
            self.assertEqual(tree.search(k), [RID(k, 0)])

        result = [k for k, _ in tree.range_scan()]
        self.assertEqual(result, list(range(n)))

        # The whole point of a B+-tree's high fanout: thousands of keys
        # should still fit in a handful of levels, not thousands of levels.
        self.assertLessEqual(tree.height(), 4)

    def test_reopen_existing_tree_from_root_page_id(self):
        tree = BPlusTree(self.bpm)
        for k in range(2000):
            tree.insert(k, RID(k, 0))
        self.bpm.flush_all()

        reopened = BPlusTree(self.bpm, root_page_id=tree.root_page_id)
        self.assertEqual(reopened.search(1500), [RID(1500, 0)])
        self.assertEqual(len([k for k, _ in reopened.range_scan()]), 2000)


class TestDeleteRebalancing(BTreeTestBase):
    def test_delete_all_keys_one_at_a_time_leaves_tree_empty(self):
        tree = BPlusTree(self.bpm)
        keys = list(range(800))
        random.Random(1).shuffle(keys)
        for k in keys:
            tree.insert(k, RID(k, 0))

        delete_order = list(keys)
        random.Random(2).shuffle(delete_order)
        for k in delete_order:
            self.assertTrue(tree.delete(k, RID(k, 0)), f"failed to delete {k}")

        self.assertEqual(list(tree.range_scan()), [])
        for k in range(800):
            self.assertEqual(tree.search(k), [])

    def test_interleaved_insert_and_delete_matches_reference_model(self):
        tree = BPlusTree(self.bpm)
        rng = random.Random(7)
        reference: dict[int, set[RID]] = {}

        for step in range(4000):
            key = rng.randint(0, 300)
            if rng.random() < 0.65 or key not in reference or not reference[key]:
                rid = RID(step, 0)
                tree.insert(key, rid)
                reference.setdefault(key, set()).add(rid)
            else:
                rid = next(iter(reference[key]))
                self.assertTrue(tree.delete(key, rid))
                reference[key].discard(rid)

        for key, rids in reference.items():
            self.assertEqual(set(tree.search(key)), rids, f"mismatch at key {key}")

        sort_key = lambda pair: (pair[0], pair[1].page_id, pair[1].slot_id)  # noqa: E731
        expected_pairs = sorted(
            ((key, rid) for key, rids in reference.items() for rid in rids), key=sort_key
        )
        actual_pairs = sorted(tree.range_scan(), key=sort_key)
        self.assertEqual(actual_pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()
