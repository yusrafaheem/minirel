import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.catalog import Catalog
from minirel.planner import build_select_plan
from minirel.sql.parser import parse
from minirel.storage.buffer_pool import BufferPoolManager
from minirel.storage.disk_manager import DiskManager
from minirel.storage.heap_file import HeapFile
from minirel.transaction import TransactionManager
from minirel.types import INFINITY_TXN, Column, ColumnType, Schema, encode_row, pack_tuple
from minirel.wal import WriteAheadLog

WIDGETS_SCHEMA = Schema(
    columns=(
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
        Column("price", ColumnType.FLOAT),
        Column("category", ColumnType.TEXT),
    )
)

ORDERS_SCHEMA = Schema(
    columns=(
        Column("id", ColumnType.INT),
        Column("widget_id", ColumnType.INT),
        Column("qty", ColumnType.INT),
    )
)

WIDGET_ROWS = [
    (1, "bolt", 0.50, "hardware"),
    (2, "nail", 0.10, "hardware"),
    (3, "widget", 9.99, "gadgets"),
    (4, "gizmo", 14.99, "gadgets"),
    (5, "sprocket", 3.25, "hardware"),
]


class PlannerExecutorTestBase(unittest.TestCase):
    def setUp(self):
        self._db_tmp = tempfile.NamedTemporaryFile(delete=False)
        self._db_tmp.close()
        self._cat_tmp = tempfile.NamedTemporaryFile(delete=False)
        self._cat_tmp.close()
        os.unlink(self._cat_tmp.name)
        self._wal_tmp = tempfile.NamedTemporaryFile(delete=False)
        self._wal_tmp.close()

        self.dm = DiskManager(self._db_tmp.name)
        self.bpm = BufferPoolManager(self.dm, pool_size=64)
        self.catalog = Catalog(self.bpm, self._cat_tmp.name)
        self.wal = WriteAheadLog(self._wal_tmp.name)
        self.txn_mgr = TransactionManager(self.wal)

        self.catalog.create_table("widgets", WIDGETS_SCHEMA)
        self.catalog.create_index("widgets", "widgets_id_idx", "id", unique=True)
        self.catalog.create_table("orders", ORDERS_SCHEMA)

        loader = self.txn_mgr.begin()
        widgets_first_page = self.catalog.get_table("widgets").heap_first_page_id
        heap = HeapFile(self.bpm, first_page_id=widgets_first_page)
        from minirel.index.btree import BPlusTree

        idx_meta = self.catalog.get_table("widgets").indexes["widgets_id_idx"]
        tree = BPlusTree(
            self.bpm, key_type=idx_meta.key_type, root_page_id=idx_meta.root_page_id, unique=True
        )
        for row in WIDGET_ROWS:
            payload = encode_row(list(row), WIDGETS_SCHEMA)
            tup = pack_tuple(loader.txn_id, INFINITY_TXN, payload)
            rid = heap.insert(tup)
            tree.insert(row[0], rid)
        self.catalog.update_index_root("widgets", "widgets_id_idx", tree.root_page_id)

        orders_first_page = self.catalog.get_table("orders").heap_first_page_id
        orders_heap = HeapFile(self.bpm, first_page_id=orders_first_page)
        for row in [(1, 3, 10), (2, 1, 100), (3, 4, 2)]:
            payload = encode_row(list(row), ORDERS_SCHEMA)
            tup = pack_tuple(loader.txn_id, INFINITY_TXN, payload)
            orders_heap.insert(tup)
        loader.commit()

    def tearDown(self):
        self.wal.close()
        self.dm.close()
        os.unlink(self._db_tmp.name)
        os.unlink(self._wal_tmp.name)
        if os.path.exists(self._cat_tmp.name):
            os.unlink(self._cat_tmp.name)

    def run_select(self, sql: str, txn=None):
        stmt = parse(sql)
        owns_txn = txn is None
        txn = txn or self.txn_mgr.begin()
        plan = build_select_plan(stmt, self.catalog, self.bpm, txn, self.txn_mgr)
        rows = list(plan)
        if owns_txn:
            txn.commit()
        return rows


class TestSeqScanAndFilter(PlannerExecutorTestBase):
    def test_select_star_returns_all_rows(self):
        rows = self.run_select("SELECT * FROM widgets")
        self.assertEqual(len(rows), 5)

    def test_where_equality_on_non_indexed_column(self):
        rows = self.run_select("SELECT name FROM widgets WHERE category = 'gadgets'")
        self.assertEqual({r["name"] for r in rows}, {"widget", "gizmo"})

    def test_where_comparison(self):
        rows = self.run_select("SELECT name FROM widgets WHERE price > 5.0")
        self.assertEqual({r["name"] for r in rows}, {"widget", "gizmo"})

    def test_where_and(self):
        sql = "SELECT name FROM widgets WHERE category = 'hardware' AND price < 1.0"
        rows = self.run_select(sql)
        self.assertEqual({r["name"] for r in rows}, {"bolt", "nail"})

    def test_where_or(self):
        rows = self.run_select("SELECT name FROM widgets WHERE id = 1 OR id = 4")
        self.assertEqual({r["name"] for r in rows}, {"bolt", "gizmo"})

    def test_select_specific_columns_only(self):
        rows = self.run_select("SELECT id, name FROM widgets WHERE id = 3")
        self.assertEqual(rows, [{"id": 3, "name": "widget"}])


class TestIndexScanPushdown(PlannerExecutorTestBase):
    def test_equality_on_indexed_column_uses_index_scan(self):
        from minirel.executor import IndexScanOperator

        stmt = parse("SELECT * FROM widgets WHERE id = 3")
        txn = self.txn_mgr.begin()
        plan = build_select_plan(stmt, self.catalog, self.bpm, txn, self.txn_mgr)
        # ProjectOperator wraps the scan directly (no FilterOperator) when
        # the whole WHERE clause was covered by the index.
        self.assertIsInstance(plan.child, IndexScanOperator)
        rows = list(plan)
        txn.commit()
        self.assertEqual(rows[0]["name"], "widget")

    def test_index_scan_result_matches_seq_scan_result(self):
        indexed = self.run_select("SELECT * FROM widgets WHERE id = 4")
        seq_sql = "SELECT * FROM widgets WHERE category = 'gadgets' OR category = 'hardware'"
        seq = self.run_select(seq_sql)
        matching = [r for r in seq if r["id"] == 4]
        self.assertEqual(indexed, matching)

    def test_residual_predicate_still_applied_after_index_scan(self):
        rows = self.run_select("SELECT name FROM widgets WHERE id = 3 AND price > 100")
        self.assertEqual(rows, [])


class TestJoin(PlannerExecutorTestBase):
    def test_inner_join_on_equality(self):
        sql = (
            "SELECT widgets.name, orders.qty FROM orders "
            "JOIN widgets ON orders.widget_id = widgets.id"
        )
        rows = self.run_select(sql)
        self.assertEqual(
            {(r["name"], r["qty"]) for r in rows},
            {("widget", 10), ("bolt", 100), ("gizmo", 2)},
        )


class TestAggregation(PlannerExecutorTestBase):
    def test_count_star_no_group_by(self):
        rows = self.run_select("SELECT COUNT(*) FROM widgets")
        self.assertEqual(rows, [{"count": 5}])

    def test_group_by_with_multiple_aggregates(self):
        rows = self.run_select(
            "SELECT category, COUNT(*), SUM(price) FROM widgets GROUP BY category"
        )
        by_category = {r["category"]: r for r in rows}
        self.assertEqual(by_category["hardware"]["count"], 3)
        self.assertAlmostEqual(by_category["hardware"]["sum"], 0.50 + 0.10 + 3.25)
        self.assertEqual(by_category["gadgets"]["count"], 2)

    def test_group_by_applies_where_filter_before_grouping(self):
        rows = self.run_select(
            "SELECT category, COUNT(*) FROM widgets WHERE price > 1.0 GROUP BY category"
        )
        by_category = {r["category"]: r["count"] for r in rows}
        self.assertEqual(by_category, {"hardware": 1, "gadgets": 2})

    def test_count_on_empty_result_is_zero_not_no_rows(self):
        rows = self.run_select("SELECT COUNT(*) FROM widgets WHERE id = 999")
        self.assertEqual(rows, [{"count": 0}])


class TestSortAndLimit(PlannerExecutorTestBase):
    def test_order_by_asc(self):
        rows = self.run_select("SELECT name FROM widgets ORDER BY price")
        self.assertEqual([r["name"] for r in rows], ["nail", "bolt", "sprocket", "widget", "gizmo"])

    def test_limit_zero_returns_no_rows(self):
        rows = self.run_select("SELECT name FROM widgets ORDER BY price LIMIT 0")
        self.assertEqual(rows, [])

    def test_order_by_multiple_columns(self):
        # category ASC (gadgets < hardware) breaking ties by price ASC.
        rows = self.run_select("SELECT name FROM widgets ORDER BY category, price")
        self.assertEqual([r["name"] for r in rows], ["widget", "gizmo", "nail", "bolt", "sprocket"])

    def test_order_by_desc_with_limit(self):
        rows = self.run_select("SELECT name FROM widgets ORDER BY price DESC LIMIT 2")
        self.assertEqual([r["name"] for r in rows], ["gizmo", "widget"])


class TestMvccVisibilityThroughExecutor(PlannerExecutorTestBase):
    def test_uncommitted_insert_invisible_to_other_transaction(self):
        from minirel.index.btree import BPlusTree

        writer = self.txn_mgr.begin()
        widgets_first_page = self.catalog.get_table("widgets").heap_first_page_id
        heap = HeapFile(self.bpm, first_page_id=widgets_first_page)
        idx_meta = self.catalog.get_table("widgets").indexes["widgets_id_idx"]
        tree = BPlusTree(
            self.bpm, key_type=idx_meta.key_type, root_page_id=idx_meta.root_page_id, unique=True
        )
        payload = encode_row([99, "secret", 1.0, "x"], WIDGETS_SCHEMA)
        rid = heap.insert(pack_tuple(writer.txn_id, INFINITY_TXN, payload))
        tree.insert(99, rid)
        self.catalog.update_index_root("widgets", "widgets_id_idx", tree.root_page_id)

        reader = self.txn_mgr.begin()
        rows = self.run_select("SELECT * FROM widgets WHERE id = 99", txn=reader)
        reader.commit()
        self.assertEqual(rows, [])

        writer.commit()
        reader2 = self.txn_mgr.begin()
        rows2 = self.run_select("SELECT name FROM widgets WHERE id = 99", txn=reader2)
        reader2.commit()
        self.assertEqual(rows2, [{"name": "secret"}])


if __name__ == "__main__":
    unittest.main()
