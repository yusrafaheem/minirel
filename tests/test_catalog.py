import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from minirel.catalog import Catalog, TableAlreadyExistsError, TableNotFoundError
from minirel.storage.buffer_pool import BufferPoolManager
from minirel.storage.disk_manager import DiskManager
from minirel.types import Column, ColumnType, Schema

USERS_SCHEMA = Schema(
    columns=(
        Column("id", ColumnType.INT),
        Column("name", ColumnType.TEXT),
    )
)


class TestCatalog(unittest.TestCase):
    def setUp(self):
        self._db_tmp = tempfile.NamedTemporaryFile(delete=False)
        self._db_tmp.close()
        self._cat_tmp = tempfile.NamedTemporaryFile(delete=False)
        self._cat_tmp.close()
        os.unlink(self._cat_tmp.name)  # catalog constructor expects "doesn't exist yet"
        self.dm = DiskManager(self._db_tmp.name)
        self.bpm = BufferPoolManager(self.dm, pool_size=16)

    def tearDown(self):
        self.dm.close()
        os.unlink(self._db_tmp.name)
        if os.path.exists(self._cat_tmp.name):
            os.unlink(self._cat_tmp.name)

    def test_create_and_get_table(self):
        cat = Catalog(self.bpm, self._cat_tmp.name)
        meta = cat.create_table("users", USERS_SCHEMA)
        self.assertEqual(cat.get_table("users").name, "users")
        self.assertEqual(meta.schema.names, ["id", "name"])

    def test_create_duplicate_table_raises(self):
        cat = Catalog(self.bpm, self._cat_tmp.name)
        cat.create_table("users", USERS_SCHEMA)
        with self.assertRaises(TableAlreadyExistsError):
            cat.create_table("users", USERS_SCHEMA)

    def test_get_missing_table_raises(self):
        cat = Catalog(self.bpm, self._cat_tmp.name)
        with self.assertRaises(TableNotFoundError):
            cat.get_table("nope")

    def test_create_index_and_persist_across_reopen(self):
        cat = Catalog(self.bpm, self._cat_tmp.name)
        cat.create_table("users", USERS_SCHEMA)
        cat.create_index("users", "users_pkey", "id", unique=True)
        self.bpm.flush_all()

        reopened = Catalog(self.bpm, self._cat_tmp.name)
        table = reopened.get_table("users")
        self.assertIn("users_pkey", table.indexes)
        self.assertEqual(table.indexes["users_pkey"].column, "id")
        self.assertTrue(table.indexes["users_pkey"].unique)

    def test_heap_first_page_id_persists_and_is_reusable(self):
        cat = Catalog(self.bpm, self._cat_tmp.name)
        meta = cat.create_table("users", USERS_SCHEMA)
        self.bpm.flush_all()

        reopened = Catalog(self.bpm, self._cat_tmp.name)
        self.assertEqual(reopened.get_table("users").heap_first_page_id, meta.heap_first_page_id)

    def test_create_duplicate_index_name_raises(self):
        cat = Catalog(self.bpm, self._cat_tmp.name)
        cat.create_table("users", USERS_SCHEMA)
        cat.create_index("users", "users_pkey", "id")
        with self.assertRaises(ValueError):
            cat.create_index("users", "users_pkey", "id")


if __name__ == "__main__":
    unittest.main()
