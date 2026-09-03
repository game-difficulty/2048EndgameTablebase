from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backend import tablebase_catalog


class GuestTablebaseCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_limit = os.environ.get("GUEST_MAX_TABLE_MULTIPLIER")
        os.environ["GUEST_MAX_TABLE_MULTIPLIER"] = "5"

    def tearDown(self) -> None:
        if self.previous_limit is None:
            os.environ.pop("GUEST_MAX_TABLE_MULTIPLIER", None)
        else:
            os.environ["GUEST_MAX_TABLE_MULTIPLIER"] = self.previous_limit

    def test_only_local_low_cost_tables_are_guest_available(self) -> None:
        self.assertTrue(
            tablebase_catalog._entry_guest_available(
                {"_provider": "local", "_full_pattern": "L3_256"}
            )
        )
        self.assertFalse(
            tablebase_catalog._entry_guest_available(
                {"_provider": "local", "_full_pattern": "free10_256"}
            )
        )
        self.assertFalse(
            tablebase_catalog._entry_guest_available(
                {"_provider": "remote", "_full_pattern": "L3_256"}
            )
        )

    def test_public_catalog_exposes_boolean_but_never_storage_metadata(self) -> None:
        entry = {
            "pattern": "L3",
            "target": "256",
            "dtype": "uint32",
            "spawn_rate": 0.1,
            "_full_pattern": "L3_256",
            "_provider": "local",
            "_absolute_path": "/secret/tablebase/path",
        }
        with patch.object(tablebase_catalog, "_iter_available_entries", return_value=[entry]):
            payload = tablebase_catalog.get_available_tablebases()
        self.assertEqual(payload[0]["guest_available"], True)
        self.assertNotIn("_provider", payload[0])
        self.assertNotIn("_absolute_path", payload[0])
        self.assertNotIn("/secret/tablebase/path", str(payload))


if __name__ == "__main__":
    unittest.main()
