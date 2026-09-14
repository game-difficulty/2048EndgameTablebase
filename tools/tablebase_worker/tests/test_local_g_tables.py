from __future__ import annotations

import os
import unittest

from tools.tablebase_worker.config import load_worker_config, table_path_status
from tools.tablebase_worker.reader_pool import ExistingBookReaderAdapter, _normalized_rate


@unittest.skipUnless(os.getenv("RUN_LOCAL_TABLE_TESTS") == "1", "Requires local G-drive tables")
class LocalGTableTests(unittest.TestCase):
    def test_random_boards_are_queryable_with_the_configured_dtype(self):
        config = load_worker_config(require_auth=False)
        for name in ("4442f_1024", "free11_2048"):
            with self.subTest(table=name):
                table = config.tables[name]
                self.assertTrue(table_path_status(table)[0])
                reader = ExistingBookReaderAdapter(table)
                for _ in range(3):
                    board = reader.random_state()
                    self.assertNotEqual(board, 0)
                    results, dtype = reader.lookup(board, use_variant=False, board_is_lookup=False)
                    self.assertEqual(dtype, table.dtype)
                    rates = [_normalized_rate(value, dtype) for value in results.values()
                             if isinstance(value, (int, float))]
                    self.assertTrue(rates)
                    self.assertGreater(max(rates), 0)


if __name__ == "__main__":
    unittest.main()
