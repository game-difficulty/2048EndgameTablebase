from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(os.getenv("RUN_LOCAL_TABLE_TESTS") == "1", "Requires local free12 tables")
class LocalBCRawTests(unittest.TestCase):
    def test_raw_and_compressed_layers_from_both_drives(self):
        from native_core import formation_core as native
        from tools.tablebase_worker.config import load_worker_config
        from tools.tablebase_worker.reader_pool import ExistingBookReaderAdapter

        table = load_worker_config(require_auth=False).tables["free12_2048"]
        reader = ExistingBookReaderAdapter(table)
        for root in table.paths:
            for extension in (".bcraw", ".bccmp"):
                with self.subTest(root=str(root), extension=extension):
                    path = next(root.glob(f"free12_2048_*{extension}"))
                    board = native.sample_bc_compressed_random_state(str(path), 11, 0.1)
                    self.assertNotEqual(board, 0)
                    # The native sampler adds one tile to the archived board.
                    # Removing that tile must recover an actual archive record.
                    found = False
                    for position in range(16):
                        if (board >> (position * 4)) & 15 not in (1, 2):
                            continue
                        key = board & ~(15 << (position * 4))
                        result = native.lookup_bc_compressed_result_cold(str(path), 11, key)
                        if result["found"]:
                            found = True
                            break
                    self.assertTrue(found, str(path))
                    results, dtype = reader.lookup(board, use_variant=False, board_is_lookup=False)
                    self.assertEqual(dtype, "uint32")
                    self.assertTrue(any(isinstance(v, (float, int)) and v > 0 for v in results.values()))
                    print(f"{path.name}: {board:016x} {results}", flush=True)

    def test_truncated_raw_archive_is_rejected(self):
        from native_core import formation_core as native

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "free12_2048_0.bcraw"
            path.write_bytes(b"BCRAW1\0\0")
            with self.assertRaises(RuntimeError):
                native.lookup_bc_compressed_result_cold(str(path), 11, 0)


if __name__ == "__main__":
    unittest.main()
