"""Negative file-layer regression tests. Run with the freshly built native module.

python tests/test_free_negative_layers.py -v
Optional FREE_LAYER_MODULE_DIR selects an isolated build without replacing a loaded DLL.
"""
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("FREE_LAYER_MODULE_DIR"):
    directory = Path(os.environ["FREE_LAYER_MODULE_DIR"])
    module_path = next(directory.glob("formation_core*.pyd"))
    _dll_handle = os.add_dll_directory("C:/Apps/mingw64/bin") if os.name == "nt" else None
    import native_core
    spec = importlib.util.spec_from_file_location("native_core.formation_core", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native_core.formation_core = module

import numpy as np
from native_core import formation_core as f
from engine_core import mover_runtime
from Config import pattern_catalog


def options(root, offset, *, chunked=False, compress=False, optimal=False):
    o = f.RunOptions()
    o.target, o.steps, o.docheck_step = 2, 5, 0
    o.pathname = str(root / "free12_4_")
    o.num_threads, o.is_free = 2, True
    o.chunked_solve, o.compress = chunked, compress
    o.optimal_branch_only = optimal
    o.direct_io = False
    return o


def pattern(advanced):
    p = f.AdvancedPatternSpec() if advanced else f.PatternSpec()
    p.name, p.symm_mode = "free12", 1
    p.success_shifts = list(range(0, 64, 4))
    if advanced:
        p.target, p.num_free_32k, p.small_tile_sum_limit = 2, 4, 96
    return p


class NegativeReaders(unittest.TestCase):
    def test_classic_negative_lookup_and_legacy_positive_path(self):
        # Independent handcrafted .book fixture tests the reader's four directions.
        p = pattern(False)
        reader = f.ClassicBookReader(p, False)
        board = np.uint64(0x000000000001ffff)
        moved = set(int(mover_runtime.canonical_full(b)) for b in mover_runtime.std.move_all_dir(board) if b != board)
        raw = b"".join(struct.pack("<QI", b, 3000000000) for b in sorted(moved))
        matrix = mover_runtime.decode_board(board).tolist()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "free12_4_-10.book").write_bytes(raw)
            result, _ = reader.move_on_dic(matrix, [(tmp, "uint32")], "free12_4", -(4*32768+22))
            self.assertTrue(any(isinstance(v, (float, int)) and v > 0 for v in result.values()), result)
            (path / "free12_4_0.book").write_bytes(raw)
            old, _ = reader.move_on_dic(matrix, [(tmp, "uint32")], "free12_4", -(4*32768+2))
            self.assertEqual(dict(result), dict(old))

    def test_ad_negative_lookup(self):
        reader = f.AdvancedBookReader(pattern(True), False)
        board = np.uint64(0x000000000001ffff)
        moved = sorted(set(int(mover_runtime.canonical_full(b)) for b in mover_runtime.std.move_all_dir(board) if b != board))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"free12_4_-10b"
            folder.mkdir()
            (folder/"4.i").write_bytes(struct.pack(f"<{len(moved)}Q", *moved))
            (folder/"4.b").write_bytes(struct.pack(f"<{len(moved)}I", *([3000000000]*len(moved))))
            result, _ = reader.move_on_dic(mover_runtime.decode_board(board).tolist(),
                                          [(tmp,"uint32")], "free12_4", -(4*32768+22))
            self.assertTrue(any(isinstance(v,(float,int)) and v > 0 for v in result.values()), result)


    def test_ex_readers_negative_compressed_and_raw(self):
        for advanced in (False, True):
            for compressed in (False, True):
                with self.subTest(advanced=advanced, compressed=compressed), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    o = options(root, -2, chunked=advanced, compress=compressed)
                    o.docheck_step = 999
                    p = pattern(advanced)
                    build = f.run_pattern_build_exad if advanced else f.run_pattern_build_zmask
                    build(np.array([0x2ffff], dtype=np.uint64), p, o)
                    # Build a tiny legacy fixture, then move its result filenames to
                    # negative logical layers without changing payload encoding.
                    for file in list(root.iterdir()):
                        suffix = file.name.removeprefix("free12_4_")
                        number, dot, rest = suffix.partition(".")
                        if number.isdigit():
                            file.rename(root / f"free12_4_{int(number)-2}.{rest}")
                    reader = f.EXADBookReader(p, False) if advanced else f.EXBookReader(p, False)
                    # Initial sum + one spawn-2 => internal layer 1 => file -1.
                    board = np.uint64(0x12ffff)
                    result, _ = reader.move_on_dic(mover_runtime.decode_board(board).tolist(),
                        [(tmp,"uint32")], "free12_4", -(4*32768+8))
                    self.assertTrue(any(isinstance(v,(float,int)) and v >= 0 for v in result.values()), result)


if __name__ == "__main__":
    unittest.main()
