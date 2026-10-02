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
from engine_core.BookBuilder import free_layer_offset, generate_free_empty_inits, generate_free_inits
from engine_core import mover_runtime
from Config import pattern_catalog


def options(root, offset, *, chunked=False, compress=False, optimal=False):
    o = f.RunOptions()
    o.target, o.steps, o.docheck_step = 2, 5, 0
    o.pathname = str(root / "free12_4_")
    o.layer_offset = offset
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


class FreeLayers(unittest.TestCase):
    def test_seed_policy_and_legacy_zero_sum(self):
        for n in range(10, 17):
            with self.subTest(n=n):
                seeds = generate_free_empty_inits(n)
                self.assertGreater(len(seeds), 0)
                for board in seeds:
                    tiles = [(int(board) >> (4*i)) & 15 for i in range(16)]
                    self.assertEqual(tiles.count(15), 16-n)
                    self.assertTrue(all(v in (0, 15) for v in tiles))
                    self.assertEqual(board, mover_runtime.canonical_full(board))
                old_sum = (16-n)*32768+2*(n-1)
                self.assertEqual((16-n)*32768 - 2*free_layer_offset(f"free{n}"), old_sum)
                if f"free{n}" in pattern_catalog:
                    self.assertEqual(-pattern_catalog[f"free{n}"]["nums_adjust"], old_sum)
        for name in ("free8", "free9", "3x4free9", "free10x", "free17"):
            self.assertEqual(free_layer_offset(name), 0)
        self.assertEqual(len(generate_free_inits(7, 8)), 21283)

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

    def test_native_algorithms_shifted_files_and_resume(self):
        # The exact same native computation at offsets 0 and -2 must serialize
        # identical results, including through the logical -1/0 boundary.
        seeds = np.array([0x2ffff], dtype=np.uint64)
        for mode, advanced, chunked in (("classic", False, False), ("ad", True, False),
                                       ("ad", True, True), ("ex", False, False),
                                       ("exad", True, False), ("exad", True, True)):
            with self.subTest(mode=mode, chunked=chunked), tempfile.TemporaryDirectory() as tmp:
                build = getattr(f, {"classic":"run_pattern_build", "ad":"run_pattern_build_ad",
                                    "ex":"run_pattern_build_zmask", "exad":"run_pattern_build_exad"}[mode])
                dirs = [Path(tmp)/"zero", Path(tmp)/"negative"]
                for d, offset in zip(dirs, (0,-2)):
                    d.mkdir()
                    o = options(d, offset, chunked=chunked)
                    build(seeds, pattern(advanced), o)
                    snapshot = {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*") if p.is_file() and
                                (p.suffix in (".book", ".zbook", ".exadbook", ".i", ".b"))}
                    self.assertTrue(snapshot or list(d.glob("*b")))
                    build(seeds, pattern(advanced), o)
                    for name, value in snapshot.items():
                        self.assertEqual((d/name).read_bytes(), value, name)
                for p in dirs[0].rglob("*"):
                    if not p.is_file() or p.suffix not in (".book", ".zbook", ".exadbook", ".i", ".b"):
                        continue
                    rel = str(p.relative_to(dirs[0]))
                    import re
                    shifted = re.sub(r"free12_4_(\d+)", lambda m: "free12_4_"+str(int(m[1])-2), rel)
                    self.assertEqual(p.read_bytes(), (dirs[1]/shifted).read_bytes(), shifted)

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
                    reader = f.EXADBookReader(p, False) if advanced else f.EXBookReader(p, False)
                    # Initial sum + one spawn-2 => internal layer 1 => file -1.
                    board = np.uint64(0x12ffff)
                    result, _ = reader.move_on_dic(mover_runtime.decode_board(board).tolist(),
                        [(tmp,"uint32")], "free12_4", -(4*32768+8))
                    self.assertTrue(any(isinstance(v,(float,int)) and v >= 0 for v in result.values()), result)
                    build(np.array([0x2ffff], dtype=np.uint64), p, o)

    def test_python_run_options_preserve_terminal_sum(self):
        from unittest.mock import patch
        from engine_core import BookBuilder as b
        with patch.object(b, "write_runtime_deletion_threshold_signal"):
            old = b._build_native_run_options(8, 164, "unused", 117, True, False, .1)
            new = b._build_native_run_options(8, 164, "unused", 117, True, False, .1, layer_offset=-11)
        self.assertEqual(new.steps, old.steps+11)
        self.assertEqual(new.docheck_step, old.docheck_step+11)
        self.assertEqual(4*32768+2*(new.steps-1), 4*32768+22+2*(old.steps-1))

    def test_compression_cold_paths_and_optimal_resume(self):
        for mode in ("classic", "ad", "ex", "exad"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                hot, cold = root/"hot", root/"cold"
                hot.mkdir(); cold.mkdir()
                advanced = mode in ("ad", "exad")
                o = options(hot, -2, chunked=advanced, compress=True, optimal=not advanced)
                o.cold_pathnames = [str(cold/"free12_4_")]
                o.compress_temp_files = True
                build = getattr(f, {"classic":"run_pattern_build", "ad":"run_pattern_build_ad",
                                    "ex":"run_pattern_build_zmask", "exad":"run_pattern_build_exad"}[mode])
                seed = np.array([0x2ffff], dtype=np.uint64)
                build(seed, pattern(advanced), o)
                self.assertTrue(list(hot.glob("free12_4_-*"))+list(cold.glob("free12_4_-*")))
                build(seed, pattern(advanced), o)

    def test_extend_existing_nonnegative_results(self):
        for mode in ("classic", "ad", "ex", "exad"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                advanced = mode in ("ad", "exad")
                old = options(root, 0, chunked=advanced, compress=mode != "ad", optimal=not advanced)
                build = getattr(f, {"classic":"run_pattern_build", "ad":"run_pattern_build_ad",
                                    "ex":"run_pattern_build_zmask", "exad":"run_pattern_build_exad"}[mode])
                # Legacy layer 0 sum is 4*32768+6; new seed is smaller by 2.
                build(np.array([0x12ffff], dtype=np.uint64), pattern(advanced), old)
                suffixes = (".b", ".i", ".book", ".z", ".book.7z", ".zbook", ".exzbook", ".exadbook", ".exadzbook")
                positive = {p:p.read_bytes() for p in root.rglob("*")
                            if p.is_file() and p.name.endswith(suffixes)}
                self.assertTrue(positive)
                if mode == "ad":
                    # The old two exact frontier layers undergo normal retirement
                    # when extending backwards; compare already finalized results.
                    positive = {p:data for p,data in positive.items()
                                if p.parent.name not in ("free12_4_0b", "free12_4_1b")}
                if mode == "ex":
                    # Interrupted legacy optimal phase: its marker must not bypass new layers.
                    (root/"free12_4_ex_optimal_complete").unlink(missing_ok=True)
                new = options(root, -1, chunked=advanced, compress=mode != "ad", optimal=not advanced)
                new.steps = old.steps+1
                build(np.array([0x2ffff], dtype=np.uint64), pattern(advanced), new)
                self.assertTrue(list(root.glob("free12_4_-1*")))
                for p, data in positive.items():
                    self.assertEqual(p.read_bytes(), data, str(p))

    def test_bc_final_spawn4_terminal_matches_ex(self):
        # Three generated layers plus seed: the last spawn-4 output must not
        # be replaced by the solver's virtual empty sentinel.
        import csv
        import itertools
        terminal_row_count = None
        for generation_route, solve_route in itertools.product(("resident", "single", "family"), repeat=2):
            with self.subTest(generation=generation_route, solve=solve_route), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ("ex", "gen", "solve", "bc"):
                    (root/name).mkdir()
                seed = int(mover_runtime.canonical_full(np.uint64(0x0001ffffffffffff)))
                p = f.PatternSpec()
                p.name, p.symm_mode = "boundary", 1
                p.success_shifts = list(range(0, 64, 4))
                o = f.RunOptions()
                o.target, o.steps, o.docheck_step = 2, 4, 0
                o.pathname, o.num_threads, o.is_free = str(root/"ex/boundary_4_"), 2, True
                o.direct_io = False
                f.run_pattern_build_zmask(np.array([seed], dtype=np.uint64), p, o)
                f.run_bc_family_build(dict(pattern="boundary", target_rank=2, extra_steps=2,
                    seed_boards=[seed], canonical_symm_mode=1, success_shifts=p.success_shifts,
                    prefix="boundary_4_", threads=2, family_modulus=13, direct_io=False, compress=False,
                    family_route=generation_route, solve_route=solve_route,
                    generation_stats_csv=str(root/"generation.csv"), solve_stats_csv=str(root/"solve.csv"),
                    generated_dir=str(root/"gen"), solved_dir=str(root/"solve"), archive_dir=str(root/"bc")))
                with (root/"generation.csv").open() as stream:
                    actual_generation = {row["route"] for row in csv.DictReader(stream) if row["row_type"] == "layer"}
                with (root/"solve.csv").open() as stream:
                    actual_solve = {row["solve_route"] for row in csv.DictReader(stream) if row["kind"] == "solve"}
                self.assertEqual(actual_generation, {generation_route})
                self.assertEqual(actual_solve, {solve_route})
                # BCSuccessHeader stores payload offset/size at bytes 32/40.
                # The final real layer must contain successful spawn-4 outcomes;
                # a metadata-only virtual layer here reproduces the production bug.
                terminal_bytes = (root/"bc/boundary_4_3.bcsuc").read_bytes()
                payload_offset, payload_bytes = struct.unpack_from("<QQ", terminal_bytes, 32)
                self.assertGreater(payload_bytes, 0)
                if terminal_row_count is None:
                    terminal_row_count = payload_bytes // 4
                self.assertEqual(payload_bytes // 4, terminal_row_count)
                terminal_values = np.frombuffer(terminal_bytes, dtype="<u4", count=payload_bytes//4, offset=payload_offset)
                self.assertTrue(np.all(terminal_values == 4000000000))
                board = mover_runtime.decode_board(np.uint64(seed)).tolist()
                adjust = -sum(sum(row) for row in board)
                ex, _ = f.EXBookReader(p, False).move_on_dic(board, [(str(root/"ex"), "uint32")], "boundary_4", adjust)
                bc, _ = f.BCBookReader(p, 2, False).move_on_dic(board, [(str(root/"bc"), "uint32")], "boundary_4", adjust)
                self.assertTrue(any(isinstance(v, (int, float)) and v > 0 for v in ex.values()), ex)
                self.assertEqual(set(ex), set(bc))
                for direction in ex:
                    if isinstance(ex[direction], (int, float)) and ex[direction] > 0:
                        self.assertIsInstance(bc[direction], (int, float), (direction, dict(ex), dict(bc)))
                        self.assertAlmostEqual(ex[direction], bc[direction], delta=6e-9)
                    else:
                        self.assertIn(bc[direction], (None, 0))

    def test_bc_compressed_negative_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gen", "solve", "archive"):
                (root/name).mkdir()
            o = dict(pattern="free12", target_rank=2, success_target_rank=2,
                     extra_steps=0, prefix="free12_4_", threads=2, family_modulus=17,
                     direct_io=False, compress=True, success_check_min_source_layer_sum=4*32768,
                     generated_dir=str(root/"gen"), solved_dir=str(root/"solve"),
                     archive_dir=str(root/"archive"), resume=True)
            self.assertTrue(f.run_bc_family_build(o)["solve_completed"])
            self.assertTrue(list((root/"archive").glob("free12_4_-11.bc*")))
            reader = f.BCBookReader(pattern(False), 2, False)
            result, _ = reader.move_on_dic(mover_runtime.decode_board(np.uint64(0x1ffff)).tolist(),
                                          [(str(root/"archive"),"uint32")], "free12_4", -(4*32768+22))
            self.assertTrue(any(isinstance(v,(float,int)) and v >= 0 for v in result.values()), result)
            o["skip_generation"] = True
            self.assertTrue(f.run_bc_family_build(o)["solve_completed"])

    def test_bc_fixed_only_generation_solve_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gen", "solve", "archive"):
                (root/name).mkdir()
            o = dict(pattern="free12", target_rank=2, success_target_rank=2,
                     extra_steps=0, prefix="free12_4_", threads=2, family_modulus=17,
                     direct_io=False, compress=False,
                     success_check_min_source_layer_sum=4*32768,
                     generated_dir=str(root/"gen"), solved_dir=str(root/"solve"),
                     archive_dir=str(root/"archive"), resume=True)
            result = f.run_bc_family_build(o)
            self.assertTrue(result["solve_completed"])
            negative = root/"archive"/"free12_4_-11.bcpos"
            self.assertTrue(negative.exists())
            # Whole solve includes the legacy layer 0 under its original name.
            self.assertTrue((root/"archive"/"free12_4_0.bcpos").exists())
            snapshot = {p.name:p.read_bytes() for p in (root/"archive").iterdir() if p.is_file()}
            o["skip_generation"] = True
            result = f.run_bc_family_build(o)
            self.assertTrue(result["solve_completed"])
            self.assertEqual(snapshot, {p.name:p.read_bytes() for p in (root/"archive").iterdir() if p.is_file()})
            # Model an old completed table: keep all nonnegative files and its
            # logical -1 completion marker, remove only the new negative prefix.
            positive = {p:p.read_bytes() for p in (root/"archive").iterdir()
                        if p.name.startswith("free12_4_") and not p.name.startswith("free12_4_-")}
            for d in (root/"archive", root/"solve"):
                for p in d.glob("free12_4_-*"):
                    if p.is_file():
                        p.unlink()
            checkpoint = root/"solve"/"free12_4_family_checkpoint.csv"
            checkpoint.write_text("next_ordinal,exact_future2_ordinal,exact_future4_ordinal,dtype,family_modulus\n-1,-1,-1,1,17\n")
            o["skip_generation"] = False
            result = f.run_bc_family_build(o)
            self.assertTrue(result["solve_completed"])
            for p, data in positive.items():
                self.assertEqual(p.read_bytes(), data)
            self.assertTrue(negative.exists())


if __name__ == "__main__":
    unittest.main()
