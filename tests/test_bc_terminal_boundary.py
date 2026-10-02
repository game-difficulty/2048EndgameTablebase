"""BC terminal spawn-4 regression across all generation and solve routes."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from native_core import formation_core as f
from engine_core import mover_runtime


class BCTerminalBoundaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
