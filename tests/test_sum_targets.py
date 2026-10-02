"""Sum-goal integration and independent small-board Bellman oracle."""
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if os.environ.get("SUM_GOAL_MODULE_DIR"):
    import native_core
    _dll = os.add_dll_directory("C:/Apps/mingw64/bin") if os.name == "nt" else None
    module_file = next(Path(os.environ["SUM_GOAL_MODULE_DIR"]).glob("formation_core*.pyd"))
    spec = importlib.util.spec_from_file_location("native_core.formation_core", module_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    native_core.formation_core = module

import numpy as np
from native_core import formation_core as f
from engine_core.GoalSpec import GoalSpec
from engine_core import mover_runtime
from engine_core.BookBuilder import free_layer_offset


# A 2x2 board embedded in the low two rows, separated from walls.
CELLS = (0, 1, 4, 5)
WALLS = sum(15 << (4*p) for p in range(16) if p not in CELLS)


def moves(board):
    # Independent Python 2x2 slide/merge, not the native mover being tested.
    values = [((board >> (4*p)) & 15) for p in CELLS]
    result = []
    for lines in (((0, 1), (2, 3)), ((1, 0), (3, 2)),
                  ((0, 2), (1, 3)), ((2, 0), (3, 1))):
        out = values.copy()
        for line in lines:
            tiles = [values[p] for p in line if values[p]]
            if len(tiles) == 2 and tiles[0] == tiles[1]:
                tiles = [tiles[0]+1]
            tiles += [0]*(2-len(tiles))
            for p, tile in zip(line, tiles):
                out[p] = tile
        result.append(WALLS | sum(tile << (4*p) for p, tile in zip(CELLS, out)))
    return result


@lru_cache(None)
def oracle(board, target=18):
    total = sum((1 << ((board >> (4*p)) & 15)) if (board >> (4*p)) & 15 else 0 for p in CELLS)
    if total >= target-2:
        return 1.0
    empty = [p for p in CELLS if not (board >> (4*p)) & 15]
    if not empty:
        return 0.0
    result = 0.0
    for p in empty:
        for rank, prob in ((1, .9), (2, .1)):
            spawned = board | rank << (4*p)
            result += prob*max((oracle(b, target) for b in moves(spawned) if b != spawned), default=0)
    return result/len(empty)


def build_small(root, mode, compress=False):
    p = f.PatternSpec()
    p.name = "small"
    p.symm_mode = 0
    p.success_shifts = [4*i for i in CELLS]
    o = f.RunOptions()
    o.target, o.sum_target = 4, 18
    o.steps, o.docheck_step = 10, 7
    o.pathname = str(root/"small_sum-18_")
    o.num_threads, o.is_variant = 2, True
    o.direct_io, o.compress = False, compress
    build = f.run_pattern_build if mode == "classic" else f.run_pattern_build_zmask
    build(np.array([WALLS], dtype=np.uint64), p, o)
    return p, o, build


class SumGoals(unittest.TestCase):
    def test_goal_contract_and_legacy_layer_alignment(self):
        g = GoalSpec.parse("sum-1792")
        self.assertEqual(g.encoding_rank, 10)
        self.assertEqual(g.build_range(7*32768, 60), (897, 894))
        steps, check = g.build_range(4*32768+22, 36, start_offset=free_layer_offset("free12"))
        self.assertEqual(4*32768+22+2*(steps-2), 4*32768+1790)
        self.assertEqual(check, steps-3)
        self.assertTrue(g.reached((15 << 60) | (14 << 56) | (10 << 8) | (9 << 4) | 8))
        for value in ("sum-0", "sum-3", "sum-16384", "sum-1793"):
            with self.assertRaises(ValueError):
                GoalSpec.parse(value)
        for algorithm in ("ad", "exad"):
            with self.assertRaises(ValueError):
                g.validate_algorithm(algorithm)

    def test_classic_against_independent_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, o, build = build_small(Path(tmp), "classic")
            nontrivial = 0
            for path in Path(tmp).glob("*.book"):
                for board, raw in struct.iter_unpack("<QI", path.read_bytes()):
                    expected = oracle(board)
                    nontrivial += 0 < expected < 1
                    self.assertAlmostEqual(raw/4e9, expected, delta=1e-8)
            self.assertGreater(nontrivial, 0)
            snapshot = {p.name:p.read_bytes() for p in Path(tmp).glob("*.book")}
            build(np.array([WALLS], dtype=np.uint64), p, o)
            self.assertEqual(snapshot, {p.name:p.read_bytes() for p in Path(tmp).glob("*.book")})

    def test_ex_readers_raw_compressed_and_dtype(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as tmp:
                p, o, build = build_small(Path(tmp), "ex", compressed)
                reader = f.EXBookReader(p, True)
                for board in (WALLS | 1, WALLS | 2 | (1 << 4), WALLS | 3 | (3 << 4)):
                    result, _ = reader.move_on_dic(mover_runtime.decode_board(np.uint64(board)).tolist(),
                        [(tmp,"uint32")], "small_sum-18", -12*32768)
                    expected = sorted(oracle(b) for b in moves(board) if b != board)
                    actual = sorted(v for v in result.values() if isinstance(v,(float,int)))
                    self.assertEqual(len(actual), len(expected), result)
                    for a, b in zip(actual, expected):
                        self.assertAlmostEqual(a, b, delta=1e-7)
                build(np.array([WALLS], dtype=np.uint64), p, o)

    def test_bc_goal_runs_all_routes_and_compression(self):
        for route in ("resident", "single", "family"):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ("gen", "solve", "archive"):
                    (root/name).mkdir()
                opts = dict(pattern="free12", target_rank=3, sum_target=8, success_target_rank=0,
                    prefix="free12_sum-8_", threads=2, family_modulus=17,
                    direct_io=False, compress=True, solve_route=route,
                    generated_dir=str(root/"gen"), solved_dir=str(root/"solve"),
                    archive_dir=str(root/"archive"), resume=True)
                self.assertTrue(f.run_bc_family_build(opts)["solve_completed"])
                p = f.PatternSpec(); p.name="free12"; p.symm_mode=1
                p.success_shifts=list(range(0,64,4))
                reader = f.BCBookReader(p, 3, False)
                for board in (0x1ffff, 0x2ffff, 0x21ffff):
                    result, _ = reader.move_on_dic(mover_runtime.decode_board(np.uint64(board)).tolist(),
                        [(str(root/"archive"),"uint32")], "free12_sum-8", -(4*32768+22))
                    self.assertTrue(any(v == 1 for v in result.values()), result)
                opts["skip_generation"] = True
                self.assertTrue(f.run_bc_family_build(opts)["solve_completed"])

    def test_native_rejects_ad_and_terminal_dtype_and_legal_moves(self):
        for name in ("run_pattern_build_ad", "run_pattern_build_exad", "run_pattern_solve_exad"):
            o = f.RunOptions(); o.sum_target = 18
            with self.assertRaises(ValueError):
                getattr(f, name)(np.array([WALLS], dtype=np.uint64), f.AdvancedPatternSpec(), o)
        p = f.PatternSpec(); p.name = "small"; p.symm_mode = 0
        board = WALLS | 3 | (3 << 4)
        for reader in (f.ClassicBookReader(p, True), f.EXBookReader(p, True)):
            with tempfile.TemporaryDirectory() as tmp:
                for dtype in ("uint32", "uint64", "float32", "float64", "1-float32", "1-float64"):
                    result, actual_dtype = reader.move_on_dic(mover_runtime.decode_board(np.uint64(board)).tolist(),
                        [(tmp,dtype)], "small_sum-18", -12*32768)
                    values = [v for v in result.values() if isinstance(v,(float,int))]
                    self.assertEqual(len(values), sum(b != board for b in moves(board)))
                    self.assertTrue(all(v == (0 if dtype.startswith("1-") else 1) for v in values))
                    self.assertEqual(actual_dtype, dtype)
                dead = WALLS | 2 | (3 << 4) | (3 << 16) | (2 << 20)
                result, _ = reader.move_on_dic(mover_runtime.decode_board(np.uint64(dead)).tolist(),
                    [(tmp,"uint32")], "small_sum-18", -12*32768)
                self.assertFalse(any(isinstance(v,(float,int)) for v in result.values()))

    def test_public_builder_and_metadata(self):
        from unittest.mock import patch
        from engine_core import BookBuilder as builder
        from Config import SingletonConfig
        for mode in ("classic", "ex"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                config = dict(SingletonConfig().config, algorithm_mode=mode, advanced_algo=False,
                    zmask_algo=mode == "ex", direct_io=False, compress=False,
                    compress_temp_files=False, optimal_branch_only=False, deletion_threshold=0,
                    deletion_threshold_mode="off")
                prefix = str(Path(tmp)/"3x3_sum-8_")
                with patch.object(SingletonConfig(), "config", config), patch.object(builder,
                    "write_runtime_deletion_threshold_signal"), patch.object(builder,
                    "RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH", str(Path(tmp)/"threshold")):
                    builder.v_start_build("3x3", "sum-8", prefix)
                    self.assertTrue(Path(prefix+"goal.json").exists())
                    self.assertEqual(builder.estimate_build_progress("3x3", "sum-8", prefix), (10,10))
                    builder.v_start_build("3x3", "sum-8", prefix)
                with self.assertRaises(ValueError):
                    GoalSpec.parse("sum-10").save_metadata(prefix, 7*32768, 60, 0, mode)

    def test_backend_goal_parsing_and_board_preservation(self):
        from backend.analysis import normalize_target_value
        from backend.trainer_helpers import replace_board_for_lookup
        self.assertEqual(normalize_target_value("sum-1792"), ("sum-1792",)*3)
        board = np.uint64(0xabcffff)
        self.assertEqual(replace_board_for_lookup(board, "free12", 4, "sum-1792", False), board)

    def test_completed_trainer_does_not_request_another_move(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from backend.handlers.trainer import handle_trainer_action
        from backend.actions import Action, Message
        session = SimpleNamespace(current_pattern="free12_sum-8", board_encoded=0x21ffff,
            sum_goal_completed_at=("free12_sum-8", 0x21ffff), trainer_results={"left": 1.0})
        socket = SimpleNamespace(send_json=AsyncMock())
        asyncio.run(handle_trainer_action(Action.TRAINER_STEP, {}, session, socket, None))
        socket.send_json.assert_awaited_once_with({"action": Message.TRAINER_STEP_FAILED, "data": {}})

    def test_trainer_complement_dtype_can_play_certain_move(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from backend.handlers.trainer import handle_trainer_action
        from backend.actions import Action, Message
        for value, expected in ((0.0, Message.DO_AI_MOVE_CMD), (-1.0, Message.TRAINER_STEP_FAILED)):
            session = SimpleNamespace(current_pattern="free12_sum-8", board_encoded=0x21ffff,
                trainer_results={"left": value}, success_rate_dtype="1-float64")
            socket = SimpleNamespace(send_json=AsyncMock())
            asyncio.run(handle_trainer_action(Action.TRAINER_STEP, {}, session, socket, None))
            self.assertEqual(socket.send_json.call_args.args[0]["action"], expected)

    def test_sum_lookup_tester_analysis_and_replay_workflow(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import patch, AsyncMock
        from Config import SingletonConfig, pattern_catalog
        from engine_core import BookBuilder as builder, VBoardMover as mover
        from engine_core.BookReader import BookReaderDispatcher
        from engine_core.replay_utils import load_replay_file_with_terminal_board
        from backend.session import GameSession
        from backend.actions import Action
        from backend.tester import _tester_start_practice, _tester_compute_results
        from backend.handlers import tester as handler
        from backend.analysis_core import Analyzer, ReplayDecoder
        from backend.replay import _replay_pattern_from_path

        def spawn_two(board, _rate=.1):
            empty = [p for p in range(16) if (int(board) >> (4*p)) & 15 == 0]
            p = empty[0]
            return np.uint64(int(board) | (1 << (4*p))), len(empty)-1, 15-p, 1

        for mode in ("classic", "ex"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                full = "3x3_sum-8"
                cfg = dict(SingletonConfig().config, algorithm_mode=mode, advanced_algo=False,
                    zmask_algo=mode == "ex", direct_io=False, compress=False, optimal_branch_only=False,
                    compress_temp_files=False, deletion_threshold=0, deletion_threshold_mode="off",
                    filepath_map={(full, .1): [(tmp, "uint32")]}, **{"4_spawn_rate": .1})
                with patch.object(SingletonConfig(), "config", cfg), patch.object(builder,
                    "write_runtime_deletion_threshold_signal"), patch.object(builder,
                    "RUNTIME_DELETION_THRESHOLD_SIGNAL_PATH", str(Path(tmp)/"threshold")):
                    builder.v_start_build("3x3", "sum-8", str(Path(tmp)/(full+"_")))
                    session = GameSession("sum_workflow_test")
                    reader = session.ensure_book_reader()
                    reader.dispatch([(tmp,"uint32")], "3x3", "sum-8")
                    session.tester_pattern = ["3x3", "sum-8"]
                    session.tester_full_pattern = full
                    session.tester_table_found = session.use_variant = True
                    seed = int(pattern_catalog["3x3"]["seed_boards"][0])
                    initial = spawn_two(seed)[0]
                    _tester_start_practice(session, initial, "Test")
                    other_tab = GameSession("trainer_other_goal")
                    other_tab.ensure_book_reader().dispatch([], "free12", "2048")
                    _tester_compute_results(session)
                    self.assertEqual(session.tester_results[session.tester_best_move], 1.0)
                    rows = []
                    with patch.object(handler, "v_gen_new_num", side_effect=spawn_two), patch.object(
                        handler, "_cache_tester_replay"), patch.object(handler, "send_tester_state", new=AsyncMock()):
                        for step in range(3):
                            direction = session.tester_best_move
                            index = {"left":1,"right":2,"up":3,"down":4}[direction]
                            board = session.board_encoded
                            moved, _ = mover.s_move_board(board, index)
                            _, _, pos, rank = spawn_two(moved)
                            rows.append((board, 0, index, rank, pos))
                            asyncio.run(handler.handle_tester_action(Action.TESTER_MOVE,
                                {"dir":direction}, session, SimpleNamespace()))
                            self.assertEqual(session.tester_ready, step < 2)
                            self.assertFalse(any("target tile" in line for line in session.tester_logs))
                        self.assertEqual(session.tester_status, "Board sum goal reached.")
                        asyncio.run(handler.handle_tester_action(Action.TESTER_MOVE,
                            {"dir":"left"}, session, SimpleNamespace()))
                        self.assertEqual(session.tester_step_count, 3)
                    # Extra input steps must not be analyzed after the actual successful move.
                    def decoded(decoder):
                        decoder.record_list = rows + rows[-1:]
                        decoder.variant = "3x3"
                    with patch.object(ReplayDecoder, "decode", decoded):
                        analyzer = Analyzer(str(Path(tmp)/"input.txt"), "3x3", "sum-8", full, tmp)
                        analyzer.generate_reports()
                    report = next(Path(tmp).glob(full+"_input_*.txt"))
                    self.assertIn("in 3 moves", report.read_text(encoding="utf-8"))
                    replay_path = next(Path(tmp).glob("*.rpl"))
                    self.assertIn("_3_", replay_path.name)
                    self.assertEqual(_replay_pattern_from_path(str(replay_path)), full)
                    record, terminal = load_replay_file_with_terminal_board(str(replay_path))
                    self.assertEqual(len(record), 3)
                    self.assertEqual(int(terminal), int(session.board_encoded))


if __name__ == "__main__":
    unittest.main()
