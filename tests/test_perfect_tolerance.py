import json
from types import SimpleNamespace

import numpy as np
import pytest

from engine_core import performance_evaluation as performance
from engine_core.replay_utils import (
    REPLAY_DTYPE, analyze_replay, evaluation_of_performance, replay_step_goodness_ratio,
)

DTYPES = ["uint32", "float32", "1-float32", "uint64", "float64", "1-float64"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("best", [0.5, 0.01, 0.00003, 1e-20])
def test_perfect_relative_loss_is_independent_of_goal_probability(dtype, best):
    epsilon = performance.perfect_tolerance(dtype)
    assert performance.is_perfect_result(best * (1 - epsilon / 4), best, dtype)
    assert not performance.is_perfect_result(best * (1 - epsilon * 4), best, dtype)


def test_baactiba_move_six_is_not_perfect_for_either_goal():
    # Actual float64 values from the same archived move. Previously the smaller
    # sum-goal probability swallowed the loss under the absolute epsilon.
    for selected, best in [(0.011337880073218879, 0.011337880075745934),
                           (0.000036304308949195786, 0.0000363043089572875)]:
        assert not performance.is_perfect_result(selected, best, "float64")


@pytest.mark.parametrize("dtype", DTYPES + ["1-f32", "1-f64"])
def test_dtype_policy_and_shared_verdict(dtype):
    epsilon = 1e-14 if "64" in dtype else 3e-10
    assert performance.perfect_tolerance(dtype) == epsilon
    for difference, expected in [(epsilon / 2, True), (epsilon * 2, False)]:
        best, selected = .5, .5 - difference * .5
        assert performance.is_perfect_result(selected, best, dtype) == expected
        ratio = replay_step_goodness_ratio(selected, best, dtype)
        assert (ratio == 1) == expected
        label = evaluation_of_performance(ratio, selected, best, dtype)
        assert (label == performance.PERFORMANCE_PERFECT_LABEL) == expected
    assert not performance.is_perfect_result(0, epsilon, dtype)
    assert performance.is_perfect_result(0, 0, dtype)


def test_configuration_overrides_and_invalid_values(tmp_path, monkeypatch):
    config = tmp_path / "performance.json"
    config.write_text(json.dumps({"perfect": {"tolerance": 2e-8,
        "tolerance_by_dtype": {"float64": 2e-14, "1-f64": 4e-14,
                              "float32": -1, "uint64": "nan", "uint32": "oops"}}}))
    monkeypatch.setattr(performance, "CONFIG_PATH", config)
    loaded = performance.load_performance_evaluation_config()["perfect"]
    monkeypatch.setattr(performance, "PERFORMANCE_PERFECT_TOLERANCE", loaded["tolerance"])
    monkeypatch.setattr(performance, "PERFORMANCE_PERFECT_TOLERANCES", loaded["tolerance_by_dtype"])
    assert performance.perfect_tolerance("float64") == 2e-14
    assert performance.perfect_tolerance("1-float64") == 4e-14
    assert performance.perfect_tolerance("uint64") == 1e-14
    assert performance.perfect_tolerance("uint32") == 3e-10
    assert performance.perfect_tolerance("float32") == 3e-10
    assert performance.perfect_tolerance("unknown") == 2e-8


def test_missing_configuration_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(performance, "CONFIG_PATH", tmp_path / "missing.json")
    loaded = performance.load_performance_evaluation_config()["perfect"]
    assert loaded["tolerance_by_dtype"] == performance.DEFAULT_PERFECT_TOLERANCES


@pytest.mark.parametrize("invalid", [None, "oops", float("nan"), float("inf")])
def test_invalid_success_rates_are_not_perfect(invalid):
    assert not performance.is_perfect_result(invalid, .5, "float64")
    assert not performance.is_perfect_result(.5, invalid, "float64")


def test_legacy_replay_uses_uint32_and_one_relative_verdict():
    record = np.zeros(1, dtype=REPLAY_DTYPE)
    record[0]["f1"] = 1 << 5
    record[0]["f2"] = 4_000_000_000
    record[0]["f3"] = 3_999_999_999
    analysis = analyze_replay(record)
    assert analysis["summary"]["final_gof"] == 1
    assert analysis["summary"]["max_combo"] == 1
    assert analysis["summary"]["counts"] == {performance.PERFORMANCE_PERFECT_LABEL: 1}


@pytest.mark.parametrize("dtype", DTYPES)
def test_analyzer_uses_reader_dtype_for_perfect_combo_and_gof(dtype):
    from backend.analysis_core import Analyzer
    from Config import pattern_catalog
    from engine_core import BoardMover as mover
    from engine_core.GoalSpec import GoalSpec
    from engine_core.VBoardMover import decode_board

    analyzer = Analyzer.__new__(Analyzer)
    analyzer.goal = GoalSpec.parse("sum-1790")
    analyzer.variant = "3x3"
    analyzer.pattern = "3x3"
    analyzer.full_pattern = "3x3_sum-1790"
    analyzer.bm = mover
    analyzer.record = np.zeros(4000, dtype=REPLAY_DTYPE)
    analyzer.clear_analysis()
    best, selected = .5, .5 - 1e-12
    offset = -1 if dtype.startswith("1-") else 0
    rates = {"left": best + offset, "right": selected + offset, "up": None, "down": None}
    analyzer.book_reader = SimpleNamespace(move_on_dic=lambda *args: (rates, dtype))
    board = decode_board(np.uint64(pattern_catalog["3x3"]["seed_boards"][0])).copy()
    board[0, 0] = board[0, 1] = 2
    assert analyzer._analyze_one_step(board, board, "Right", 1, 0)
    expected = "32" in dtype
    assert analyzer.combo == int(expected)
    assert analyzer.max_combo == int(expected)
    assert analyzer.performance_stats[performance.markdown_label(performance.PERFORMANCE_PERFECT_LABEL)] == int(expected)
    assert (analyzer.goodness_of_fit == 1) == expected
    if not expected:
        assert analyzer.goodness_of_fit == selected / best


@pytest.mark.parametrize("pattern,target", [("3x3", 10), ("3x3", "sum-1790"),
                                          ("2x4", 9), ("2x4", "sum-894")])
def test_small_board_goals_grade_and_record_first_four_moves(pattern, target):
    from backend.analysis_core import Analyzer
    from Config import pattern_catalog
    from engine_core import BoardMover as mover
    from engine_core.GoalSpec import GoalSpec
    from engine_core.VBoardMover import decode_board

    analyzer = Analyzer.__new__(Analyzer)
    analyzer.goal = GoalSpec.parse(target, rank=isinstance(target, int))
    analyzer.variant = analyzer.pattern = pattern
    analyzer.full_pattern = pattern + "_" + analyzer.goal.token
    analyzer.bm = mover
    analyzer.record = np.zeros(4000, dtype=REPLAY_DTYPE)
    analyzer.book_reader = SimpleNamespace(move_on_dic=lambda *args:
        ({"left": .01, "right": .009, "up": None, "down": None}, "float64"))
    analyzer.clear_analysis()
    board = decode_board(np.uint64(pattern_catalog[pattern]["seed_boards"][0])).copy()
    board[0, 0] = board[0, 1] = 2
    for _ in range(4):
        assert analyzer._analyze_one_step(board, board, "Left", 1, 0)
    assert analyzer.step_count == analyzer.rec_step_count == 4
    assert analyzer.performance_stats["**Perfect!**"] == 4
    assert analyzer.max_combo == 4


@pytest.mark.parametrize("pattern", ["3x4", "free10", "free12"])
@pytest.mark.parametrize("target", [9, "sum-1790"])
def test_other_formations_keep_opening_omission(pattern, target):
    from backend.analysis_core import Analyzer
    from engine_core.GoalSpec import GoalSpec
    analyzer = Analyzer.__new__(Analyzer)
    analyzer.pattern = pattern
    analyzer.goal = GoalSpec.parse(target, rank=isinstance(target, int))
    assert analyzer._skip_opening_moves()
