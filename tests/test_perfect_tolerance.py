import json
from types import SimpleNamespace

import numpy as np
import pytest

from engine_core import performance_evaluation as performance
from engine_core.replay_utils import (
    REPLAY_DTYPE, analyze_replay, evaluation_of_performance, replay_step_goodness_ratio,
)

DTYPES = ["uint32", "float32", "1-float32", "uint64", "float64", "1-float64"]


@pytest.mark.parametrize("dtype", DTYPES + ["1-f32", "1-f64"])
def test_dtype_policy_and_shared_verdict(dtype):
    epsilon = 1e-14 if "64" in dtype else 3e-10
    assert performance.perfect_tolerance(dtype) == epsilon
    for difference, expected in [(epsilon / 2, True), (epsilon * 2, False)]:
        best, selected = .5, .5 - difference
        assert performance.is_perfect_result(selected, best, dtype) == expected
        ratio = replay_step_goodness_ratio(selected, best, dtype)
        assert (ratio == 1) == expected
        label = evaluation_of_performance(ratio, selected, best, dtype)
        assert (label == performance.PERFORMANCE_PERFECT_LABEL) == expected
    assert performance.is_perfect_result(0, epsilon, dtype)


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


def test_legacy_replay_uses_uint32_and_one_absolute_verdict():
    record = np.zeros(1, dtype=REPLAY_DTYPE)
    record[0]["f1"] = 1 << 5
    record[0]["f2"] = 2_000_000_000
    record[0]["f3"] = 1_999_999_999
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
