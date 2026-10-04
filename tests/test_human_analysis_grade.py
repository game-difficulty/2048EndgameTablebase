import pytest

from backend.human_play.analysis_grade import FIT_ANCHORS, SCORE_ANCHORS, SPEED_ANCHORS, _interpolate, grade_for_summary, grade_result
from backend.human_play.analysis_summary import poster_goal_tile


@pytest.mark.parametrize("pattern,target,goal", [
    ("L3", "256", 16384),
    ("free10", "512", 32768),
    ("free12", "4096", 65536),
])
def test_formation_and_target_determine_endgame_tier(pattern, target, goal):
    assert poster_goal_tile(pattern, target, "4x4") == goal
    assert poster_goal_tile(pattern, target, "3x4") is None


@pytest.mark.parametrize("anchors", [*FIT_ANCHORS.values(), SCORE_ANCHORS, SPEED_ANCHORS])
def test_every_anchor_and_midpoint_is_linear(anchors):
    for x, expected in anchors:
        assert _interpolate(x, anchors) == pytest.approx(expected)
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
        midpoint = (left_x + right_x) / 2
        assert _interpolate(midpoint, anchors) == pytest.approx((left_y + right_y) / 2)
    assert _interpolate(anchors[0][0] - 100, anchors) == anchors[0][1]
    assert _interpolate(anchors[-1][0] + 100, anchors) == anchors[-1][1]


@pytest.mark.parametrize("fit,score,seconds,expected", [
    (.99, 850000, 1, "X"),
    (.982, 850000, 1, "X"),
    (.98199999, 850000, 1, "SSS"),
    (.98, 800000, 2, "SS"),
    (.95, 800000, 3, "S"),
    (.90, 800000, 3, "A"),
    (.90, 700000, 5, "B"),
    (.80, 600000, 8, "C"),
    (.60, 200000, 64, "D"),
    (.60, 0, 256, "E"),
    (.59, 0, 256, "F"),
])
def test_grade_boundaries(fit, score, seconds, expected):
    aggregate = {"stage_count": 1, "speed_grade_eligible": True,
                 "mean_goodness_of_fit": fit, "mean_ms_per_timed_move": seconds * 1000}
    assert grade_for_summary(variant="4x4", goal_tile=32768, score=score,
                             aggregate=aggregate) == expected


def test_tier_specific_fit_and_missing_speed():
    aggregate = {"stage_count": 1, "speed_grade_eligible": True,
                 "mean_goodness_of_fit": .95, "mean_ms_per_timed_move": 1000}
    assert grade_for_summary(variant="4x4", goal_tile=16384, score=850000,
                             aggregate=aggregate) == "S"
    assert grade_for_summary(variant="4x4", goal_tile=65536, score=850000,
                             aggregate=aggregate) == "SSS"
    assert grade_for_summary(variant="3x3", goal_tile=65536, score=850000,
                             aggregate=aggregate) is None
    assert grade_for_summary(variant="4x4", goal_tile=8192, score=850000,
                             aggregate=aggregate) is None
    aggregate["speed_grade_eligible"] = False
    assert grade_for_summary(variant="4x4", goal_tile=32768, score=850000,
                             aggregate=aggregate) == "SSS"
    aggregate["mean_ms_per_timed_move"] = None
    assert grade_for_summary(variant="4x4", goal_tile=32768, score=850000,
                             aggregate=aggregate) is None


@pytest.mark.parametrize('goal,fit', [(16384, .9958), (32768, .982), (65536, .956)])
def test_x_threshold_is_inclusive_for_all_4x4_endgame_tiers(goal, fit):
    aggregate = dict(stage_count=1, mean_goodness_of_fit=fit, mean_ms_per_timed_move=1000)
    points, grade = grade_result(variant='4x4', goal_tile=goal, score=850000, aggregate=aggregate)
    assert points == pytest.approx(96)
    assert grade == 'X'
    aggregate['mean_goodness_of_fit'] = fit - 1e-7
    points, grade = grade_result(variant='4x4', goal_tile=goal, score=850000, aggregate=aggregate)
    assert points < 96
    assert grade == 'SSS'
