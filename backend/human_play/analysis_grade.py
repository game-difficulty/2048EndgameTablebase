"""Private, variant-specific grading rules for analysis result cards."""
from __future__ import annotations

import math


GRADE_VERSION = 4
THREE_BY_THREE_PROFILE = "3x3-v1"
COMBO_3X3 = (13, 17, 27, 90, 158, 186, 203, 225, 290)
PERFECT_3X3 = (.63, .67, .73, .85, .88, .895, .91, .92, .93)
ACCURACY_3X3 = (.9940000, .9979000, .9992000, .9999300, .9999840,
                .9999925, .9999957, .9999976, .9999990)
GRADES_3X3 = ((20, "X"), (18, "SSS"), (15, "SS"), (12, "S"), (9, "A"),
              (7, "B"), (5, "C"), (3, "D"), (1, "E"))


def points_3x3(value: float, thresholds: tuple) -> float:
    if value < thresholds[0]:
        return 0.0
    return _interpolate(value, tuple((threshold, index + 1)
                                    for index, threshold in enumerate(thresholds)))


def time_bonus_3x3(board_sum: int, elapsed_ms: float | None) -> float:
    if board_sum >= 1536:
        return 3
    if elapsed_ms is None or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        return 0
    limits = (270, 195, 120) if board_sum < 1024 else (660, 480, 300)
    if elapsed_ms > limits[0] * 1000:
        return 0.0
    return _interpolate(elapsed_ms, tuple((seconds * 1000, points)
                                        for points, seconds in reversed(list(enumerate(limits, 1)))))


def _grade_3x3(aggregate: dict) -> tuple[float, str] | None:
    if aggregate.get("grading_profile") != THREE_BY_THREE_PROFILE:
        return None
    try:
        accuracy = float(aggregate["mean_single_step_accuracy"])
        perfect = float(aggregate["perfect_rate"])
        combo = int(aggregate["max_combo"])
        board_sum = int(aggregate["run_board_sum"])
        moves = int(aggregate["evaluated_moves"])
        elapsed = aggregate.get("run_elapsed_ms")
        elapsed = float(elapsed) if elapsed is not None else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (0 <= accuracy <= 1 and 0 <= perfect <= 1 and 0 <= combo <= moves
            and moves > 0 and board_sum > 0):
        return None
    points = (points_3x3(combo, COMBO_3X3)
              + points_3x3(perfect, PERFECT_3X3)
              + points_3x3(accuracy, ACCURACY_3X3)
              + time_bonus_3x3(board_sum, elapsed))
    return points, next((grade for threshold, grade in GRADES_3X3 if points >= threshold), "F")

# Anchor points are ordered by the observed metric, not by the awarded score.
FIT_ANCHORS = {
    16384: ((0, 0), (.80, 40), (.90, 50), (.95, 60), (.975, 70),
            (.99, 80), (.995, 90), (.999, 100)),
    32768: ((0, 0), (.60, 40), (.70, 50), (.80, 60), (.90, 70),
            (.95, 80), (.98, 90), (.99, 100)),
    65536: ((0, 0), (.20, 40), (.40, 50), (.60, 60), (.80, 70),
            (.90, 80), (.95, 90), (.98, 100)),
}
SCORE_ANCHORS = ((0, 0), (200_000, 40), (400_000, 50), (600_000, 60),
                 (700_000, 70), (800_000, 80), (830_000, 90), (850_000, 100))
SPEED_ANCHORS = ((1, 100), (2, 90), (3, 80), (5, 70), (8, 60),
                 (16, 50), (64, 40), (256, 0))
GRADE_THRESHOLDS = ((90, "SSS"), (85, "SS"), (80, "S"), (75, "A"),
                    (70, "B"), (60, "C"), (40, "D"), (20, "E"))


def _interpolate(value: float, anchors: tuple[tuple[float, int], ...]) -> float:
    if value <= anchors[0][0]:
        return float(anchors[0][1])
    for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
        if value <= right_x:
            return left_y + (value - left_x) * (right_y - left_y) / (right_x - left_x)
    return float(anchors[-1][1])


def grade_result(*, variant: str, goal_tile: int | None, score: int,
                 aggregate: dict) -> tuple[float, str] | None:
    """Return the private weighted score and its public letter grade."""
    if variant == "3x3":
        return _grade_3x3(aggregate)
    if variant != "4x4" or goal_tile not in FIT_ANCHORS or not aggregate.get("stage_count"):
        return None
    try:
        fit = float(aggregate["mean_goodness_of_fit"])
        mean_seconds = float(aggregate["mean_ms_per_timed_move"]) / 1000
        final_score = float(score)
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (not all(math.isfinite(value) for value in (fit, mean_seconds, final_score))
            or fit <= 0 or mean_seconds < 0 or final_score < 0):
        return None
    weighted = (0.50 * _interpolate(fit, FIT_ANCHORS[goal_tile])
                + 0.25 * _interpolate(final_score, SCORE_ANCHORS)
                + 0.25 * _interpolate(mean_seconds, SPEED_ANCHORS))
    grade = next((grade for threshold, grade in GRADE_THRESHOLDS if weighted >= threshold), "F")
    return weighted, grade


def grade_for_summary(*, variant: str, goal_tile: int | None, score: int,
                      aggregate: dict) -> str | None:
    """Return only the public grade; the weighted value must stay server-side."""
    result = grade_result(variant=variant, goal_tile=goal_tile, score=score,
                          aggregate=aggregate)
    return result[1] if result else None
