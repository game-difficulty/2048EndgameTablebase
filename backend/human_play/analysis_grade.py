"""Private grading rules for eligible 4x4 analysis result cards."""
from __future__ import annotations

import math


GRADE_VERSION = 2

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
