"""Tester-compatible Battle goodness-of-fit scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Mapping, Sequence

from .route_codec import DIRECTION_NAMES, RATE_SCALE, RouteCodecError


ABSOLUTE_DIFFERENCE_TOLERANCE = 3e-10


@dataclass(frozen=True, slots=True)
class ScoreUpdate:
    selected_direction: str
    best_direction: str
    selected_rate: float
    best_rate: float
    absolute_difference: float
    step_ratio: float
    goodness_of_fit: float
    is_best: bool

    @property
    def goodness_drop(self) -> float:
        return 1.0 - self.step_ratio


def _normalized_rate(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite number")
    try:
        rate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return rate


def _current_goodness(value: object) -> float:
    goodness = _normalized_rate(value, "current_goodness")
    return goodness


def calculate_step_ratio(
    selected_rate: float,
    best_rate: float,
) -> float:
    """Apply Tester's exact per-step ``selected / best`` semantics."""

    selected = _normalized_rate(selected_rate, "selected_rate")
    best = _normalized_rate(best_rate, "best_rate")
    difference = abs(best - selected)
    if difference <= ABSOLUTE_DIFFERENCE_TOLERANCE:
        return 1.0
    if selected > best:
        raise ValueError("selected_rate cannot exceed best_rate")
    return selected / best if best > 0.0 else 1.0


def accumulate_goodness(
    current_goodness: float,
    selected_rate: float,
    best_rate: float,
) -> float:
    """Multiply cumulative GOF directly; logarithmic scoring is not used."""

    current = _current_goodness(current_goodness)
    return current * calculate_step_ratio(
        selected_rate,
        best_rate,
    )


def _ordered_rates(
    rates: Mapping[str, float] | Sequence[float],
) -> tuple[float, float, float, float]:
    if isinstance(rates, Mapping):
        missing = [direction for direction in DIRECTION_NAMES if direction not in rates]
        if missing:
            raise ValueError(f"missing direction rates: {', '.join(missing)}")
        raw = tuple(rates[direction] for direction in DIRECTION_NAMES)
    else:
        if isinstance(rates, (str, bytes, bytearray, memoryview)) or len(rates) != 4:
            raise ValueError("rates must contain four direction values")
        raw = tuple(rates)
    return tuple(
        _normalized_rate(rate, f"rates[{DIRECTION_NAMES[index]}]")
        for index, rate in enumerate(raw)
    )  # type: ignore[return-value]


def score_choice(
    rates: Mapping[str, float] | Sequence[float],
    selected_direction: str,
    *,
    current_goodness: float = 1.0,
) -> ScoreUpdate:
    """Score one selected direction against a four-direction rate record."""

    direction = str(selected_direction).lower()
    if direction not in DIRECTION_NAMES:
        raise ValueError(f"unknown direction: {selected_direction}")

    ordered = _ordered_rates(rates)
    selected_index = DIRECTION_NAMES.index(direction)
    best_rate = max(ordered)
    best_index = ordered.index(best_rate)
    selected_rate = ordered[selected_index]
    difference = abs(best_rate - selected_rate)
    ratio = calculate_step_ratio(
        selected_rate,
        best_rate,
    )
    goodness = _current_goodness(current_goodness) * ratio
    return ScoreUpdate(
        selected_direction=direction,
        best_direction=DIRECTION_NAMES[best_index],
        selected_rate=selected_rate,
        best_rate=best_rate,
        absolute_difference=difference,
        step_ratio=ratio,
        goodness_of_fit=goodness,
        is_best=difference <= ABSOLUTE_DIFFERENCE_TOLERANCE,
    )


def score_recorded_choice(
    rates: Sequence[int],
    selected_direction: str,
    *,
    current_goodness: float = 1.0,
) -> ScoreUpdate:
    """Score raw Trainer uint32 rates after their canonical 4e9 conversion."""

    if isinstance(rates, (str, bytes, bytearray, memoryview)) or len(rates) != 4:
        raise RouteCodecError("rates must contain exactly four uint32 values")
    normalized = []
    for index, rate in enumerate(rates):
        if isinstance(rate, bool) or not isinstance(rate, Integral):
            raise RouteCodecError(f"rates[{index}] must be an integer")
        value = int(rate)
        if not 0 <= value <= RATE_SCALE:
            raise RouteCodecError(
                f"rates[{index}] must be between 0 and {RATE_SCALE}"
            )
        normalized.append(value / RATE_SCALE)
    return score_choice(
        normalized,
        selected_direction,
        current_goodness=current_goodness,
    )


__all__ = [
    "ABSOLUTE_DIFFERENCE_TOLERANCE",
    "ScoreUpdate",
    "accumulate_goodness",
    "calculate_step_ratio",
    "score_choice",
    "score_recorded_choice",
]
