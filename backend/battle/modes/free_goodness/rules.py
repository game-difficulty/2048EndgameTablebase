from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from typing import Iterable, Mapping


MOVE_RISK_LIMIT = 1.15
SPAWN_RISK_LIMIT = 1.15
SPAWN_DRAWDOWN_LIMIT = 1.20
RISK_EPSILON = 1e-12
TIE_ORDER = ("left", "right", "down", "up")
DIRECTIONS = ("up", "down", "left", "right")


def clamp_probability(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


def best_direction(
    results: Mapping[str, object],
    legal_directions: Iterable[str],
) -> tuple[str | None, float]:
    legal = set(legal_directions)
    winner = None
    winner_rate = -1.0
    for direction in TIE_ORDER:
        if direction not in legal:
            continue
        rate = clamp_probability(results.get(direction))
        if rate > winner_rate:
            winner = direction
            winner_rate = rate
    return winner, max(0.0, winner_rate)


def death_risk(success: object) -> float:
    return max(0.0, 1.0 - clamp_probability(success))


def risk_multiplier(before_success: object, after_success: object) -> float:
    before = death_risk(before_success)
    after = death_risk(after_success)
    if before <= RISK_EPSILON:
        return 1.0 if after <= RISK_EPSILON else math.inf
    return after / before


def step_goodness(best_success: object, selected_success: object) -> float:
    best = clamp_probability(best_success)
    selected = clamp_probability(selected_success)
    if best <= RISK_EPSILON:
        return 0.0
    return max(0.0, min(1.0, selected / best))


@dataclass(frozen=True, slots=True)
class MoveDecision:
    selected_direction: str
    executed_direction: str
    best_direction: str
    best_success: float
    selected_success: float
    goodness: float
    risk_multiplier: float
    corrected: bool
    correction_reason: str | None


def decide_move(
    *,
    selected_direction: str,
    results: Mapping[str, object],
    legal_directions: Iterable[str],
    move_risk_limit: float = MOVE_RISK_LIMIT,
) -> MoveDecision | None:
    selected = str(selected_direction or "").lower()
    legal = set(legal_directions)
    if selected not in legal:
        return None
    optimal, best_success = best_direction(results, legal)
    if optimal is None or best_success <= RISK_EPSILON:
        return None
    selected_success = clamp_probability(results.get(selected))
    goodness = step_goodness(best_success, selected_success)
    multiplier = risk_multiplier(best_success, selected_success)
    reason = None
    if selected_success <= RISK_EPSILON:
        reason = "zero_success"
    elif multiplier > float(move_risk_limit) + RISK_EPSILON:
        reason = "risk_limit"
    return MoveDecision(
        selected_direction=selected,
        executed_direction=optimal if reason else selected,
        best_direction=optimal,
        best_success=best_success,
        selected_success=selected_success,
        goodness=goodness,
        risk_multiplier=multiplier,
        corrected=reason is not None,
        correction_reason=reason,
    )


@dataclass(frozen=True, slots=True)
class SpawnRiskState:
    log_index: float = 0.0
    log_floor: float = 0.0

    @property
    def drawdown(self) -> float:
        return math.exp(min(700.0, self.log_index - self.log_floor))

    def apply(self, multiplier: float) -> "SpawnRiskState":
        if multiplier <= 0 or not math.isfinite(multiplier):
            next_index = math.inf
        else:
            next_index = self.log_index + math.log(multiplier)
        return SpawnRiskState(next_index, min(self.log_floor, next_index))


def evaluate_spawn(
    *,
    executed_success: object,
    next_success: object,
    risk_state: SpawnRiskState,
    spawn_risk_limit: float = SPAWN_RISK_LIMIT,
    drawdown_limit: float = SPAWN_DRAWDOWN_LIMIT,
) -> tuple[bool, float, SpawnRiskState]:
    multiplier = risk_multiplier(executed_success, next_success)
    next_state = risk_state.apply(multiplier)
    accepted = (
        multiplier <= float(spawn_risk_limit) + RISK_EPSILON
        and next_state.drawdown <= float(drawdown_limit) + RISK_EPSILON
    )
    return accepted, multiplier, next_state


def deterministic_ticket(seed_hex: str, step_index: int, attempt_index: int) -> bytes:
    seed = bytes.fromhex(str(seed_hex))
    payload = (
        int(step_index).to_bytes(8, "big", signed=False)
        + int(attempt_index).to_bytes(8, "big", signed=False)
    )
    return hmac.new(seed, payload, hashlib.sha256).digest()


def deterministic_spawn_choice(
    seed_hex: str,
    step_index: int,
    attempt_index: int,
    empty_indices: Iterable[int],
    *,
    spawn_rate: float,
) -> tuple[int, int]:
    empty = tuple(sorted(int(index) for index in empty_indices))
    if not empty:
        raise ValueError("no_spawn_cell")
    ticket = deterministic_ticket(seed_hex, step_index, attempt_index)
    position_roll = int.from_bytes(ticket[:8], "big")
    value_roll = int.from_bytes(ticket[8:16], "big") / float(1 << 64)
    return empty[position_roll % len(empty)], 4 if value_roll < float(spawn_rate) else 2
