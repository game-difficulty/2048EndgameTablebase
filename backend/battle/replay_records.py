from __future__ import annotations

from numbers import Integral
import re
import struct
from typing import Mapping, Sequence


REPLAY_RECORD_BYTES = 25
REPLAY_RATE_SCALE = 4_000_000_000
MAX_REPLAY_BYTES = 500 * 1024
MAX_REPLAY_MOVES = (MAX_REPLAY_BYTES // REPLAY_RECORD_BYTES) - 1
REPLAY_DIRECTIONS = ("left", "right", "up", "down")
REPLAY_DIRECTION_BITS = {
    direction: index for index, direction in enumerate(REPLAY_DIRECTIONS)
}
REPLAY_SENTINEL = (
    88,
    (666_666_666, 233_333_333, 314_159_265, 987_654_321),
)

_RECORD_STRUCT = struct.Struct("<QB4I")
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9_-]+")


class BattleReplayError(ValueError):
    pass


def _scaled_rate(value: object, *, already_scaled: bool) -> int:
    if already_scaled:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise BattleReplayError("scaled replay rates must be integers")
        return max(0, min(REPLAY_RATE_SCALE, int(value)))
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise BattleReplayError("replay rates must be numeric") from exc
    if numeric != numeric:
        numeric = 0.0
    return max(0, min(REPLAY_RATE_SCALE, round(numeric * REPLAY_RATE_SCALE)))


def _ordered_rates(
    rates: Mapping[str, object] | Sequence[object],
    *,
    already_scaled: bool,
) -> tuple[int, int, int, int]:
    if isinstance(rates, Mapping):
        values = tuple(rates.get(direction, 0) for direction in REPLAY_DIRECTIONS)
    else:
        values = tuple(rates)
        if len(values) != 4:
            raise BattleReplayError("replay rates must contain four values")
    return tuple(
        _scaled_rate(value, already_scaled=already_scaled) for value in values
    )  # type: ignore[return-value]


def encode_replay_step(
    *,
    board: int,
    selected_direction: str,
    spawn_index: int,
    spawn_value: int,
    rates: Mapping[str, object] | Sequence[object],
    rates_already_scaled: bool = False,
) -> bytes:
    board_value = int(board)
    if not 0 <= board_value <= 0xFFFF_FFFF_FFFF_FFFF:
        raise BattleReplayError("replay board must fit in uint64")
    direction = str(selected_direction or "").lower()
    if direction not in REPLAY_DIRECTION_BITS:
        raise BattleReplayError("invalid replay direction")
    position = int(spawn_index)
    if not 0 <= position <= 15:
        raise BattleReplayError("replay spawn index must be between 0 and 15")
    value = int(spawn_value)
    if value not in (2, 4):
        raise BattleReplayError("replay spawn value must be 2 or 4")
    change = (
        (REPLAY_DIRECTION_BITS[direction] << 5)
        | (position << 1)
        | (1 if value == 4 else 0)
    )
    return _RECORD_STRUCT.pack(
        board_value,
        change,
        *_ordered_rates(rates, already_scaled=rates_already_scaled),
    )


def append_replay_step(blob: bytes | None, step: bytes) -> tuple[bytes, bool]:
    current = bytes(blob or b"")
    if len(current) % REPLAY_RECORD_BYTES:
        raise BattleReplayError("stored battle replay is corrupted")
    if len(step) != REPLAY_RECORD_BYTES:
        raise BattleReplayError("battle replay step has an invalid size")
    if len(current) // REPLAY_RECORD_BYTES >= MAX_REPLAY_MOVES:
        return current, False
    return current + step, True


def finalize_replay(blob: bytes | None, *, terminal_board: int) -> bytes:
    payload = bytes(blob or b"")
    if not payload or len(payload) % REPLAY_RECORD_BYTES:
        raise BattleReplayError("battle replay contains no valid moves")
    board_value = int(terminal_board)
    if not 0 <= board_value <= 0xFFFF_FFFF_FFFF_FFFF:
        raise BattleReplayError("terminal replay board must fit in uint64")
    change, rates = REPLAY_SENTINEL
    result = payload + _RECORD_STRUCT.pack(board_value, change, *rates)
    if len(result) > MAX_REPLAY_BYTES:
        raise BattleReplayError("battle replay exceeds the size limit")
    return result


def battle_replay_filename(
    *,
    mode_key: str,
    full_pattern: str,
    goodness_of_fit: float,
) -> str:
    mode = _SAFE_FILENAME.sub("_", str(mode_key or "battle")).strip("_-") or "battle"
    pattern = (
        _SAFE_FILENAME.sub("_", str(full_pattern or "tablebase")).strip("_-")
        or "tablebase"
    )
    goodness = max(0.0, min(1.0, float(goodness_of_fit or 0.0)))
    return f"battle_{mode}_{pattern}_{goodness:.4f}.rpl"


__all__ = [
    "BattleReplayError",
    "MAX_REPLAY_BYTES",
    "MAX_REPLAY_MOVES",
    "REPLAY_DIRECTIONS",
    "REPLAY_RECORD_BYTES",
    "append_replay_step",
    "battle_replay_filename",
    "encode_replay_step",
    "finalize_replay",
]
