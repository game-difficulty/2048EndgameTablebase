"""Compact Battle route codec compatible with Trainer ``.rec`` records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
import struct
from typing import Iterable, Sequence


RECORD_SIZE = 17
MAX_ROUTE_RECORDS = 10_000
MAX_ROUTE_STEPS = MAX_ROUTE_RECORDS - 1
MAX_ROUTE_BYTES = MAX_ROUTE_RECORDS * RECORD_SIZE
RATE_SCALE = 4_000_000_000

DIRECTION_NAMES = ("up", "down", "left", "right")
DIRECTION_CODES = {name: code for code, name in enumerate(DIRECTION_NAMES)}

_RECORD_STRUCT = struct.Struct("<B4I")
_RESERVED_CHANGES_MASK = 0b1000_0000


class RouteCodecError(ValueError):
    """Raised when a Battle route is malformed or exceeds codec limits."""


def _require_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RouteCodecError(f"{field} must be an integer")
    return int(value)


def _validate_changes(changes: object) -> int:
    value = _require_int(changes, "changes")
    if not 0 <= value <= 0xFF:
        raise RouteCodecError("changes must fit in one byte")
    if value & _RESERVED_CHANGES_MASK:
        raise RouteCodecError("changes uses the reserved high bit")
    return value


def _validate_rates(rates: Sequence[int]) -> tuple[int, int, int, int]:
    if isinstance(rates, (bytes, bytearray, memoryview, str)):
        raise RouteCodecError("rates must contain exactly four uint32 values")
    try:
        valid_length = len(rates) == 4
    except TypeError as exc:
        raise RouteCodecError(
            "rates must contain exactly four uint32 values"
        ) from exc
    if not valid_length:
        raise RouteCodecError("rates must contain exactly four uint32 values")

    validated = []
    for index, rate in enumerate(rates):
        value = _require_int(rate, f"rates[{index}]")
        if not 0 <= value <= RATE_SCALE:
            raise RouteCodecError(
                f"rates[{index}] must be between 0 and {RATE_SCALE}"
            )
        validated.append(value)
    return tuple(validated)  # type: ignore[return-value]


def _validated_max_steps(max_steps: int) -> int:
    value = _require_int(max_steps, "max_steps")
    if not 0 <= value <= MAX_ROUTE_STEPS:
        raise RouteCodecError(
            f"max_steps must be between 0 and {MAX_ROUTE_STEPS}"
        )
    return value


@dataclass(frozen=True, slots=True)
class DecodedChanges:
    direction_code: int
    direction: str
    spawn_index: int
    spawn_exponent: int
    spawn_value: int


@dataclass(frozen=True, slots=True)
class RouteStep:
    changes: int
    rates: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "changes", _validate_changes(self.changes))
        object.__setattr__(self, "rates", _validate_rates(self.rates))

    @property
    def decoded_changes(self) -> DecodedChanges:
        return decode_changes(self.changes)

    @property
    def normalized_rates(self) -> tuple[float, float, float, float]:
        return tuple(rate / RATE_SCALE for rate in self.rates)


@dataclass(frozen=True, slots=True)
class BattleRoute:
    initial_board: int
    steps: tuple[RouteStep, ...]


def encode_changes(
    direction: str | int,
    spawn_index: int,
    spawn_value: int,
) -> int:
    """Encode Trainer direction/spawn metadata into its one-byte layout."""

    if isinstance(direction, str):
        try:
            direction_code = DIRECTION_CODES[direction.lower()]
        except KeyError as exc:
            raise RouteCodecError(f"unknown direction: {direction}") from exc
    else:
        direction_code = _require_int(direction, "direction")
        if not 0 <= direction_code < len(DIRECTION_NAMES):
            raise RouteCodecError("direction must be between 0 and 3")

    position = _require_int(spawn_index, "spawn_index")
    if not 0 <= position <= 15:
        raise RouteCodecError("spawn_index must be between 0 and 15")

    value = _require_int(spawn_value, "spawn_value")
    if value not in (2, 4):
        raise RouteCodecError("spawn_value must be 2 or 4")

    return direction_code | (position << 2) | ((value == 4) << 6)


def decode_changes(changes: int) -> DecodedChanges:
    """Decode a Trainer changes byte into direction and spawn metadata."""

    value = _validate_changes(changes)
    direction_code = value & 0b11
    spawn_index = (value >> 2) & 0b1111
    spawn_exponent = ((value >> 6) & 0b1) + 1
    return DecodedChanges(
        direction_code=direction_code,
        direction=DIRECTION_NAMES[direction_code],
        spawn_index=spawn_index,
        spawn_exponent=spawn_exponent,
        spawn_value=1 << spawn_exponent,
    )


def decode_direction(changes: int) -> str:
    return decode_changes(changes).direction


def decode_spawn(changes: int) -> tuple[int, int]:
    decoded = decode_changes(changes)
    return decoded.spawn_index, decoded.spawn_value


def encode_route(
    initial_board: int,
    steps: Iterable[RouteStep],
    *,
    max_steps: int = MAX_ROUTE_STEPS,
) -> bytes:
    """Encode a route using the exact 17-byte Trainer record layout."""

    board = _require_int(initial_board, "initial_board")
    if not 0 <= board <= 0xFFFF_FFFF_FFFF_FFFF:
        raise RouteCodecError("initial_board must fit in uint64")
    limit = _validated_max_steps(max_steps)

    records = [
        _RECORD_STRUCT.pack(
            0,
            board & 0xFFFF,
            (board >> 16) & 0xFFFF,
            (board >> 32) & 0xFFFF,
            (board >> 48) & 0xFFFF,
        )
    ]
    for index, step in enumerate(steps):
        if index >= limit:
            raise RouteCodecError(f"route exceeds the {limit}-step limit")
        if not isinstance(step, RouteStep):
            raise RouteCodecError("steps must contain RouteStep instances")
        records.append(_RECORD_STRUCT.pack(step.changes, *step.rates))
    return b"".join(records)


def decode_route(
    payload: bytes | bytearray | memoryview,
    *,
    max_steps: int = MAX_ROUTE_STEPS,
) -> BattleRoute:
    """Decode and strictly validate a compact Trainer-compatible route."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise RouteCodecError("payload must be bytes-like")
    raw = bytes(payload)
    limit = _validated_max_steps(max_steps)

    if len(raw) < RECORD_SIZE:
        raise RouteCodecError("route is missing its initial-board record")
    if len(raw) % RECORD_SIZE:
        raise RouteCodecError("route length must be a multiple of 17 bytes")
    if len(raw) > MAX_ROUTE_BYTES:
        raise RouteCodecError(
            f"route exceeds the {MAX_ROUTE_BYTES}-byte codec limit"
        )

    record_count = len(raw) // RECORD_SIZE
    step_count = record_count - 1
    if step_count > limit:
        raise RouteCodecError(f"route exceeds the {limit}-step limit")

    iterator = _RECORD_STRUCT.iter_unpack(raw)
    header_changes, *board_parts = next(iterator)
    if header_changes != 0:
        raise RouteCodecError("initial-board record changes byte must be zero")
    if any(part > 0xFFFF for part in board_parts):
        raise RouteCodecError("initial-board record contains non-canonical chunks")

    initial_board = sum(part << (16 * index) for index, part in enumerate(board_parts))
    steps = tuple(RouteStep(changes, tuple(rates)) for changes, *rates in iterator)
    return BattleRoute(initial_board=initial_board, steps=steps)


def route_sha256(payload: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 hex digest of the encoded route bytes."""

    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise RouteCodecError("payload must be bytes-like")
    return hashlib.sha256(bytes(payload)).hexdigest()


__all__ = [
    "BattleRoute",
    "DecodedChanges",
    "DIRECTION_CODES",
    "DIRECTION_NAMES",
    "MAX_ROUTE_BYTES",
    "MAX_ROUTE_RECORDS",
    "MAX_ROUTE_STEPS",
    "RATE_SCALE",
    "RECORD_SIZE",
    "RouteCodecError",
    "RouteStep",
    "decode_changes",
    "decode_direction",
    "decode_route",
    "decode_spawn",
    "encode_changes",
    "encode_route",
    "route_sha256",
]
