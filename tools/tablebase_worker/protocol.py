from __future__ import annotations

import json
import math
import numbers
import re
from dataclasses import dataclass
from typing import Any, Iterable


PROTOCOL_VERSION = 1
CAPABILITY_BATTLE_ROUTE_V1 = "battle_route_v1"
CAPABILITY_GAMER_ROUTE_V1 = "gamer_route_v1"
WORKER_CAPABILITIES = (CAPABILITY_BATTLE_ROUTE_V1, CAPABILITY_GAMER_ROUTE_V1)
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
BOARD_RE = re.compile(r"^[0-9a-fA-F]{16}$")
SEED_RE = re.compile(r"^[0-9a-fA-F]{32}$")
REQUEST_TYPES = {
    "LOOKUP",
    "LOOKUP_BATCH",
    "RANDOM_STATE",
    "GENERATE_BATTLE_ROUTE",
    "GENERATE_GAMER_ROUTE",
    "CANCEL",
}
MAX_BATTLE_ROUTE_STEPS = 9_999


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, request_id: str | None = None):
        self.code = code
        self.request_id = request_id
        super().__init__(message)


@dataclass(frozen=True)
class Request:
    message_type: str
    request_id: str
    full_pattern: str | None = None
    pattern: str | None = None
    target: str | None = None
    boards: tuple[int, ...] = ()
    use_variant: bool = False
    board_is_lookup: bool = False
    initial_board: int | None = None
    max_steps: int | None = None
    min_steps: int = 0
    spawn_rate: float = 0.1
    seed_hex: str = ""
    gamer_options: dict | None = None


def encode_message(message_type: str, **fields: Any) -> str:
    return json.dumps(
        {"type": message_type, **fields},
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def decode_message(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("INVALID_MESSAGE", "Message must be UTF-8 JSON") from exc
    if not isinstance(raw, str) or not raw or len(raw.encode("utf-8")) > 2 * 1024 * 1024:
        raise ProtocolError("INVALID_MESSAGE", "Message is empty or too large")
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError("INVALID_MESSAGE", "Message must be valid JSON") from exc
    if not isinstance(message, dict):
        raise ProtocolError("INVALID_MESSAGE", "Message must be an object")
    return message


def validate_hello_ack(raw: str | bytes, *, worker_id: str) -> dict[str, Any]:
    message = decode_message(raw)
    allowed = {
        "type",
        "protocol_version",
        "worker_id",
        "heartbeat_timeout_seconds",
        "tables",
    }
    if set(message) - allowed:
        raise ProtocolError("INVALID_MESSAGE", "HELLO_ACK contains unsupported fields")
    if str(message.get("type") or "").upper() != "HELLO_ACK":
        raise ProtocolError("HELLO_REJECTED", "Expected HELLO_ACK")
    if message.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("PROTOCOL_VERSION_MISMATCH", "Unsupported protocol version")
    if str(message.get("worker_id") or "") != worker_id:
        raise ProtocolError("HELLO_REJECTED", "HELLO_ACK worker_id does not match")
    if not isinstance(message.get("tables"), list):
        raise ProtocolError("INVALID_MESSAGE", "HELLO_ACK tables must be an array")
    return message


def _request_id(value: Any) -> str:
    request_id = str(value or "")
    if not REQUEST_ID_RE.fullmatch(request_id):
        raise ProtocolError("INVALID_REQUEST_ID", "Invalid request_id")
    return request_id


def _board(value: Any, request_id: str) -> int:
    if not isinstance(value, str) or not BOARD_RE.fullmatch(value):
        raise ProtocolError(
            "INVALID_BOARD", "board must contain exactly 16 hexadecimal digits", request_id
        )
    return int(value, 16)


def _require_bool(value: Any, field: str, request_id: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolError("INVALID_REQUEST", f"{field} must be boolean", request_id)
    return value


def _bounded_int(
    value: Any,
    field: str,
    request_id: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ProtocolError("INVALID_REQUEST", f"{field} must be an integer", request_id)
    parsed = int(value)
    if parsed < minimum or parsed > MAX_BATTLE_ROUTE_STEPS:
        raise ProtocolError(
            "INVALID_REQUEST",
            f"{field} must be between {minimum} and {MAX_BATTLE_ROUTE_STEPS}",
            request_id,
        )
    return parsed


def _spawn_rate(value: Any, request_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ProtocolError("INVALID_REQUEST", "spawn_rate must be numeric", request_id)
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ProtocolError(
            "INVALID_REQUEST", "spawn_rate must be between 0 and 1", request_id
        )
    return parsed


def validate_request(
    message: dict[str, Any],
    *,
    allowed_tables: Iterable[str],
    table_metadata: dict[str, tuple[str, str]] | None = None,
    max_batch_size: int,
) -> Request:
    message_type = str(message.get("type") or "").upper()
    if message_type not in REQUEST_TYPES:
        raise ProtocolError("UNSUPPORTED_ACTION", "Unsupported worker action")
    request_id = _request_id(message.get("request_id"))

    if message_type == "CANCEL":
        if set(message) != {"type", "request_id"}:
            raise ProtocolError(
                "INVALID_REQUEST", "CANCEL accepts only type and request_id", request_id
            )
        return Request(message_type=message_type, request_id=request_id)

    common = {"type", "request_id", "full_pattern", "pattern", "target"}
    allowed_fields = {
        "LOOKUP": common | {"board", "use_variant", "board_is_lookup"},
        "LOOKUP_BATCH": common | {"boards", "use_variant", "board_is_lookup"},
        "RANDOM_STATE": common,
        "GENERATE_GAMER_ROUTE": common | {"options"},
        "GENERATE_BATTLE_ROUTE": common
        | {
            "initial_board",
            "max_steps",
            "min_steps",
            "spawn_rate",
            "seed_hex",
        },
    }[message_type]
    if set(message) != allowed_fields:
        raise ProtocolError(
            "INVALID_REQUEST",
            f"{message_type} contains missing or unsupported fields",
            request_id,
        )

    full_pattern = str(message.get("full_pattern") or "")
    if full_pattern not in set(allowed_tables):
        raise ProtocolError("TABLE_NOT_ALLOWED", "Table is not allowlisted", request_id)
    pattern = str(message.get("pattern") or "")
    target = str(message.get("target") or "")
    expected = (table_metadata or {}).get(full_pattern)
    if expected is not None and (pattern, target) != expected:
        raise ProtocolError(
            "TABLE_METADATA_MISMATCH",
            "pattern/target do not match the local allowlist",
            request_id,
        )

    if message_type == "GENERATE_GAMER_ROUTE":
        from backend.gamer_tablebase_route import validate_options
        try:
            options = validate_options(message.get('options'))
        except ValueError as exc:
            raise ProtocolError('INVALID_REQUEST', str(exc), request_id) from exc
        return Request(message_type, request_id, full_pattern, pattern, target,
                       gamer_options=options)

    if message_type == "RANDOM_STATE":
        return Request(
            message_type=message_type,
            request_id=request_id,
            full_pattern=full_pattern,
            pattern=pattern,
            target=target,
        )

    if message_type == "GENERATE_BATTLE_ROUTE":
        raw_initial_board = message.get("initial_board")
        initial_board = (
            None if raw_initial_board is None else _board(raw_initial_board, request_id)
        )
        raw_max_steps = message.get("max_steps")
        max_steps = (
            None
            if raw_max_steps is None
            else _bounded_int(
                raw_max_steps,
                "max_steps",
                request_id,
                minimum=1,
            )
        )
        min_steps = _bounded_int(
            message.get("min_steps"),
            "min_steps",
            request_id,
            minimum=0,
        )
        if max_steps is not None and min_steps > max_steps:
            raise ProtocolError(
                "INVALID_REQUEST",
                "min_steps cannot exceed max_steps",
                request_id,
            )
        seed_hex = str(message.get("seed_hex") or "").lower()
        if not SEED_RE.fullmatch(seed_hex):
            raise ProtocolError(
                "INVALID_REQUEST",
                "seed_hex must contain exactly 32 hexadecimal digits",
                request_id,
            )
        return Request(
            message_type=message_type,
            request_id=request_id,
            full_pattern=full_pattern,
            pattern=pattern,
            target=target,
            initial_board=initial_board,
            max_steps=max_steps,
            min_steps=min_steps,
            spawn_rate=_spawn_rate(message.get("spawn_rate"), request_id),
            seed_hex=seed_hex,
        )

    use_variant = _require_bool(message.get("use_variant"), "use_variant", request_id)
    board_is_lookup = _require_bool(
        message.get("board_is_lookup"), "board_is_lookup", request_id
    )
    if message_type == "LOOKUP":
        boards = (_board(message.get("board"), request_id),)
    else:
        raw_boards = message.get("boards")
        if not isinstance(raw_boards, list) or not raw_boards:
            raise ProtocolError("INVALID_BATCH", "boards must be a non-empty array", request_id)
        if len(raw_boards) > max_batch_size:
            raise ProtocolError("BATCH_TOO_LARGE", "Batch exceeds the configured limit", request_id)
        boards = tuple(_board(value, request_id) for value in raw_boards)
    return Request(
        message_type=message_type,
        request_id=request_id,
        full_pattern=full_pattern,
        pattern=pattern,
        target=target,
        boards=boards,
        use_variant=use_variant,
        board_is_lookup=board_is_lookup,
    )


def sanitize_results(raw_results: Any) -> dict[str, str | float | int | None]:
    if not isinstance(raw_results, dict):
        return {}
    results: dict[str, str | float | int | None] = {}
    for raw_key, raw_value in raw_results.items():
        key = str(raw_key)
        if len(key) > 32:
            continue
        if raw_value is None:
            results[key] = None
        elif isinstance(raw_value, bool):
            results[key] = int(raw_value)
        elif isinstance(raw_value, numbers.Integral):
            results[key] = int(raw_value)
        elif isinstance(raw_value, numbers.Real):
            numeric = float(raw_value)
            results[key] = numeric if math.isfinite(numeric) else None
        else:
            value = str(raw_value)
            results[key] = value[:128]
    return results


def error_message(error: ProtocolError, request_id: str | None = None) -> str:
    return encode_message(
        "ERROR",
        request_id=request_id if request_id is not None else error.request_id,
        message=str(error),
        code=error.code,
    )
