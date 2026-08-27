from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import uuid

from .catalog import MINIGAMES


MGO1_PREFIX = "MINIGAME_v1MGO_B64_"
MGO1_MAGIC = b"MGO1"
MGO1_FORMAT_VERSION = 1
MAX_MGO1_BYTES = 256 * 1024
MAX_MGO1_ACTIONS = 50_000
MAX_SAFE_INTEGER = (1 << 53) - 1

_GAME_BY_CODE = {
    index: game_id
    for index, (game_id, _title) in enumerate(MINIGAMES, start=1)
}
_MOVE_TAGS = {0x00, 0x01, 0x02, 0x03}
_PAYLOAD_LENGTHS = {
    0x10: 1,  # bomb
    0x11: 2,  # glove
    0x12: 1,  # twist
    0x20: 2,  # custom
    0x30: 0,  # tick
    0x70: 8,  # digest
    0x7F: 1,  # end
}


@dataclass(frozen=True)
class Mgo1Envelope:
    decoded_size: int
    rules_version: int
    game_id: str
    difficulty: int
    flags: int
    run_id: str
    seed_hex: str
    action_count: int
    elapsed_ms: int


def _read_uleb128(data: bytes, offset: int, limit: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for index in range(8):
        if offset >= limit:
            raise ValueError("truncated_record")
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if index > 0 and byte & 0x7F == 0:
                raise ValueError("non_canonical_record")
            if value > MAX_SAFE_INTEGER:
                raise ValueError("record_integer_too_large")
            return value, offset
        shift += 7
    raise ValueError("record_integer_too_long")


def parse_mgo1_envelope(record_encoding: str) -> Mgo1Envelope:
    text = str(record_encoding or "").strip()
    if not text.startswith(MGO1_PREFIX):
        raise ValueError("invalid_record")
    encoded = text[len(MGO1_PREFIX):]
    if not encoded or len(encoded) % 4:
        raise ValueError("invalid_record")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid_record") from exc
    if base64.b64encode(data).decode("ascii") != encoded:
        raise ValueError("non_canonical_record")
    if len(data) > MAX_MGO1_BYTES:
        raise ValueError("record_too_large")
    if len(data) < 45:
        raise ValueError("record_too_short")

    content_limit = len(data) - 4
    expected_crc = int.from_bytes(data[content_limit:], "little")
    if binascii.crc32(data[:content_limit]) & 0xFFFFFFFF != expected_crc:
        raise ValueError("record_crc_mismatch")
    if data[:4] != MGO1_MAGIC or data[4] != MGO1_FORMAT_VERSION:
        raise ValueError("unsupported_record_format")

    offset = 5
    rules_version, offset = _read_uleb128(data, offset, content_limit)
    if offset + 35 > content_limit:
        raise ValueError("truncated_record_header")
    game_code = data[offset]
    game_id = _GAME_BY_CODE.get(game_code)
    if game_id is None:
        raise ValueError("unknown_record_game")
    difficulty = data[offset + 1]
    flags = data[offset + 2]
    if difficulty not in (0, 1) or flags != 0:
        raise ValueError("unsupported_record_flags")
    offset += 3
    run_id = str(uuid.UUID(bytes=data[offset:offset + 16]))
    offset += 16
    seed_hex = data[offset:offset + 16].hex()
    offset += 16

    total_actions = 0
    mutable_actions = 0
    elapsed_ms = 0
    ended = False
    while offset < content_limit:
        if total_actions >= MAX_MGO1_ACTIONS or ended:
            raise ValueError("invalid_record_actions")
        tag = data[offset]
        offset += 1
        delta_ms, offset = _read_uleb128(data, offset, content_limit)
        elapsed_ms += delta_ms
        if elapsed_ms > MAX_SAFE_INTEGER:
            raise ValueError("record_elapsed_too_large")
        if tag in _MOVE_TAGS:
            payload_length = 0
        elif tag in _PAYLOAD_LENGTHS:
            payload_length = _PAYLOAD_LENGTHS[tag]
        else:
            raise ValueError("unknown_record_action")
        if offset + payload_length > content_limit:
            raise ValueError("truncated_record_action")
        offset += payload_length
        total_actions += 1
        if tag == 0x7F:
            ended = True
        elif tag != 0x70:
            mutable_actions += 1

    if not ended:
        raise ValueError("record_missing_end")
    return Mgo1Envelope(
        decoded_size=len(data),
        rules_version=rules_version,
        game_id=game_id,
        difficulty=difficulty,
        flags=flags,
        run_id=run_id,
        seed_hex=seed_hex,
        action_count=mutable_actions,
        elapsed_ms=elapsed_ms,
    )


__all__ = ["Mgo1Envelope", "parse_mgo1_envelope"]
