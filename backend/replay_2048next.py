from __future__ import annotations

import base64
import binascii
import zlib
from dataclasses import dataclass
from typing import Union


REPLAY_PREFIX = "REPLAY_v1RPL_B64_"

FLAG_HAS_START_UNIX_MS = 1
FLAG_EXTENDED_INIT_TILES = 8

RECORD_UNDO1 = 128
RECORD_UNDON = 129
RECORD_CHECKPOINT = 130
RECORD_EXT = 131
RECORD_END = 132
RECORD_MOVE8 = 133


class Replay2048NextError(ValueError):
    pass


@dataclass(frozen=True)
class MoveRecord:
    direction: int
    spawn_index: int
    spawn_value_bit: int
    delta_ms: int


@dataclass(frozen=True)
class UndoRecord:
    count: int
    delta_ms: int


@dataclass(frozen=True)
class CheckpointRecord:
    board_codes: tuple[int, ...]


@dataclass(frozen=True)
class ExtensionRecord:
    extension_type: int
    payload: bytes


@dataclass(frozen=True)
class EndRecord:
    pass


ReplayRecord = Union[
    MoveRecord,
    UndoRecord,
    CheckpointRecord,
    ExtensionRecord,
    EndRecord,
]


@dataclass(frozen=True)
class Replay2048Next:
    width: int
    height: int
    flags: int
    initial_tiles: tuple[tuple[int, int], ...]
    start_unix_ms: int | None
    records: tuple[ReplayRecord, ...]

    def text_extension(self, extension_type: int) -> str | None:
        for record in self.records:
            if (
                isinstance(record, ExtensionRecord)
                and record.extension_type == extension_type
            ):
                try:
                    return record.payload.decode("utf-8")
                except UnicodeDecodeError:
                    return None
        return None


def is_2048next_replay(text: str) -> bool:
    return text.strip().startswith(REPLAY_PREFIX)


def _decode_uleb128(data: bytes, offset: int, limit: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < limit:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
        if shift > 56:
            raise Replay2048NextError("2048next replay ULEB128 value is too large")
    raise Replay2048NextError("Unexpected end of 2048next replay ULEB128 value")


def _decode_checkpoint(
    payload: bytes, width: int, height: int
) -> tuple[int, ...]:
    board_size = width * height
    codes = []
    bit_offset = 0
    for _ in range(board_size):
        code = 0
        for bit in range(5):
            absolute_bit = bit_offset + bit
            code |= ((payload[absolute_bit // 8] >> (absolute_bit % 8)) & 1) << bit
        codes.append(code)
        bit_offset += 5
    return tuple(codes)


def decode_2048next_replay(text: str) -> Replay2048Next:
    replay_text = text.strip()
    if not replay_text.startswith(REPLAY_PREFIX):
        raise Replay2048NextError("Missing 2048next replay prefix")

    encoded = "".join(replay_text[len(REPLAY_PREFIX) :].split())
    if not encoded:
        raise Replay2048NextError("Empty 2048next replay payload")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise Replay2048NextError("Invalid 2048next replay Base64 payload") from exc

    if len(data) < 11:
        raise Replay2048NextError("2048next replay payload is too short")
    if data[:4] != b"RPL1":
        raise Replay2048NextError("Invalid 2048next replay magic")

    dimensions = data[4]
    width = dimensions & 0x0F
    height = (dimensions >> 4) & 0x0F
    if width == 0 or height == 0:
        raise Replay2048NextError("Invalid 2048next replay board dimensions")

    flags = data[5]
    initial_tile_count = data[6]
    payload_end = len(data) - 4
    expected_crc = int.from_bytes(data[payload_end:], "little")
    computed_crc = zlib.crc32(data[:payload_end]) & 0xFFFFFFFF
    if expected_crc != computed_crc:
        raise Replay2048NextError("2048next replay CRC32 mismatch")

    offset = 7
    start_unix_ms = None
    if flags & FLAG_HAS_START_UNIX_MS:
        start_unix_ms, offset = _decode_uleb128(data, offset, payload_end)

    board_size = width * height
    initial_tiles = []
    for _ in range(initial_tile_count):
        if flags & FLAG_EXTENDED_INIT_TILES:
            packed, offset = _decode_uleb128(data, offset, payload_end)
            cell_index = packed >> 1
            value_bit = packed & 1
        else:
            if offset >= payload_end:
                raise Replay2048NextError(
                    "Unexpected end of 2048next replay initial tiles"
                )
            packed = data[offset]
            offset += 1
            cell_index = packed & 0x0F
            value_bit = (packed >> 4) & 1
        if cell_index >= board_size:
            raise Replay2048NextError(
                "Invalid 2048next replay initial tile position"
            )
        initial_tiles.append((cell_index, value_bit))

    records: list[ReplayRecord] = []
    checkpoint_size = (board_size * 5 + 7) // 8
    while offset < payload_end:
        record_type = data[offset]
        if record_type < 128:
            offset += 1
            delta_ms, offset = _decode_uleb128(data, offset, payload_end)
            direction = record_type & 0x03
            spawn_index = (record_type >> 2) & 0x0F
            spawn_value_bit = (record_type >> 6) & 1
            if spawn_index >= board_size:
                raise Replay2048NextError(
                    "Invalid 2048next replay move spawn position"
                )
            records.append(
                MoveRecord(direction, spawn_index, spawn_value_bit, delta_ms)
            )
            continue

        offset += 1
        if record_type == RECORD_UNDO1:
            delta_ms, offset = _decode_uleb128(data, offset, payload_end)
            records.append(UndoRecord(1, delta_ms))
        elif record_type == RECORD_UNDON:
            count, offset = _decode_uleb128(data, offset, payload_end)
            delta_ms, offset = _decode_uleb128(data, offset, payload_end)
            if count < 1:
                raise Replay2048NextError("Invalid 2048next replay undo count")
            records.append(UndoRecord(count, delta_ms))
        elif record_type == RECORD_CHECKPOINT:
            checkpoint_end = offset + checkpoint_size
            if checkpoint_end > payload_end:
                raise Replay2048NextError(
                    "Unexpected end of 2048next replay checkpoint"
                )
            records.append(
                CheckpointRecord(
                    _decode_checkpoint(data[offset:checkpoint_end], width, height)
                )
            )
            offset = checkpoint_end
        elif record_type == RECORD_EXT:
            extension_type, offset = _decode_uleb128(data, offset, payload_end)
            extension_length, offset = _decode_uleb128(data, offset, payload_end)
            extension_end = offset + extension_length
            if extension_end > payload_end:
                raise Replay2048NextError(
                    "Unexpected end of 2048next replay extension"
                )
            records.append(
                ExtensionRecord(extension_type, data[offset:extension_end])
            )
            offset = extension_end
        elif record_type == RECORD_END:
            records.append(EndRecord())
        elif record_type == RECORD_MOVE8:
            if offset >= payload_end:
                raise Replay2048NextError(
                    "Unexpected end of 2048next extended move direction"
                )
            direction = data[offset]
            offset += 1
            spawn_index, offset = _decode_uleb128(data, offset, payload_end)
            if offset >= payload_end:
                raise Replay2048NextError(
                    "Unexpected end of 2048next extended move value"
                )
            spawn_value_bit = data[offset]
            offset += 1
            delta_ms, offset = _decode_uleb128(data, offset, payload_end)
            if direction > 7 or spawn_index >= board_size or spawn_value_bit > 1:
                raise Replay2048NextError("Invalid 2048next extended move")
            records.append(
                MoveRecord(direction, spawn_index, spawn_value_bit, delta_ms)
            )
        else:
            raise Replay2048NextError(
                f"Unsupported 2048next replay record type: {record_type}"
            )

    return Replay2048Next(
        width=width,
        height=height,
        flags=flags,
        initial_tiles=tuple(initial_tiles),
        start_unix_ms=start_unix_ms,
        records=tuple(records),
    )
