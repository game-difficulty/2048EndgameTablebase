"""Convert a sealed human replay to the analyzer's compact RPL1 input."""
from __future__ import annotations

import base64
import gzip
import struct
import zlib

from . import engine


def _uleb(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 127
        value >>= 7
        result.append(byte | (128 if value else 0))
        if not value:
            return bytes(result)


def archived_to_analysis_text(archive: bytes, *, expected_run_id: str,
                              expected_variant: str, expected_seq: int) -> str:
    if len(archive) > engine.MAX_BYTES + 131072:
        raise ValueError("archive_too_large")
    raw = gzip.decompress(archive)
    if len(raw) > engine.MAX_BYTES + 65544:
        raise ValueError("archive_too_large")
    header, events = engine.parse_replay(raw)
    if (header.get("run_id") != expected_run_id or header.get("variant") != expected_variant
            or len(events) // engine.EVENT.size != expected_seq):
        raise ValueError("archive_mismatch")
    rows, cols = engine.VARIANTS[expected_variant]
    initial = engine.initial(expected_run_id, expected_variant, header["seed"])["board"]
    tiles = [(index, tile) for index, tile in enumerate(initial) if tile]
    if len(tiles) != 2 or any(tile not in (2, 4) for _, tile in tiles):
        raise ValueError("invalid_initial_board")
    payload = bytearray(b"RPL1")
    payload.extend(((rows << 4) | cols, 0, len(tiles)))
    payload.extend(index | (16 if tile == 4 else 0) for index, tile in tiles)
    for code, delta in engine.EVENT.iter_unpack(events):
        if code >= 128:
            raise ValueError("invalid_event")
        payload.append(code)
        payload.extend(_uleb(delta))
    payload.append(132)
    payload.extend(struct.pack("<I", zlib.crc32(payload)))
    return "REPLAY_v1RPL_B64_" + base64.b64encode(payload).decode("ascii")
