"""Bounded validation and normalization of user-supplied Verse replays."""
from __future__ import annotations

import base64
import binascii
import gzip
import json
import re
import struct
import zlib

from . import engine, service, statistics
from .store import database
from .codec import gunzip_limited

MAX_INPUT = 2 * 1024 * 1024
ALPHABET = ("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            + "".join(chr(value) for value in range(0xC0, 0x100)) + "\xa4\xbe")
VALUES = {character: index for index, character in enumerate(ALPHABET)}
HEADER = re.compile(r"^(\d+)x(\d+)-([^_]*)_")
PREFIX = "REPLAY_v1RPL_B64_"
UNKNOWN_TIMING_MS = 0xFFFFFFFF
TIMING_BUCKETS = ((1, 255), (3, 1021), (12, 4084), (48, 16336),
                  (192, 65344), (768, 265966), (3072, 1050094), (12288, 4186606))


def _uleb(value: int) -> bytes:
    result = bytearray()
    while True:
        byte = value & 127
        value >>= 7
        result.append(byte | (128 if value else 0))
        if not value:
            return bytes(result)


def _read_uleb(raw: bytes, offset: int, end: int):
    result = 0
    for shift in range(0, 56, 7):
        if offset >= end:
            raise ValueError("replay_truncated")
        byte = raw[offset]
        offset += 1
        result |= (byte & 127) << shift
        if byte < 128:
            return result, offset
    raise ValueError("replay_integer_invalid")


def _legacy(text: str):
    text = text.lstrip("\ufeff").strip()
    header = HEADER.match(text)
    if not header:
        raise ValueError("replay_header_invalid")
    variant = f"{header.group(1)}x{header.group(2)}"
    if variant not in engine.VARIANTS:
        raise ValueError("replay_variant_invalid")
    rows, cols = engine.VARIANTS[variant]
    payload = text[header.end():]
    if len(payload) % 3 or len(payload) < 6 or len(payload) // 3 > engine.MAX_MOVES + 2:
        raise ValueError("replay_length_invalid")
    records = []
    for offset in range(0, len(payload), 3):
        try:
            a, b, c = (VALUES[character] for character in payload[offset:offset + 3])
        except KeyError as exc:
            raise ValueError("replay_character_invalid") from exc
        encoded = (a << 14) | (b << 7) | c
        spawn_code = (encoded >> 2) & 3
        x, y = (encoded >> 4) & 7, (encoded >> 7) & 7
        bucket, timing = (encoded >> 10) & 7, (encoded >> 13) & 255
        if spawn_code > 1 or x >= cols or y >= rows:
            raise ValueError("replay_spawn_invalid")
        delta = None if bucket == 7 and timing == 255 else (
            (0 if bucket == 0 else TIMING_BUCKETS[bucket - 1][1] + TIMING_BUCKETS[bucket - 1][0])
            + TIMING_BUCKETS[bucket][0] * timing)
        records.append((encoded & 3, y * cols + x, 4 if spawn_code else 2,
                        UNKNOWN_TIMING_MS if delta is None else delta))
    initial = [0] * (rows * cols)
    for _, index, value, _ in records[:2]:
        if initial[index]:
            raise ValueError("replay_initial_invalid")
        initial[index] = value
    # Verse's legacy directions are up/down/left/right; RPL1 uses up/right/down/left.
    moves = [((0, 2, 3, 1)[direction], index, value, delta)
             for direction, index, value, delta in records[2:]]
    return variant, initial, moves


def _rpl1(raw: bytes):
    if len(raw) < 11 or raw[:4] != b"RPL1":
        raise ValueError("replay_header_invalid")
    end = len(raw) - 4
    if zlib.crc32(raw[:end]) != struct.unpack_from("<I", raw, end)[0]:
        raise ValueError("replay_crc_invalid")
    rows, cols = raw[4] >> 4, raw[4] & 15
    variant = f"{rows}x{cols}"
    if variant not in engine.VARIANTS or raw[5] != 0 or raw[6] != 2:
        raise ValueError("replay_variant_invalid")
    initial = [0] * (rows * cols)
    offset = 7
    for _ in range(2):
        if offset >= end:
            raise ValueError("replay_truncated")
        code = raw[offset]
        offset += 1
        index, value = code & 15, 4 if code & 16 else 2
        if code & 0xE0 or index >= len(initial) or initial[index]:
            raise ValueError("replay_initial_invalid")
        initial[index] = value
    moves, ended = [], False
    while offset < end:
        code = raw[offset]
        offset += 1
        if code == 132:
            ended = True
            break
        if code >= 128 or len(moves) >= engine.MAX_MOVES:
            raise ValueError("replay_event_invalid")
        delta, offset = _read_uleb(raw, offset, end)
        if delta > 0xFFFFFFFF:
            raise ValueError("replay_timing_invalid")
        moves.append((code & 3, (code >> 2) & 15, 4 if code & 64 else 2, delta))
    if not ended or offset != end:
        raise ValueError("replay_end_invalid")
    return variant, initial, moves


def encode_rpl1(variant, initial, moves) -> bytes:
    rows, cols = engine.VARIANTS[variant]
    output = bytearray(b"RPL1")
    output.extend(((rows << 4) | cols, 0, 2))
    output.extend(index | (16 if value == 4 else 0)
                  for index, value in enumerate(initial) if value)
    for direction, index, value, delta in moves:
        output.append(direction | (index << 2) | (64 if value == 4 else 0))
        output.extend(_uleb(delta))
    output.append(132)
    output.extend(struct.pack("<I", zlib.crc32(output)))
    return bytes(output)


# Backward-compatible private name used by older validation tests and tools.
_encode = encode_rpl1


def inspect_replay(raw: bytes, expected_variant: str | None = None) -> dict:
    if not raw or len(raw) > MAX_INPUT:
        raise ValueError("replay_size_invalid")
    if raw.startswith(b"RPL1"):
        variant, board, moves = _rpl1(raw)
    else:
        text = raw.decode("latin-1").strip()
        if text.startswith(PREFIX):
            try:
                variant, board, moves = _rpl1(base64.b64decode(text[len(PREFIX):], validate=True))
            except (ValueError, binascii.Error) as exc:
                raise ValueError("replay_base64_invalid") from exc
        else:
            variant, board, moves = _legacy(text)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError("replay_variant_mismatch")
    if sum(bool(value) for value in board) != 2:
        raise ValueError("replay_initial_invalid")
    initial = board.copy()
    rows, cols = engine.VARIANTS[variant]
    score = 0
    elapsed = 0
    timed_moves = 0
    nodes = {}
    spawn_count = 2
    four_count = initial.count(4)
    rate_tracker = statistics.Rate32kTracker(variant)
    rate_tracker.observe({"board": board})
    for move_number, (direction, index, value, delta) in enumerate(moves, 1):
        next_board, gained = engine.move(board, rows, cols, direction)
        if next_board == board or index >= len(board) or next_board[index] or value not in (2, 4):
            raise ValueError("replay_move_invalid")
        next_board[index] = value
        board = next_board
        score += gained
        spawn_count += 1
        four_count += int(value == 4)
        if delta != UNKNOWN_TIMING_MS:
            elapsed += delta
            timed_moves += 1
        for tile in engine.NODES[variant]:
            if tile in board and str(tile) not in nodes:
                nodes[str(tile)] = {"seq": move_number, "elapsed": elapsed}
        rate_tracker.observe({"board": board})
    return {"normalized": encode_rpl1(variant, initial, moves), "variant": variant,
            "initial_board": initial, "board": board, "score": score,
            "moves": len(moves), "elapsed": elapsed, "timed_moves": timed_moves,
            "nodes": nodes, "spawn_count": spawn_count, "four_count": four_count,
            "game_over": engine.game_over(board, rows, cols), "rate": rate_tracker.result()}


def validate(raw: bytes, expected_variant: str, expected_score: int, expected_board: list[int]):
    result = inspect_replay(raw, expected_variant)
    if result["score"] != expected_score or result["board"] != expected_board:
        raise ValueError("replay_result_mismatch")
    return result["normalized"], result["moves"], result["rate"]


def attach(run_id: str, user_id: int, raw: bytes):
    with database() as db:
        row = db.execute("""SELECT id,user_id,variant,state,archive,source,visible,status,ended
            FROM human_runs WHERE id=?""", (run_id,)).fetchone()
    if (not row or row["user_id"] != user_id or row["source"] != "verse"
            or row["status"] != "sealed" or not row["visible"]):
        raise service.RunError("replay_run_not_found", 404)
    state = json.loads(row["state"])
    try:
        normalized, moves, rate = validate(raw, row["variant"], state["score"], state["board"])
    except ValueError as exc:
        raise service.RunError(str(exc), 400) from exc
    archive = gzip.compress(normalized, compresslevel=6, mtime=0)
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("""SELECT archive,source,state,has_replay FROM human_runs
            WHERE id=? AND user_id=? AND visible=1""", (run_id, user_id)).fetchone()
        if not current or current["source"] != "verse" or json.loads(current["state"])["score"] != state["score"]:
            raise service.RunError("replay_run_changed")
        if current["archive"] is not None:
            if current["archive"] == archive:
                return {"id": run_id, "has_replay": True, "moves": moves}
            raise service.RunError("replay_already_attached", 409)
        state["seq"] = moves
        # Version 2 preserves unknown Verse intervals as a sentinel, rather than
        # silently replacing them with an artificial 100 ms interval.
        state["replay_timing_version"] = 2
        db.execute("""UPDATE human_runs SET archive=?,has_replay=1,state=?
            WHERE id=?""", (archive, json.dumps(state, separators=(",", ":")), run_id))
        db.execute("UPDATE rolling_candidates SET has_replay=1,replay_id=? WHERE run_id=?",
                   (run_id, run_id))
        statistics.upsert_fact(db, dict(row), state["board"], state["score"], rate)
        statistics.rebuild_player(db, user_id, row["variant"])
    return {"id": run_id, "has_replay": True, "moves": moves}


def archived_to_analysis_text(archive: bytes, expected_variant: str, expected_moves: int) -> str:
    raw = gunzip_limited(archive, MAX_INPUT)
    variant, _, moves = _rpl1(raw)
    if variant != expected_variant or len(moves) != expected_moves:
        raise ValueError("archive_mismatch")
    return PREFIX + base64.b64encode(raw).decode("ascii")
