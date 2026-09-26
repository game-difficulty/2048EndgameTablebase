"""Version 1 rules and compact replay codec, independent of native AI modules."""
from __future__ import annotations

import hashlib
import json
import struct
from .codec import to_planes, from_planes

from backend.gamer_ranked.prng import Xoshiro128StarStar

VARIANTS = {"4x4": (4, 4), "3x4": (3, 4), "2x4": (2, 4), "3x3": (3, 3)}
THRESHOLDS = {"4x4": 800000, "3x4": 70000, "2x4": 5000, "3x3": 10000}
RESTART_THRESHOLDS = {"4x4": 10000, "3x4": 2500, "2x4": 500, "3x3": 1000}
NODES = {variant: [2 ** exponent for exponent in range(5, 32)] for variant in VARIANTS}
EVENT = struct.Struct("<BI")
MAX_MOVES = 200000
MAX_BYTES = MAX_MOVES * EVENT.size


def move(board, rows, cols, direction):
    if direction not in range(4):
        raise ValueError("invalid_direction")
    result = list(board)
    score = 0
    horizontal = direction in (1, 3)
    for line in range(rows if horizontal else cols):
        positions = ([line * cols + x for x in range(cols)] if horizontal
                     else [y * cols + line for y in range(rows)])
        if direction in (1, 2):
            positions.reverse()
        source = [board[p] for p in positions if board[p]]
        merged = []
        i = 0
        while i < len(source):
            value = source[i]
            if i + 1 < len(source) and source[i + 1] == value:
                value *= 2
                if value > 2**31:
                    raise ValueError("tile_limit")
                score += value
                i += 1
            merged.append(value)
            i += 1
        for i, p in enumerate(positions):
            result[p] = merged[i] if i < len(merged) else 0
    return result, score


def spawn(board, rng):
    empty = [i for i, value in enumerate(board) if not value]
    if not empty:
        raise ValueError("no_spawn_space")
    index = empty[rng.choose_index(len(empty))]
    value = 4 if rng.next_float() < 0.1 else 2
    board[index] = value
    return index, value


def initial(run_id, variant, seed):
    rows, cols = VARIANTS[variant]
    rng = Xoshiro128StarStar.from_seed_hex(seed)
    board = [0] * (rows * cols)
    spawn(board, rng)
    spawn(board, rng)
    return {"board": board, "rng": rng.state, "score": 0, "seq": 0, "elapsed": 0,
            "fourCount": board.count(4), "spawnCount": 2,
            "nodes": {}, "first_over": None,
            "hash": hashlib.sha256(f"HPR1|{run_id}|{variant}|{seed}".encode()).hexdigest()}


def game_over(board, rows, cols):
    return all(move(board, rows, cols, d)[0] == board for d in range(4))


def advance(state, variant, data, threshold=None, observer=None):
    if len(data) % EVENT.size or len(data) > MAX_BYTES:
        raise ValueError("invalid_binary_length")
    state = json.loads(json.dumps(state))
    rows, cols = VARIANTS[variant]
    rng = Xoshiro128StarStar(state["rng"])
    for offset in range(0, len(data), EVENT.size):
        code, delta = EVENT.unpack_from(data, offset)
        if code >= 128 or state["seq"] >= MAX_MOVES:
            raise ValueError("invalid_event")
        board, score = move(state["board"], rows, cols, code & 3)
        if board == state["board"]:
            raise ValueError("invalid_move")
        index, value = spawn(board, rng)
        if (code >> 2) & 15 != index or (4 if code & 64 else 2) != value:
            raise ValueError("spawn_mismatch")
        state["board"] = board
        state["score"] += score
        state["seq"] += 1
        if "spawnCount" in state:
            state["spawnCount"] += 1
            state["fourCount"] += int(value == 4)
        if threshold is not None and state["score"] > threshold and state["first_over"] is None:
            state["first_over"] = state["seq"]
        state["elapsed"] += delta
        state["hash"] = hashlib.sha256(bytes.fromhex(state["hash"]) + data[offset:offset + 5]).hexdigest()
        for tile in NODES[variant]:
            if tile in board and str(tile) not in state["nodes"]:
                state["nodes"][str(tile)] = {"seq": state["seq"], "elapsed": state["elapsed"]}
        if observer is not None:
            observer(state)
    state["rng"] = rng.state
    return state


def replay_bytes(run, events, version=1):
    if version not in (1, 2):
        raise ValueError('unsupported_replay')
    header = json.dumps({"version": version, "rules_version": 1, "run_id": run["id"],
                         "variant": run["variant"], "seed": run["seed"],
                         "reason": run["reason"], "started_at": run["created"],
                         "timing": "continuous-client-ms"}, separators=(",", ":")).encode()
    return (b'HPR2' if version == 2 else b'HPR1') + struct.pack("<I", len(header)) + header + (to_planes(events) if version == 2 else events)


def parse_replay(binary):
    if len(binary) < 8 or binary[:4] not in (b'HPR1', b'HPR2'):
        raise ValueError('invalid_replay')
    size = struct.unpack_from('<I', binary, 4)[0]
    if size > 65536 or size + 8 > len(binary):
        raise ValueError('invalid_replay')
    header = json.loads(binary[8:8 + size])
    version = 2 if binary[:4] == b'HPR2' else 1
    if header.get('version') != version or header.get('rules_version') != 1 or header.get('variant') not in VARIANTS:
        raise ValueError('unsupported_replay')
    raw = binary[8 + size:]
    if len(raw) % 5 or len(raw) > MAX_BYTES:
        raise ValueError('invalid_binary_length')
    return header, from_planes(raw) if version == 2 else raw
