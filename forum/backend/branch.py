"""Explicitly hypothetical replay files: arbitrary start, every spawn recorded."""

import json
from backend.human_play.engine import VARIANTS, move


def parse(raw):
    data = json.loads(raw[4:])
    if (
        set(data) - {"variant", "initial", "moves", "origin"}
        or data["variant"] not in VARIANTS
    ):
        raise ValueError("branch format")
    rows, cols = VARIANTS[data["variant"]]
    board = data["initial"]
    if (
        not isinstance(board, list)
        or len(board) != rows * cols
        or any(
            type(v) != int or v < 0 or v > 2**20 or v == 1 or (v and v & (v - 1))
            for v in board
        )
    ):
        raise ValueError("branch board")
    events = data["moves"]
    if not isinstance(events, list) or not 1 <= len(events) <= 1000:
        raise ValueError("branch length")
    score = 0
    for event in events:
        if (
            not isinstance(event, list)
            or len(event) != 4
            or any(type(v) != int for v in event)
        ):
            raise ValueError("branch event")
        direction, index, value, delta = event
        if (
            direction not in range(4)
            or index not in range(len(board))
            or value not in (2, 4)
            or delta != 0
        ):
            raise ValueError("branch spawn")
        nxt, gain = move(board, rows, cols, direction)
        if nxt == board or nxt[index]:
            raise ValueError("branch move")
        nxt[index] = value
        board = nxt
        score += gain
    # Origin is descriptive only, never evidence of source ownership or authenticity.
    origin = data.get("origin", "")
    if not isinstance(origin, str) or len(origin) > 200:
        raise ValueError("branch origin")
    return {
        "variant": data["variant"],
        "initial": data["initial"],
        "moves": events,
        "score": score,
        "steps": len(events),
        "verification": "hypothetical",
        "origin": origin,
    }
