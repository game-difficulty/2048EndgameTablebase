from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


AI_MOVE_MAP = {1: "left", 2: "right", 3: "up", 4: "down"}
VALID_AI_MOVES = frozenset(AI_MOVE_MAP.values())


@dataclass(frozen=True)
class AIDecision:
    direction: str | None
    source: str


FallbackProvider = Callable[[int, float, float], tuple[Any, Any]]


def choose_full_ai_move(
    *,
    board_encoded: int,
    board: np.ndarray,
    dispatcher: Any,
    fallback_provider: FallbackProvider,
    spawn_rate4: float,
    time_limit_ratio: float,
    allow_resolve_32768: bool,
) -> AIDecision:
    """Run the production tablebase-first AI decision chain."""
    dispatcher.reset(board, board_encoded)
    table_move = dispatcher.dispatcher()
    if table_move != "AI":
        direction = str(table_move or "").lower()
        if direction not in VALID_AI_MOVES:
            direction = None
        source = str(getattr(dispatcher, "current_table", "") or "table")
        return AIDecision(direction=direction, source=source)

    search_board = int(board_encoded)
    if allow_resolve_32768:
        from native_core import ai_core

        search_board = int(ai_core.resolve_32768_doubles(search_board))

    player, logic = fallback_provider(
        search_board,
        float(spawn_rate4),
        float(time_limit_ratio),
    )
    move_code = logic.calculate_step(player, board, dispatcher.counts)
    direction = AI_MOVE_MAP.get(int(move_code))
    source = str(getattr(logic, "last_table", "") or "AI")
    return AIDecision(direction=direction, source=source)
