from __future__ import annotations

from typing import Any, Optional

import numpy as np
from Config import SingletonConfig
from engine_core.VBoardMover import decode_board
from fastapi import WebSocket

from .cloud_safety import is_cloud_mode
from .serialization import sanitize_config
from .session import GameSession, normalize_gamer_special_tiles, np_u64, safe_hex, u64
from .trainer_helpers import _get_current_record_results


def _apply_gamer_special_tiles(board_array, special_tiles):
    if not special_tiles:
        return board_array
    board_array = board_array.copy()
    flat = board_array.reshape(-1)
    for index, value in special_tiles.items():
        if 0 <= int(index) < flat.size and int(flat[int(index)]) == 32768:
            flat[int(index)] = int(value)
    return board_array


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, GameSession] = {}

    async def connect(self, websocket: WebSocket, client_id: str = "") -> None:
        await websocket.accept()
        self.active_connections[websocket] = GameSession(client_id)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.pop(websocket, None)

    async def broadcast(self, message: str) -> None:
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

    async def send_state(
        self, websocket: WebSocket, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        session = self.active_connections[websocket]
        board_encoded = np_u64(session.board_encoded)
        board_array = decode_board(board_encoded)
        if session.client_id.startswith("gamer_"):
            board_array = _apply_gamer_special_tiles(
                board_array, getattr(session, "gamer_special_tiles", {})
            )
        config = SingletonConfig().config
        tablebase_status = "not_selected"
        tablebase_full_pattern = ""
        tablebase_dtype = ""
        if session.client_id.startswith("trainer_"):
            tablebase_full_pattern = str(getattr(session, "current_pattern", "") or "")
            if tablebase_full_pattern:
                tablebase_status = "loaded"
                tablebase_dtype = str(getattr(session, "success_rate_dtype", "") or "")
        record_results, record_results_dtype = _get_current_record_results(session)

        await websocket.send_json(
            {
                "action": "UPDATE_STATE",
                "data": {
                    "board": board_array.flatten().tolist(),
                    "animation": sanitize_config(metadata or {}),
                    "score": {
                        "current": int(session.score),
                        "best": int(session.best_score),
                    },
                    "hex_str": safe_hex(session.board_encoded),
                    "record_step": getattr(session, "played_length", 0),
                    "record_max": len(getattr(session, "history", [])),
                    "recording_length": getattr(session, "record_length", 0),
                    "record_playback_loaded": bool(
                        getattr(session, "record_playback_loaded", False)
                    ),
                    "history": [
                        safe_hex(h[0]) for h in getattr(session, "history", [])
                    ],
                    "moves": getattr(session, "move_history", []),
                    "gamer_special_tiles": [
                        [int(index), int(value)]
                        for index, value in sorted(
                            getattr(session, "gamer_special_tiles", {}).items()
                        )
                    ],
                    "record_results_mode": (
                        "embedded"
                        if getattr(session, "record_result_history", [])
                        else None
                    ),
                    "record_results": record_results,
                    "record_results_dtype": record_results_dtype,
                    "awaiting_spawn": (session.spawn_mode == 3 and session.moved == 1),
                    "tablebase_status": tablebase_status,
                    "tablebase_full_pattern": tablebase_full_pattern,
                    "tablebase_dtype": tablebase_dtype,
                    "settings": {
                        "difficulty": getattr(session, "difficulty", 0) * 100.0,
                        "speed": getattr(session, "speed", 100.0),
                        "colors": config.get("colors", []),
                        "theme": config.get("theme", "Default"),
                        "do_animation": config.get("do_animation", True),
                        "dis_32k": config.get("dis_32k", False),
                        "font_size_factor": config.get("font_size_factor", 100),
                    },
                },
            }
        )


def save_game_state(session_or_data: GameSession | dict[str, Any]) -> None:
    try:
        if is_cloud_mode():
            return
        config = SingletonConfig().config
        if isinstance(session_or_data, dict):
            if not session_or_data.get("is_gamer", True):
                return
            config["game_state"] = [
                u64(session_or_data.get("board_encoded", 0)),
                int(session_or_data.get("score", 0)),
                int(session_or_data.get("best_score", 0)),
                [
                    [int(index), int(value)]
                    for index, value in sorted(
                        normalize_gamer_special_tiles(
                            session_or_data.get("special_tiles", [])
                        ).items()
                    )
                ],
            ]
        else:
            if not session_or_data.client_id.startswith("gamer_"):
                return
            config["game_state"] = [
                u64(session_or_data.board_encoded),
                int(session_or_data.score),
                int(session_or_data.best_score),
                [
                    [int(index), int(value)]
                    for index, value in sorted(
                        getattr(session_or_data, "gamer_special_tiles", {}).items()
                    )
                ],
            ]
        SingletonConfig().save_config(config)
    except Exception as e:
        print(f"Failed to save game state: {e}")
