from __future__ import annotations

from typing import Any, Optional

import numpy as np
from Config import SingletonConfig
from egtb_core.VBoardMover import decode_board
from fastapi import WebSocket

from .serialization import sanitize_config
from .session import GameSession, safe_hex, u64
from .trainer_helpers import _get_current_record_results


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
        board_encoded = np.uint64(u64(session.board_encoded))
        board_array = decode_board(board_encoded)
        config = SingletonConfig().config
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
                    "history": [
                        safe_hex(h[0]) for h in getattr(session, "history", [])
                    ],
                    "moves": getattr(session, "move_history", []),
                    "record_results_mode": (
                        "embedded"
                        if getattr(session, "record_result_history", [])
                        else None
                    ),
                    "record_results": record_results,
                    "record_results_dtype": record_results_dtype,
                    "awaiting_spawn": (session.spawn_mode == 3 and session.moved == 1),
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
        if isinstance(session_or_data, GameSession):
            minigame_state = getattr(session_or_data, "minigame_session", None)
            engine = getattr(minigame_state, "engine", None)
            if engine is not None:
                engine.save_to_config()
        config = SingletonConfig().config
        if isinstance(session_or_data, dict):
            if not session_or_data.get("is_gamer", True):
                return
            config["game_state"] = [
                u64(session_or_data.get("board_encoded", 0)),
                int(session_or_data.get("score", 0)),
                int(session_or_data.get("best_score", 0)),
            ]
        else:
            if not session_or_data.client_id.startswith("gamer_"):
                return
            config["game_state"] = [
                u64(session_or_data.board_encoded),
                int(session_or_data.score),
                int(session_or_data.best_score),
            ]
        SingletonConfig().save_config(config)
    except Exception as e:
        print(f"Failed to save game state: {e}")
