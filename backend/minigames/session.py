from __future__ import annotations

from dataclasses import dataclass, field

from Config import SingletonConfig

from ..cloud_safety import is_cloud_mode


def default_minigame_difficulty() -> int:
    if is_cloud_mode():
        return 1
    return int(SingletonConfig().config.get("minigame_difficulty", 1))


@dataclass
class MinigameSessionState:
    difficulty: int = field(default_factory=default_minigame_difficulty)
    current_game_id: str = ""
    engine: object | None = None
    powerup_counts: dict[str, int] = field(
        default_factory=lambda: {"bomb": 0, "glove": 0, "twist": 0}
    )
    active_mode: str | None = None
    interaction_phase: int = 0
    selection_cache: dict | None = None

    def reset_runtime(self) -> None:
        self.current_game_id = ""
        self.engine = None
        self.powerup_counts = {"bomb": 0, "glove": 0, "twist": 0}
        self.active_mode = None
        self.interaction_phase = 0
        self.selection_cache = None
