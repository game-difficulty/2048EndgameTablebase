from __future__ import annotations

from typing import Any

from Config import SingletonConfig

from .games.blitzkrieg import BlitzkriegEngine
from .games.column_chaos import ColumnChaosEngine
from .games.design_master import DesignMasterEngine
from .games.endless_family import EndlessFamilyEngine
from .games.ferris_wheel import FerrisWheelEngine
from .games.gravity_twist import GravityTwistEngine
from .games.ice_age import IceAgeEngine
from .games.isolated_island import IsolatedIslandEngine
from .games.mystery_merge import MysteryMergeEngine
from .games.shape_shifter import ShapeShifterEngine
from .games.tricky_tiles import TrickyTilesEngine
from .powerups import (
    activate_powerup,
    apply_target_action,
    build_interaction_payload,
    build_powerups_payload,
    cancel_powerup_interaction,
    load_powerup_counts,
    maybe_award_random_powerup,
    reset_powerup_counts,
    save_powerup_counts,
)
from .registry import MINIGAME_BY_ID, MINIGAME_REGISTRY, MinigameDefinition
from .session import MinigameSessionState


ENGINE_BY_MODULE = {
    "design_master": DesignMasterEngine,
    "gravity_twist": GravityTwistEngine,
    "column_chaos": ColumnChaosEngine,
    "blitzkrieg": BlitzkriegEngine,
    "mystery_merge": MysteryMergeEngine,
    "ice_age": IceAgeEngine,
    "isolated_island": IsolatedIslandEngine,
    "shape_shifter": ShapeShifterEngine,
    "endless_family": EndlessFamilyEngine,
    "ferris_wheel": FerrisWheelEngine,
    "tricky_tiles": TrickyTilesEngine,
}


def _summary_for_game(definition: MinigameDefinition, difficulty: int) -> dict[str, Any]:
    config = SingletonConfig().config
    game_state = config.get("minigame_state", [dict(), dict()])
    saved = game_state[difficulty].get(definition.legacy_name)
    best_score = 0
    highest_exp = 0
    trophy = 0
    if saved:
        try:
            main_state = saved[0]
            best_score = int(main_state[2])
            highest_exp = int(main_state[3])
            trophy = int(main_state[4])
        except Exception:
            best_score = 0
            highest_exp = 0
            trophy = 0
    return {
        "id": definition.id,
        "title": definition.title,
        "description": definition.description,
        "coverUrl": definition.cover_url,
        "difficultyAware": bool(definition.difficulty_aware),
        "supportsPowerups": bool(definition.supports_powerups),
        "implemented": bool(definition.implemented),
        "summary": {
            "bestScore": best_score,
            "highestTile": 0 if highest_exp <= 0 else int(2**highest_exp),
            "highestExp": highest_exp,
            "trophy": trophy,
        },
    }


def build_menu_payload(state: MinigameSessionState) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    section_map: dict[str, list[dict[str, Any]]] = {}
    for definition in MINIGAME_REGISTRY:
        section_map.setdefault(definition.section, []).append(
            _summary_for_game(definition, state.difficulty)
        )
    for section_name, items in section_map.items():
        sections.append({"title": section_name, "items": items})
    return {
        "difficulty": int(state.difficulty),
        "sections": sections,
        "currentGameId": state.current_game_id,
    }


def create_engine(definition: MinigameDefinition, difficulty: int):
    engine_cls = ENGINE_BY_MODULE.get(definition.module_key)
    if engine_cls is None:
        raise ValueError(f"Minigame '{definition.title}' is not implemented yet")
    return engine_cls(definition, difficulty)


def start_minigame(state: MinigameSessionState, game_id: str):
    definition = MINIGAME_BY_ID.get(game_id)
    if definition is None:
        raise ValueError("Unknown minigame")
    if not definition.implemented:
        raise ValueError(f"Minigame '{definition.title}' is not implemented yet")
    engine = create_engine(definition, state.difficulty)
    state.current_game_id = definition.id
    state.engine = engine
    state.active_mode = None
    state.interaction_phase = 0
    state.selection_cache = None
    state.powerup_counts = load_powerup_counts(state)
    engine.save_to_config()
    return engine


def save_current_game(state: MinigameSessionState) -> None:
    if state.engine is not None:
        state.engine.save_to_config()
        save_powerup_counts(state)


def build_state_payload(state: MinigameSessionState) -> dict[str, Any]:
    if state.engine is None:
        raise ValueError("No active minigame")
    payload = state.engine.serialize_state()
    payload["powerups"] = build_powerups_payload(state)
    payload["interaction"] = build_interaction_payload(state)
    return payload


def reset_powerups_for_new_game(state: MinigameSessionState) -> None:
    cancel_powerup_interaction(state)
    reset_powerup_counts(state)


def request_powerup(state: MinigameSessionState, mode: str) -> bool:
    return activate_powerup(state, mode)


def resolve_powerup_target(state: MinigameSessionState, index: int) -> bool:
    return apply_target_action(state, index)


def cancel_powerup(state: MinigameSessionState) -> None:
    cancel_powerup_interaction(state)


def trigger_custom_action(state: MinigameSessionState, key: str, phase: str = "trigger") -> bool:
    if state.engine is None:
        return False
    return bool(state.engine.handle_custom_action(str(key or ""), str(phase or "trigger")))


def maybe_reward_powerup_from_score(state: MinigameSessionState, score_delta: int) -> str | None:
    return maybe_award_random_powerup(state, score_delta)
