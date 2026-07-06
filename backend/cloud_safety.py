from __future__ import annotations

import os

from .actions import Action


def env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def is_cloud_mode() -> bool:
    return env_flag("CLOUD_MODE", "1") and os.getenv("APP_MODE", "cloud").lower() != "desktop"


CLOUD_BLOCKED_ACTIONS = frozenset(
    {
        Action.INIT_GAME,
        Action.USER_MOVE,
        Action.SAVE_GAME_STATE,
        Action.AI_STEP,
        Action.TESTER_SAVE_LOG,
        Action.TESTER_SAVE_REPLAY,
        Action.TESTER_TRIGGER_SAVE_LOG,
        Action.TESTER_TRIGGER_SAVE_REPLAY,
        Action.REPLAY_LOAD_FILE,
        Action.REPLAY_TRIGGER_OPEN_FILE,
        Action.ANALYSIS_START,
        Action.ANALYSIS_TRIGGER_SELECT_FILES,
        Action.START_BUILD,
        Action.SELECT_FOLDER_CMD,
        Action.SETTINGS_TRIGGER_SELECT_FOLDER,
        Action.TRIGGER_SELECT_FOLDER,
        Action.TRIGGER_RECORD_OPEN,
        Action.TRIGGER_RECORD_SAVE,
        Action.RECORD_OPEN,
        Action.RECORD_SAVE,
        Action.START_RECORDING,
        Action.STOP_RECORDING,
        Action.PREPARE_STOP_RECORDING,
        Action.RECORD_STEP,
    }
)


def is_cloud_action_blocked(action: str | None) -> bool:
    if not is_cloud_mode() or not action:
        return False
    action_name = str(action)
    return (
        action in CLOUD_BLOCKED_ACTIONS
        or action_name.startswith("NOTEBOOK_")
        or action_name.startswith("MINIGAME_")
    )


def cloud_disabled_message(action: str | None) -> str:
    return f"Action {action} is disabled in cloud mode."
