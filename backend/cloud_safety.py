from __future__ import annotations

import os

from .actions import Action


def env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off"}


def is_cloud_mode() -> bool:
    return env_flag("CLOUD_MODE", "1") and os.getenv("APP_MODE", "cloud").lower() != "desktop"


CLOUD_BLOCKED_ACTIONS = frozenset(
    {
        Action.GET_STATE,
        Action.INIT_GAME,
        Action.USER_MOVE,
        Action.SAVE_GAME_STATE,
        Action.AI_STEP,
        Action.TESTER_SAVE_LOG,
        Action.TESTER_SAVE_REPLAY,
        Action.TESTER_TRIGGER_SAVE_LOG,
        Action.TESTER_TRIGGER_SAVE_REPLAY,
        Action.TESTER_MOVE,
        Action.TESTER_SET_BOARD,
        Action.TESTER_SET_TEXT_VISIBLE,
        Action.TESTER_EXPORT_LOG,
        Action.TESTER_EXPORT_REPLAY,
        Action.REPLAY_LOAD_FILE,
        Action.REPLAY_TRIGGER_OPEN_FILE,
        Action.REPLAY_GET_INIT,
        Action.REPLAY_LOAD_UPLOAD,
        Action.REPLAY_LOAD_LATEST,
        Action.REPLAY_SET_STEP,
        Action.REPLAY_STEP,
        Action.REPLAY_NEXT_POINT,
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
        Action.TRAINER_GET_RESULTS,
        Action.TRAINER_MOVE,
        Action.TRAINER_MANUAL_SPAWN,
        Action.TRAINER_STEP,
        Action.SET_BOARD,
        Action.SET_CELL,
        Action.UNDO,
        Action.SET_SPAWN_MODE,
        Action.ROTATE,
    }
)


# These actions still have desktop-compatible branches that use GameSession as
# the board owner. Cloud clients must opt into the stateless/client-owned
# protocol so those legacy branches are unreachable from the public service.
CLOUD_CLIENT_LOCAL_BOARD_ACTIONS = frozenset(
    {
        Action.TRAINER_SET_EMPTY_PATTERN,
        Action.TRAINER_SET_FILEPATH,
        Action.TRAINER_DEFAULT,
        Action.TESTER_GET_INIT,
        Action.TESTER_SELECT_PATTERN,
        Action.TESTER_RESET_RANDOM,
        Action.TABLEBASE_QUERY,
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


def cloud_payload_error(action: str | None, payload: object) -> str | None:
    """Reject cloud requests that could fall back to server-owned board state."""
    if not is_cloud_mode() or not action:
        return None
    if action not in CLOUD_CLIENT_LOCAL_BOARD_ACTIONS:
        return None
    if not isinstance(payload, dict) or payload.get("client_local_board") is not True:
        return (
            f"Action {action} requires the client-owned board protocol "
            "in cloud mode."
        )
    return None
