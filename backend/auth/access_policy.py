from __future__ import annotations

from enum import IntEnum

from backend.actions import Action

from .principal import ActorRef


class AccessLevel(IntEnum):
    PUBLIC = 0
    GUEST_OR_USER = 1
    REGISTERED_USER = 2
    ROOM_HOST = 3
    ADMIN = 4


_GUEST_OR_USER_ACTIONS = {
    Action.TRAINER_SET_FILEPATH,
    Action.TRAINER_SET_EMPTY_PATTERN,
    Action.TRAINER_DEFAULT,
    Action.TRAINER_MOVE,
    Action.TRAINER_MANUAL_SPAWN,
    Action.TRAINER_STEP,
    Action.SET_SPAWN_MODE,
    Action.SET_BOARD,
    Action.SET_CELL,
    Action.UNDO,
    Action.TABLEBASE_QUERY,
    Action.BATTLE_SUBSCRIBE,
    Action.BATTLE_ACTION,
    Action.BATTLE_PROGRESS,
    Action.BATTLE_HEARTBEAT,
    Action.BATTLE_CHAT_SEND,
}

_REGISTERED_USER_ACTIONS = {
    Action.TRAINER_GET_RESULTS,
    Action.TRAINER_SPAWN_QUERY,
    Action.TESTER_SELECT_PATTERN,
    Action.TESTER_RESET_RANDOM,
    Action.TESTER_MOVE,
    Action.TESTER_SET_BOARD,
    Action.TESTER_EXPORT_LOG,
    Action.TESTER_EXPORT_REPLAY,
    Action.REPLAY_LOAD_UPLOAD,
    Action.REPLAY_LOAD_LATEST,
    Action.ANALYSIS_SUBSCRIBE,
}


def websocket_access_level(action: str | None) -> AccessLevel:
    if action in _REGISTERED_USER_ACTIONS:
        return AccessLevel.REGISTERED_USER
    if action in _GUEST_OR_USER_ACTIONS:
        return AccessLevel.GUEST_OR_USER
    return AccessLevel.PUBLIC


def actor_satisfies(level: AccessLevel, actor: ActorRef | None) -> bool:
    if level == AccessLevel.PUBLIC:
        return True
    if actor is None:
        return False
    if level == AccessLevel.GUEST_OR_USER:
        return actor.is_guest or actor.is_user
    if level == AccessLevel.REGISTERED_USER:
        return actor.is_user
    if level == AccessLevel.ADMIN:
        return actor.is_user and actor.role == "admin"
    return actor.is_user
