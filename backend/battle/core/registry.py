from __future__ import annotations

from types import MappingProxyType

from .contracts import BattleMode, BattleModeError


_modes: dict[str, BattleMode] = {}


def register_battle_mode(mode: BattleMode, *, replace: bool = False) -> BattleMode:
    key = str(getattr(mode, "key", "") or "").strip().lower()
    if not key or len(key) > 64:
        raise BattleModeError("invalid_battle_mode_key")
    if int(getattr(mode, "version", 0) or 0) <= 0:
        raise BattleModeError("invalid_battle_mode_version")
    if key in _modes and _modes[key] is not mode and not replace:
        raise BattleModeError(f"battle_mode_already_registered:{key}")
    _modes[key] = mode
    return mode


def get_battle_mode(mode_key: str | None) -> BattleMode:
    key = str(mode_key or "goodness").strip().lower()
    try:
        return _modes[key]
    except KeyError as exc:
        raise BattleModeError(f"battle_mode_not_available:{key}") from exc


def list_battle_modes() -> MappingProxyType:
    return MappingProxyType(dict(_modes))


def clear_battle_modes_for_tests() -> None:
    _modes.clear()
