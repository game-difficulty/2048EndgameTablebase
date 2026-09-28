from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CompetitionStatus(str, Enum):
    CREATED = "CREATED"
    SEATING = "SEATING"
    READY_CHECK = "READY_CHECK"
    DRAW = "DRAW"
    FIRST_PICK_BAN = "FIRST_PICK_BAN"
    SECOND_PICK_BAN = "SECOND_PICK_BAN"
    BLIND_PICK = "BLIND_PICK"
    C_DRAW = "C_DRAW"
    LINEUP = "LINEUP"
    GAME_A_READY = "GAME_A_READY"
    GAME_A_PLAYING = "GAME_A_PLAYING"
    GAME_A_RESULT = "GAME_A_RESULT"
    GAME_B_READY = "GAME_B_READY"
    GAME_B_PLAYING = "GAME_B_PLAYING"
    GAME_B_RESULT = "GAME_B_RESULT"
    GAME_C_READY = "GAME_C_READY"
    GAME_C_PLAYING = "GAME_C_PLAYING"
    GAME_C_RESULT = "GAME_C_RESULT"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"


class TeamSide(str, Enum):
    YELLOW = "yellow"
    WHITE = "white"


class StaffRole(str, Enum):
    ORGANIZER = "organizer"
    REFEREE = "referee"


@dataclass(frozen=True)
class Principal:
    user_id: int
    display_name: str
    site_role: str = "user"
    session_id: int | None = None

    @property
    def is_platform_organizer(self) -> bool:
        return self.site_role.lower() in {"admin", "organizer"}


SEAT_POSITIONS = (1, 2, 3)
SIDES = (TeamSide.YELLOW, TeamSide.WHITE)
