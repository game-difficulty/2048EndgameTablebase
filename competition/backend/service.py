from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .auth import avatar_urls_for_users
from .db import CompetitionDatabase
from .domain import CompetitionStatus, Principal, SEAT_POSITIONS, StaffRole, TeamSide
from .errors import CompetitionError
from .projects import (
    ProjectRegistry,
    Standard2048Adapter,
    TOURNAMENT_ADAPTER_FACTORIES,
)
from .projects.contracts import ProjectState
from . import client_runtime


ROOM_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
ROOM_CODE_RE = re.compile(r"^[A-Z2-9]{4,12}$")
COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
PROJECT_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
DRAW_ALGORITHM_VERSION = "hmac-sha256-v1"
GAME_KEYS = ("A", "B", "C")
GAME_READY_STATUS = {
    "A": CompetitionStatus.GAME_A_READY.value,
    "B": CompetitionStatus.GAME_B_READY.value,
    "C": CompetitionStatus.GAME_C_READY.value,
}
GAME_PLAYING_STATUS = {
    "A": CompetitionStatus.GAME_A_PLAYING.value,
    "B": CompetitionStatus.GAME_B_PLAYING.value,
    "C": CompetitionStatus.GAME_C_PLAYING.value,
}
GAME_RESULT_STATUS = {
    "A": CompetitionStatus.GAME_A_RESULT.value,
    "B": CompetitionStatus.GAME_B_RESULT.value,
    "C": CompetitionStatus.GAME_C_RESULT.value,
}
MATCH_ACTIVE_STATUSES = frozenset(
    (*GAME_READY_STATUS.values(), *GAME_PLAYING_STATUS.values(), *GAME_RESULT_STATUS.values())
)
CLOSABLE_ROOM_STATUSES = frozenset({
    CompetitionStatus.SEATING.value, CompetitionStatus.READY_CHECK.value,
})
ISSUE_CATEGORIES = frozenset({"network_device", "project", "rules", "other"})
SUSPENSION_REASON_CODES = frozenset(
    {"network_device", "project", "rules", "medical", "other"}
)
DEFAULT_PROJECTS = tuple(
    {
        "key": f"project-{index:02d}",
        "name": "标准 2048" if index == 1 else f"项目 {index:02d}（待配置）",
        "description": "标准四乘四棋盘" if index == 1 else "在创建比赛时替换为正式项目定义",
        "project_ref": "standard-2048-test",
        "rules_version": "standard-v1",
    }
    for index in range(1, 9)
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


class CompetitionService:
    def __init__(
        self,
        database: CompetitionDatabase,
        *,
        bootstrap_organizer_ids: frozenset[int] = frozenset(),
        room_creator_ids: frozenset[int] = frozenset(),
        draw_reveal_seconds: int = 10,
        draft_turn_seconds: int = 60,
        c_draw_reveal_seconds: int = 10,
        result_rest_seconds: int = 30,
        ready_preview_seconds: int = 10,
        lineup_seconds: int = 180,
        team_clock_seconds: int = 1800,
        test_project_target_tile: int = 2048,
        project_registry: ProjectRegistry | None = None,
        live_result_retention_seconds: int = 1800,
    ):
        self.database = database
        from .event_catalog import EventCatalog
        self.events = EventCatalog(self)
        from .event_schedule import EventSchedule
        self.schedule = EventSchedule(self)
        self.bootstrap_organizer_ids = bootstrap_organizer_ids
        self.room_creator_ids = room_creator_ids
        self.draw_reveal_seconds = max(1, int(draw_reveal_seconds))
        self.draft_turn_seconds = max(5, int(draft_turn_seconds))
        self.c_draw_reveal_seconds = max(1, int(c_draw_reveal_seconds))
        self.result_rest_seconds = max(0, int(result_rest_seconds))
        self.ready_preview_seconds = max(0, int(ready_preview_seconds))
        self.lineup_seconds = max(5, int(lineup_seconds))
        self.team_clock_ms = max(30, int(team_clock_seconds)) * 1000
        self.live_result_retention_seconds = max(
            60, int(live_result_retention_seconds)
        )
        if project_registry is None:
            project_registry = ProjectRegistry()
            project_registry.register(
                lambda: Standard2048Adapter(target_tile=test_project_target_tile)
            )
            for factory in TOURNAMENT_ADAPTER_FACTORIES:
                project_registry.register(factory)
        self.project_registry = project_registry

    def initialize(self) -> None:
        self.database.initialize()

    def _flow(self, db, cid):
        from .final_flow import VERSION
        return dict(db.execute('SELECT * FROM competition_flow_rules WHERE competition_id=?', (cid,)).fetchone() or
                    {'version': VERSION, 'team_clock_ms': self.team_clock_ms, 'late_minutes': 15, 'ready_seconds': 60})

    def _is_platform_organizer(self, principal: Principal) -> bool:
        return (
            principal.is_platform_organizer
            or principal.user_id in self.bootstrap_organizer_ids
        )

    def _can_create_competition(self, principal: Principal) -> bool:
        return self._is_platform_organizer(principal) or principal.user_id in self.room_creator_ids

    def _new_room_code(self) -> str:
        return "".join(secrets.choice(ROOM_CODE_ALPHABET) for _ in range(6))

    def _new_public_key(self) -> str:
        # Public URLs must not reveal the player join code or an internal UUID.
        return "m" + secrets.token_hex(10)

    def _new_phase_token(self) -> str:
        return secrets.token_urlsafe(24)

    def _normalize_projects(
        self, projects: list[dict[str, Any]] | None
    ) -> list[dict[str, Any]]:
        source = list(projects) if projects is not None else list(DEFAULT_PROJECTS)
        if not 5 <= len(source) <= 32:
            raise CompetitionError(
                "INVALID_PROJECT_POOL", "Project pool must contain 5-32 projects."
            )
        normalized: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        seen_names: set[str] = set()
        for index, item in enumerate(source, start=1):
            name = " ".join(str(item.get("name") or "").split())
            if not 1 <= len(name) <= 80:
                raise CompetitionError(
                    "INVALID_PROJECT", "Each project name must contain 1-80 characters."
                )
            key = str(item.get("key") or f"project-{index:02d}").strip().lower()
            if not PROJECT_KEY_RE.fullmatch(key):
                raise CompetitionError(
                    "INVALID_PROJECT_KEY",
                    "Project keys must use lowercase letters, numbers, dot, dash or underscore.",
                )
            name_key = name.casefold()
            if key in seen_keys or name_key in seen_names:
                raise CompetitionError(
                    "DUPLICATE_PROJECT", "Project keys and names must be unique."
                )
            seen_keys.add(key)
            seen_names.add(name_key)
            project_ref = str(
                item.get("project_ref") or "standard-2048-test"
            ).strip()
            rules_version = str(
                item.get("rules_version") or "standard-v1"
            ).strip()
            adapter_rules_version = str(
                item.get("adapter_rules_version") or "standard-v1"
            ).strip()
            descriptor = self.project_registry.snapshot(
                project_ref, adapter_rules_version
            )
            normalized.append(
                {
                    "key": key,
                    "name": name,
                    "description": " ".join(
                        str(item.get("description") or "").split()
                    )[:240],
                    "project_ref": project_ref,
                    "adapter_rules_version": adapter_rules_version,
                    "rules_version": rules_version,
                    "adapter_snapshot": descriptor,
                }
            )
        return normalized

    def _insert_projects(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        projects: list[dict[str, Any]],
    ) -> None:
        db.executemany(
            """
            INSERT INTO competition_projects
              (competition_id, project_key, name, description, project_ref,
               adapter_rules_version, rules_version, adapter_snapshot_json,
               sort_order, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            [
                (
                    competition_id,
                    item["key"],
                    item["name"],
                    item["description"],
                    item["project_ref"],
                    item["adapter_rules_version"],
                    item["rules_version"],
                    json.dumps(
                        item["adapter_snapshot"], ensure_ascii=False, separators=(",", ":")
                    ),
                    index,
                )
                for index, item in enumerate(projects, start=1)
            ],
        )

    def _project_keys(self, db: sqlite3.Connection, competition_id: str) -> list[str]:
        return [
            str(row["project_key"])
            for row in db.execute(
                """
                SELECT project_key FROM competition_projects
                WHERE competition_id = ? AND enabled = 1
                ORDER BY sort_order
                """,
                (competition_id,),
            ).fetchall()
        ]

    def _available_project_keys(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        draft: sqlite3.Row,
    ) -> list[str]:
        excluded = {
            str(value)
            for value in (
                draft["project_a"],
                draft["ban_m"],
                draft["project_b"],
                draft["ban_n"],
            )
            if value
        }
        return [
            key for key in self._project_keys(db, competition_id) if key not in excluded
        ]

    def _captain_side(
        self, db: sqlite3.Connection, competition_id: str, principal: Principal
    ) -> str:
        row = db.execute(
            """
            SELECT side FROM competition_seats
            WHERE competition_id = ? AND user_id = ? AND position = 1
            """,
            (competition_id, principal.user_id),
        ).fetchone()
        if row is None:
            raise CompetitionError(
                "CAPTAIN_REQUIRED", "Only a player in seat 1 can perform captain actions.", 403
            )
        return str(row["side"])

    def _draft_row(self, db: sqlite3.Connection, competition_id: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM competition_drafts WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if row is None:
            raise CompetitionError("DRAFT_NOT_INITIALIZED", "Draft is not initialized.", 409)
        return row

    def _initialize_draw(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        *,
        now: datetime,
    ) -> None:
        seed = secrets.token_bytes(32)
        commitment = hashlib.sha256(competition_id.encode("utf-8") + b":" + seed).hexdigest()
        first_digest = hmac.new(seed, b"first-side", hashlib.sha256).digest()
        first_side = (
            TeamSide.YELLOW.value if first_digest[0] % 2 == 0 else TeamSide.WHITE.value
        )
        deadline = (now + timedelta(seconds=self.draw_reveal_seconds)).isoformat()
        db.execute(
            """
            INSERT INTO competition_drafts
              (competition_id, random_seed_hex, commitment, algorithm_version,
               first_side, phase_token, phase_started_at,
               yellow_deadline_at, white_deadline_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                competition_id,
                seed.hex(),
                commitment,
                DRAW_ALGORITHM_VERSION,
                first_side,
                self._new_phase_token(),
                now.isoformat(),
                deadline,
                deadline,
                now.isoformat(),
            ),
        )
        self._append_event(
            db,
            competition_id,
            "draft.first_side_drawn",
            None,
            {
                "first_side": first_side,
                "commitment": commitment,
                "algorithm_version": DRAW_ALGORITHM_VERSION,
                "reveal_ends_at": deadline,
            },
        )

    def _change_draft_status(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        next_status: str,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        if str(room["status"]) != next_status:
            self._append_event(
                db,
                competition_id,
                "competition.status_changed",
                None,
                {"from": str(room["status"]), "to": next_status},
            )
            self._touch(db, competition_id, status=next_status)
        return db.execute(
            "SELECT * FROM competitions WHERE id = ?", (competition_id,)
        ).fetchone()

    def _start_first_pick(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        draft: sqlite3.Row,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        db.execute(
            "INSERT OR IGNORE INTO competition_prediction_windows (competition_id, opened_at, minimum_until) VALUES (?, ?, ?)",
            (str(room["id"]), now.isoformat(), (now + timedelta(seconds=60)).isoformat()),
        )
        deadline = (now + timedelta(seconds=self.draft_turn_seconds)).isoformat()
        first_side = str(draft["first_side"])
        db.execute(
            """
            UPDATE competition_drafts
            SET phase_token = ?, phase_started_at = ?,
                yellow_deadline_at = ?, white_deadline_at = ?, updated_at = ?
            WHERE competition_id = ?
            """,
            (
                self._new_phase_token(),
                now.isoformat(),
                deadline if first_side == TeamSide.YELLOW.value else None,
                deadline if first_side == TeamSide.WHITE.value else None,
                now.isoformat(),
                room["id"],
            ),
        )
        self._append_event(
            db,
            str(room["id"]),
            "draft.phase_started",
            None,
            {
                "phase": CompetitionStatus.FIRST_PICK_BAN.value,
                "active_side": first_side,
                "deadline_at": deadline,
            },
        )
        return self._change_draft_status(
            db, room, CompetitionStatus.FIRST_PICK_BAN.value
        )

    def _apply_pick_ban(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        draft: sqlite3.Row,
        *,
        side: str,
        pick_project_key: str,
        ban_project_key: str,
        actor_user_id: int | None,
        automatic: bool,
        now: datetime,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        status = str(room["status"])
        if status == CompetitionStatus.FIRST_PICK_BAN.value:
            db.execute(
                """
                UPDATE competition_drafts
                SET project_a = ?, ban_m = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (pick_project_key, ban_project_key, now.isoformat(), competition_id),
            )
            next_status = CompetitionStatus.SECOND_PICK_BAN.value
            next_side = (
                TeamSide.WHITE.value
                if side == TeamSide.YELLOW.value
                else TeamSide.YELLOW.value
            )
        elif status == CompetitionStatus.SECOND_PICK_BAN.value:
            db.execute(
                """
                UPDATE competition_drafts
                SET project_b = ?, ban_n = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (pick_project_key, ban_project_key, now.isoformat(), competition_id),
            )
            next_status = CompetitionStatus.BLIND_PICK.value
            next_side = None
        else:
            raise CompetitionError("INVALID_DRAFT_PHASE", "Pick/BAN is not active.", 409)
        self._append_event(
            db,
            competition_id,
            "draft.pick_ban_submitted",
            actor_user_id,
            {
                "phase": status,
                "side": side,
                "pick_project_key": pick_project_key,
                "ban_project_key": ban_project_key,
                "automatic": automatic,
            },
        )
        deadline = (now + timedelta(seconds=self.draft_turn_seconds)).isoformat()
        db.execute(
            """
            UPDATE competition_drafts
            SET phase_token = ?, phase_started_at = ?,
                yellow_deadline_at = ?, white_deadline_at = ?, updated_at = ?
            WHERE competition_id = ?
            """,
            (
                self._new_phase_token(),
                now.isoformat(),
                deadline
                if next_side in {None, TeamSide.YELLOW.value}
                else None,
                deadline
                if next_side in {None, TeamSide.WHITE.value}
                else None,
                now.isoformat(),
                competition_id,
            ),
        )
        room = self._change_draft_status(db, room, next_status)
        self._append_event(
            db,
            competition_id,
            "draft.phase_started",
            None,
            {
                "phase": next_status,
                "active_side": next_side,
                "deadline_at": deadline,
            },
        )
        return room

    def _finalize_blind(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        draft: sqlite3.Row,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        yellow = str(draft["blind_yellow"])
        white = str(draft["blind_white"])
        if yellow == white:
            project_c = yellow
        else:
            seed = bytes.fromhex(str(draft["random_seed_hex"]))
            label = f"blind-project-c:{yellow}:{white}".encode("utf-8")
            digest = hmac.new(seed, label, hashlib.sha256).digest()
            project_c = (yellow, white)[digest[0] % 2]
        reveal_ends_at = (now + timedelta(seconds=2 * self.c_draw_reveal_seconds)).isoformat()
        self._set_hold(db, str(room['id']), 'BLIND_CANDIDATES', now, self.c_draw_reveal_seconds)
        db.execute(
            """
            UPDATE competition_drafts
            SET project_c = ?, phase_token = ?, phase_started_at = ?,
                yellow_deadline_at = ?, white_deadline_at = ?, updated_at = ?
            WHERE competition_id = ?
            """,
            (
                project_c,
                self._new_phase_token(),
                now.isoformat(),
                reveal_ends_at,
                reveal_ends_at,
                now.isoformat(),
                room["id"],
            ),
        )
        self._append_event(
            db,
            str(room["id"]),
            "draft.blind_revealed",
            None,
            {
                "yellow_project_key": yellow,
                "white_project_key": white,
                "project_c": project_c,
                "reveal_ends_at": reveal_ends_at,
            },
        )
        return self._change_draft_status(db, room, CompetitionStatus.C_DRAW.value)

    def _lineup_state_row(
        self, db: sqlite3.Connection, competition_id: str
    ) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM competition_lineup_state WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if row is None:
            raise CompetitionError(
                "LINEUP_NOT_INITIALIZED", "Lineup phase is not initialized.", 409
            )
        return row

    def _lineup_submission_exists(
        self, db: sqlite3.Connection, competition_id: str, side: str
    ) -> bool:
        count = int(
            db.execute(
                """
                SELECT COUNT(*) AS count FROM competition_lineups
                WHERE competition_id = ? AND side = ?
                """,
                (competition_id, side),
            ).fetchone()["count"]
        )
        return count == 3

    def _start_lineup(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        deadline = (now + timedelta(seconds=self.lineup_seconds)).isoformat()
        db.execute(
            """
            INSERT INTO competition_lineup_state
              (competition_id, phase_token, phase_started_at,
               yellow_deadline_at, white_deadline_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                competition_id,
                self._new_phase_token(),
                now.isoformat(),
                deadline,
                deadline,
                now.isoformat(),
            ),
        )
        self._append_event(
            db,
            competition_id,
            "lineup.phase_started",
            None,
            {
                "yellow_deadline_at": deadline,
                "white_deadline_at": deadline,
            },
        )
        return self._change_draft_status(db, room, CompetitionStatus.LINEUP.value)

    def _insert_lineup(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        side: str,
        assignments: dict[str, int],
        actor_user_id: int | None,
        automatic: bool,
        now: datetime,
    ) -> None:
        competition_id = str(room["id"])
        seat_rows = db.execute(
            """
            SELECT position, user_id FROM competition_seats
            WHERE competition_id = ? AND side = ?
            ORDER BY position
            """,
            (competition_id, side),
        ).fetchall()
        users_by_position = {
            int(row["position"]): int(row["user_id"]) for row in seat_rows
        }
        if set(users_by_position) != {1, 2, 3}:
            raise CompetitionError(
                "INCOMPLETE_TEAM", "All three team seats must remain occupied.", 409
            )
        db.executemany(
            """
            INSERT INTO competition_lineups
              (competition_id, side, game_key, position, player_user_id,
               automatic, submitted_by_user_id, submitted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    competition_id,
                    side,
                    game_key,
                    assignments[game_key],
                    users_by_position[assignments[game_key]],
                    1 if automatic else 0,
                    actor_user_id,
                    now.isoformat(),
                )
                for game_key in ("A", "B", "C")
            ],
        )
        # This pre-reveal event deliberately excludes the secret mapping.
        self._append_event(
            db,
            competition_id,
            "lineup.submitted",
            actor_user_id,
            {"side": side, "automatic": automatic},
        )

    def _finalize_lineups(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        now: datetime,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        revealed_at = now.isoformat()
        db.execute(
            """
            UPDATE competition_lineups SET revealed_at = ?
            WHERE competition_id = ? AND revealed_at IS NULL
            """,
            (revealed_at, competition_id),
        )
        self._append_event(
            db,
            competition_id,
            "lineup.finalized",
            None,
            {"finalized_at": revealed_at},
        )
        phase_token = self._new_phase_token()
        db.execute(
            """
            INSERT INTO competition_match_control
              (competition_id, current_game_key, phase_token, created_at, updated_at)
            VALUES (?, 'A', ?, ?, ?)
            """,
            (competition_id, phase_token, revealed_at, revealed_at),
        )
        db.executemany(
            """
            INSERT INTO competition_game_readiness
              (competition_id, game_key, side)
            VALUES (?, 'A', ?)
            """,
            [
                (competition_id, TeamSide.YELLOW.value),
                (competition_id, TeamSide.WHITE.value),
            ],
        )
        db.executemany(
            """
            INSERT INTO competition_team_clocks
              (competition_id, side, remaining_ms_base, running_since,
               state, revision, updated_at)
            VALUES (?, ?, ?, NULL, 'stopped', 1, ?)
            """,
            [
                (competition_id, TeamSide.YELLOW.value, self._flow(db,competition_id)['team_clock_ms'], revealed_at),
                (competition_id, TeamSide.WHITE.value, self._flow(db,competition_id)['team_clock_ms'], revealed_at),
            ],
        )
        db.execute(
            """
            INSERT INTO competition_suspensions
              (competition_id, active, updated_at)
            VALUES (?, 0, ?)
            """,
            (competition_id, revealed_at),
        )
        self._append_event(
            db,
            competition_id,
            "game.ready_check_started",
            None,
            {"game_key": "A"},
        )
        self._set_hold(db, competition_id, 'GAME_A_READY', now, self.ready_preview_seconds)
        self._set_hold(db, competition_id, 'GAME_A_READY_TIMEOUT', now, self._flow(db,competition_id)['ready_seconds'])
        return self._change_draft_status(
            db, room, CompetitionStatus.GAME_A_READY.value
        )

    def _match_control_row(
        self, db: sqlite3.Connection, competition_id: str
    ) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM competition_match_control WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if row is None:
            raise CompetitionError(
                "MATCH_NOT_INITIALIZED", "Match control is not initialized.", 409
            )
        return row

    def _clock_remaining_ms(self, clock: sqlite3.Row, now: datetime) -> int:
        remaining = int(clock["remaining_ms_base"])
        if str(clock["state"]) != "running":
            return max(0, remaining)
        running_since = parse_time(str(clock["running_since"] or ""))
        if running_since is None:
            return max(0, remaining)
        elapsed = max(0, int((now - running_since).total_seconds() * 1000))
        return max(0, remaining - elapsed)

    def _stop_clock(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        side: str,
        *,
        now: datetime,
        state: str = "stopped",
    ) -> int:
        clock = db.execute(
            """
            SELECT * FROM competition_team_clocks
            WHERE competition_id = ? AND side = ?
            """,
            (competition_id, side),
        ).fetchone()
        if clock is None:
            raise CompetitionError("CLOCK_NOT_INITIALIZED", "Team clock is missing.", 500)
        remaining = self._clock_remaining_ms(clock, now)
        db.execute(
            """
            UPDATE competition_team_clocks
            SET remaining_ms_base = ?, running_since = NULL, state = ?,
                resume_after_suspension = 0,
                revision = revision + 1, updated_at = ?
            WHERE competition_id = ? AND side = ?
            """,
            (remaining, state, now.isoformat(), competition_id, side),
        )
        return remaining

    def _require_match_official(
        self, db: sqlite3.Connection, room: sqlite3.Row, principal: Principal
    ) -> None:
        if self._is_platform_organizer(principal) or int(room['created_by_user_id']) == principal.user_id:
            return
        roles = self._staff_roles(db, str(room["id"]), principal.user_id)
        if not ({StaffRole.ORGANIZER.value, StaffRole.REFEREE.value} & set(roles)):
            raise CompetitionError(
                "REFEREE_REQUIRED", "Organizer or referee permission is required.", 403
            )

    def _suspension_row(
        self, db: sqlite3.Connection, competition_id: str
    ) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM competition_suspensions WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if row is None:
            raise CompetitionError(
                "MATCH_NOT_INITIALIZED", "Match suspension control is missing.", 409
            )
        return row

    def _ensure_not_suspended(
        self, db: sqlite3.Connection, competition_id: str
    ) -> None:
        if bool(self._suspension_row(db, competition_id)["active"]):
            raise CompetitionError(
                "MATCH_SUSPENDED", "The match is suspended by a referee.", 409
            )

    def _require_phase_token(
        self, control: sqlite3.Row, phase_token: str
    ) -> None:
        if not secrets.compare_digest(
            str(control["phase_token"]), str(phase_token or "")
        ):
            raise CompetitionError(
                "STALE_PHASE", "The match phase has changed. Refresh the room state.", 409
            )

    def _normalize_reason(self, value: str) -> str:
        normalized = " ".join(str(value or "").split())
        if not 3 <= len(normalized) <= 500:
            raise CompetitionError(
                "INVALID_REASON", "Reason must contain 3-500 characters."
            )
        return normalized

    def _session_state(
        self, row: sqlite3.Row, db: sqlite3.Connection | None = None,
        *, now: datetime | None = None,
    ) -> ProjectState:
        raw_board = json.loads(str(row["board_json"]))
        started_at = parse_time(str(row["started_at"] or ""))
        ended_at = parse_time(str(row["completed_at"] or ""))
        elapsed_ms = 0
        if started_at:
            elapsed_ms = max(
                0,
                int(((ended_at or datetime.now(timezone.utc)) - started_at).total_seconds() * 1000),
            )
        try:
            extra = json.loads(str(row["adapter_state_json"] or "{}"))
        except (TypeError, ValueError):
            extra = {}
        if db is not None and isinstance(extra, dict) and "project_clock_start_ms" in extra:
            clock = db.execute(
                "SELECT * FROM competition_team_clocks WHERE competition_id = ? AND side = ?",
                (str(row["competition_id"]), str(row["side"])),
            ).fetchone()
            if clock is not None:
                elapsed_ms = max(
                    0, int(extra["project_clock_start_ms"])
                    - self._clock_remaining_ms(clock, now or datetime.now(timezone.utc)),
                )
        if str(row["state"]) == "completed" and extra.get("client_completed"):
            elapsed_ms = int(extra.get("client_elapsed_ms", elapsed_ms))
        return ProjectState(
            board=tuple(tuple(int(value) for value in items) for items in raw_board),
            score=int(row["score"]),
            elapsed_ms=elapsed_ms,
            finished=str(row["state"]) == "completed",
            outcome=str(row["outcome_reason"]) if row["outcome_reason"] else None,
            seed=str(row["seed_hex"]),
            move_count=int(row["move_count"]),
            rng_counter=int(row["rng_counter"]),
            extra=extra if isinstance(extra, dict) else {},
        )

    def _adapter_for_project_row(self, row: sqlite3.Row):
        return self.project_registry.resolve(
            str(row["project_ref"]), str(row["adapter_rules_version"])
        )

    def _adapter_for_session_row(self, row: sqlite3.Row):
        return self.project_registry.resolve(
            str(row["project_ref"]), str(row["rules_version"])
        )

    def _private_session_payload(self, row, db, *, now):
        state = self._session_state(row, db, now=now)
        adapter = self._adapter_for_session_row(row)
        return {
            **client_runtime.public_payload(state),
            "runtime": {
                "protocol": client_runtime.PROTOCOL, "instance_id": str(row["instance_id"]),
                "project_ref": str(row["project_ref"]), "rules_version": str(row["rules_version"]),
                "side": str(row["side"]), "seed": str(row["seed_hex"]),
                "sequence": int(state.extra.get("client_sequence", 0)),
                "team_remaining_at_start_ms": state.extra.get("project_clock_start_ms", 0),
                "checkpoint": state.extra.get("checkpoint"),
                "race_stop_requested": bool(state.extra.get("race_stop_at")),
                "target_tile": getattr(adapter, "target_tile", None),
            },
        }

    def _attach_result_timings(self, db, competition_id, results):
        """Expose submitted race times without re-evaluating any board."""
        by_game = {item['game_key']: item for item in results}
        if not by_game:
            return
        for row in db.execute(
            "SELECT * FROM competition_game_sessions WHERE competition_id=? AND state='completed'",
            (competition_id,),
        ).fetchall():
            result = by_game.get(str(row['game_key']))
            if result is not None:
                state = self._session_state(row, db)
                result[f"{row['side']}_elapsed_ms"] = state.elapsed_ms
                result[f"{row['side']}_outcome"] = state.outcome
        for refund in db.execute('SELECT * FROM competition_time_refunds WHERE competition_id=?',(competition_id,)):
            result=by_game.get(refund['game_key'])
            if result is not None:
                result[f"{refund['side']}_refund_ms"]=refund['amount_ms']
                result[f"{refund['side']}_decisive_ms"]=refund['decisive_ms']
        control=db.execute('SELECT m.*,c.status FROM competition_match_control m JOIN competitions c ON c.id=m.competition_id WHERE m.competition_id=?',(competition_id,)).fetchone()
        official=db.execute("SELECT 1 FROM competition_schedule WHERE competition_id=? AND yellow_team_id!='' AND white_team_id!=''",(competition_id,)).fetchone()
        eligible_finish = bool(control and control['finish_reason'] in ('completed', 'yellow_clock_expired', 'white_clock_expired', 'both_clocks_expired'))
        for result in results:
            result['record_eligible_side']=control['winner_side'] if official and eligible_finish and control['status']=='FINISHED' and control['winner_side'] in ('yellow','white') and result.get('reason') not in ('late_forfeit','referee_override') and result.get(f"{control['winner_side']}_outcome") in ('no_moves','target_reached','tile_limit') else None

    def _publish_game_result(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        game_key: str,
        now: datetime,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        sessions = {
            str(row["side"]): row
            for row in db.execute(
                """
                SELECT * FROM competition_game_sessions
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, game_key),
            ).fetchall()
        }
        if set(sessions) != {TeamSide.YELLOW.value, TeamSide.WHITE.value} or any(
            str(row["state"]) != "completed" for row in sessions.values()
        ):
            raise CompetitionError(
                "GAME_NOT_COMPLETE", "Both project sessions must be complete.", 409
            )
        adapter = self._adapter_for_session_row(sessions[TeamSide.YELLOW.value])
        yellow_state = self._session_state(sessions[TeamSide.YELLOW.value], db, now=now)
        white_state = self._session_state(sessions[TeamSide.WHITE.value], db, now=now)
        yellow_score, white_score, winner, result_reason = client_runtime.resolve_result(
            yellow_state, white_state,
            race=bool(getattr(getattr(adapter, "rules", None), "race", False)),
        )
        if bool(getattr(getattr(adapter,'rules',None),'race',False)):
            targets=[s.elapsed_ms for s in (yellow_state,white_state) if s.outcome=='target_reached']
            if targets:
                stop_ms=min(targets)
                for side,state in [('yellow',yellow_state),('white',white_state)]:
                    db.execute('UPDATE competition_team_clocks SET remaining_ms_base=MAX(0,?),running_since=NULL,state=\'stopped\',revision=revision+1 WHERE competition_id=? AND side=?',
                               (int(state.extra['project_clock_start_ms'])-stop_ms,competition_id,side))
                    if state.outcome!='target_reached':
                        extra={**state.extra,'client_completed':True,'client_elapsed_ms':stop_ms}
                        db.execute('UPDATE competition_game_sessions SET adapter_state_json=? WHERE competition_id=? AND game_key=? AND side=?',(json.dumps(extra),competition_id,game_key,side))
        if result_reason == 'score':
            if getattr(getattr(adapter, 'rules', None), 'result_metric', None) == 'board_sum':
                result_reason = 'board_sum'
            elif adapter.descriptor.view_protocol == 'cargo-transport-v1':
                result_reason = 'delivered_cargo'
        from .final_flow import refund_decision
        for side,state,other in [('yellow',yellow_state,white_state),('white',white_state,yellow_state)]:
            decision=refund_decision(state,other) if winner==side and not bool(getattr(getattr(adapter,'rules',None),'race',False)) else None
            amount=decision[1] if decision else 0
            if amount:
                db.execute('UPDATE competition_team_clocks SET remaining_ms_base=MAX(0,remaining_ms_base+?),revision=revision+1 WHERE competition_id=? AND side=?',(amount,competition_id,side))
                db.execute('INSERT INTO competition_time_refunds VALUES(?,?,?,?,?)',(competition_id,game_key,side,decision[0] if decision else 0,amount))
        db.execute(
            """
            INSERT INTO competition_game_results
              (competition_id, game_key, yellow_score, white_score,
               winner_side, reason, result_revision, published_at)
            VALUES (?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                competition_id,
                game_key,
                yellow_score,
                white_score,
                winner,
                result_reason,
                now.isoformat(),
            ),
        )
        score_column = (
            "yellow_wins"
            if winner == TeamSide.YELLOW.value
            else "white_wins"
            if winner == TeamSide.WHITE.value
            else "draws"
        )
        db.execute(
            f"""
            UPDATE competition_match_control
            SET {score_column} = {score_column} + 1,
                phase_token = ?, updated_at = ?
            WHERE competition_id = ?
            """,
            (self._new_phase_token(), now.isoformat(), competition_id),
        )
        self._append_event(
            db,
            competition_id,
            "game.result_published",
            None,
            {
                "game_key": game_key,
                "yellow_score": yellow_score,
                "white_score": white_score,
                "winner_side": winner,
                "result_revision": 1,
            },
        )
        return self._change_draft_status(db, room, GAME_RESULT_STATUS[game_key])

    def _finish_by_clock_expiry(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        expired_sides: set[str],
        now: datetime,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        control = self._match_control_row(db, competition_id)
        current_game = str(control["current_game_key"])
        for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value):
            self._stop_clock(
                db,
                competition_id,
                side,
                now=now,
                state="expired" if side in expired_sides else "stopped",
            )
        current_index = GAME_KEYS.index(current_game)
        session_scores = {
            str(row["side"]): int(row["score"])
            for row in db.execute(
                """
                SELECT side, score FROM competition_game_sessions
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, current_game),
            ).fetchall()
        }
        if len(expired_sides) == 1:
            expired_side = next(iter(expired_sides))
            default_winner = (
                TeamSide.WHITE.value
                if expired_side == TeamSide.YELLOW.value
                else TeamSide.YELLOW.value
            )
            finish_reason = f"{expired_side}_clock_expired"
        else:
            default_winner = "draw"
            finish_reason = "both_clocks_expired"
        for game_key in GAME_KEYS[current_index:]:
            if db.execute(
                """
                SELECT 1 FROM competition_game_results
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, game_key),
            ).fetchone():
                continue
            db.execute(
                """
                INSERT INTO competition_game_results
                  (competition_id, game_key, yellow_score, white_score,
                   winner_side, reason, result_revision, published_at)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    competition_id,
                    game_key,
                    session_scores.get(TeamSide.YELLOW.value, 0)
                    if game_key == current_game
                    else 0,
                    session_scores.get(TeamSide.WHITE.value, 0)
                    if game_key == current_game
                    else 0,
                    default_winner,
                    finish_reason,
                    now.isoformat(),
                ),
            )
        db.execute(
            """
            UPDATE competition_game_sessions
            SET state = 'completed', outcome_reason = ?, completed_at = ?, updated_at = ?
            WHERE competition_id = ? AND game_key = ? AND state = 'playing'
            """,
            (finish_reason, now.isoformat(), now.isoformat(), competition_id, current_game),
        )
        result_rows = db.execute(
            """
            SELECT winner_side FROM competition_game_results
            WHERE competition_id = ?
            """,
            (competition_id,),
        ).fetchall()
        yellow_wins = sum(
            1 for row in result_rows if str(row["winner_side"]) == TeamSide.YELLOW.value
        )
        white_wins = sum(
            1 for row in result_rows if str(row["winner_side"]) == TeamSide.WHITE.value
        )
        draws = sum(1 for row in result_rows if str(row["winner_side"]) == "draw")
        winner_side = (
            TeamSide.YELLOW.value
            if yellow_wins > white_wins
            else TeamSide.WHITE.value
            if white_wins > yellow_wins
            else "draw"
        )
        db.execute(
            """
            UPDATE competition_match_control
            SET yellow_wins = ?, white_wins = ?, draws = ?, winner_side = ?,
                finish_reason = ?, phase_token = ?, updated_at = ?
            WHERE competition_id = ?
            """,
            (
                yellow_wins,
                white_wins,
                draws,
                winner_side,
                finish_reason,
                self._new_phase_token(),
                now.isoformat(),
                competition_id,
            ),
        )
        self._append_event(
            db,
            competition_id,
            "match.finished",
            None,
            {
                "reason": finish_reason,
                "winner_side": winner_side,
                "expired_sides": sorted(expired_sides),
            },
        )
        return self._change_draft_status(db, room, CompetitionStatus.FINISHED.value)

    def _settle_game_clocks(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        now: datetime,
    ) -> tuple[sqlite3.Row, bool]:
        competition_id = str(room["id"])
        clocks = db.execute(
            "SELECT * FROM competition_team_clocks WHERE competition_id = ?",
            (competition_id,),
        ).fetchall()
        expired = {
            str(clock["side"])
            for clock in clocks
            if str(clock["state"]) == "running"
            and self._clock_remaining_ms(clock, now) == 0
        }
        # The budget itself is unchanged. Local input stops at zero; allow a
        # final checkpoint generated before zero a short transmission window.
        if expired:
            for clock in clocks:
                if str(clock['side']) in expired:
                    deadline = parse_time(clock['running_since']) + timedelta(milliseconds=int(clock['remaining_ms_base']) + 5000)
                    if now < deadline:
                        expired.discard(str(clock['side']))
        if not expired:
            return room, False
        return self._finish_by_clock_expiry(
            db, room, expired_sides=expired, now=now
        ), True

    def _settle_race_ack_deadlines(
        self, db: sqlite3.Connection, room: sqlite3.Row, *, now: datetime
    ) -> tuple[sqlite3.Row, bool]:
        competition_id = str(room["id"])
        if bool(self._suspension_row(db, competition_id)['active']):
            return room, False
        control = self._match_control_row(db, competition_id)
        game_key = str(control["current_game_key"])
        sessions = db.execute(
            """
            SELECT * FROM competition_game_sessions
            WHERE competition_id = ? AND game_key = ?
            """,
            (competition_id, game_key),
        ).fetchall()
        changed = False
        for session in sessions:
            if str(session["state"]) != "playing":
                continue
            # Race peers acknowledge the stop with their final local state.
            # A disconnected peer must not keep the match stuck forever.
            extra = json.loads(session["adapter_state_json"])
            race_stop = parse_time(extra.get("race_stop_at"))
            if race_stop is None or now < race_stop:
                continue
            db.execute(
                """UPDATE competition_game_sessions SET state = 'completed', outcome_reason = 'opponent_finished',
                   completed_at = ?, updated_at = ? WHERE competition_id = ? AND instance_id = ?""",
                (now.isoformat(), now.isoformat(), competition_id, str(session["instance_id"])),
            )
            self._stop_clock(db, competition_id, str(session["side"]), now=now)
            changed = True
        if not changed:
            return room, False
        completed_count = int(db.execute(
            """
            SELECT COUNT(*) AS count FROM competition_game_sessions
            WHERE competition_id = ? AND game_key = ? AND state = 'completed'
            """,
            (competition_id, game_key),
        ).fetchone()["count"])
        if completed_count == 2:
            return self._publish_game_result(db, room, game_key=game_key, now=now), True
        self._touch(db, competition_id)
        return self._room_row(db, str(room["room_code"])), True

    def _settle_due_in_transaction(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        now: datetime,
    ) -> tuple[sqlite3.Row, bool]:
        status = str(room["status"])
        cid = str(room['id'])
        if status in ('SEATING', 'READY_CHECK'):
            return self.schedule.settle(db, room, now)
        if db.execute('SELECT 1 FROM competition_expulsions e JOIN competition_seats s ON s.competition_id=e.competition_id AND s.user_id=e.user_id WHERE e.competition_id=?', (cid,)).fetchone():
            return room, False
        if status == 'C_DRAW' and self._hold_until(db, cid, 'BLIND_CANDIDATES'):
            if now >= parse_time(self._hold_until(db, cid, 'BLIND_CANDIDATES')):
                db.execute("DELETE FROM competition_stage_holds WHERE competition_id=? AND stage='BLIND_CANDIDATES'", (cid,))
                self._touch(db, cid)
                fresh = self._room_row(db, str(room['room_code']))
                settled, _ = self._settle_due_in_transaction(db, fresh, now=now)
                return settled, True
        if status in GAME_RESULT_STATUS.values():
            game = str(self._match_control_row(db, cid)['current_game_key'])
            result = db.execute('SELECT * FROM competition_game_results WHERE competition_id=? AND game_key=?', (cid, game)).fetchone()
            if not self._suspension_row(db, cid)['active'] and now >= parse_time(result['published_at']) + timedelta(seconds=self.result_rest_seconds):
                return self._advance_after_result(db, room, game_key=game, now=now, actor_user_id=None), True
            return room, False
        if status in GAME_READY_STATUS.values():
            game = str(self._match_control_row(db, cid)['current_game_key'])
            expires = self._hold_until(db,cid,status+'_TIMEOUT')
            if expires and now >= parse_time(expires) and not self._suspension_row(db,cid)['active']:
                db.execute('''UPDATE competition_game_readiness SET
                    player_ready_by_user_id=COALESCE(player_ready_by_user_id,(SELECT player_user_id FROM competition_lineups l WHERE l.competition_id=? AND l.game_key=? AND l.side=competition_game_readiness.side)),
                    captain_ready_by_user_id=COALESCE(captain_ready_by_user_id,(SELECT user_id FROM competition_seats s WHERE s.competition_id=? AND s.position=1 AND s.side=competition_game_readiness.side)),
                    player_ready_at=COALESCE(player_ready_at,?),captain_ready_at=COALESCE(captain_ready_at,?)
                    WHERE competition_id=? AND game_key=?''',(cid,game,cid,expires,expires,cid,game))
            window = self._prediction_window(db, room)
            ready_count = db.execute(
                "SELECT COUNT(*) FROM competition_game_readiness WHERE competition_id = ? AND game_key = ? AND player_ready_by_user_id IS NOT NULL AND captain_ready_by_user_id IS NOT NULL",
                (cid, game),
            ).fetchone()[0]
            suspended = self._suspension_row(db, str(room["id"]))['active']
            deadline = self._hold_until(db, cid, status)
            if ready_count == 2 and not suspended and (not deadline or now >= parse_time(deadline)) and (game != 'A' or not window or now >= parse_time(window['minimum_until'])):
                return self._start_game_in_transaction(db, room, game_key=game, now=now, actor_user_id=None), True
            return room, False
        if status in set(GAME_PLAYING_STATUS.values()):
            room, project_changed = self._settle_race_ack_deadlines(db, room, now=now)
            if str(room["status"]) not in set(GAME_PLAYING_STATUS.values()):
                return room, True
            room, clock_changed = self._settle_game_clocks(db, room, now=now)
            return room, project_changed or clock_changed
        if status not in {
            CompetitionStatus.DRAW.value,
            CompetitionStatus.FIRST_PICK_BAN.value,
            CompetitionStatus.SECOND_PICK_BAN.value,
            CompetitionStatus.BLIND_PICK.value,
            CompetitionStatus.C_DRAW.value,
            CompetitionStatus.LINEUP.value,
        }:
            return room, False
        draft = self._draft_row(db, str(room["id"]))
        if status == CompetitionStatus.DRAW.value:
            deadline = parse_time(str(draft["yellow_deadline_at"] or ""))
            if deadline is None or now < deadline:
                return room, False
            return self._start_first_pick(db, room, draft, now=now), True

        if status == CompetitionStatus.C_DRAW.value:
            deadline = parse_time(str(draft["yellow_deadline_at"] or ""))
            if deadline is None or now < deadline:
                return room, False
            return self._start_lineup(db, room, now=now), True

        if status == CompetitionStatus.LINEUP.value:
            competition_id = str(room["id"])
            lineup_state = self._lineup_state_row(db, competition_id)
            changed = False
            for side, deadline_field in (
                (TeamSide.YELLOW.value, "yellow_deadline_at"),
                (TeamSide.WHITE.value, "white_deadline_at"),
            ):
                if self._lineup_submission_exists(db, competition_id, side):
                    continue
                deadline = parse_time(str(lineup_state[deadline_field] or ""))
                if deadline is None or now < deadline:
                    continue
                self._insert_lineup(
                    db,
                    room,
                    side=side,
                    assignments={"A": 1, "B": 2, "C": 3},
                    actor_user_id=None,
                    automatic=True,
                    now=now,
                )
                changed = True
            if changed:
                if all(
                    self._lineup_submission_exists(db, competition_id, side)
                    for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value)
                ):
                    room = self._finalize_lineups(db, room, now=now)
                else:
                    self._touch(db, competition_id)
                    room = db.execute(
                        "SELECT * FROM competitions WHERE id = ?", (competition_id,)
                    ).fetchone()
            return room, changed

        if status in {
            CompetitionStatus.FIRST_PICK_BAN.value,
            CompetitionStatus.SECOND_PICK_BAN.value,
        }:
            first_side = str(draft["first_side"])
            active_side = (
                first_side
                if status == CompetitionStatus.FIRST_PICK_BAN.value
                else (
                    TeamSide.WHITE.value
                    if first_side == TeamSide.YELLOW.value
                    else TeamSide.YELLOW.value
                )
            )
            deadline = parse_time(
                str(
                    draft[
                        "yellow_deadline_at"
                        if active_side == TeamSide.YELLOW.value
                        else "white_deadline_at"
                    ]
                    or ""
                )
            )
            if deadline is None or now < deadline:
                return room, False
            available = self._available_project_keys(db, str(room["id"]), draft)
            if len(available) < 2:
                raise CompetitionError(
                    "PROJECT_POOL_EXHAUSTED", "Not enough projects remain for timeout selection.", 500
                )
            room = self._apply_pick_ban(
                db,
                room,
                draft,
                side=active_side,
                pick_project_key=available[0],
                ban_project_key=available[1],
                actor_user_id=None,
                automatic=True,
                now=now,
            )
            return room, True

        changed = False
        available = self._available_project_keys(db, str(room["id"]), draft)
        if not available:
            raise CompetitionError(
                "PROJECT_POOL_EXHAUSTED", "No project remains for blind selection.", 500
            )
        for side, field, deadline_field in (
            (TeamSide.YELLOW.value, "blind_yellow", "yellow_deadline_at"),
            (TeamSide.WHITE.value, "blind_white", "white_deadline_at"),
        ):
            if draft[field]:
                continue
            deadline = parse_time(str(draft[deadline_field] or ""))
            if deadline is None or now < deadline:
                continue
            db.execute(
                f"UPDATE competition_drafts SET {field} = ?, updated_at = ? WHERE competition_id = ?",
                (available[0], now.isoformat(), room["id"]),
            )
            self._append_event(
                db,
                str(room["id"]),
                "draft.blind_submitted",
                None,
                {"side": side, "automatic": True},
            )
            changed = True
        if changed:
            draft = self._draft_row(db, str(room["id"]))
            if draft["blind_yellow"] and draft["blind_white"]:
                room = self._finalize_blind(db, room, draft, now=now)
            else:
                self._touch(db, str(room["id"]))
                room = db.execute(
                    "SELECT * FROM competitions WHERE id = ?", (room["id"],)
                ).fetchone()
        return room, changed

    def _normalize_room_code(self, room_code: str) -> str:
        value = str(room_code or "").strip().upper()
        if not ROOM_CODE_RE.fullmatch(value):
            raise CompetitionError("ROOM_NOT_FOUND", "Competition room not found.", 404)
        return value

    def _set_hold(self, db, cid, stage, now, seconds):
        db.execute('INSERT OR REPLACE INTO competition_stage_holds VALUES(?,?,?)', (cid, stage, (now + timedelta(seconds=seconds)).isoformat()))

    def _hold_until(self, db, cid, stage):
        row = db.execute('SELECT until_at FROM competition_stage_holds WHERE competition_id=? AND stage=?', (cid, stage)).fetchone()
        return row['until_at'] if row else None

    def _ensure_admitted(self, db, cid, principal):
        if db.execute('SELECT 1 FROM competition_expulsions WHERE competition_id=? AND user_id=?', (cid, principal.user_id)).fetchone():
            raise CompetitionError('REMOVED_FROM_ROOM', 'You have been removed from this room.', 403)

    def manage_member(self, room_code, principal, *, user_id, remove, command_id):
        action = 'member.remove' if remove else 'member.readmit'
        command_id = self._normalize_command_id(command_id)
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            cid = str(room['id'])
            if self._check_command(db, cid, principal, command_id, action):
                return self._snapshot(db, room, principal)
            if not (self._is_platform_organizer(principal) or int(room['created_by_user_id']) == principal.user_id):
                raise CompetitionError('ROOM_MANAGER_REQUIRED', 'Only a tournament administrator or the room owner may manage members.', 403)
            if user_id == principal.user_id:
                raise CompetitionError('CANNOT_REMOVE_SELF', 'Cannot remove yourself.', 409)
            seat = db.execute('SELECT * FROM competition_seats WHERE competition_id=? AND user_id=?', (cid, user_id)).fetchone()
            if remove:
                db.execute('INSERT OR REPLACE INTO competition_expulsions VALUES(?,?,?,?)', (cid, user_id, principal.user_id, now.isoformat()))
                if seat and room['status'] in ('SEATING', 'READY_CHECK'):
                    db.execute('UPDATE competition_scheduled_players SET arrived_at=NULL WHERE competition_id=? AND user_id=?', (cid, user_id))
                    db.execute('DELETE FROM competition_seats WHERE competition_id=? AND user_id=?', (cid, user_id))
                    db.execute('DELETE FROM competition_team_readiness WHERE competition_id=?', (cid,))
                    self._touch(db, cid, status='SEATING')
                elif seat and room['status'] in MATCH_ACTIVE_STATUSES:
                    for clock in db.execute('SELECT * FROM competition_team_clocks WHERE competition_id=?', (cid,)).fetchall():
                        if clock['state'] == 'running':
                            self._stop_clock(db, cid, clock['side'], now=now)
                            db.execute('UPDATE competition_team_clocks SET resume_after_suspension=1 WHERE competition_id=? AND side=?', (cid, clock['side']))
                    db.execute('DELETE FROM competition_suspension_readiness WHERE competition_id=?', (cid,))
                    db.execute("UPDATE competition_suspensions SET active=1,reason_code='other',reason_text='参赛人员被移出，等待管理员处理',started_by_user_id=?,started_by_display_name=?,started_at=?,updated_at=? WHERE competition_id=?", (principal.user_id, principal.display_name, now.isoformat(), now.isoformat(), cid))
            else:
                db.execute('DELETE FROM competition_expulsions WHERE competition_id=? AND user_id=?', (cid, user_id))
            self._append_event(db, cid, action, principal.user_id, {'user_id': user_id})
            self._record_command(db, cid, principal, command_id, action)
            self._touch(db, cid)
            return self._snapshot(db, self._room_row(db, room_code), principal)

    def _normalize_command_id(self, command_id: str) -> str:
        value = str(command_id or "").strip()
        if not COMMAND_ID_RE.fullmatch(value):
            raise CompetitionError(
                "INVALID_COMMAND_ID",
                "command_id must contain 8-160 safe characters.",
            )
        return value

    def _room_row(self, db: sqlite3.Connection, room_code: str) -> sqlite3.Row:
        row = db.execute(
            "SELECT * FROM competitions WHERE room_code = ?",
            (self._normalize_room_code(room_code),),
        ).fetchone()
        if row is None:
            raise CompetitionError("ROOM_NOT_FOUND", "Competition room not found.", 404)
        return row

    def _staff_roles(
        self, db: sqlite3.Connection, competition_id: str, user_id: int
    ) -> list[str]:
        return [
            str(row["role"])
            for row in db.execute(
                "SELECT role FROM competition_staff WHERE competition_id = ? AND user_id = ? ORDER BY role",
                (competition_id, user_id),
            ).fetchall()
        ]

    def _require_room_organizer(
        self, db: sqlite3.Connection, room: sqlite3.Row, principal: Principal
    ) -> None:
        if self._is_platform_organizer(principal):
            return
        roles = self._staff_roles(db, str(room["id"]), principal.user_id)
        if StaffRole.ORGANIZER.value not in roles:
            raise CompetitionError(
                "ORGANIZER_REQUIRED", "Organizer permission is required.", 403
            )

    def _append_event(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        event_type: str,
        actor_user_id: int | None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        row = db.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence FROM competition_events WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        db.execute(
            """
            INSERT INTO competition_events
              (competition_id, sequence, event_type, actor_user_id, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                competition_id,
                sequence,
                event_type,
                actor_user_id,
                json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":")),
                utc_now(),
            ),
        )
        return sequence

    def _touch(self, db: sqlite3.Connection, competition_id: str, *, status: str | None = None) -> None:
        now = utc_now()
        current = db.execute(
            "SELECT status FROM competitions WHERE id = ?", (competition_id,)
        ).fetchone()
        next_status = status or (str(current["status"]) if current else "")
        live_status = next_status not in {
            CompetitionStatus.CREATED.value,
            CompetitionStatus.SEATING.value,
            CompetitionStatus.READY_CHECK.value,
            CompetitionStatus.CANCELLED.value,
        }
        ended = next_status in {
            CompetitionStatus.FINISHED.value,
            CompetitionStatus.CANCELLED.value,
        }
        db.execute(
            """
            UPDATE competitions
            SET status = COALESCE(?, status), version = version + 1,
                live_started_at = CASE
                  WHEN ? AND live_started_at IS NULL THEN ? ELSE live_started_at END,
                live_ended_at = CASE WHEN ? THEN COALESCE(live_ended_at, ?)
                  ELSE live_ended_at END,
                updated_at = ?
            WHERE id = ?
            """,
            (status, int(live_status), now, int(ended), now, now, competition_id),
        )
        room = db.execute(
            """
            SELECT status, version, live_generation, live_started_at
            FROM competitions WHERE id = ?
            """,
            (competition_id,),
        ).fetchone()
        if room and room["live_started_at"]:
            sequence = int(
                db.execute(
                    """
                    SELECT COALESCE(MAX(sequence), 0) + 1 AS value
                    FROM competition_live_outbox
                    WHERE competition_id = ? AND generation = ?
                    """,
                    (competition_id, int(room["live_generation"])),
                ).fetchone()["value"]
            )
            db.execute(
                """
                INSERT INTO competition_live_outbox
                  (competition_id, generation, sequence, event_type,
                   payload_json, created_at)
                VALUES (?, ?, ?, 'projection.changed', ?, ?)
                """,
                (
                    competition_id,
                    int(room["live_generation"]),
                    sequence,
                    json.dumps(
                        {"phase": str(room["status"]), "revision": int(room["version"])},
                        separators=(",", ":"),
                    ),
                    now,
                ),
            )

    def _check_command(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        principal: Principal,
        command_id: str,
        action: str,
    ) -> bool:
        self._ensure_admitted(db, competition_id, principal)
        if (action.startswith(('draft.', 'lineup.', 'game.readiness.', 'team.'))
                or action in {'game.start', 'game.result.confirm', 'game.result.force_advance', 'match.resume'}):
            if db.execute('SELECT 1 FROM competition_expulsions e JOIN competition_seats s ON s.competition_id=e.competition_id AND s.user_id=e.user_id WHERE e.competition_id=?', (competition_id,)).fetchone():
                raise CompetitionError('MEMBER_REMOVAL_HOLD', '参赛人员已被移出，需房主或赛事管理员处理后继续。', 409)
        row = db.execute(
            """
            SELECT action FROM competition_commands
            WHERE competition_id = ? AND user_id = ? AND command_id = ?
            """,
            (competition_id, principal.user_id, command_id),
        ).fetchone()
        if row is None:
            return False
        if str(row["action"]) != action:
            raise CompetitionError(
                "COMMAND_ID_REUSED",
                "This command_id was already used for another action.",
                409,
            )
        return True

    def _record_command(
        self,
        db: sqlite3.Connection,
        competition_id: str,
        principal: Principal,
        command_id: str,
        action: str,
    ) -> None:
        db.execute(
            """
            INSERT INTO competition_commands
              (competition_id, user_id, command_id, action, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (competition_id, principal.user_id, command_id, action, utc_now()),
        )

    def create_competition(
        self,
        principal: Principal,
        *,
        name: str,
        room_code: str | None = None,
        projects: list[dict[str, Any]] | None = None,
        event_slug: str | None = None,
        starts_at: str | None = None,
        yellow_team_id: str | None = None,
        white_team_id: str | None = None,
    ) -> dict[str, Any]:
        with self.database.transaction() as permission_db:
            event_manager = bool(event_slug and self.events._manager(self.events._event(permission_db, event_slug), principal))
        if not self._can_create_competition(principal) and not event_manager:
            raise CompetitionError(
                "ORGANIZER_REQUIRED",
                "Only a platform organizer can create a competition room.",
                403,
            )
        normalized_name = " ".join(str(name or "").split())
        if not 2 <= len(normalized_name) <= 100:
            raise CompetitionError(
                "INVALID_NAME", "Competition name must contain 2-100 characters."
            )
        requested_code = None
        if room_code:
            requested_code = self._normalize_room_code(room_code)
        normalized_projects = self._normalize_projects(projects)
        competition_id = uuid.uuid4().hex
        public_key = self._new_public_key()
        now = utc_now()
        for _attempt in range(12):
            code = requested_code or self._new_room_code()
            try:
                with self.database.transaction(immediate=True) as db:
                    db.execute(
                        """
                        INSERT INTO competitions
                          (id, public_key, room_code, name, status,
                           created_by_user_id, version, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                        """,
                        (
                            competition_id,
                            public_key,
                            code,
                            normalized_name,
                            CompetitionStatus.SEATING.value,
                            principal.user_id,
                            now,
                            now,
                        ),
                    )
                    self._insert_projects(db, competition_id, normalized_projects)
                    db.execute('INSERT INTO competition_flow_rules(competition_id,version,team_clock_ms) VALUES(?,?,?)',
                               (competition_id, '819984-final-v1', self.team_clock_ms))
                    db.execute(
                        """
                        INSERT INTO competition_staff
                          (competition_id, user_id, role, assigned_by_user_id, assigned_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            competition_id,
                            principal.user_id,
                            StaffRole.ORGANIZER.value,
                            principal.user_id,
                            now,
                        ),
                    )
                    self._append_event(
                        db,
                        competition_id,
                        "competition.created",
                        principal.user_id,
                        {
                            "room_code": code,
                            "name": normalized_name,
                            "project_count": len(normalized_projects),
                        },
                    )
                    room = self._room_row(db, code)
                    if event_slug:
                        self.events.link_in_transaction(db, event_slug, room, principal)
                        room = self._room_row(db, code)
                    if starts_at or yellow_team_id or white_team_id:
                        self.schedule.bind(db, room, event_slug, yellow_team_id, white_team_id, starts_at)
                        room = self._room_row(db, code)
                    return self._snapshot(db, room, principal)
            except sqlite3.IntegrityError as exc:
                if requested_code:
                    raise CompetitionError(
                        "ROOM_CODE_TAKEN", "The requested room code is already in use.", 409
                    ) from exc
        raise CompetitionError(
            "ROOM_CODE_EXHAUSTED", "Could not allocate a room code.", 503
        )

    def rematch_before_lineup(self, room_code, principal, *, command_id):
        command_id=self._normalize_command_id(command_id)
        with self.database.transaction(immediate=True) as db:
            room=self._room_row(db,room_code)
            self._require_match_official(db,room,principal)
            previous=db.execute("SELECT payload_json FROM competition_events WHERE competition_id=? AND event_type='competition.rematch'",(room['id'],)).fetchone()
            if previous:
                return self._snapshot(db,self._room_row(db,json.loads(previous[0])['replacement_room_code']),principal)
            if room['status'] not in ('SEATING','READY_CHECK','DRAW','FIRST_PICK_BAN','SECOND_PICK_BAN','BLIND_PICK','C_DRAW'):
                raise CompetitionError('REMATCH_WINDOW_CLOSED','已进入布阵，超过落位错误重赛期限。',409)
            cid,code,stamp=uuid.uuid4().hex,self._new_room_code(),utc_now()
            db.execute('INSERT INTO competitions(id,public_key,room_code,name,status,created_by_user_id,version,created_at,updated_at) VALUES(?,?,?,?,\'SEATING\',?,1,?,?)',
                       (cid,self._new_public_key(),code,room['name'],principal.user_id,stamp,stamp))
            for table in ('competition_projects','competition_flow_rules','competition_scheduled_players','competition_schedule','tournament_room_links'):
                columns=[r[1] for r in db.execute(f'PRAGMA table_info({table})')]
                for source in db.execute(f'SELECT * FROM {table} WHERE competition_id=?',(room['id'],)).fetchall():
                    values=dict(source);values['competition_id']=cid
                    if table=='competition_scheduled_players':values['arrived_at']=None
                    if table=='competition_schedule':values.update(attendance_resolved=0,exception=None,starts_at=max(values['starts_at'],stamp))
                    db.execute(f'INSERT INTO {table} ({",".join(columns)}) VALUES({",".join("?" for _ in columns)})',[values[c] for c in columns])
            db.execute('INSERT INTO competition_staff VALUES(?,?,\'organizer\',?,?)',(cid,principal.user_id,principal.user_id,stamp))
            self._append_event(db,room['id'],'competition.rematch',principal.user_id,{'replacement_room_code':code,'command_id':command_id})
            self._touch(db,room['id'],status='CANCELLED')
            self._append_event(db,cid,'competition.created',principal.user_id,{'replaces':room_code})
            return self._snapshot(db,self._room_row(db,code),principal)

    def close_competition(
        self, room_code: str, principal: Principal, *, command_id: str
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "competition.close"
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            self._require_room_organizer(db, room, principal)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            if str(room["status"]) not in CLOSABLE_ROOM_STATUSES:
                raise CompetitionError(
                    "ROOM_CLOSE_UNAVAILABLE",
                    "A room can only be closed before the draw begins.",
                    409,
                )
            self._append_event(
                db, competition_id, "competition.closed", principal.user_id,
                {"previous_status": str(room["status"])},
            )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id, status=CompetitionStatus.CANCELLED.value)
            return self._snapshot(db, self._room_row(db, room_code), principal)

    def list_competitions(self, principal: Principal) -> list[dict[str, Any]]:
        with self.database.transaction() as db:
            if self._is_platform_organizer(principal):
                rows = db.execute(
                    "SELECT * FROM competitions ORDER BY created_at DESC LIMIT 100"
                ).fetchall()
            else:
                rows = db.execute(
                    """
                    SELECT DISTINCT competition.*
                    FROM competitions AS competition
                    LEFT JOIN competition_staff AS staff
                      ON staff.competition_id = competition.id
                    LEFT JOIN competition_seats AS seat
                      ON seat.competition_id = competition.id
                    WHERE competition.created_by_user_id = ?
                       OR staff.user_id = ?
                       OR seat.user_id = ?
                    ORDER BY competition.created_at DESC
                    LIMIT 100
                    """,
                    (principal.user_id, principal.user_id, principal.user_id),
                ).fetchall()
            return [self._summary(db, row, principal) for row in rows]

    def list_live_rooms(self, *, now: datetime | None = None) -> list[dict[str, Any]]:
        moment = now or datetime.now(timezone.utc)
        cutoff = (moment - timedelta(seconds=self.live_result_retention_seconds)).isoformat()
        with self.database.transaction() as db:
            rows = db.execute(
                """
                SELECT * FROM competitions
                WHERE live_started_at IS NOT NULL
                  AND (live_ended_at IS NULL OR live_ended_at >= ?)
                ORDER BY live_started_at DESC
                """,
                (cutoff,),
            ).fetchall()
            return [self._live_directory_entry(db, row) for row in rows]

    def live_projection(self, public_key: str) -> dict[str, Any]:
        with self.database.transaction() as db:
            room = db.execute(
                "SELECT * FROM competitions WHERE public_key = ?",
                (str(public_key),),
            ).fetchone()
            if room is None or not room["live_started_at"]:
                raise CompetitionError("LIVE_ROOM_NOT_FOUND", "Live match not found.", 404)
            if room["live_ended_at"]:
                ended = parse_time(str(room["live_ended_at"]))
                if ended and datetime.now(timezone.utc) - ended > timedelta(
                    seconds=self.live_result_retention_seconds
                ):
                    raise CompetitionError("LIVE_ROOM_NOT_FOUND", "Live match not found.", 404)
            return self._public_match_projection(db, room)

    def _prediction_window(self, db, room):
        row = db.execute("SELECT * FROM competition_prediction_windows WHERE competition_id = ?", (str(room['id']),)).fetchone()
        if row is None:
            return None
        return {"opened_at": row['opened_at'], "minimum_until": row['minimum_until'],
                "closed_at": row['closed_at'], "open": not row['closed_at'] and str(room['status']) in {
                    'FIRST_PICK_BAN', 'SECOND_PICK_BAN', 'BLIND_PICK', 'C_DRAW', 'LINEUP', 'GAME_A_READY'}}

    def live_prediction_facts(self, public_key):
        """Authenticated settlement facts remain available after lobby retention."""
        with self.database.transaction() as db:
            room = db.execute("SELECT * FROM competitions WHERE public_key = ?", (public_key,)).fetchone()
            if room is None or not room['live_started_at']:
                raise CompetitionError('LIVE_ROOM_NOT_FOUND', 'Live match not found.', 404)
            projection = self._public_match_projection(db, room)
            return {key: projection[key] for key in (
                'match_public_key', 'generation', 'content_sequence', 'phase', 'prediction_window',
                'teams', 'public_result', 'suspended', 'server_time')}

    def _live_directory_entry(
        self, db: sqlite3.Connection, room: sqlite3.Row
    ) -> dict[str, Any]:
        projection = self._public_match_projection(db, room)
        score = projection["score"]
        current = next((game for game in projection['games'] if game['game_key'] == projection['current_game']), None)
        project = next((item for item in projection['projects'] if current and item['key'] == current['project_key']), None)
        team_names = {side: ' / '.join(item['display_name'] for item in projection['teams'][side]['roster'])
                      for side in ('yellow', 'white')}
        return {
            "room_id": f"competition-{room['public_key']}",
            "public_key": str(room["public_key"]),
            "generation": int(room["live_generation"]),
            "content_sequence": int(projection["content_sequence"]),
            "content_kind": "competition-match",
            "protocol": "competition-match-v1",
            "category_label": {"zh": "赛事直播", "en": "TOURNAMENT"},
            "badge": {"zh": "已结束" if room['live_ended_at'] else "赛事直播", "en": "ENDED" if room['live_ended_at'] else "TOURNAMENT"},
            "title": {"zh": str(room["name"]), "en": str(room["name"])},
            "subtitle": {
                "zh": f"{team_names['yellow']} 对阵 {team_names['white']}",
                "en": f"{team_names['yellow']} vs {team_names['white']}",
            },
            "preview": {
                "kind": "competition-score",
                "phase": projection["phase"],
                "yellow_score": score["yellow"],
                "white_score": score["white"],
                "teams": projection['teams'],
                "current_project": project,
                "current_game": projection['current_game'],
                "prediction_open": bool((projection['prediction_window'] or {}).get('open')),
            },
            "started_at": str(room["live_started_at"]),
            "ended": bool(room["live_ended_at"]),
        }

    def _public_match_projection(
        self, db: sqlite3.Connection, room: sqlite3.Row
    ) -> dict[str, Any]:
        competition_id = str(room["id"])
        status = str(room["status"])
        sequence_row = db.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) AS value
            FROM competition_live_outbox
            WHERE competition_id = ? AND generation = ?
            """,
            (competition_id, int(room["live_generation"])),
        ).fetchone()
        seat_rows = db.execute(
            """
            SELECT side, position, user_id, display_name_snapshot
            FROM competition_seats WHERE competition_id = ?
            ORDER BY CASE side WHEN 'yellow' THEN 0 ELSE 1 END, position
            """,
            (competition_id,),
        ).fetchall()
        avatar_urls = avatar_urls_for_users(row["user_id"] for row in seat_rows)
        roster = {
            side: [
                {
                    "position": int(row["position"]),
                    "display_name": str(row["display_name_snapshot"]),
                    "avatar_url": avatar_urls.get(int(row["user_id"])),
                    "is_captain": int(row["position"]) == 1,
                }
                for row in seat_rows
                if str(row["side"]) == side
            ]
            for side in ("yellow", "white")
        }
        projects = [
            {
                "key": str(row["project_key"]),
                "name": str(row["name"]),
                "description": str(row["description"]),
                "project_ref": str(row["project_ref"]),
                "sort_order": int(row["sort_order"]),
                "adapter": json.loads(str(row["adapter_snapshot_json"] or "{}")),
            }
            for row in db.execute(
                """
                SELECT * FROM competition_projects
                WHERE competition_id = ? AND enabled = 1 ORDER BY sort_order
                """,
                (competition_id,),
            ).fetchall()
        ]
        project_names = {item["key"]: item["name"] for item in projects}
        draft = db.execute(
            "SELECT * FROM competition_drafts WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        public_draft = None
        phase_timing: dict[str, Any] | None = None
        if draft is not None:
            public_draft = {
                "first_side": str(draft["first_side"]),
                "project_a": draft["project_a"],
                "ban_m": draft["ban_m"],
                "project_b": draft["project_b"],
                "ban_n": draft["ban_n"],
                "blind_submissions": {
                    "yellow": bool(draft["blind_yellow"]),
                    "white": bool(draft["blind_white"]),
                },
                "project_c": draft["project_c"],
            }
            if draft["project_c"]:
                public_draft["blind_choices"] = {
                    "yellow": str(draft["blind_yellow"]),
                    "white": str(draft["blind_white"]),
                }
            public_draft['c_reveal_at'] = self._hold_until(db, competition_id, 'BLIND_CANDIDATES') if status == 'C_DRAW' else None
            if public_draft['c_reveal_at']:
                public_draft['project_c'] = None
            draft_deadlines = {
                "yellow": str(draft["yellow_deadline_at"])
                if draft["yellow_deadline_at"]
                else None,
                "white": str(draft["white_deadline_at"])
                if draft["white_deadline_at"]
                else None,
            }
            first_side = str(draft["first_side"])
            second_side = (
                TeamSide.WHITE.value
                if first_side == TeamSide.YELLOW.value
                else TeamSide.YELLOW.value
            )
            if status in {
                CompetitionStatus.DRAW.value,
                CompetitionStatus.FIRST_PICK_BAN.value,
                CompetitionStatus.SECOND_PICK_BAN.value,
                CompetitionStatus.BLIND_PICK.value,
                CompetitionStatus.C_DRAW.value,
            }:
                active_side = None
                mode = "reveal"
                if status == CompetitionStatus.FIRST_PICK_BAN.value:
                    active_side = first_side
                    mode = "turn"
                elif status == CompetitionStatus.SECOND_PICK_BAN.value:
                    active_side = second_side
                    mode = "turn"
                elif status == CompetitionStatus.BLIND_PICK.value:
                    mode = "simultaneous"
                deadline_candidates = [
                    value for value in draft_deadlines.values() if value is not None
                ]
                phase_timing = {
                    "mode": mode,
                    "started_at": str(draft["phase_started_at"]),
                    "deadline_at": max(deadline_candidates)
                    if deadline_candidates
                    else None,
                    "deadlines": draft_deadlines,
                    "active_side": active_side,
                }
        lineup_rows = db.execute(
            """
            SELECT side, game_key, position, automatic, revealed_at
            FROM competition_lineups WHERE competition_id = ?
            ORDER BY side, game_key
            """,
            (competition_id,),
        ).fetchall()
        lineup_status = {
            side: {
                "submitted": sum(1 for row in lineup_rows if row["side"] == side) == 3,
                "revealed": any(
                    row["side"] == side and bool(row["revealed_at"])
                    for row in lineup_rows
                ),
            }
            for side in ("yellow", "white")
        }
        lineup_state = db.execute(
            "SELECT * FROM competition_lineup_state WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        if status == CompetitionStatus.LINEUP.value and lineup_state is not None:
            lineup_deadlines = {
                "yellow": str(lineup_state["yellow_deadline_at"]),
                "white": str(lineup_state["white_deadline_at"]),
            }
            phase_timing = {
                "mode": "simultaneous",
                "started_at": str(lineup_state["phase_started_at"]),
                "deadline_at": max(lineup_deadlines.values()),
                "deadlines": lineup_deadlines,
                "active_side": None,
            }
        revealed_players: dict[str, dict[str, Any]] = {"yellow": {}, "white": {}}
        seat_names = {
            (str(row["side"]), int(row["position"])): str(row["display_name_snapshot"])
            for row in seat_rows
        }
        seat_avatars = {
            (str(row["side"]), int(row["position"])): avatar_urls.get(int(row["user_id"]))
            for row in seat_rows
        }
        control = db.execute(
            "SELECT * FROM competition_match_control WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        current_game = str(control["current_game_key"]) if control else None
        game_readiness = {
            side: {"player_ready": False, "captain_ready": False}
            for side in ("yellow", "white")
        }
        if current_game and status == GAME_READY_STATUS[current_game]:
            for row in db.execute(
                """
                SELECT side, player_ready_by_user_id, captain_ready_by_user_id
                FROM competition_game_readiness
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, current_game),
            ).fetchall():
                game_readiness[str(row["side"])] = {
                    "player_ready": bool(row["player_ready_by_user_id"]),
                    "captain_ready": bool(row["captain_ready_by_user_id"]),
                }
        public_game_keys = set(GAME_KEYS)
        for row in lineup_rows:
            if row["revealed_at"] and str(row["game_key"]) in public_game_keys:
                side = str(row["side"])
                position = int(row["position"])
                revealed_players[side][str(row["game_key"])] = {
                    "position": position,
                    "display_name": seat_names.get((side, position), "Unknown player"),
                    "avatar_url": seat_avatars.get((side, position)),
                    "automatic": bool(row["automatic"]),
                }
        results = [dict(row) for row in db.execute(
            "SELECT * FROM competition_game_results WHERE competition_id = ? ORDER BY game_key",
            (competition_id,),
        ).fetchall()]
        public_results = [
            {
                "game_key": str(row["game_key"]),
                "yellow_score": int(row["yellow_score"]),
                "white_score": int(row["white_score"]),
                "winner_side": str(row["winner_side"]),
                "reason": str(row["reason"]),
                "result_revision": int(row["result_revision"]),
                "corrected": bool(row["corrected_at"]),
                "published_at": str(row['published_at']),
            }
            for row in results
        ]
        self._attach_result_timings(db, competition_id, public_results)
        score = {
            "yellow": int(control["yellow_wins"]) if control else 0,
            "white": int(control["white_wins"]) if control else 0,
            "draws": int(control["draws"]) if control else 0,
        }
        confirmations = {"yellow": False, "white": False}
        if control:
            for row in db.execute(
                """
                SELECT side FROM competition_result_confirmations
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, current_game),
            ).fetchall():
                confirmations[str(row["side"])] = True
        now = datetime.now(timezone.utc)
        clocks: dict[str, Any] = {}
        for clock in db.execute(
            "SELECT * FROM competition_team_clocks WHERE competition_id = ?",
            (competition_id,),
        ).fetchall():
            clocks[str(clock["side"])] = {
                "state": str(clock["state"]),
                "remaining_ms": self._clock_remaining_ms(clock, now),
                "revision": int(clock["revision"]),
            }
        sessions = db.execute(
            """
            SELECT * FROM competition_game_sessions
            WHERE competition_id = ? AND game_key = ? ORDER BY side
            """,
            (competition_id, current_game),
        ).fetchall() if current_game else []
        session_status: dict[str, Any] = {}
        project_views: dict[str, Any] = {}
        for session in sessions:
            side = str(session["side"])
            session_status[side] = {
                "state": str(session["state"]),
                "finished": str(session["state"]) == "completed",
            }
            try:
                project_views[side] = client_runtime.public_view(
                    self._adapter_for_session_row(session), self._session_state(session, db, now=now),
                    int(session["public_generation"]),
                )
            except Exception:
                # A public renderer failure cannot affect authoritative match state.
                project_views[side] = None
        games = []
        for game_key in GAME_KEYS:
            project_key = (
                str(draft[f"project_{game_key.lower()}"])
                if draft is not None and draft[f"project_{game_key.lower()}"]
                else None
            )
            result = next((item for item in public_results if item["game_key"] == game_key), None)
            if game_key == 'C' and public_draft and public_draft.get('c_reveal_at'):
                project_key = None
            games.append({
                "game_key": game_key,
                "project_key": project_key,
                "project_name": project_names.get(project_key),
                "players": {
                    side: revealed_players[side].get(game_key)
                    for side in ("yellow", "white")
                },
                "result": result,
            })
        schedule = self.schedule.view(db, competition_id) or {}
        return {
            "match_public_key": str(room["public_key"]),
            "generation": int(room["live_generation"]),
            "content_sequence": int(sequence_row["value"]),
            "phase": status,
            "phase_timing": phase_timing,
            "prediction_window": self._prediction_window(db, room),
            "name": str(room["name"]),
            "teams": {
                "yellow": {"name": schedule.get('yellow_name') or '黄方', "roster": roster["yellow"]},
                "white": {"name": schedule.get('white_name') or '白方', "roster": roster["white"]},
            },
            "projects": projects,
            "public_draft": public_draft,
            "lineup_submission_status": lineup_status,
            "revealed_players": revealed_players,
            "games": games,
            "current_game": current_game,
            "game_readiness": game_readiness,
            "preview_until": self._hold_until(db, competition_id, status),
            "ready_deadline_at": self._hold_until(db, competition_id, status+'_TIMEOUT'),
            "member_hold": bool(db.execute('SELECT 1 FROM competition_expulsions e JOIN competition_seats s ON s.competition_id=e.competition_id AND s.user_id=e.user_id WHERE e.competition_id=?', (competition_id,)).fetchone()),
            "rest_until": next(((parse_time(item['published_at']) + timedelta(seconds=self.result_rest_seconds)).isoformat() for item in public_results if item['game_key'] == current_game), None),
            "score": score,
            "series_points": {'yellow':2*score['yellow']+score.get('draws',0),'white':2*score['white']+score.get('draws',0)},
            "team_clocks": clocks,
            "captain_confirmation_status": confirmations,
            "session_status": session_status,
            "public_result": {
                "games": public_results,
                "winner_side": str(control["winner_side"]) if control and control["winner_side"] else None,
                "finish_reason": str(control["finish_reason"]) if control and control["finish_reason"] else None,
            },
            "project_public_views": project_views,
            "suspended": bool(
                db.execute(
                    "SELECT active FROM competition_suspensions WHERE competition_id = ?",
                    (competition_id,),
                ).fetchone()["active"]
            ) if control else False,
            "server_time": now.isoformat(),
        }

    def snapshot(self, room_code: str, principal: Principal) -> dict[str, Any]:
        self.settle_deadline(room_code)
        with self.database.transaction() as db:
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def settle_deadline(
        self, room_code: str, *, now: datetime | None = None
    ) -> bool:
        moment = now or datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            _room, changed = self._settle_due_in_transaction(db, room, now=moment)
            return changed

    def process_due_rooms(self, *, now: datetime | None = None) -> list[str]:
        moment = now or datetime.now(timezone.utc)
        with self.database.transaction() as db:
            codes = [
                str(row["room_code"])
                for row in db.execute(
                    """
                    SELECT room_code FROM competitions
                    WHERE status IN (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SEATING', 'READY_CHECK')
                    ORDER BY updated_at
                    """,
                    (
                        CompetitionStatus.DRAW.value,
                        CompetitionStatus.FIRST_PICK_BAN.value,
                        CompetitionStatus.SECOND_PICK_BAN.value,
                        CompetitionStatus.BLIND_PICK.value,
                        CompetitionStatus.C_DRAW.value,
                        CompetitionStatus.LINEUP.value,
                        CompetitionStatus.GAME_A_READY.value,
                        CompetitionStatus.GAME_A_PLAYING.value,
                        CompetitionStatus.GAME_B_PLAYING.value,
                        CompetitionStatus.GAME_C_PLAYING.value,
                        'GAME_B_READY', 'GAME_C_READY', 'GAME_A_RESULT', 'GAME_B_RESULT', 'GAME_C_RESULT',
                    ),
                ).fetchall()
            ]
        changed: list[str] = []
        for code in codes:
            if self.settle_deadline(code, now=moment):
                changed.append(code)
        return changed

    def assign_staff(
        self,
        room_code: str,
        principal: Principal,
        *,
        user_id: int,
        role: str,
    ) -> dict[str, Any]:
        try:
            staff_role = StaffRole(str(role).lower())
        except ValueError as exc:
            raise CompetitionError("INVALID_STAFF_ROLE", "Invalid staff role.") from exc
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            self._require_room_organizer(db, room, principal)
            db.execute(
                """
                INSERT OR IGNORE INTO competition_staff
                  (competition_id, user_id, role, assigned_by_user_id, assigned_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    room["id"],
                    int(user_id),
                    staff_role.value,
                    principal.user_id,
                    utc_now(),
                ),
            )
            self._append_event(
                db,
                str(room["id"]),
                "staff.assigned",
                principal.user_id,
                {"user_id": int(user_id), "role": staff_role.value},
            )
            self._touch(db, str(room["id"]))
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def claim_seat(
        self,
        room_code: str,
        principal: Principal,
        *,
        side: str,
        position: int,
        command_id: str,
    ) -> dict[str, Any]:
        try:
            team_side = TeamSide(str(side).lower())
        except ValueError as exc:
            raise CompetitionError("INVALID_SIDE", "Invalid team side.") from exc
        if int(position) not in SEAT_POSITIONS:
            raise CompetitionError("INVALID_POSITION", "Seat position must be 1, 2 or 3.")
        normalized_command = self._normalize_command_id(command_id)
        action = "seat.claim"
        self.settle_deadline(room_code)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            schedule = self.schedule.view(db, competition_id)
            if schedule and schedule['players']:
                if not any(p['user_id']==principal.user_id and p['side']==team_side.value and p['position']==int(position) for p in schedule['players']):
                    raise CompetitionError('SCHEDULE_SEATS_FIXED', '请按报名表队内序号落座。', 409)
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            if room["status"] not in {
                CompetitionStatus.SEATING.value,
                CompetitionStatus.READY_CHECK.value,
            }:
                raise CompetitionError(
                    "SEATING_CLOSED", "Seat selection is closed for this room.", 409
                )
            target = db.execute(
                """
                SELECT user_id FROM competition_seats
                WHERE competition_id = ? AND side = ? AND position = ?
                """,
                (competition_id, team_side.value, int(position)),
            ).fetchone()
            if target is not None and int(target["user_id"]) != principal.user_id:
                raise CompetitionError("SEAT_TAKEN", "This seat is already occupied.", 409)
            current = db.execute(
                "SELECT side, position FROM competition_seats WHERE competition_id = ? AND user_id = ?",
                (competition_id, principal.user_id),
            ).fetchone()
            if (
                current is not None
                and str(current["side"]) == team_side.value
                and int(current["position"]) == int(position)
            ):
                self._record_command(
                    db, competition_id, principal, normalized_command, action
                )
                return self._snapshot(db, room, principal)
            db.execute('DELETE FROM competition_team_readiness WHERE competition_id=? AND (side=? OR side IN (SELECT side FROM competition_seats WHERE competition_id=? AND user_id=?))',
                       (competition_id,team_side.value,competition_id,principal.user_id))
            if current is not None:
                db.execute(
                    "DELETE FROM competition_seats WHERE competition_id = ? AND user_id = ?",
                    (competition_id, principal.user_id),
                )
            db.execute(
                """
                INSERT INTO competition_seats
                  (competition_id, side, position, user_id, display_name_snapshot, seated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition_id,
                    team_side.value,
                    int(position),
                    principal.user_id,
                    principal.display_name,
                    utc_now(),
                ),
            )
            occupied = int(
                db.execute(
                    "SELECT COUNT(*) AS count FROM competition_seats WHERE competition_id = ?",
                    (competition_id,),
                ).fetchone()["count"]
            )
            next_status = (
                CompetitionStatus.READY_CHECK.value
                if occupied == 6
                else CompetitionStatus.SEATING.value
            )
            self._append_event(
                db,
                competition_id,
                "seat.claimed",
                principal.user_id,
                {
                    "side": team_side.value,
                    "position": int(position),
                    "previous_side": str(current["side"]) if current else None,
                    "previous_position": int(current["position"]) if current else None,
                    "display_name": principal.display_name,
                },
            )
            if next_status != room["status"]:
                self._append_event(
                    db,
                    competition_id,
                    "competition.status_changed",
                    None,
                    {"from": str(room["status"]), "to": next_status},
                )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id, status=next_status)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def leave_seat(
        self,
        room_code: str,
        principal: Principal,
        *,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "seat.leave"
        self.settle_deadline(room_code)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            if room["status"] not in {
                CompetitionStatus.SEATING.value,
                CompetitionStatus.READY_CHECK.value,
            }:
                raise CompetitionError(
                    "SEATING_CLOSED", "Seat selection is closed for this room.", 409
                )
            current = db.execute(
                "SELECT side, position FROM competition_seats WHERE competition_id = ? AND user_id = ?",
                (competition_id, principal.user_id),
            ).fetchone()
            if current is None:
                raise CompetitionError("NOT_SEATED", "You do not occupy a seat.", 409)
            db.execute('DELETE FROM competition_team_readiness WHERE competition_id=? AND side=?',(competition_id,current['side']))
            db.execute(
                "DELETE FROM competition_seats WHERE competition_id = ? AND user_id = ?",
                (competition_id, principal.user_id),
            )
            self._append_event(
                db,
                competition_id,
                "seat.left",
                principal.user_id,
                {"side": str(current["side"]), "position": int(current["position"])},
            )
            if room["status"] != CompetitionStatus.SEATING.value:
                self._append_event(
                    db,
                    competition_id,
                    "competition.status_changed",
                    None,
                    {
                        "from": str(room["status"]),
                        "to": CompetitionStatus.SEATING.value,
                    },
                )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id, status=CompetitionStatus.SEATING.value)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def set_ready(
        self,
        room_code: str,
        principal: Principal,
        *,
        ready: bool,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "team.ready" if ready else "team.unready"
        self.settle_deadline(room_code)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            if room["status"] not in ('SEATING','READY_CHECK'):
                raise CompetitionError(
                    "READY_CHECK_CLOSED",
                    "The room is not accepting readiness changes.",
                    409,
                )
            captain = db.execute(
                """
                SELECT side FROM competition_seats
                WHERE competition_id = ? AND user_id = ? AND position = 1
                """,
                (competition_id, principal.user_id),
            ).fetchone()
            if captain is None:
                raise CompetitionError(
                    "CAPTAIN_REQUIRED", "Only a player in seat 1 can change team readiness.", 403
                )
            side = str(captain["side"])
            if db.execute('SELECT COUNT(*) FROM competition_seats WHERE competition_id=? AND side=?',(competition_id,side)).fetchone()[0] != 3:
                raise CompetitionError('SEATS_INCOMPLETE','本队三位选手均需落座。',409)
            if ready:
                db.execute(
                    """
                    INSERT OR REPLACE INTO competition_team_readiness
                      (competition_id, side, ready_by_user_id, ready_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (competition_id, side, principal.user_id, utc_now()),
                )
                event_type = "team.ready"
            else:
                db.execute(
                    "DELETE FROM competition_team_readiness WHERE competition_id = ? AND side = ?",
                    (competition_id, side),
                )
                event_type = "team.unready"
            self._append_event(
                db,
                competition_id,
                event_type,
                principal.user_id,
                {"side": side},
            )
            ready_count = int(
                db.execute(
                    "SELECT COUNT(*) AS count FROM competition_team_readiness WHERE competition_id = ?",
                    (competition_id,),
                ).fetchone()["count"]
            )
            next_status = (
                CompetitionStatus.DRAW.value
                if ready_count == 2 and self.schedule.may_draw(db, competition_id, datetime.now(timezone.utc))
                else CompetitionStatus.READY_CHECK.value
            )
            if next_status != room["status"]:
                self._append_event(
                    db,
                    competition_id,
                    "competition.status_changed",
                    None,
                    {"from": str(room["status"]), "to": next_status},
                )
            if (
                next_status == CompetitionStatus.DRAW.value
                and room["status"] != CompetitionStatus.DRAW.value
            ):
                self._initialize_draw(
                    db,
                    competition_id,
                    now=datetime.now(timezone.utc),
                )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id, status=next_status)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def submit_pick_ban(
        self,
        room_code: str,
        principal: Principal,
        *,
        pick_project_key: str,
        ban_project_key: str,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        pick_key = str(pick_project_key or "").strip().lower()
        ban_key = str(ban_project_key or "").strip().lower()
        if pick_key == ban_key:
            raise CompetitionError(
                "PICK_BAN_CONFLICT", "The selected and banned projects must differ."
            )
        normalized_command = self._normalize_command_id(command_id)
        action = "draft.pick_ban"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            room, _changed = self._settle_due_in_transaction(db, room, now=now)
            status = str(room["status"])
            if status not in {
                CompetitionStatus.FIRST_PICK_BAN.value,
                CompetitionStatus.SECOND_PICK_BAN.value,
            }:
                raise CompetitionError(
                    "INVALID_DRAFT_PHASE", "Pick/BAN is not active.", 409
                )
            draft = self._draft_row(db, competition_id)
            if not secrets.compare_digest(
                str(draft["phase_token"]), str(phase_token or "")
            ):
                raise CompetitionError(
                    "STALE_PHASE", "The draft phase has changed. Refresh the room state.", 409
                )
            captain_side = self._captain_side(db, competition_id, principal)
            first_side = str(draft["first_side"])
            active_side = (
                first_side
                if status == CompetitionStatus.FIRST_PICK_BAN.value
                else (
                    TeamSide.WHITE.value
                    if first_side == TeamSide.YELLOW.value
                    else TeamSide.YELLOW.value
                )
            )
            if captain_side != active_side:
                raise CompetitionError(
                    "NOT_ACTIVE_SIDE", "The other team is currently drafting.", 403
                )
            available = set(self._available_project_keys(db, competition_id, draft))
            if pick_key not in available or ban_key not in available:
                raise CompetitionError(
                    "PROJECT_UNAVAILABLE", "One or more projects are no longer available.", 409
                )
            room = self._apply_pick_ban(
                db,
                room,
                draft,
                side=captain_side,
                pick_project_key=pick_key,
                ban_project_key=ban_key,
                actor_user_id=principal.user_id,
                automatic=False,
                now=now,
            )
            self._record_command(
                db, competition_id, principal, normalized_command, action
            )
            return self._snapshot(db, room, principal)

    def submit_blind_pick(
        self,
        room_code: str,
        principal: Principal,
        *,
        project_key: str,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        selected_key = str(project_key or "").strip().lower()
        normalized_command = self._normalize_command_id(command_id)
        action = "draft.blind"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            room, _changed = self._settle_due_in_transaction(db, room, now=now)
            if room["status"] != CompetitionStatus.BLIND_PICK.value:
                raise CompetitionError(
                    "INVALID_DRAFT_PHASE", "Blind selection is not active.", 409
                )
            draft = self._draft_row(db, competition_id)
            if not secrets.compare_digest(
                str(draft["phase_token"]), str(phase_token or "")
            ):
                raise CompetitionError(
                    "STALE_PHASE", "The draft phase has changed. Refresh the room state.", 409
                )
            side = self._captain_side(db, competition_id, principal)
            field = (
                "blind_yellow"
                if side == TeamSide.YELLOW.value
                else "blind_white"
            )
            if draft[field]:
                raise CompetitionError(
                    "BLIND_ALREADY_SUBMITTED", "Your team already submitted its blind choice.", 409
                )
            available = set(self._available_project_keys(db, competition_id, draft))
            if selected_key not in available:
                raise CompetitionError(
                    "PROJECT_UNAVAILABLE", "This project is not available for blind selection.", 409
                )
            db.execute(
                f"UPDATE competition_drafts SET {field} = ?, updated_at = ? WHERE competition_id = ?",
                (selected_key, now.isoformat(), competition_id),
            )
            self._append_event(
                db,
                competition_id,
                "draft.blind_submitted",
                principal.user_id,
                {"side": side, "automatic": False},
            )
            self._record_command(
                db, competition_id, principal, normalized_command, action
            )
            draft = self._draft_row(db, competition_id)
            if draft["blind_yellow"] and draft["blind_white"]:
                room = self._finalize_blind(db, room, draft, now=now)
            else:
                self._touch(db, competition_id)
                room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def submit_lineup(
        self,
        room_code: str,
        principal: Principal,
        *,
        assignments: dict[str, int],
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        try:
            normalized_assignments = {
                str(game_key).upper(): int(position)
                for game_key, position in assignments.items()
            }
        except (AttributeError, TypeError, ValueError) as exc:
            raise CompetitionError(
                "INVALID_LINEUP", "Lineup must assign games A, B and C to positions 1, 2 and 3."
            ) from exc
        if (
            len(assignments) != 3
            or set(normalized_assignments) != {"A", "B", "C"}
            or sorted(normalized_assignments.values()) != [1, 2, 3]
        ):
            raise CompetitionError(
                "INVALID_LINEUP",
                "Each game A, B and C must use one distinct team position.",
            )
        normalized_command = self._normalize_command_id(command_id)
        action = "lineup.submit"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            room, _changed = self._settle_due_in_transaction(db, room, now=now)
            if room["status"] != CompetitionStatus.LINEUP.value:
                raise CompetitionError(
                    "INVALID_LINEUP_PHASE", "Secret lineup is not active.", 409
                )
            lineup_state = self._lineup_state_row(db, competition_id)
            if not secrets.compare_digest(
                str(lineup_state["phase_token"]), str(phase_token or "")
            ):
                raise CompetitionError(
                    "STALE_PHASE", "The lineup phase has changed. Refresh the room state.", 409
                )
            side = self._captain_side(db, competition_id, principal)
            if self._lineup_submission_exists(db, competition_id, side):
                raise CompetitionError(
                    "LINEUP_ALREADY_SUBMITTED",
                    "Your team already submitted its lineup.",
                    409,
                )
            self._insert_lineup(
                db,
                room,
                side=side,
                assignments=normalized_assignments,
                actor_user_id=principal.user_id,
                automatic=False,
                now=now,
            )
            self._record_command(
                db, competition_id, principal, normalized_command, action
            )
            if all(
                self._lineup_submission_exists(db, competition_id, team_side)
                for team_side in (TeamSide.YELLOW.value, TeamSide.WHITE.value)
            ):
                room = self._finalize_lineups(db, room, now=now)
            else:
                self._touch(db, competition_id)
                room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def set_game_readiness(
        self,
        room_code: str,
        principal: Principal,
        *,
        readiness_role: str,
        ready: bool,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        role = str(readiness_role or "").strip().lower()
        if role not in {"player", "captain"}:
            raise CompetitionError(
                "INVALID_READINESS_ROLE", "Readiness role must be player or captain."
            )
        normalized_command = self._normalize_command_id(command_id)
        action = f"game.readiness.{role}"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            control = self._match_control_row(db, competition_id)
            self._ensure_not_suspended(db, competition_id)
            game_key = str(control["current_game_key"])
            if str(room["status"]) != GAME_READY_STATUS[game_key]:
                raise CompetitionError(
                    "INVALID_GAME_PHASE", "The current game is not in ready check.", 409
                )
            if not secrets.compare_digest(
                str(control["phase_token"]), str(phase_token or "")
            ):
                raise CompetitionError(
                    "STALE_PHASE", "The game phase has changed. Refresh the room state.", 409
                )
            if role == "captain":
                side = self._captain_side(db, competition_id, principal)
                by_column = "captain_ready_by_user_id"
                at_column = "captain_ready_at"
            else:
                lineup = db.execute(
                    """
                    SELECT side FROM competition_lineups
                    WHERE competition_id = ? AND game_key = ? AND player_user_id = ?
                    """,
                    (competition_id, game_key, principal.user_id),
                ).fetchone()
                if lineup is None:
                    raise CompetitionError(
                        "ACTIVE_PLAYER_REQUIRED",
                        "Only the player assigned to this game can mark player readiness.",
                        403,
                    )
                side = str(lineup["side"])
                by_column = "player_ready_by_user_id"
                at_column = "player_ready_at"
            db.execute(
                f"""
                UPDATE competition_game_readiness
                SET {by_column} = ?, {at_column} = ?
                WHERE competition_id = ? AND game_key = ? AND side = ?
                """,
                (
                    principal.user_id if ready else None,
                    now.isoformat() if ready else None,
                    competition_id,
                    game_key,
                    side,
                ),
            )
            self._append_event(
                db,
                competition_id,
                "game.readiness_changed",
                principal.user_id,
                {
                    "game_key": game_key,
                    "side": side,
                    "readiness_role": role,
                    "ready": bool(ready),
                },
            )
            self._record_command(
                db, competition_id, principal, normalized_command, action
            )
            all_ready = bool(ready) and int(db.execute(
                """
                SELECT COUNT(*) AS count FROM competition_game_readiness
                WHERE competition_id = ? AND game_key = ?
                  AND player_ready_by_user_id IS NOT NULL
                  AND captain_ready_by_user_id IS NOT NULL
                """,
                (competition_id, game_key),
            ).fetchone()["count"]) == 2
            if all_ready:
                room = self._start_game_in_transaction(
                    db, room, game_key=game_key, now=now, actor_user_id=None
                )
            else:
                self._touch(db, competition_id)
                room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def start_current_game(
        self,
        room_code: str,
        principal: Principal,
        *,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "game.start"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            control = self._match_control_row(db, competition_id)
            self._ensure_not_suspended(db, competition_id)
            game_key = str(control["current_game_key"])
            if str(room["status"]) != GAME_READY_STATUS[game_key]:
                raise CompetitionError(
                    "INVALID_GAME_PHASE", "The current game is not ready to start.", 409
                )
            if not secrets.compare_digest(
                str(control["phase_token"]), str(phase_token or "")
            ):
                raise CompetitionError(
                    "STALE_PHASE", "The game phase has changed. Refresh the room state.", 409
                )
            readiness = db.execute(
                """
                SELECT * FROM competition_game_readiness
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, game_key),
            ).fetchall()
            if len(readiness) != 2 or any(
                not row["player_ready_by_user_id"] or not row["captain_ready_by_user_id"]
                for row in readiness
            ):
                raise CompetitionError(
                    "READINESS_INCOMPLETE",
                    "Both active players and both captains must be ready.",
                    409,
                )
            room = self._start_game_in_transaction(
                db, room, game_key=game_key, now=now, actor_user_id=principal.user_id
            )
            self._record_command(
                db, competition_id, principal, normalized_command, action
            )
            return self._snapshot(db, room, principal)

    def _start_game_in_transaction(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        game_key: str,
        now: datetime,
        actor_user_id: int | None,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        preview_until = self._hold_until(db, competition_id, GAME_READY_STATUS[game_key])
        if preview_until and now < parse_time(preview_until):
            self._touch(db, competition_id)
            return self._room_row(db, str(room['room_code']))
        if game_key == 'A':
            window = self._prediction_window(db, room)
            if window and now < parse_time(window['minimum_until']):
                self._touch(db, competition_id)
                return self._room_row(db, str(room['room_code']))
            db.execute("UPDATE competition_prediction_windows SET closed_at = COALESCE(closed_at, ?) WHERE competition_id = ?",
                       (now.isoformat(), competition_id))
        draft = self._draft_row(db, competition_id)
        project_key = str(draft[f"project_{game_key.lower()}"])
        project = db.execute(
            """
            SELECT * FROM competition_projects
            WHERE competition_id = ? AND project_key = ? AND enabled = 1
            """,
            (competition_id, project_key),
        ).fetchone()
        if project is None:
            raise CompetitionError(
                "PROJECT_NOT_FOUND", "Selected project is not in the frozen pool.", 500
            )
        adapter = self._adapter_for_project_row(project)
        shared_seed = hmac.new(
            bytes.fromhex(str(draft["random_seed_hex"])),
            f"test-project:{game_key}".encode("ascii"),
            hashlib.sha256,
        ).hexdigest()
        lineups = db.execute(
            """
            SELECT side, player_user_id FROM competition_lineups
            WHERE competition_id = ? AND game_key = ?
            """,
            (competition_id, game_key),
        ).fetchall()
        if len(lineups) != 2:
            raise CompetitionError("LINEUP_INCOMPLETE", "Game lineup is incomplete.", 500)
        project_clock_starts = {
            str(clock["side"]): int(clock["remaining_ms_base"])
            for clock in db.execute(
                "SELECT side, remaining_ms_base FROM competition_team_clocks WHERE competition_id = ?",
                (competition_id,),
            ).fetchall()
        }
        instance_id = uuid.uuid4().hex
        db.executemany(
            """
            INSERT INTO competition_game_sessions
              (competition_id, game_key, side, project_key, project_ref,
               rules_version, instance_id, public_generation, player_user_id,
               seed_hex, board_json, score, move_count, rng_counter, adapter_state_json,
               state, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, 'playing', ?, ?)
            """,
            [
                (
                    competition_id,
                    game_key,
                    str(row["side"]),
                    project_key,
                    str(project["project_ref"]),
                    str(project["adapter_rules_version"]),
                    f"{instance_id}:{str(row['side'])}",
                    int(row["player_user_id"]),
                    initial.seed,
                    json.dumps(initial.board, separators=(",", ":")),
                    initial.score,
                    initial.move_count,
                    initial.rng_counter,
                    json.dumps(
                        {
                            **initial.extra,
                            "project_clock_start_ms": project_clock_starts[str(row["side"])],
                        },
                        separators=(",", ":"),
                    ),
                    now.isoformat(),
                    now.isoformat(),
                )
                for row in lineups
                for initial in [client_runtime.initial_state(adapter, shared_seed)]
            ],
        )
        db.execute(
            """
            UPDATE competition_team_clocks
            SET running_since = ?, state = 'running', revision = revision + 1,
                updated_at = ?
            WHERE competition_id = ? AND state = 'stopped'
            """,
            (now.isoformat(), now.isoformat(), competition_id),
        )
        db.execute(
            """
            UPDATE competition_match_control
            SET phase_token = ?, updated_at = ? WHERE competition_id = ?
            """,
            (self._new_phase_token(), now.isoformat(), competition_id),
        )
        self._append_event(
            db,
            competition_id,
            "game.started",
            actor_user_id,
            {
                "game_key": game_key,
                "project_key": project_key,
                "adapter": adapter.project_id,
                "rules_version": adapter.rules_version,
                "instance_id": instance_id,
                "automatic": actor_user_id is None,
            },
        )
        room = self._change_draft_status(db, room, GAME_PLAYING_STATUS[game_key])
        return room

    def sync_client_game(self, room_code, principal, *, instance_id, sequence, phase_token,
                         payload, checkpoint, result_value, elapsed_ms, finished, outcome):
        """Accept a player's latest complete state; no move, RNG or WASM calls."""
        now = datetime.now(timezone.utc)
        if len(json.dumps({"payload": payload, "checkpoint": checkpoint})) > 1024 * 1024:
            raise CompetitionError("STATE_TOO_LARGE", "Project state is too large.", 413)
        board = payload.get("board")
        history = checkpoint.get('metric_history', [])
        if (not isinstance(history, list) or any(
                not isinstance(item, list) or len(item) != 2
                or any(type(value) is not int or value < 0 for value in item)
                or item[0] > elapsed_ms for item in history)
                or any(history[i][0] > history[i + 1][0] for i in range(len(history) - 1))):
            raise CompetitionError('INVALID_CLIENT_STATE', 'Malformed metric timeline.')
        # Aftershock may grow beyond the original rectangle; the 1 MiB packet
        # limit above bounds this dense view without imposing an arbitrary axis.
        if (not isinstance(board, list) or not board
                or not all(isinstance(row, list) and row for row in board)
                or any(len(row) != len(board[0]) for row in board)
                or any(type(cell) is not int for row in board for cell in row)
                or type(payload.get("score")) is not int or payload["score"] < 0
                or type(payload.get("move_count")) is not int or payload["move_count"] < 0
                or checkpoint.get("version") != 1 or not isinstance(checkpoint.get("state"), dict)):
            raise CompetitionError("INVALID_CLIENT_STATE", "Malformed project state.")
        if finished and outcome not in {"target_reached", "no_moves", "tile_limit", "time_limit", "opponent_finished", "surrendered"}:
            raise CompetitionError("INVALID_CLIENT_STATE", "Unknown completion reason.")
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            session = db.execute(
                "SELECT * FROM competition_game_sessions WHERE competition_id = ? AND instance_id = ? AND player_user_id = ?",
                (competition_id, instance_id, principal.user_id),
            ).fetchone()
            self._ensure_admitted(db, competition_id, principal)
            if session is None:
                raise CompetitionError("ACTIVE_PLAYER_REQUIRED", "Only this project's active player may upload.", 403)
            extra = json.loads(session["adapter_state_json"])
            accepted = int(extra.get("client_sequence", 0))
            if sequence <= accepted:
                result = {"instance_id": instance_id, "accepted_sequence": accepted, "duplicate": True}
                if finished or str(session['state']) == 'completed':
                    result['competition'] = self._snapshot(db, room, principal)
                return result
            control = self._match_control_row(db, competition_id)
            game_key = str(session["game_key"])
            if (str(room["status"]) != GAME_PLAYING_STATUS[game_key]
                    or str(session["state"]) != "playing"):
                return {"instance_id": instance_id, "accepted_sequence": accepted,
                        "stopped": True, "competition": self._snapshot(db, room, principal)}
            self._ensure_not_suspended(db, competition_id)
            if not secrets.compare_digest(str(control["phase_token"]), str(phase_token or "")):
                raise CompetitionError("STALE_PHASE", "Refresh the current match phase.", 409)
            room, settled = self._settle_game_clocks(db, room, now=now)
            if settled:
                return {"instance_id": instance_id, "accepted_sequence": accepted,
                        "stopped": True, "competition": self._snapshot(db, room, principal)}
            if elapsed_ms >= int(extra['project_clock_start_ms']):
                raise CompetitionError('TEAM_CLOCK_EXPIRED', 'The team time budget has expired.', 409)
            if outcome == 'surrendered':
                opponent = db.execute("SELECT state,outcome_reason FROM competition_game_sessions WHERE competition_id=? AND game_key=? AND side!=?", (competition_id, game_key, session['side'])).fetchone()
                if not finished or not opponent or opponent['state'] != 'completed' or opponent['outcome_reason'] == 'surrendered':
                    raise CompetitionError('SURRENDER_NOT_ALLOWED', 'Only an unfinished player whose opponent has finished may surrender.', 409)
            extra.update(client_sequence=sequence, client_payload=payload, checkpoint=checkpoint,
                         result_value=result_value, client_elapsed_ms=elapsed_ms, client_completed=bool(finished))
            db.execute(
                """UPDATE competition_game_sessions SET board_json = ?, score = ?, move_count = ?,
                   adapter_state_json = ?, state = ?, outcome_reason = ?, completed_at = ?, updated_at = ?
                   WHERE competition_id = ? AND instance_id = ?""",
                (json.dumps(board, separators=(",", ":")), payload["score"], payload["move_count"],
                 json.dumps(extra, separators=(",", ":")), "completed" if finished else "playing",
                 outcome if finished else None, now.isoformat() if finished else None, now.isoformat(),
                 competition_id, instance_id),
            )
            if finished:
                # Freeze the team clock at the reported local completion time,
                # excluding upload latency. The server still owns match clocks.
                self._stop_clock(db, competition_id, str(session["side"]), now=now)
                remaining = max(0, int(extra["project_clock_start_ms"]) - elapsed_ms)
                db.execute("UPDATE competition_team_clocks SET remaining_ms_base = ? WHERE competition_id = ? AND side = ?",
                           (remaining, competition_id, str(session["side"])))
                adapter = self._adapter_for_session_row(session)
                if outcome == "target_reached" and bool(getattr(getattr(adapter, "rules", None), "race", False)):
                    opponent = db.execute(
                        "SELECT * FROM competition_game_sessions WHERE competition_id = ? AND game_key = ? AND side != ? AND state = 'playing'",
                        (competition_id, game_key, str(session["side"])),
                    ).fetchone()
                    if opponent is not None:
                        opponent_extra = json.loads(opponent["adapter_state_json"])
                        opponent_extra["race_stop_at"] = (now + timedelta(seconds=5)).isoformat()
                        self._stop_clock(db,competition_id,str(opponent['side']),now=now)
                        db.execute('UPDATE competition_team_clocks SET remaining_ms_base=MAX(0,?) WHERE competition_id=? AND side=?',
                                   (int(opponent_extra['project_clock_start_ms'])-elapsed_ms,competition_id,opponent['side']))
                        db.execute("UPDATE competition_game_sessions SET adapter_state_json = ? WHERE competition_id = ? AND instance_id = ?",
                                   (json.dumps(opponent_extra), competition_id, str(opponent["instance_id"])))
                self._append_event(db, competition_id, "project.completed", principal.user_id,
                                   {"game_key": game_key, "side": str(session["side"]), "score": result_value,
                                    "outcome": outcome, "client_sequence": sequence})
            completed = db.execute(
                "SELECT count(*) AS count FROM competition_game_sessions WHERE competition_id = ? AND game_key = ? AND state = 'completed'",
                (competition_id, game_key),
            ).fetchone()["count"]
            if completed == 2:
                room = self._publish_game_result(db, room, game_key=game_key, now=now)
            else:
                self._touch(db, competition_id)
                room = self._room_row(db, room_code)
            result = {"instance_id": instance_id, "accepted_sequence": sequence}
            if finished:
                result["competition"] = self._snapshot(db, room, principal)
            else:
                state = self._session_state(db.execute(
                    "SELECT * FROM competition_game_sessions WHERE competition_id = ? AND instance_id = ?",
                    (competition_id, instance_id)).fetchone(), db, now=now)
                adapter = self._adapter_for_session_row(session)
                result["update"] = {
                    "room_code": str(room["room_code"]), "game_key": game_key,
                    "instance_id": instance_id, "side": str(session["side"]),
                    "version": int(room["version"]), "server_time": now.isoformat(),
                    "public_view": client_runtime.public_view(adapter, state, int(session["public_generation"])),
                }
            return result

    def _advance_after_result(
        self,
        db: sqlite3.Connection,
        room: sqlite3.Row,
        *,
        game_key: str,
        now: datetime,
        actor_user_id: int | None,
        forced: bool = False,
        reason: str | None = None,
    ) -> sqlite3.Row:
        competition_id = str(room["id"])
        control = self._match_control_row(db, competition_id)
        if forced:
            result = db.execute('SELECT published_at FROM competition_game_results WHERE competition_id=? AND game_key=?', (competition_id, game_key)).fetchone()
            if result and now < parse_time(result['published_at']) + timedelta(seconds=self.result_rest_seconds):
                raise CompetitionError('RESULT_REST_REQUIRED', '本局休整尚未结束，请等待 30 秒展示完毕。', 409)
            self._append_event(
                db,
                competition_id,
                "game.result_force_advanced",
                actor_user_id,
                {"game_key": game_key, "reason": reason},
            )
        if game_key != "C":
            next_game = GAME_KEYS[GAME_KEYS.index(game_key) + 1]
            self._set_hold(db, competition_id, GAME_READY_STATUS[next_game], now, self.ready_preview_seconds)
            self._set_hold(db, competition_id, GAME_READY_STATUS[next_game]+'_TIMEOUT', now, self._flow(db,competition_id)['ready_seconds'])
            db.executemany(
                """
                INSERT INTO competition_game_readiness
                  (competition_id, game_key, side)
                VALUES (?, ?, ?)
                """,
                [
                    (competition_id, next_game, TeamSide.YELLOW.value),
                    (competition_id, next_game, TeamSide.WHITE.value),
                ],
            )
            db.execute(
                """
                UPDATE competition_match_control
                SET current_game_key = ?, phase_token = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (next_game, self._new_phase_token(), now.isoformat(), competition_id),
            )
            self._append_event(
                db,
                competition_id,
                "game.ready_check_started",
                actor_user_id if forced else None,
                {"game_key": next_game},
            )
            return self._change_draft_status(db, room, GAME_READY_STATUS[next_game])

        yellow_wins = int(control["yellow_wins"])
        white_wins = int(control["white_wins"])
        winner_side = (
            TeamSide.YELLOW.value
            if yellow_wins > white_wins
            else TeamSide.WHITE.value
            if white_wins > yellow_wins
            else "draw"
        )
        finish_reason = "referee_force_advance" if forced else "completed"
        db.execute(
            """
            UPDATE competition_match_control
            SET winner_side = ?, finish_reason = ?, phase_token = ?, updated_at = ?
            WHERE competition_id = ?
            """,
            (
                winner_side,
                finish_reason,
                self._new_phase_token(),
                now.isoformat(),
                competition_id,
            ),
        )
        self._append_event(
            db,
            competition_id,
            "match.finished",
            actor_user_id if forced else None,
            {
                "reason": finish_reason,
                "reason_text": reason if forced else None,
                "winner_side": winner_side,
                "yellow_wins": yellow_wins,
                "white_wins": white_wins,
            },
        )
        return self._change_draft_status(db, room, CompetitionStatus.FINISHED.value)

    def confirm_current_result(
        self,
        room_code: str,
        principal: Principal,
        *,
        result_revision: int,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "game.result.confirm"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(
                db, competition_id, principal, normalized_command, action
            ):
                return self._snapshot(db, room, principal)
            control = self._match_control_row(db, competition_id)
            self._ensure_not_suspended(db, competition_id)
            game_key = str(control["current_game_key"])
            if str(room["status"]) != GAME_RESULT_STATUS[game_key]:
                raise CompetitionError(
                    "INVALID_GAME_PHASE", "No current result is awaiting confirmation.", 409
                )
            if not secrets.compare_digest(
                str(control["phase_token"]), str(phase_token or "")
            ):
                raise CompetitionError(
                    "STALE_PHASE", "The result phase has changed. Refresh the room state.", 409
                )
            result = db.execute(
                """
                SELECT * FROM competition_game_results
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, game_key),
            ).fetchone()
            if result is None or int(result["result_revision"]) != int(result_revision):
                raise CompetitionError(
                    "STALE_RESULT", "The result has changed. Review it again.", 409
                )
            side = self._captain_side(db, competition_id, principal)
            if db.execute(
                """
                SELECT 1 FROM competition_result_confirmations
                WHERE competition_id = ? AND game_key = ? AND side = ?
                """,
                (competition_id, game_key, side),
            ).fetchone():
                raise CompetitionError(
                    "RESULT_ALREADY_CONFIRMED", "Your team already confirmed this result.", 409
                )
            db.execute(
                """
                INSERT INTO competition_result_confirmations
                  (competition_id, game_key, side, result_revision,
                   confirmed_by_user_id, confirmed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    competition_id,
                    game_key,
                    side,
                    int(result_revision),
                    principal.user_id,
                    now.isoformat(),
                ),
            )
            self._append_event(
                db,
                competition_id,
                "game.result_confirmed",
                principal.user_id,
                {
                    "game_key": game_key,
                    "side": side,
                    "result_revision": int(result_revision),
                },
            )
            self._record_command(
                db, competition_id, principal, normalized_command, action
            )
            confirmation_count = int(
                db.execute(
                    """
                    SELECT COUNT(*) AS count FROM competition_result_confirmations
                    WHERE competition_id = ? AND game_key = ?
                      AND result_revision = ?
                    """,
                    (competition_id, game_key, int(result_revision)),
                ).fetchone()["count"]
            )
            if confirmation_count < 2 or now < parse_time(result['published_at']) + timedelta(seconds=self.result_rest_seconds):
                self._touch(db, competition_id)
                room = self._room_row(db, room_code)
                return self._snapshot(db, room, principal)

            room = self._advance_after_result(
                db,
                room,
                game_key=game_key,
                now=now,
                actor_user_id=None,
            )
            return self._snapshot(db, room, principal)

    def report_issue(
        self,
        room_code: str,
        principal: Principal,
        *,
        category: str,
        details: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_category = str(category or "").strip().lower()
        if normalized_category not in ISSUE_CATEGORIES:
            raise CompetitionError("INVALID_ISSUE_CATEGORY", "Unknown issue category.")
        normalized_details = self._normalize_reason(details)
        normalized_command = self._normalize_command_id(command_id)
        action = "issue.report"
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            if str(room["status"]) not in MATCH_ACTIVE_STATUSES:
                raise CompetitionError(
                    "MATCH_NOT_ACTIVE", "Issues can only be reported during an active match.", 409
                )
            if db.execute(
                "SELECT 1 FROM competition_seats WHERE competition_id = ? AND user_id = ?",
                (competition_id, principal.user_id),
            ).fetchone() is None:
                raise CompetitionError(
                    "PLAYER_REQUIRED", "Only seated players can report match issues.", 403
                )
            existing = db.execute(
                """
                SELECT id FROM competition_issue_reports
                WHERE competition_id = ? AND reported_by_user_id = ?
                  AND category = ? AND status = 'open'
                """,
                (competition_id, principal.user_id, normalized_category),
            ).fetchone()
            if existing is None:
                issue_id = uuid.uuid4().hex
                db.execute(
                    """
                    INSERT INTO competition_issue_reports
                      (id, competition_id, reported_by_user_id, category,
                       details, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        issue_id,
                        competition_id,
                        principal.user_id,
                        normalized_category,
                        normalized_details,
                        utc_now(),
                    ),
                )
                self._append_event(
                    db,
                    competition_id,
                    "issue.reported",
                    principal.user_id,
                    {"issue_id": issue_id, "category": normalized_category},
                )
                self._touch(db, competition_id)
                room = self._room_row(db, room_code)
            self._record_command(db, competition_id, principal, normalized_command, action)
            return self._snapshot(db, room, principal)

    def resolve_issue(
        self,
        room_code: str,
        principal: Principal,
        *,
        issue_id: str,
        status: str,
        resolution_note: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"resolved", "dismissed"}:
            raise CompetitionError(
                "INVALID_ISSUE_STATUS", "Issue status must be resolved or dismissed."
            )
        normalized_note = self._normalize_reason(resolution_note)
        normalized_command = self._normalize_command_id(command_id)
        action = "issue.resolve"
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            issue = db.execute(
                "SELECT * FROM competition_issue_reports WHERE id = ? AND competition_id = ?",
                (str(issue_id), competition_id),
            ).fetchone()
            if issue is None:
                raise CompetitionError("ISSUE_NOT_FOUND", "Issue report not found.", 404)
            if str(issue["status"]) != "open":
                raise CompetitionError("ISSUE_ALREADY_CLOSED", "Issue is already closed.", 409)
            now = utc_now()
            db.execute(
                """
                UPDATE competition_issue_reports
                SET status = ?, resolved_by_user_id = ?, resolution_note = ?, resolved_at = ?
                WHERE id = ?
                """,
                (normalized_status, principal.user_id, normalized_note, now, str(issue_id)),
            )
            self._append_event(
                db,
                competition_id,
                f"issue.{normalized_status}",
                principal.user_id,
                {"issue_id": str(issue_id), "resolution_note": normalized_note},
            )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def suspend_match(
        self,
        room_code: str,
        principal: Principal,
        *,
        reason_code: str,
        reason_text: str,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_code = str(reason_code or "").strip().lower()
        if normalized_code not in SUSPENSION_REASON_CODES:
            raise CompetitionError("INVALID_SUSPENSION_REASON", "Unknown suspension reason.")
        normalized_reason = self._normalize_reason(reason_text)
        normalized_command = self._normalize_command_id(command_id)
        action = "match.suspend"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            room, settled = self._settle_due_in_transaction(db, room, now=now)
            if settled and str(room["status"]) == CompetitionStatus.FINISHED.value:
                self._record_command(db, competition_id, principal, normalized_command, action)
                return self._snapshot(db, room, principal)
            if str(room["status"]) not in MATCH_ACTIVE_STATUSES:
                raise CompetitionError("MATCH_NOT_ACTIVE", "The match cannot be suspended now.", 409)
            control = self._match_control_row(db, competition_id)
            self._require_phase_token(control, phase_token)
            suspension = self._suspension_row(db, competition_id)
            if bool(suspension["active"]):
                raise CompetitionError("MATCH_ALREADY_SUSPENDED", "The match is already suspended.", 409)
            for clock in db.execute(
                "SELECT * FROM competition_team_clocks WHERE competition_id = ?",
                (competition_id,),
            ).fetchall():
                if str(clock["state"]) == "running":
                    side = str(clock["side"])
                    self._stop_clock(db, competition_id, side, now=now)
                    db.execute(
                        """
                        UPDATE competition_team_clocks SET resume_after_suspension = 1
                        WHERE competition_id = ? AND side = ?
                        """,
                        (competition_id, side),
                    )
            db.execute(
                "DELETE FROM competition_suspension_readiness WHERE competition_id = ?",
                (competition_id,),
            )
            db.execute(
                """
                UPDATE competition_suspensions
                SET active = 1, reason_code = ?, reason_text = ?,
                    started_by_user_id = ?, started_by_display_name = ?, started_at = ?,
                    ended_by_user_id = NULL, ended_at = NULL, updated_at = ?
                WHERE competition_id = ?
                """,
                (
                    normalized_code,
                    normalized_reason,
                    principal.user_id,
                    principal.display_name,
                    now.isoformat(),
                    now.isoformat(),
                    competition_id,
                ),
            )
            db.execute(
                """
                UPDATE competition_match_control SET phase_token = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (self._new_phase_token(), now.isoformat(), competition_id),
            )
            self._append_event(
                db,
                competition_id,
                "match.suspended",
                principal.user_id,
                {"reason_code": normalized_code, "reason_text": normalized_reason},
            )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def set_suspension_readiness(
        self,
        room_code: str,
        principal: Principal,
        *,
        ready: bool,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "match.resume_readiness"
        now = utc_now()
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            if not bool(self._suspension_row(db, competition_id)["active"]):
                raise CompetitionError("MATCH_NOT_SUSPENDED", "The match is not suspended.", 409)
            control = self._match_control_row(db, competition_id)
            self._require_phase_token(control, phase_token)
            side = self._captain_side(db, competition_id, principal)
            if ready:
                db.execute(
                    """
                    INSERT INTO competition_suspension_readiness
                      (competition_id, side, ready_by_user_id, ready_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(competition_id, side) DO UPDATE SET
                      ready_by_user_id = excluded.ready_by_user_id,
                      ready_at = excluded.ready_at
                    """,
                    (competition_id, side, principal.user_id, now),
                )
            else:
                db.execute(
                    "DELETE FROM competition_suspension_readiness WHERE competition_id = ? AND side = ?",
                    (competition_id, side),
                )
            self._append_event(
                db,
                competition_id,
                "match.resume_readiness_changed",
                principal.user_id,
                {"side": side, "ready": bool(ready)},
            )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def resume_match(
        self,
        room_code: str,
        principal: Principal,
        *,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_command = self._normalize_command_id(command_id)
        action = "match.resume"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            if not bool(self._suspension_row(db, competition_id)["active"]):
                raise CompetitionError("MATCH_NOT_SUSPENDED", "The match is not suspended.", 409)
            control = self._match_control_row(db, competition_id)
            self._require_phase_token(control, phase_token)
            ready_sides = {
                str(row["side"])
                for row in db.execute(
                    "SELECT side FROM competition_suspension_readiness WHERE competition_id = ?",
                    (competition_id,),
                ).fetchall()
            }
            if ready_sides != {TeamSide.YELLOW.value, TeamSide.WHITE.value}:
                raise CompetitionError(
                    "RESUME_READINESS_INCOMPLETE", "Both captains must be ready to resume.", 409
                )
            suspended_at=parse_time(self._suspension_row(db,competition_id)['started_at'])
            if suspended_at:
                pause=now-suspended_at
                for hold in db.execute('SELECT stage,until_at FROM competition_stage_holds WHERE competition_id=?',(competition_id,)).fetchall():
                    until=parse_time(hold['until_at'])
                    if until and until>suspended_at:
                        db.execute('UPDATE competition_stage_holds SET until_at=? WHERE competition_id=? AND stage=?',((until+pause).isoformat(),competition_id,hold['stage']))
                if str(room['status']) in GAME_RESULT_STATUS.values():
                    db.execute('UPDATE competition_game_results SET published_at=? WHERE competition_id=? AND game_key=?',
                               ((parse_time(db.execute('SELECT published_at FROM competition_game_results WHERE competition_id=? AND game_key=?',(competition_id,control['current_game_key'])).fetchone()[0])+pause).isoformat(),competition_id,control['current_game_key']))
            for session in db.execute(
                "SELECT instance_id, adapter_state_json FROM competition_game_sessions WHERE competition_id=? AND state='playing'",
                (competition_id,),
            ).fetchall():
                extra = json.loads(session['adapter_state_json'])
                if extra.get('race_stop_at'):
                    extra['race_stop_at'] = (now + timedelta(seconds=5)).isoformat()
                    db.execute('UPDATE competition_game_sessions SET adapter_state_json=? WHERE instance_id=?',
                               (json.dumps(extra, separators=(',', ':')), session['instance_id']))
            db.execute(
                """
                UPDATE competition_team_clocks
                SET state = 'running', running_since = ?, resume_after_suspension = 0,
                    revision = revision + 1, updated_at = ?
                WHERE competition_id = ? AND resume_after_suspension = 1
                  AND remaining_ms_base > 0
                """,
                (now.isoformat(), now.isoformat(), competition_id),
            )
            db.execute(
                """
                UPDATE competition_suspensions
                SET active = 0, ended_by_user_id = ?, ended_at = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (principal.user_id, now.isoformat(), now.isoformat(), competition_id),
            )
            db.execute(
                "DELETE FROM competition_suspension_readiness WHERE competition_id = ?",
                (competition_id,),
            )
            db.execute(
                """
                UPDATE competition_match_control SET phase_token = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (self._new_phase_token(), now.isoformat(), competition_id),
            )
            self._append_event(db, competition_id, "match.resumed", principal.user_id, {})
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def override_current_result(
        self,
        room_code: str,
        principal: Principal,
        *,
        yellow_score: int,
        white_score: int,
        winner_side: str,
        reason: str,
        result_revision: int,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_winner = str(winner_side or "").strip().lower()
        if normalized_winner not in {TeamSide.YELLOW.value, TeamSide.WHITE.value, "draw"}:
            raise CompetitionError("INVALID_WINNER", "Winner must be yellow, white or draw.")
        if int(yellow_score) < 0 or int(white_score) < 0:
            raise CompetitionError("INVALID_SCORE", "Scores cannot be negative.")
        normalized_reason = self._normalize_reason(reason)
        normalized_command = self._normalize_command_id(command_id)
        action = "game.result.override"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            self._ensure_not_suspended(db, competition_id)
            control = self._match_control_row(db, competition_id)
            game_key = str(control["current_game_key"])
            if str(room["status"]) != GAME_RESULT_STATUS[game_key]:
                raise CompetitionError("INVALID_GAME_PHASE", "No current result can be corrected.", 409)
            self._require_phase_token(control, phase_token)
            result = db.execute(
                "SELECT * FROM competition_game_results WHERE competition_id = ? AND game_key = ?",
                (competition_id, game_key),
            ).fetchone()
            if result is None or int(result["result_revision"]) != int(result_revision):
                raise CompetitionError("STALE_RESULT", "The result has changed. Review it again.", 409)
            next_revision = int(result_revision) + 1
            for refund in db.execute('SELECT * FROM competition_time_refunds WHERE competition_id=? AND game_key=?',(competition_id,game_key)).fetchall():
                if refund['side'] != normalized_winner:
                    db.execute('UPDATE competition_team_clocks SET remaining_ms_base=MAX(0,remaining_ms_base-?),revision=revision+1 WHERE competition_id=? AND side=?',(refund['amount_ms'],competition_id,refund['side']))
                    db.execute('DELETE FROM competition_time_refunds WHERE competition_id=? AND game_key=? AND side=?',(competition_id,game_key,refund['side']))
            db.execute(
                """
                UPDATE competition_game_results
                SET yellow_score = ?, white_score = ?, winner_side = ?,
                    reason = 'referee_override', result_revision = ?,
                    corrected_by_user_id = ?, correction_reason = ?, corrected_at = ?
                WHERE competition_id = ? AND game_key = ?
                """,
                (
                    int(yellow_score),
                    int(white_score),
                    normalized_winner,
                    next_revision,
                    principal.user_id,
                    normalized_reason,
                    now.isoformat(),
                    competition_id,
                    game_key,
                ),
            )
            counts = db.execute(
                """
                SELECT
                  SUM(CASE WHEN winner_side = 'yellow' THEN 1 ELSE 0 END) AS yellow_wins,
                  SUM(CASE WHEN winner_side = 'white' THEN 1 ELSE 0 END) AS white_wins,
                  SUM(CASE WHEN winner_side = 'draw' THEN 1 ELSE 0 END) AS draws
                FROM competition_game_results WHERE competition_id = ?
                """,
                (competition_id,),
            ).fetchone()
            new_token = self._new_phase_token()
            db.execute(
                """
                UPDATE competition_match_control
                SET yellow_wins = ?, white_wins = ?, draws = ?,
                    phase_token = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (
                    int(counts["yellow_wins"] or 0),
                    int(counts["white_wins"] or 0),
                    int(counts["draws"] or 0),
                    new_token,
                    now.isoformat(),
                    competition_id,
                ),
            )
            db.execute(
                "DELETE FROM competition_result_confirmations WHERE competition_id = ? AND game_key = ?",
                (competition_id, game_key),
            )
            self._append_event(
                db,
                competition_id,
                "game.result_overridden",
                principal.user_id,
                {
                    "game_key": game_key,
                    "old": {
                        "yellow_score": int(result["yellow_score"]),
                        "white_score": int(result["white_score"]),
                        "winner_side": str(result["winner_side"]),
                        "result_revision": int(result["result_revision"]),
                    },
                    "new": {
                        "yellow_score": int(yellow_score),
                        "white_score": int(white_score),
                        "winner_side": normalized_winner,
                        "result_revision": next_revision,
                    },
                    "reason": normalized_reason,
                },
            )
            self._record_command(db, competition_id, principal, normalized_command, action)
            self._touch(db, competition_id)
            room = self._room_row(db, room_code)
            return self._snapshot(db, room, principal)

    def force_advance_current_result(
        self,
        room_code: str,
        principal: Principal,
        *,
        reason: str,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_reason = self._normalize_reason(reason)
        normalized_command = self._normalize_command_id(command_id)
        action = "game.result.force_advance"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            self._ensure_not_suspended(db, competition_id)
            control = self._match_control_row(db, competition_id)
            game_key = str(control["current_game_key"])
            if str(room["status"]) != GAME_RESULT_STATUS[game_key]:
                raise CompetitionError("INVALID_GAME_PHASE", "No result is awaiting advance.", 409)
            self._require_phase_token(control, phase_token)
            self._record_command(db, competition_id, principal, normalized_command, action)
            room = self._advance_after_result(
                db,
                room,
                game_key=game_key,
                now=now,
                actor_user_id=principal.user_id,
                forced=True,
                reason=normalized_reason,
            )
            return self._snapshot(db, room, principal)

    def force_finish_match(
        self,
        room_code: str,
        principal: Principal,
        *,
        winner_side: str,
        reason: str,
        phase_token: str,
        command_id: str,
    ) -> dict[str, Any]:
        normalized_winner = str(winner_side or "").strip().lower()
        if normalized_winner not in {TeamSide.YELLOW.value, TeamSide.WHITE.value, "draw"}:
            raise CompetitionError("INVALID_WINNER", "Winner must be yellow, white or draw.")
        normalized_reason = self._normalize_reason(reason)
        normalized_command = self._normalize_command_id(command_id)
        action = "match.force_finish"
        now = datetime.now(timezone.utc)
        with self.database.transaction(immediate=True) as db:
            room = self._room_row(db, room_code)
            competition_id = str(room["id"])
            if self._check_command(db, competition_id, principal, normalized_command, action):
                return self._snapshot(db, room, principal)
            self._require_match_official(db, room, principal)
            if str(room["status"]) not in MATCH_ACTIVE_STATUSES:
                raise CompetitionError("MATCH_NOT_ACTIVE", "The match is not active.", 409)
            control = self._match_control_row(db, competition_id)
            self._require_phase_token(control, phase_token)
            current_game = str(control["current_game_key"])
            for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value):
                clock = db.execute(
                    "SELECT state FROM competition_team_clocks WHERE competition_id = ? AND side = ?",
                    (competition_id, side),
                ).fetchone()
                if clock is not None and str(clock["state"]) == "running":
                    self._stop_clock(db, competition_id, side, now=now)
            db.execute(
                """
                UPDATE competition_team_clocks SET resume_after_suspension = 0
                WHERE competition_id = ?
                """,
                (competition_id,),
            )
            session_scores = {
                str(row["side"]): int(row["score"])
                for row in db.execute(
                    """
                    SELECT side, score FROM competition_game_sessions
                    WHERE competition_id = ? AND game_key = ?
                    """,
                    (competition_id, current_game),
                ).fetchall()
            }
            for game_key in GAME_KEYS:
                if db.execute(
                    "SELECT 1 FROM competition_game_results WHERE competition_id = ? AND game_key = ?",
                    (competition_id, game_key),
                ).fetchone():
                    continue
                db.execute(
                    """
                    INSERT INTO competition_game_results
                      (competition_id, game_key, yellow_score, white_score,
                       winner_side, reason, result_revision,
                       corrected_by_user_id, correction_reason, corrected_at, published_at)
                    VALUES (?, ?, ?, ?, ?, 'referee_force_finish', 1, ?, ?, ?, ?)
                    """,
                    (
                        competition_id,
                        game_key,
                        session_scores.get(TeamSide.YELLOW.value, 0)
                        if game_key == current_game else 0,
                        session_scores.get(TeamSide.WHITE.value, 0)
                        if game_key == current_game else 0,
                        normalized_winner,
                        principal.user_id,
                        normalized_reason,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            db.execute(
                """
                UPDATE competition_game_sessions
                SET state = 'completed', outcome_reason = 'referee_force_finish',
                    completed_at = COALESCE(completed_at, ?), updated_at = ?
                WHERE competition_id = ? AND state = 'playing'
                """,
                (now.isoformat(), now.isoformat(), competition_id),
            )
            counts = db.execute(
                """
                SELECT
                  SUM(CASE WHEN winner_side = 'yellow' THEN 1 ELSE 0 END) AS yellow_wins,
                  SUM(CASE WHEN winner_side = 'white' THEN 1 ELSE 0 END) AS white_wins,
                  SUM(CASE WHEN winner_side = 'draw' THEN 1 ELSE 0 END) AS draws
                FROM competition_game_results WHERE competition_id = ?
                """,
                (competition_id,),
            ).fetchone()
            db.execute(
                """
                UPDATE competition_match_control
                SET yellow_wins = ?, white_wins = ?, draws = ?, winner_side = ?,
                    finish_reason = 'referee_force_finish', phase_token = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (
                    int(counts["yellow_wins"] or 0),
                    int(counts["white_wins"] or 0),
                    int(counts["draws"] or 0),
                    normalized_winner,
                    self._new_phase_token(),
                    now.isoformat(),
                    competition_id,
                ),
            )
            db.execute(
                """
                UPDATE competition_suspensions
                SET active = 0, ended_by_user_id = ?, ended_at = ?, updated_at = ?
                WHERE competition_id = ?
                """,
                (principal.user_id, now.isoformat(), now.isoformat(), competition_id),
            )
            db.execute(
                "DELETE FROM competition_suspension_readiness WHERE competition_id = ?",
                (competition_id,),
            )
            self._append_event(
                db,
                competition_id,
                "match.force_finished",
                principal.user_id,
                {"winner_side": normalized_winner, "reason": normalized_reason},
            )
            self._append_event(
                db,
                competition_id,
                "match.finished",
                principal.user_id,
                {"winner_side": normalized_winner, "reason": "referee_force_finish"},
            )
            self._record_command(db, competition_id, principal, normalized_command, action)
            room = self._change_draft_status(db, room, CompetitionStatus.FINISHED.value)
            return self._snapshot(db, room, principal)

    def _summary(
        self, db: sqlite3.Connection, room: sqlite3.Row, principal: Principal
    ) -> dict[str, Any]:
        competition_id = str(room["id"])
        staff_roles = self._staff_roles(db, competition_id, principal.user_id)
        occupied = int(
            db.execute(
                "SELECT COUNT(*) AS count FROM competition_seats WHERE competition_id = ?",
                (competition_id,),
            ).fetchone()["count"]
        )
        return {
            "id": competition_id,
            "room_code": str(room["room_code"]),
            "name": str(room["name"]),
            "status": str(room["status"]),
            "occupied_seats": occupied,
            "event": self.events.room_event(db, competition_id),
            "schedule": self.schedule.view(db, competition_id),
            "series_score": dict(db.execute('SELECT yellow_wins AS yellow,white_wins AS white FROM competition_match_control WHERE competition_id=?', (competition_id,)).fetchone() or {}),
            "version": int(room["version"]),
            "created_at": str(room["created_at"]),
            "updated_at": str(room["updated_at"]),
            "my_staff_roles": staff_roles,
            "can_close": str(room["status"]) in CLOSABLE_ROOM_STATUSES and (
                self._is_platform_organizer(principal)
                or StaffRole.ORGANIZER.value in staff_roles
            ),
        }

    def _snapshot(
        self, db: sqlite3.Connection, room: sqlite3.Row, principal: Principal
    ) -> dict[str, Any]:
        competition_id = str(room["id"])
        self._ensure_admitted(db, competition_id, principal)
        seat_rows = db.execute(
            """
            SELECT side, position, user_id, display_name_snapshot, seated_at
            FROM competition_seats
            WHERE competition_id = ?
            ORDER BY CASE side WHEN 'yellow' THEN 0 ELSE 1 END, position
            """,
            (competition_id,),
        ).fetchall()
        avatar_urls = avatar_urls_for_users(row["user_id"] for row in seat_rows)
        readiness_rows = db.execute(
            """
            SELECT side, ready_by_user_id, ready_at
            FROM competition_team_readiness WHERE competition_id = ?
            """,
            (competition_id,),
        ).fetchall()
        readiness = {
            str(row["side"]): {
                "ready": True,
                "ready_by_user_id": int(row["ready_by_user_id"]),
                "ready_at": str(row["ready_at"]),
            }
            for row in readiness_rows
        }
        for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value):
            readiness.setdefault(
                side, {"ready": False, "ready_by_user_id": None, "ready_at": None}
            )
        seats = [
            {
                "side": str(row["side"]),
                "position": int(row["position"]),
                "user_id": int(row["user_id"]),
                "display_name": str(row["display_name_snapshot"]),
                "avatar_url": avatar_urls.get(int(row["user_id"])),
                "is_captain": int(row["position"]) == 1,
                "seated_at": str(row["seated_at"]),
            }
            for row in seat_rows
        ]
        my_seat = next(
            (seat for seat in seats if seat["user_id"] == principal.user_id), None
        )
        staff_roles = self._staff_roles(db, competition_id, principal.user_id)
        latest_sequence_row = db.execute(
            "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM competition_events WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        status = str(room["status"])
        has_ready_team = any(item["ready"] for item in readiness.values())
        is_captain = bool(my_seat and my_seat["position"] == 1)
        my_side = str(my_seat["side"]) if my_seat else None
        can_manage = self._is_platform_organizer(principal) or StaffRole.ORGANIZER.value in staff_roles
        project_rows = db.execute(
            """
            SELECT project_key, name, description, project_ref,
                   adapter_rules_version, rules_version,
                   adapter_snapshot_json, sort_order
            FROM competition_projects
            WHERE competition_id = ? AND enabled = 1
            ORDER BY sort_order
            """,
            (competition_id,),
        ).fetchall()
        projects = [
            {
                "key": str(row["project_key"]),
                "name": str(row["name"]),
                "description": str(row["description"]),
                "project_ref": str(row["project_ref"]),
                "adapter_rules_version": str(row["adapter_rules_version"]),
                "rules_version": str(row["rules_version"]),
                "adapter": json.loads(str(row["adapter_snapshot_json"] or "{}")),
                "sort_order": int(row["sort_order"]),
            }
            for row in project_rows
        ]
        draft_row = db.execute(
            "SELECT * FROM competition_drafts WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        draft_payload: dict[str, Any] | None = None
        can_submit_pick_ban = False
        can_submit_blind = False
        if draft_row is not None:
            first_side = str(draft_row["first_side"])
            second_side = (
                TeamSide.WHITE.value
                if first_side == TeamSide.YELLOW.value
                else TeamSide.YELLOW.value
            )
            active_side = None
            if status == CompetitionStatus.FIRST_PICK_BAN.value:
                active_side = first_side
            elif status == CompetitionStatus.SECOND_PICK_BAN.value:
                active_side = second_side
            available_keys = self._available_project_keys(
                db, competition_id, draft_row
            )
            can_submit_pick_ban = bool(
                is_captain
                and my_side == active_side
                and status
                in {
                    CompetitionStatus.FIRST_PICK_BAN.value,
                    CompetitionStatus.SECOND_PICK_BAN.value,
                }
            )
            my_blind_field = (
                "blind_yellow"
                if my_side == TeamSide.YELLOW.value
                else "blind_white"
            ) if my_side else None
            my_blind_choice = (
                str(draft_row[my_blind_field])
                if is_captain and my_blind_field and draft_row[my_blind_field]
                else None
            )
            can_submit_blind = bool(
                is_captain
                and status == CompetitionStatus.BLIND_PICK.value
                and my_blind_field
                and not draft_row[my_blind_field]
            )
            deadline_at = None
            if active_side == TeamSide.YELLOW.value:
                deadline_at = draft_row["yellow_deadline_at"]
            elif active_side == TeamSide.WHITE.value:
                deadline_at = draft_row["white_deadline_at"]
            elif status == CompetitionStatus.DRAW.value:
                deadline_at = draft_row["yellow_deadline_at"]
            elif status == CompetitionStatus.C_DRAW.value:
                deadline_at = draft_row["yellow_deadline_at"]
            elif status == CompetitionStatus.BLIND_PICK.value:
                deadline_candidates = [
                    str(value)
                    for value in (
                        draft_row["yellow_deadline_at"],
                        draft_row["white_deadline_at"],
                    )
                    if value
                ]
                deadline_at = max(deadline_candidates) if deadline_candidates else None
            draft_payload = {
                "algorithm_version": str(draft_row["algorithm_version"]),
                "commitment": str(draft_row["commitment"]),
                "first_side": first_side,
                "second_side": second_side,
                "active_side": active_side,
                "phase_started_at": str(draft_row["phase_started_at"]),
                "deadline_at": str(deadline_at) if deadline_at else None,
                "deadlines": {
                    "yellow": str(draft_row["yellow_deadline_at"])
                    if draft_row["yellow_deadline_at"]
                    else None,
                    "white": str(draft_row["white_deadline_at"])
                    if draft_row["white_deadline_at"]
                    else None,
                },
                "phase_token": str(draft_row["phase_token"])
                if is_captain
                else None,
                "project_a": str(draft_row["project_a"])
                if draft_row["project_a"]
                else None,
                "ban_m": str(draft_row["ban_m"])
                if draft_row["ban_m"]
                else None,
                "project_b": str(draft_row["project_b"])
                if draft_row["project_b"]
                else None,
                "ban_n": str(draft_row["ban_n"])
                if draft_row["ban_n"]
                else None,
                "available_project_keys": available_keys,
                "blind_submissions": {
                    "yellow": bool(draft_row["blind_yellow"]),
                    "white": bool(draft_row["blind_white"]),
                },
                "my_blind_choice": my_blind_choice,
                "project_c": str(draft_row["project_c"])
                if draft_row["project_c"]
                else None,
            }
            if draft_row["project_c"]:
                draft_payload["blind_choices"] = {
                    "yellow": str(draft_row["blind_yellow"]),
                    "white": str(draft_row["blind_white"]),
                }
            draft_payload['c_reveal_at'] = self._hold_until(db, competition_id, 'BLIND_CANDIDATES') if status == 'C_DRAW' else None
            if draft_payload['c_reveal_at']:
                draft_payload['project_c'] = None
            # Public provenance without exposing either blind choice before C_DRAW.
            draft_sources: dict[str, str] = {}
            for event in db.execute(
                """
                SELECT event_type, payload_json FROM competition_events
                WHERE competition_id = ? AND event_type IN
                    ('draft.pick_ban_submitted', 'draft.blind_submitted')
                ORDER BY sequence
                """,
                (competition_id,),
            ).fetchall():
                payload = json.loads(str(event["payload_json"] or "{}"))
                source = "timeout" if payload.get("automatic") else "captain"
                if event["event_type"] == "draft.pick_ban_submitted":
                    key = "A" if payload.get("phase") == CompetitionStatus.FIRST_PICK_BAN.value else "B"
                    draft_sources[key] = source
                elif payload.get("side") in (TeamSide.YELLOW.value, TeamSide.WHITE.value):
                    draft_sources[str(payload["side"])] = source
            draft_payload["sources"] = draft_sources

        lineup_state = db.execute(
            "SELECT * FROM competition_lineup_state WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        lineup_payload: dict[str, Any] | None = None
        can_submit_lineup = False
        if lineup_state is not None:
            lineup_rows = db.execute(
                """
                SELECT side, game_key, position, player_user_id, automatic,
                       submitted_at, revealed_at
                FROM competition_lineups
                WHERE competition_id = ?
                ORDER BY CASE side WHEN 'yellow' THEN 0 ELSE 1 END, game_key
                """,
                (competition_id,),
            ).fetchall()
            submissions = {
                side: sum(1 for row in lineup_rows if str(row["side"]) == side) == 3
                for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value)
            }
            seat_names = {
                (seat["side"], seat["position"]): seat["display_name"]
                for seat in seats
            }
            game_projects = {
                "A": str(draft_row["project_a"]) if draft_row and draft_row["project_a"] else None,
                "B": str(draft_row["project_b"]) if draft_row and draft_row["project_b"] else None,
                "C": str(draft_row["project_c"]) if draft_row and draft_row["project_c"] else None,
            }

            def lineup_for(side: str) -> dict[str, dict[str, Any]]:
                return {
                    str(row["game_key"]): {
                        "game_key": str(row["game_key"]),
                        "project_key": game_projects[str(row["game_key"])],
                        "position": int(row["position"]),
                        "player_user_id": int(row["player_user_id"]),
                        "display_name": seat_names.get(
                            (side, int(row["position"])), "Unknown player"
                        ),
                        "avatar_url": avatar_urls.get(int(row["player_user_id"])),
                        "automatic": bool(row["automatic"]),
                    }
                    for row in lineup_rows
                    if str(row["side"]) == side
                }

            revealed = bool(lineup_rows) and all(row["revealed_at"] for row in lineup_rows)
            own_deadline = (
                lineup_state[
                    "yellow_deadline_at"
                    if my_side == TeamSide.YELLOW.value
                    else "white_deadline_at"
                ]
                if my_side
                else max(
                    str(lineup_state["yellow_deadline_at"]),
                    str(lineup_state["white_deadline_at"]),
                )
            )
            can_submit_lineup = bool(
                is_captain
                and status == CompetitionStatus.LINEUP.value
                and my_side
                and not submissions[my_side]
            )
            lineup_payload = {
                "phase_started_at": str(lineup_state["phase_started_at"]),
                "deadline_at": str(own_deadline),
                "deadlines": {
                    "yellow": str(lineup_state["yellow_deadline_at"]),
                    "white": str(lineup_state["white_deadline_at"]),
                },
                "phase_token": str(lineup_state["phase_token"])
                if is_captain and status == CompetitionStatus.LINEUP.value
                else None,
                "submissions": submissions,
                "revealed": revealed,
                "my_lineup": lineup_for(my_side)
                if my_side and submissions[my_side]
                else None,
                # Both sealed submissions are public once finalization is complete.
                "revealed_lineups": {side: lineup_for(side) for side in ('yellow', 'white')} if revealed else None,
            }

        match_control = db.execute(
            "SELECT * FROM competition_match_control WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()
        match_payload: dict[str, Any] | None = None
        can_mark_player_ready = False
        can_mark_captain_ready = False
        can_start_game = False
        can_move = False
        can_confirm_result = False
        is_match_official = False
        can_report_issue = False
        can_suspend = False
        can_mark_resume_ready = False
        can_resume = False
        can_override_result = False
        can_force_advance = False
        can_force_finish = False
        suspension_payload: dict[str, Any] | None = None
        snapshot_now = datetime.now(timezone.utc)
        if match_control is not None:
            current_game = str(match_control["current_game_key"])
            suspension = self._suspension_row(db, competition_id)
            suspended = bool(suspension["active"])
            suspension_ready_rows = db.execute(
                """
                SELECT side, ready_by_user_id, ready_at
                FROM competition_suspension_readiness
                WHERE competition_id = ?
                """,
                (competition_id,),
            ).fetchall()
            suspension_readiness = {
                side: {
                    "ready": any(str(row["side"]) == side for row in suspension_ready_rows),
                    "ready_by_user_id": next(
                        (
                            int(row["ready_by_user_id"])
                            for row in suspension_ready_rows
                            if str(row["side"]) == side
                        ),
                        None,
                    ),
                }
                for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value)
            }
            readiness_rows = db.execute(
                """
                SELECT * FROM competition_game_readiness
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, current_game),
            ).fetchall()
            game_readiness = {
                str(row["side"]): {
                    "player_ready": bool(row["player_ready_by_user_id"]),
                    "captain_ready": bool(row["captain_ready_by_user_id"]),
                }
                for row in readiness_rows
            }
            for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value):
                game_readiness.setdefault(
                    side, {"player_ready": False, "captain_ready": False}
                )
            current_lineups = {
                str(row["side"]): {
                    "player_user_id": int(row["player_user_id"]),
                    "position": int(row["position"]),
                    "display_name": seat_names.get(
                        (str(row["side"]), int(row["position"])), "Unknown player"
                    ),
                    "avatar_url": avatar_urls.get(int(row["player_user_id"])),
                }
                for row in db.execute(
                    """
                    SELECT side, player_user_id, position FROM competition_lineups
                    WHERE competition_id = ? AND game_key = ?
                    """,
                    (competition_id, current_game),
                ).fetchall()
            }
            clock_rows = db.execute(
                """
                SELECT * FROM competition_team_clocks
                WHERE competition_id = ? ORDER BY side
                """,
                (competition_id,),
            ).fetchall()
            clocks: dict[str, dict[str, Any]] = {}
            for clock in clock_rows:
                running_since = parse_time(str(clock["running_since"] or ""))
                deadline_at = (
                    (
                        running_since
                        + timedelta(milliseconds=int(clock["remaining_ms_base"]))
                    ).isoformat()
                    if running_since and str(clock["state"]) == "running"
                    else None
                )
                clocks[str(clock["side"])] = {
                    "state": str(clock["state"]),
                    "remaining_ms": self._clock_remaining_ms(clock, snapshot_now),
                    "deadline_at": deadline_at,
                    "revision": int(clock["revision"]),
                }
            sessions = db.execute(
                """
                SELECT * FROM competition_game_sessions
                WHERE competition_id = ? AND game_key = ?
                ORDER BY side
                """,
                (competition_id, current_game),
            ).fetchall()
            public_sessions = {}
            for row in sessions:
                adapter = self._adapter_for_session_row(row)
                state = self._session_state(row, db, now=snapshot_now)
                side = str(row["side"])
                limit_ms = getattr(adapter, "time_limit_ms", None)
                public_sessions[str(row["side"])] = {
                    "instance_id": str(row["instance_id"]),
                    "state": str(row["state"]),
                    "finished": str(row["state"]) == "completed",
                    "project_key": str(row["project_key"]),
                    "project_clock": {
                        "mode": "countdown" if limit_ms is not None else "elapsed",
                        "elapsed_ms": state.elapsed_ms,
                        "limit_ms": int(limit_ms) if limit_ms is not None else None,
                        "running": (
                            str(row["state"]) == "playing"
                            and clocks.get(side, {}).get("state") == "running"
                            and not suspended
                        ),
                    },
                    "public_view": client_runtime.public_view(adapter, state, int(row["public_generation"])),
                }
            my_session_row = next(
                (
                    row
                    for row in sessions
                    if int(row["player_user_id"]) == principal.user_id
                ),
                None,
            )
            my_session = (
                {
                    "side": str(my_session_row["side"]),
                    "player_user_id": int(my_session_row["player_user_id"]),
                    "project_key": str(my_session_row["project_key"]),
                    **self._private_session_payload(my_session_row, db, now=snapshot_now),
                }
                if my_session_row is not None
                else None
            )
            result_rows = db.execute(
                """
                SELECT * FROM competition_game_results
                WHERE competition_id = ? ORDER BY game_key
                """,
                (competition_id,),
            ).fetchall()
            results = [
                {
                    "game_key": str(row["game_key"]),
                    "project_key": (
                        str(draft_row[f"project_{str(row['game_key']).lower()}"])
                        if draft_row
                        else None
                    ),
                    "yellow_score": int(row["yellow_score"]),
                    "white_score": int(row["white_score"]),
                    "winner_side": str(row["winner_side"]),
                    "reason": str(row["reason"]),
                    "result_revision": int(row["result_revision"]),
                    "corrected": bool(row["corrected_at"]),
                    "corrected_by_user_id": int(row["corrected_by_user_id"])
                    if row["corrected_by_user_id"]
                    else None,
                    "correction_reason": str(row["correction_reason"])
                    if row["correction_reason"]
                    else None,
                    "corrected_at": str(row["corrected_at"])
                    if row["corrected_at"]
                    else None,
                    "published_at": str(row["published_at"]),
                }
                for row in result_rows
            ]
            self._attach_result_timings(db, competition_id, results)
            current_result = next(
                (item for item in results if item["game_key"] == current_game), None
            )
            confirmation_rows = db.execute(
                """
                SELECT side, result_revision FROM competition_result_confirmations
                WHERE competition_id = ? AND game_key = ?
                """,
                (competition_id, current_game),
            ).fetchall()
            confirmations = {
                side: any(str(row["side"]) == side for row in confirmation_rows)
                for side in (TeamSide.YELLOW.value, TeamSide.WHITE.value)
            }
            is_ready_phase = status == GAME_READY_STATUS[current_game]
            is_playing_phase = status == GAME_PLAYING_STATUS[current_game]
            is_result_phase = status == GAME_RESULT_STATUS[current_game]
            active_side = next(
                (
                    side
                    for side, assignment in current_lineups.items()
                    if assignment["player_user_id"] == principal.user_id
                ),
                None,
            )
            can_mark_player_ready = bool(is_ready_phase and active_side and not suspended)
            can_mark_captain_ready = bool(
                is_ready_phase and is_captain and my_side and not suspended
            )
            is_official = bool(
                self._is_platform_organizer(principal)
                or int(room['created_by_user_id']) == principal.user_id
                or {StaffRole.ORGANIZER.value, StaffRole.REFEREE.value}
                & set(staff_roles)
            )
            is_match_official = is_official
            readiness_complete = all(
                state["player_ready"] and state["captain_ready"]
                for state in game_readiness.values()
            )
            can_start_game = bool(
                is_ready_phase and is_official and readiness_complete and not suspended
            )
            can_move = bool(
                is_playing_phase
                and not suspended
                and my_session is not None
                and not my_session["finished"]
                and clocks.get(str(my_session["side"]), {}).get("state") == "running"
            )
            can_confirm_result = bool(
                is_result_phase
                and not suspended
                and is_captain
                and my_side
                and not confirmations[my_side]
            )
            can_report_issue = bool(my_seat and status in MATCH_ACTIVE_STATUSES)
            can_suspend = bool(
                is_official and status in MATCH_ACTIVE_STATUSES and not suspended
            )
            can_mark_resume_ready = bool(suspended and is_captain)
            can_resume = bool(
                suspended
                and is_official
                and all(item["ready"] for item in suspension_readiness.values())
            )
            can_override_result = bool(
                is_official and is_result_phase and not suspended
            )
            can_force_advance = can_override_result
            can_force_finish = bool(is_official and status in MATCH_ACTIVE_STATUSES)
            expose_phase_token = bool(
                can_mark_player_ready
                or (is_playing_phase and active_side is not None)
                or can_mark_captain_ready
                or is_official
                or can_move
                or can_confirm_result
                or can_suspend
                or can_mark_resume_ready
                or can_resume
                or can_override_result
                or can_force_finish
            )
            suspension_payload = {
                "active": suspended,
                "reason_code": str(suspension["reason_code"])
                if suspension["reason_code"]
                else None,
                "reason_text": str(suspension["reason_text"])
                if suspension["reason_text"]
                else None,
                "started_by_user_id": int(suspension["started_by_user_id"])
                if suspension["started_by_user_id"]
                else None,
                "started_by_display_name": str(suspension["started_by_display_name"])
                if suspension["started_by_display_name"]
                else None,
                "started_at": str(suspension["started_at"])
                if suspension["started_at"]
                else None,
                "resume_readiness": suspension_readiness,
            }
            match_payload = {
                "prediction_window": self._prediction_window(db, room),
                "current_game_key": current_game,
                "project_key": (
                    str(draft_row[f"project_{current_game.lower()}"])
                    if draft_row
                    else None
                ),
                "phase_token": str(match_control["phase_token"])
                if expose_phase_token
                else None,
                "players": current_lineups,
                "preview_until": self._hold_until(db, competition_id, status),
                "rest_until": (parse_time(current_result['published_at']) + timedelta(seconds=self.result_rest_seconds)).isoformat() if current_result else None,
                "readiness": game_readiness,
                "ready_deadline_at": self._hold_until(db,competition_id,status+'_TIMEOUT'),
                "series_points": {'yellow':2*int(match_control['yellow_wins'])+int(match_control['draws']), 'white':2*int(match_control['white_wins'])+int(match_control['draws'])},
                "readiness_complete": readiness_complete,
                "clocks": clocks,
                "sessions": public_sessions,
                "my_session": my_session,
                "current_result": current_result,
                "confirmations": confirmations,
                "series_score": {
                    "yellow": int(match_control["yellow_wins"]),
                    "white": int(match_control["white_wins"]),
                    "draws": int(match_control["draws"]),
                },
                "results": results,
                "winner_side": str(match_control["winner_side"])
                if match_control["winner_side"]
                else None,
                "finish_reason": str(match_control["finish_reason"])
                if match_control["finish_reason"]
                else None,
                "adapter": (
                    self._adapter_for_session_row(sessions[0]).descriptor.snapshot()
                    if sessions
                    else next(
                        (
                            item["adapter"]
                            for item in projects
                            if item["key"]
                            == (
                                str(draft_row[f"project_{current_game.lower()}"])
                                if draft_row
                                else None
                            )
                        ),
                        None,
                    )
                ),
                "suspension": suspension_payload,
            }
        issue_where = "competition_id = ? AND status = 'open'"
        issue_params: tuple[Any, ...] = (competition_id,)
        if not is_match_official:
            issue_where += " AND reported_by_user_id = ?"
            issue_params += (principal.user_id,)
        issue_rows = db.execute(
            f"""
            SELECT * FROM competition_issue_reports
            WHERE {issue_where} ORDER BY created_at
            """,
            issue_params,
        ).fetchall()
        seat_names_by_user = {seat["user_id"]: seat["display_name"] for seat in seats}
        issues = [
            {
                "id": str(row["id"]),
                "reported_by_user_id": int(row["reported_by_user_id"]),
                "reporter_display_name": seat_names_by_user.get(
                    int(row["reported_by_user_id"]), "Unknown player"
                ),
                "category": str(row["category"]),
                "details": str(row["details"]),
                "status": str(row["status"]),
                "created_at": str(row["created_at"]),
            }
            for row in issue_rows
        ]
        rematch = db.execute("SELECT payload_json FROM competition_events WHERE competition_id=? AND event_type='competition.rematch' ORDER BY sequence DESC LIMIT 1", (competition_id,)).fetchone()
        return {
            "id": competition_id,
            "room_code": str(room["room_code"]),
            "replacement_room_code": json.loads(rematch[0])['replacement_room_code'] if rematch else None,
            "name": str(room["name"]),
            "status": status,
            "version": int(room["version"]),
            "event_sequence": int(latest_sequence_row["sequence"]),
            "event": self.events.room_event(db, competition_id),
            "schedule": self.schedule.view(db, competition_id),
            "member_hold": bool(db.execute('SELECT 1 FROM competition_expulsions e JOIN competition_seats s ON s.competition_id=e.competition_id AND s.user_id=e.user_id WHERE e.competition_id=?', (competition_id,)).fetchone()),
            "created_at": str(room["created_at"]),
            "updated_at": str(room["updated_at"]),
            "server_time": snapshot_now.isoformat(),
            "seats": seats,
            "teams": readiness,
            "projects": projects,
            "draft": draft_payload,
            "lineup": lineup_payload,
            "match": match_payload,
            "issues": issues,
            "me": {
                "user_id": principal.user_id,
                "display_name": principal.display_name,
                "site_role": principal.site_role,
                "staff_roles": staff_roles,
                "seat": my_seat,
                "is_captain": is_captain,
                "can_manage": can_manage,
                "can_close": can_manage and status in CLOSABLE_ROOM_STATUSES,
                "can_manage_members": self._is_platform_organizer(principal) or int(room['created_by_user_id']) == principal.user_id,
                "removed_members": [dict(row) for row in db.execute('SELECT user_id FROM competition_expulsions WHERE competition_id=?', (competition_id,))] if self._is_platform_organizer(principal) or int(room['created_by_user_id']) == principal.user_id else [],
                "can_claim_seat": status in {
                    CompetitionStatus.SEATING.value,
                    CompetitionStatus.READY_CHECK.value,
                }
                ,
                "can_leave_seat": my_seat is not None
                and status in {
                    CompetitionStatus.SEATING.value,
                    CompetitionStatus.READY_CHECK.value,
                }
                ,
                "can_ready": is_captain and sum(s['side']==my_seat['side'] for s in seats)==3
                and status in ('SEATING','READY_CHECK'),
                "can_submit_pick_ban": can_submit_pick_ban,
                "can_submit_blind": can_submit_blind,
                "can_submit_lineup": can_submit_lineup,
                "can_mark_player_ready": can_mark_player_ready,
                "can_mark_captain_ready": can_mark_captain_ready,
                "can_start_game": can_start_game,
                "can_move": can_move,
                "can_confirm_result": can_confirm_result,
                "can_report_issue": can_report_issue,
                "can_suspend": can_suspend,
                "can_mark_resume_ready": can_mark_resume_ready,
                "can_resume": can_resume,
                "can_override_result": can_override_result,
                "can_force_advance": can_force_advance,
                "can_force_finish": can_force_finish,
                "is_match_official": is_match_official,
            },
        }
