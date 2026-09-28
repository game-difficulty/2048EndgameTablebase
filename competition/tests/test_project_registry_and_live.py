from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.projects import ProjectRegistry, Standard2048Adapter
from competition.backend.service import CompetitionService, parse_time
from competition.tests.test_draft_service import captain_for_side, prepare_draw
from competition.tests.test_lineup_service import prepare_lineup
from competition.tests.test_match_service import prepare_game_a, ready_and_start


def player(user_id: int, *, role: str = "user") -> Principal:
    return Principal(user_id=user_id, display_name=f"Player {user_id}", site_role=role)


def test_live_directory_starts_at_draw_and_expires_after_result_retention(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "live-lifecycle.sqlite3"),
        draw_reveal_seconds=1,
        live_result_retention_seconds=60,
    )
    service.initialize()
    service.create_competition(
        player(1, role="admin"), name="Lifecycle Cup", room_code="LIVE42"
    )
    assert service.list_live_rooms() == []

    players = [player(user_id) for user_id in range(20, 26)]
    for index, participant in enumerate(players):
        service.claim_seat(
            "LIVE42",
            participant,
            side="yellow" if index < 3 else "white",
            position=(index % 3) + 1,
            command_id=f"live-seat-{participant.user_id}",
        )
    service.set_ready(
        "LIVE42", players[0], ready=True, command_id="live-ready-yellow"
    )
    assert service.list_live_rooms() == []
    service.set_ready(
        "LIVE42", players[3], ready=True, command_id="live-ready-white"
    )
    directory = service.list_live_rooms()
    assert len(directory) == 1
    public_key = directory[0]["public_key"]

    ended_at = datetime.now(timezone.utc)
    with service.database.transaction(immediate=True) as db:
        db.execute(
            """
            UPDATE competitions SET status = 'FINISHED', live_ended_at = ?
            WHERE room_code = 'LIVE42'
            """,
            (ended_at.isoformat(),),
        )
    assert len(service.list_live_rooms(now=ended_at + timedelta(seconds=59))) == 1
    assert service.list_live_rooms(now=ended_at + timedelta(seconds=61)) == []

    with service.database.transaction(immediate=True) as db:
        db.execute(
            "UPDATE competitions SET live_ended_at = ? WHERE room_code = 'LIVE42'",
            ((datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat(),),
        )
    with pytest.raises(CompetitionError) as captured:
        service.live_projection(public_key)
    assert captured.value.code == "LIVE_ROOM_NOT_FOUND"


def test_project_registry_requires_exact_unique_version() -> None:
    registry = ProjectRegistry()
    registry.register(lambda: Standard2048Adapter(target_tile=128))
    descriptor = registry.snapshot("standard-2048-test", "standard-v1")
    assert descriptor == {
        "project_ref": "standard-2048-test",
        "rules_version": "standard-v1",
        "display_name": "Standard 2048 (test adapter)",
        "view_kind": "2048-board",
        "view_protocol": "2048-board-v1",
        "test_only": True,
    }
    with pytest.raises(ValueError, match="duplicate project adapter"):
        registry.register(Standard2048Adapter)
    with pytest.raises(CompetitionError) as captured:
        registry.resolve("standard-2048-test", "missing")
    assert captured.value.code == "PROJECT_ADAPTER_UNAVAILABLE"


def test_live_projection_never_exposes_sealed_blind_choice(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "public.sqlite3"),
        draw_reveal_seconds=1,
        draft_turn_seconds=5,
    )
    service.initialize()
    players = prepare_draw(service, "PUBLIC2")
    draw = service.snapshot("PUBLIC2", players[0])
    draw_projection = service.live_projection(service.list_live_rooms()[0]["public_key"])
    assert draw_projection["phase"] == "DRAW"
    assert draw_projection["phase_timing"]["mode"] == "reveal"
    assert draw_projection["phase_timing"]["deadline_at"] == draw["draft"]["deadline_at"]
    deadline = parse_time(draw["draft"]["deadline_at"])
    assert deadline is not None
    service.settle_deadline("PUBLIC2", now=deadline + timedelta(milliseconds=1))
    current = service.snapshot("PUBLIC2", players[0])
    keys = [item["key"] for item in current["projects"]]
    first = captain_for_side(players, current["draft"]["first_side"])
    second = captain_for_side(players, current["draft"]["second_side"])
    turn_projection = service.live_projection(
        service.list_live_rooms()[0]["public_key"]
    )
    assert turn_projection["phase_timing"]["mode"] == "turn"
    assert turn_projection["phase_timing"]["active_side"] == current["draft"]["first_side"]
    first_view = service.snapshot("PUBLIC2", first)
    service.submit_pick_ban(
        "PUBLIC2", first, pick_project_key=keys[0], ban_project_key=keys[1],
        phase_token=first_view["draft"]["phase_token"], command_id="public-first-pb",
    )
    second_view = service.snapshot("PUBLIC2", second)
    second_turn_projection = service.live_projection(
        service.list_live_rooms()[0]["public_key"]
    )
    assert second_turn_projection["phase_timing"]["active_side"] == current["draft"]["second_side"]
    service.submit_pick_ban(
        "PUBLIC2", second, pick_project_key=keys[2], ban_project_key=keys[3],
        phase_token=second_view["draft"]["phase_token"], command_id="public-second-pb",
    )
    yellow_view = service.snapshot("PUBLIC2", players[0])
    service.submit_blind_pick(
        "PUBLIC2", players[0], project_key=keys[4],
        phase_token=yellow_view["draft"]["phase_token"], command_id="public-yellow-blind",
    )
    public_key = service.list_live_rooms()[0]["public_key"]
    projection = service.live_projection(public_key)
    assert projection["public_draft"]["blind_submissions"] == {
        "yellow": True,
        "white": False,
    }
    assert "blind_choices" not in projection["public_draft"]
    assert projection["phase_timing"]["mode"] == "simultaneous"
    assert projection["phase_timing"]["active_side"] is None
    assert projection["phase_timing"]["deadline_at"]
    serialized = json.dumps(projection)
    for forbidden in ("phase_token", "room_code", "random_seed_hex", "issues"):
        assert forbidden not in serialized


def test_live_projection_keeps_single_submitted_lineup_sealed(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "lineup-public.sqlite3"),
        draw_reveal_seconds=1, draft_turn_seconds=5,
        c_draw_reveal_seconds=1, lineup_seconds=5,
    )
    service.initialize()
    players, yellow_view = prepare_lineup(service, "PUBLIN")
    service.submit_lineup(
        "PUBLIN", players[0], assignments={"A": 3, "B": 1, "C": 2},
        phase_token=yellow_view["lineup"]["phase_token"],
        command_id="public-secret-lineup",
    )
    public_key = service.list_live_rooms()[0]["public_key"]
    projection = service.live_projection(public_key)
    assert projection["lineup_submission_status"]["yellow"]["submitted"] is True
    assert projection["lineup_submission_status"]["white"]["submitted"] is False
    assert projection["revealed_players"] == {"yellow": {}, "white": {}}
    assert projection["phase_timing"]["mode"] == "simultaneous"
    assert projection["phase_timing"]["deadline_at"] == max(
        projection["phase_timing"]["deadlines"].values()
    )


def test_registered_adapter_exports_versioned_public_views(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "game-public.sqlite3"),
        draw_reveal_seconds=1, draft_turn_seconds=5,
        c_draw_reveal_seconds=1, lineup_seconds=5,
        test_project_target_tile=4,
    )
    service.initialize()
    players = prepare_game_a(service, "PUBGAME")
    ready_and_start(service, players, "A", "PUBGAME")
    public_key = service.list_live_rooms()[0]["public_key"]
    projection = service.live_projection(public_key)
    assert projection["phase"] == "GAME_A_PLAYING"
    assert set(projection["revealed_players"]["yellow"]) == {"A"}
    assert set(projection["revealed_players"]["white"]) == {"A"}
    assert all(game["players"] == {"yellow": None, "white": None}
               for game in projection["games"] if game["game_key"] in {"B", "C"})
    for side in ("yellow", "white"):
        view = projection["project_public_views"][side]
        assert view["view_kind"] == "2048-board"
        assert view["view_protocol"] == "2048-board-v1"
        assert view["generation"] == 1
        assert view["sequence"] == 0
        assert len(view["payload"]["board"]) == 4
    assert "seed" not in json.dumps(projection)
