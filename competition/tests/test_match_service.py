from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import CompetitionStatus, Principal
from competition.backend.errors import CompetitionError
from competition.backend.projects.standard_2048 import Standard2048Adapter
from competition.backend.service import CompetitionService
from competition.tests.test_lineup_service import prepare_lineup


def player(user_id: int, *, role: str = "user") -> Principal:
    return Principal(user_id=user_id, display_name=f"Player {user_id}", site_role=role)


@pytest.fixture()
def service(tmp_path) -> CompetitionService:
    value = CompetitionService(
        CompetitionDatabase(tmp_path / "match.sqlite3"),
        draw_reveal_seconds=1,
        draft_turn_seconds=5,
        c_draw_reveal_seconds=1,
        lineup_seconds=5,
        team_clock_seconds=60,
        test_project_target_tile=4,
    )
    value.initialize()
    return value


def prepare_game_a(
    service: CompetitionService, code: str = "MATCH5"
) -> list[Principal]:
    players, yellow_view = prepare_lineup(service, code)
    service.submit_lineup(
        code,
        players[0],
        assignments={"A": 1, "B": 2, "C": 3},
        phase_token=yellow_view["lineup"]["phase_token"],
        command_id="match-lineup-yellow",
    )
    white_view = service.snapshot(code, players[3])
    completed = service.submit_lineup(
        code,
        players[3],
        assignments={"A": 1, "B": 2, "C": 3},
        phase_token=white_view["lineup"]["phase_token"],
        command_id="match-lineup-white",
    )
    assert completed["status"] == CompetitionStatus.GAME_A_READY.value
    return players


def ready_and_start(
    service: CompetitionService,
    players: list[Principal],
    game_key: str,
    code: str = "MATCH5",
) -> dict:
    snapshot = service.snapshot(code, players[0])
    token = snapshot["match"]["phase_token"]
    active_ids = {
        side: service.snapshot(code, players[0 if side == "yellow" else 3])
        ["match"]["players"][side]["player_user_id"]
        for side in ("yellow", "white")
    }
    by_id = {participant.user_id: participant for participant in players}
    for side in ("yellow", "white"):
        service.set_game_readiness(
            code,
            by_id[active_ids[side]],
            readiness_role="player",
            ready=True,
            phase_token=token,
            command_id=f"match-{game_key}-{side}-player-ready",
        )
    service.set_game_readiness(
        code,
        players[0],
        readiness_role="captain",
        ready=True,
        phase_token=token,
        command_id=f"match-{game_key}-yellow-captain-ready",
    )
    started = service.set_game_readiness(
        code,
        players[3],
        readiness_role="captain",
        ready=True,
        phase_token=token,
        command_id=f"match-{game_key}-white-captain-ready",
    )
    assert started["status"] == f"GAME_{game_key}_PLAYING"
    return started


def force_one_move_completion(
    service: CompetitionService,
    principal: Principal,
    game_key: str,
    side: str,
    code: str = "MATCH5",
) -> dict:
    with service.database.transaction(immediate=True) as db:
        room = db.execute(
            "SELECT id FROM competitions WHERE room_code = ?", (code,)
        ).fetchone()
        db.execute(
            """
            UPDATE competition_game_sessions
            SET board_json = ?, score = 0, move_count = 0
            WHERE competition_id = ? AND game_key = ? AND side = ?
            """,
            (json.dumps([[2, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]), room["id"], game_key, side),
        )
    snapshot = service.snapshot(code, principal)
    return service.move_current_game(
        code,
        principal,
        direction="left",
        phase_token=snapshot["match"]["phase_token"],
        command_id=f"match-{game_key}-{side}-winning-move",
    )


def test_standard_adapter_is_deterministic_and_rejects_noop() -> None:
    adapter = Standard2048Adapter(target_tile=2048)
    seed = "11" * 32
    first = adapter.initial_state(seed=seed)
    second = adapter.initial_state(seed=seed)
    assert first.board == second.board
    assert first.rng_counter == 2
    with pytest.raises(ValueError):
        adapter.apply_move(first, "diagonal")


def test_fourth_readiness_starts_game_without_official_and_retry_is_idempotent(
    service: CompetitionService,
) -> None:
    players = prepare_game_a(service)
    old_token = service.snapshot("MATCH5", players[0])["match"]["phase_token"]
    started = ready_and_start(service, players, "A")
    assert started["status"] == CompetitionStatus.GAME_A_PLAYING.value
    assert started["me"]["is_match_official"] is False
    retry = service.set_game_readiness(
        "MATCH5", players[3], readiness_role="captain", ready=True,
        phase_token=old_token, command_id="match-A-white-captain-ready",
    )
    assert retry["status"] == CompetitionStatus.GAME_A_PLAYING.value
    with service.database.transaction() as db:
        room = db.execute("SELECT id FROM competitions WHERE room_code = 'MATCH5'").fetchone()
        game_count = db.execute(
            "SELECT COUNT(*) AS count FROM competition_game_sessions WHERE competition_id = ? AND game_key = 'A'",
            (room["id"],),
        ).fetchone()["count"]
        start_count = db.execute(
            "SELECT COUNT(*) AS count FROM competition_events WHERE competition_id = ? AND event_type = 'game.started'",
            (room["id"],),
        ).fetchone()["count"]
    assert game_count == 2
    assert start_count == 1


def test_ready_gate_clocks_private_play_and_result_confirmation(
    service: CompetitionService,
) -> None:
    players = prepare_game_a(service)
    initial = service.snapshot("MATCH5", players[0])
    assert initial["match"]["current_game_key"] == "A"
    assert all(clock["state"] == "stopped" for clock in initial["match"]["clocks"].values())

    with pytest.raises(CompetitionError) as captured:
        service.start_current_game(
            "MATCH5",
            player(1, role="admin"),
            phase_token=service.snapshot("MATCH5", player(1, role="admin"))["match"]["phase_token"],
            command_id="match-start-too-early",
        )
    assert captured.value.code == "READINESS_INCOMPLETE"

    started = ready_and_start(service, players, "A")
    assert started["status"] == CompetitionStatus.GAME_A_PLAYING.value
    assert all(clock["state"] == "running" for clock in started["match"]["clocks"].values())
    assert service.snapshot("MATCH5", players[0])["match"]["my_session"]["board"]

    yellow_done = force_one_move_completion(service, players[0], "A", "yellow")
    assert yellow_done["match"]["clocks"]["yellow"]["state"] == "stopped"
    assert yellow_done["match"]["clocks"]["white"]["state"] == "running"
    white_view = service.snapshot("MATCH5", players[3])
    opponent = white_view["match"]["sessions"]["yellow"]
    assert opponent["state"] == "completed"
    assert opponent["finished"] is True
    assert opponent["public_view"]["view_kind"] == "2048-board"
    assert opponent["public_view"]["view_protocol"] == "2048-board-v1"
    assert opponent["public_view"]["payload"]["board"]
    teammate_view = service.snapshot("MATCH5", players[1])
    assert teammate_view["match"]["my_session"] is None
    assert set(teammate_view["match"]["sessions"]) == {"yellow", "white"}
    assert all(
        session["public_view"]["payload"]["board"]
        for session in teammate_view["match"]["sessions"].values()
    )

    completed = force_one_move_completion(service, players[3], "A", "white")
    assert completed["status"] == CompetitionStatus.GAME_A_RESULT.value
    assert completed["match"]["current_result"]["winner_side"] == "draw"

    yellow_result = service.snapshot("MATCH5", players[0])
    service.confirm_current_result(
        "MATCH5",
        players[0],
        result_revision=yellow_result["match"]["current_result"]["result_revision"],
        phase_token=yellow_result["match"]["phase_token"],
        command_id="match-a-confirm-yellow",
    )
    white_result = service.snapshot("MATCH5", players[3])
    next_game = service.confirm_current_result(
        "MATCH5",
        players[3],
        result_revision=white_result["match"]["current_result"]["result_revision"],
        phase_token=white_result["match"]["phase_token"],
        command_id="match-a-confirm-white",
    )
    assert next_game["status"] == CompetitionStatus.GAME_B_READY.value
    assert next_game["match"]["current_game_key"] == "B"
    assert next_game["match"]["series_score"] == {
        "yellow": 0,
        "white": 0,
        "draws": 1,
    }


def test_three_games_progress_to_finished(service: CompetitionService) -> None:
    players = prepare_game_a(service)
    active_by_game = {
        "A": (players[0], players[3]),
        "B": (players[1], players[4]),
        "C": (players[2], players[5]),
    }
    for game_key in ("A", "B", "C"):
        ready_and_start(service, players, game_key)
        yellow_player, white_player = active_by_game[game_key]
        force_one_move_completion(service, yellow_player, game_key, "yellow")
        result_view = force_one_move_completion(
            service, white_player, game_key, "white"
        )
        revision = result_view["match"]["current_result"]["result_revision"]
        service.confirm_current_result(
            "MATCH5",
            players[0],
            result_revision=revision,
            phase_token=service.snapshot("MATCH5", players[0])["match"]["phase_token"],
            command_id=f"match-{game_key}-confirm-yellow",
        )
        final = service.confirm_current_result(
            "MATCH5",
            players[3],
            result_revision=revision,
            phase_token=service.snapshot("MATCH5", players[3])["match"]["phase_token"],
            command_id=f"match-{game_key}-confirm-white",
        )
    assert final["status"] == CompetitionStatus.FINISHED.value
    assert final["match"]["winner_side"] == "draw"
    assert len(final["match"]["results"]) == 3


def test_team_clock_expiry_forfeits_all_remaining_games(
    service: CompetitionService,
) -> None:
    players = prepare_game_a(service)
    ready_and_start(service, players, "A")
    with service.database.transaction(immediate=True) as db:
        room = db.execute(
            "SELECT id FROM competitions WHERE room_code = 'MATCH5'"
        ).fetchone()
        db.execute(
            """
            UPDATE competition_team_clocks
            SET remaining_ms_base = 1, running_since = ?
            WHERE competition_id = ? AND side = 'yellow'
            """,
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), room["id"]),
        )
    assert service.settle_deadline("MATCH5") is True
    finished = service.snapshot("MATCH5", players[0])
    assert finished["status"] == CompetitionStatus.FINISHED.value
    assert finished["match"]["winner_side"] == "white"
    assert finished["match"]["finish_reason"] == "yellow_clock_expired"
    assert [result["winner_side"] for result in finished["match"]["results"]] == [
        "white",
        "white",
        "white",
    ]
