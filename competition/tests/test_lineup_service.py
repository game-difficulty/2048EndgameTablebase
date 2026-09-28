from __future__ import annotations

import json
from datetime import timedelta

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import CompetitionStatus, Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService, parse_time


def player(user_id: int, *, role: str = "user") -> Principal:
    return Principal(user_id=user_id, display_name=f"Player {user_id}", site_role=role)


@pytest.fixture()
def service(tmp_path) -> CompetitionService:
    value = CompetitionService(
        CompetitionDatabase(tmp_path / "lineup.sqlite3"),
        draw_reveal_seconds=1,
        draft_turn_seconds=5,
        c_draw_reveal_seconds=1,
        lineup_seconds=5,
    )
    value.initialize()
    return value


def prepare_lineup(
    service: CompetitionService, code: str = "LINE42"
) -> tuple[list[Principal], dict]:
    service.create_competition(
        player(1, role="admin"), name="Lineup Cup", room_code=code
    )
    players = [player(user_id) for user_id in range(10, 16)]
    for index, participant in enumerate(players):
        service.claim_seat(
            code,
            participant,
            side="yellow" if index < 3 else "white",
            position=(index % 3) + 1,
            command_id=f"lineup-seat-{participant.user_id}",
        )
    service.set_ready(code, players[0], ready=True, command_id="lineup-ready-yellow")
    snapshot = service.set_ready(
        code, players[3], ready=True, command_id="lineup-ready-white"
    )

    draw_deadline = parse_time(snapshot["draft"]["deadline_at"])
    assert draw_deadline is not None
    service.settle_deadline(code, now=draw_deadline + timedelta(milliseconds=1))
    snapshot = service.snapshot(code, players[0])
    keys = [project["key"] for project in snapshot["projects"]]
    first_captain = players[0] if snapshot["draft"]["first_side"] == "yellow" else players[3]
    second_captain = players[3] if first_captain is players[0] else players[0]

    first_view = service.snapshot(code, first_captain)
    service.submit_pick_ban(
        code,
        first_captain,
        pick_project_key=keys[0],
        ban_project_key=keys[1],
        phase_token=first_view["draft"]["phase_token"],
        command_id="lineup-first-pick-ban",
    )
    second_view = service.snapshot(code, second_captain)
    service.submit_pick_ban(
        code,
        second_captain,
        pick_project_key=keys[2],
        ban_project_key=keys[3],
        phase_token=second_view["draft"]["phase_token"],
        command_id="lineup-second-pick-ban",
    )
    yellow_view = service.snapshot(code, players[0])
    service.submit_blind_pick(
        code,
        players[0],
        project_key=keys[4],
        phase_token=yellow_view["draft"]["phase_token"],
        command_id="lineup-blind-yellow",
    )
    white_view = service.snapshot(code, players[3])
    snapshot = service.submit_blind_pick(
        code,
        players[3],
        project_key=keys[5],
        phase_token=white_view["draft"]["phase_token"],
        command_id="lineup-blind-white",
    )
    assert snapshot["status"] == CompetitionStatus.C_DRAW.value
    reveal_deadline = parse_time(snapshot["draft"]["deadline_at"])
    assert reveal_deadline is not None
    assert service.settle_deadline(
        code, now=reveal_deadline + timedelta(milliseconds=1)
    )
    snapshot = service.snapshot(code, players[0])
    assert snapshot["status"] == CompetitionStatus.LINEUP.value
    return players, snapshot


def test_lineups_stay_secret_until_both_captains_submit(
    service: CompetitionService,
) -> None:
    players, yellow_view = prepare_lineup(service)
    assignments = {"A": 2, "B": 3, "C": 1}
    submitted = service.submit_lineup(
        "LINE42",
        players[0],
        assignments=assignments,
        phase_token=yellow_view["lineup"]["phase_token"],
        command_id="lineup-yellow-submit",
    )
    assert submitted["lineup"]["my_lineup"]["A"]["position"] == 2
    assert submitted["lineup"]["submissions"] == {
        "yellow": True,
        "white": False,
    }
    assert submitted["lineup"]["revealed_lineups"] is None

    teammate = service.snapshot("LINE42", players[1])
    assert teammate["lineup"]["my_lineup"]["A"]["position"] == 2

    service.assign_staff("LINE42", player(1, role="admin"), user_id=99, role="referee")
    for viewer in (players[3], player(1, role="admin"), player(99), player(100)):
        hidden = service.snapshot("LINE42", viewer)
        assert hidden["lineup"]["my_lineup"] is None
        assert hidden["lineup"]["revealed_lineups"] is None

    with service.database.transaction() as db:
        event = db.execute(
            """
            SELECT payload_json FROM competition_events
            WHERE event_type = 'lineup.submitted'
            ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
    payload = json.loads(str(event["payload_json"]))
    assert payload == {"automatic": False, "side": "yellow"}

    retried = service.submit_lineup(
        "LINE42",
        players[0],
        assignments=assignments,
        phase_token=yellow_view["lineup"]["phase_token"],
        command_id="lineup-yellow-submit",
    )
    assert retried["lineup"]["submissions"]["yellow"] is True

    white_view = service.snapshot("LINE42", players[3])
    completed = service.submit_lineup(
        "LINE42",
        players[3],
        assignments={"A": 3, "B": 1, "C": 2},
        phase_token=white_view["lineup"]["phase_token"],
        command_id="lineup-white-submit",
    )
    assert completed["status"] == CompetitionStatus.GAME_A_READY.value
    assert completed["lineup"]["my_lineup"]["A"]["display_name"] == "Player 15"
    assert completed["lineup"]["revealed_lineups"] is None
    assert set(completed["match"]["players"]) == {"white"}
    yellow = service.snapshot("LINE42", players[2])
    assert yellow["lineup"]["my_lineup"]["A"]["display_name"] == "Player 11"
    assert set(yellow["match"]["players"]) == {"yellow"}
    for viewer in (player(1, role="admin"), player(99), player(100)):
        hidden = service.snapshot("LINE42", viewer)
        assert hidden["lineup"]["my_lineup"] is None
        assert hidden["lineup"]["revealed_lineups"] is None
        assert hidden["match"]["players"] == {}
        assert "Player 11" not in json.dumps(hidden["lineup"])
    public_key = service.list_live_rooms()[0]["public_key"]
    projection = service.live_projection(public_key)
    assert projection["revealed_players"] == {"yellow": {}, "white": {}}
    assert all(game["players"] == {"yellow": None, "white": None} for game in projection["games"])
    with service.database.transaction() as db:
        event = db.execute(
            "SELECT payload_json FROM competition_events WHERE event_type = 'lineup.finalized'"
        ).fetchone()
    assert "lineups" not in json.loads(str(event["payload_json"]))


def test_lineup_requires_captain_valid_permutation_and_current_token(
    service: CompetitionService,
) -> None:
    players, snapshot = prepare_lineup(service)
    token = snapshot["lineup"]["phase_token"]
    with pytest.raises(CompetitionError) as captured:
        service.submit_lineup(
            "LINE42",
            players[0],
            assignments={"A": 1, "B": 1, "C": 3},
            phase_token=token,
            command_id="lineup-invalid-order",
        )
    assert captured.value.code == "INVALID_LINEUP"

    with pytest.raises(CompetitionError) as captured:
        service.submit_lineup(
            "LINE42",
            players[1],
            assignments={"A": 1, "B": 2, "C": 3},
            phase_token=token,
            command_id="lineup-non-captain",
        )
    assert captured.value.code == "CAPTAIN_REQUIRED"

    with pytest.raises(CompetitionError) as captured:
        service.submit_lineup(
            "LINE42",
            players[0],
            assignments={"A": 1, "B": 2, "C": 3},
            phase_token="stale-lineup-token",
            command_id="lineup-stale-token",
        )
    assert captured.value.code == "STALE_PHASE"


def test_lineup_timeout_uses_default_seat_order(service: CompetitionService) -> None:
    players, snapshot = prepare_lineup(service)
    deadlines = [
        parse_time(value) for value in snapshot["lineup"]["deadlines"].values()
    ]
    assert all(deadlines)
    assert service.settle_deadline(
        "LINE42", now=max(deadlines) + timedelta(milliseconds=1)
    )
    completed = service.snapshot("LINE42", players[0])
    assert completed["status"] == CompetitionStatus.GAME_A_READY.value
    for side, participant in (("yellow", players[0]), ("white", players[3])):
        lineup = service.snapshot("LINE42", participant)["lineup"]["my_lineup"]
        assert {game: item["position"] for game, item in lineup.items()} == {
            "A": 1,
            "B": 2,
            "C": 3,
        }
        assert all(item["automatic"] for item in lineup.values())
