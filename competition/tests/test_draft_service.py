from __future__ import annotations

import hashlib
import hmac
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
        CompetitionDatabase(tmp_path / "draft.sqlite3"),
        draw_reveal_seconds=1,
        draft_turn_seconds=5,
    )
    value.initialize()
    return value


def prepare_draw(service: CompetitionService, code: str = "DRAFT2") -> list[Principal]:
    service.create_competition(
        player(1, role="admin"), name="Draft Cup", room_code=code
    )
    players = [player(user_id) for user_id in range(10, 16)]
    for index, participant in enumerate(players):
        service.claim_seat(
            code,
            participant,
            side="yellow" if index < 3 else "white",
            position=(index % 3) + 1,
            command_id=f"draft-seat-{participant.user_id}",
        )
    service.set_ready(
        code, players[0], ready=True, command_id="draft-ready-yellow"
    )
    result = service.set_ready(
        code, players[3], ready=True, command_id="draft-ready-white"
    )
    assert result["status"] == CompetitionStatus.DRAW.value
    assert result["draft"]["first_side"] in {"yellow", "white"}
    return players


def advance_draw(service: CompetitionService, code: str, viewer: Principal) -> dict:
    snapshot = service.snapshot(code, viewer)
    deadline = parse_time(snapshot["draft"]["deadline_at"])
    assert deadline is not None
    assert service.settle_deadline(code, now=deadline + timedelta(milliseconds=1))
    return service.snapshot(code, viewer)


def captain_for_side(players: list[Principal], side: str) -> Principal:
    return players[0] if side == "yellow" else players[3]


def test_complete_manual_draft_and_blind_secrecy(service: CompetitionService) -> None:
    players = prepare_draw(service)
    first_snapshot = advance_draw(service, "DRAFT2", players[0])
    first_side = first_snapshot["draft"]["first_side"]
    second_side = first_snapshot["draft"]["second_side"]
    first_captain = captain_for_side(players, first_side)
    second_captain = captain_for_side(players, second_side)
    first_view = service.snapshot("DRAFT2", first_captain)
    keys = [item["key"] for item in first_view["projects"]]

    after_first = service.submit_pick_ban(
        "DRAFT2",
        first_captain,
        pick_project_key=keys[0],
        ban_project_key=keys[1],
        phase_token=first_view["draft"]["phase_token"],
        command_id="draft-first-submit",
    )
    assert after_first["status"] == CompetitionStatus.SECOND_PICK_BAN.value
    assert after_first["draft"]["project_a"] == keys[0]
    assert after_first["draft"]["ban_m"] == keys[1]
    assert after_first["draft"]["sources"]["A"] == "captain"

    second_view = service.snapshot("DRAFT2", second_captain)
    after_second = service.submit_pick_ban(
        "DRAFT2",
        second_captain,
        pick_project_key=keys[2],
        ban_project_key=keys[3],
        phase_token=second_view["draft"]["phase_token"],
        command_id="draft-second-submit",
    )
    assert after_second["status"] == CompetitionStatus.BLIND_PICK.value
    assert after_second["draft"]["sources"]["B"] == "captain"

    yellow_captain = players[0]
    white_captain = players[3]
    yellow_view = service.snapshot("DRAFT2", yellow_captain)
    service.submit_blind_pick(
        "DRAFT2",
        yellow_captain,
        project_key=keys[4],
        phase_token=yellow_view["draft"]["phase_token"],
        command_id="draft-blind-yellow",
    )
    opponent_view = service.snapshot("DRAFT2", white_captain)
    assert opponent_view["draft"]["blind_submissions"]["yellow"] is True
    assert opponent_view["draft"]["my_blind_choice"] is None
    assert "blind_choices" not in opponent_view["draft"]
    assert opponent_view["draft"]["sources"]["yellow"] == "captain"

    with service.database.transaction() as db:
        event = db.execute(
            """
            SELECT payload_json FROM competition_events
            WHERE event_type = 'draft.blind_submitted'
            ORDER BY sequence DESC LIMIT 1
            """
        ).fetchone()
    assert keys[4] not in str(event["payload_json"])

    white_view = service.snapshot("DRAFT2", white_captain)
    completed = service.submit_blind_pick(
        "DRAFT2",
        white_captain,
        project_key=keys[5],
        phase_token=white_view["draft"]["phase_token"],
        command_id="draft-blind-white",
    )
    assert completed["status"] == CompetitionStatus.C_DRAW.value
    assert completed["draft"]["blind_choices"] == {
        "yellow": keys[4],
        "white": keys[5],
    }
    assert completed["draft"]["project_c"] in {keys[4], keys[5]}


def test_only_active_captain_can_submit(service: CompetitionService) -> None:
    players = prepare_draw(service)
    snapshot = advance_draw(service, "DRAFT2", players[0])
    first_side = snapshot["draft"]["first_side"]
    wrong_captain = captain_for_side(
        players, "white" if first_side == "yellow" else "yellow"
    )
    keys = [item["key"] for item in snapshot["projects"]]
    wrong_view = service.snapshot("DRAFT2", wrong_captain)
    with pytest.raises(CompetitionError) as captured:
        service.submit_pick_ban(
            "DRAFT2",
            wrong_captain,
            pick_project_key=keys[0],
            ban_project_key=keys[1],
            phase_token=wrong_view["draft"]["phase_token"],
            command_id="wrong-captain-submit",
        )
    assert captured.value.code == "NOT_ACTIVE_SIDE"

    ordinary_player = players[1]
    with pytest.raises(CompetitionError) as captured:
        service.submit_pick_ban(
            "DRAFT2",
            ordinary_player,
            pick_project_key=keys[0],
            ban_project_key=keys[1],
            phase_token="not-visible",
            command_id="ordinary-player-submit",
        )
    assert captured.value.code in {"STALE_PHASE", "CAPTAIN_REQUIRED"}


def test_all_draft_timeouts_choose_first_available(service: CompetitionService) -> None:
    players = prepare_draw(service)
    snapshot = advance_draw(service, "DRAFT2", players[0])
    keys = [item["key"] for item in snapshot["projects"]]

    for expected_status in (
        CompetitionStatus.SECOND_PICK_BAN.value,
        CompetitionStatus.BLIND_PICK.value,
        CompetitionStatus.C_DRAW.value,
    ):
        deadline_values = [
            parse_time(value)
            for value in snapshot["draft"]["deadlines"].values()
            if value
        ]
        assert deadline_values
        now = max(deadline_values) + timedelta(milliseconds=1)
        assert service.settle_deadline("DRAFT2", now=now)
        snapshot = service.snapshot("DRAFT2", players[0])
        assert snapshot["status"] == expected_status

    assert snapshot["draft"]["project_a"] == keys[0]
    assert snapshot["draft"]["ban_m"] == keys[1]
    assert snapshot["draft"]["project_b"] == keys[2]
    assert snapshot["draft"]["ban_n"] == keys[3]
    assert snapshot["draft"]["blind_choices"] == {
        "yellow": keys[4],
        "white": keys[4],
    }
    assert snapshot["draft"]["project_c"] == keys[4]
    assert snapshot["draft"]["sources"] == {
        "A": "timeout", "B": "timeout", "yellow": "timeout", "white": "timeout"
    }


def test_draws_match_stored_seed_and_commitment(service: CompetitionService) -> None:
    players = prepare_draw(service)
    snapshot = service.snapshot("DRAFT2", players[0])
    with service.database.transaction() as db:
        room = db.execute(
            "SELECT id FROM competitions WHERE room_code = 'DRAFT2'"
        ).fetchone()
        draft = db.execute(
            "SELECT * FROM competition_drafts WHERE competition_id = ?",
            (room["id"],),
        ).fetchone()
    seed = bytes.fromhex(str(draft["random_seed_hex"]))
    commitment = hashlib.sha256(str(room["id"]).encode() + b":" + seed).hexdigest()
    expected_side = "yellow" if hmac.new(seed, b"first-side", hashlib.sha256).digest()[0] % 2 == 0 else "white"
    assert commitment == snapshot["draft"]["commitment"]
    assert expected_side == snapshot["draft"]["first_side"]
    assert "random_seed_hex" not in json.dumps(snapshot)


def test_project_pool_validation_and_snapshot(service: CompetitionService) -> None:
    with pytest.raises(CompetitionError) as captured:
        service.create_competition(
            player(1, role="admin"),
            name="Too Small",
            room_code="SMALL2",
            projects=[{"name": f"P{index}"} for index in range(4)],
        )
    assert captured.value.code == "INVALID_PROJECT_POOL"

    projects = [
        {"key": f"rule-{index}", "name": f"规则 {index}", "rules_version": "v2"}
        for index in range(1, 7)
    ]
    snapshot = service.create_competition(
        player(1, role="admin"),
        name="Configured",
        room_code="RULES2",
        projects=projects,
    )
    assert [item["key"] for item in snapshot["projects"]] == [
        f"rule-{index}" for index in range(1, 7)
    ]
