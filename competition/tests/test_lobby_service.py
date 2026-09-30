from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import CompetitionStatus, Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService


def principal(user_id: int, *, role: str = "user") -> Principal:
    return Principal(user_id=user_id, display_name=f"Player {user_id}", site_role=role)


@pytest.fixture()
def service(tmp_path) -> CompetitionService:
    value = CompetitionService(CompetitionDatabase(tmp_path / "competition.sqlite3"))
    value.initialize()
    return value


def create_room(service: CompetitionService, code: str = "TEAM42") -> dict:
    return service.create_competition(
        principal(1, role="admin"),
        name="Autumn Cup",
        room_code=code,
    )


def fill_room(service: CompetitionService, code: str = "TEAM42") -> list[Principal]:
    players = [principal(user_id) for user_id in range(10, 16)]
    assignments = [
        ("yellow", 1),
        ("yellow", 2),
        ("yellow", 3),
        ("white", 1),
        ("white", 2),
        ("white", 3),
    ]
    for index, (player, assignment) in enumerate(zip(players, assignments), start=1):
        service.claim_seat(
            code,
            player,
            side=assignment[0],
            position=assignment[1],
            command_id=f"seat-command-{index}",
        )
    return players


def test_only_platform_organizer_can_create(service: CompetitionService) -> None:
    with pytest.raises(CompetitionError) as captured:
        service.create_competition(principal(7), name="No Access")
    assert captured.value.code == "ORGANIZER_REQUIRED"


def test_room_creator_can_open_only_their_own_room(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "room-creators.sqlite3"),
        room_creator_ids=frozenset({89}),
    )
    service.initialize()
    creator = principal(89)
    admin = principal(1, role="admin")
    assert service._can_create_competition(creator)
    assert not service._is_platform_organizer(creator)

    service.create_competition(admin, name="Other Cup", room_code="TEST77")
    own = service.create_competition(creator, name="Own Cup", room_code="MINE23")
    assert own["me"]["staff_roles"] == ["organizer"]
    assert [room["room_code"] for room in service.list_competitions(creator)] == ["MINE23"]
    with pytest.raises(CompetitionError) as denied:
        service.close_competition("TEST77", creator, command_id="creator-close-other")
    assert denied.value.code == "ORGANIZER_REQUIRED"
    assert service.close_competition("MINE23", creator, command_id="creator-close-own")["status"] == CompetitionStatus.CANCELLED.value


def test_organizer_can_close_room_before_draw_without_creating_live_room(
    service: CompetitionService,
) -> None:
    create_room(service)
    players = fill_room(service)
    service.set_ready("TEAM42", players[0], ready=True, command_id="ready-before-close")

    with pytest.raises(CompetitionError) as unauthorized:
        service.close_competition(
            "TEAM42", players[1], command_id="unauthorized-close"
        )
    assert unauthorized.value.code == "ORGANIZER_REQUIRED"

    closed = service.close_competition(
        "TEAM42", principal(1, role="admin"), command_id="close-room-1"
    )
    assert closed["status"] == CompetitionStatus.CANCELLED.value
    assert closed["me"]["can_close"] is False
    assert closed["me"]["can_claim_seat"] is False
    assert service.list_live_rooms() == []
    listing = service.list_competitions(principal(1, role="admin"))
    assert listing[0]["status"] == CompetitionStatus.CANCELLED.value
    assert listing[0]["can_close"] is False
    assert service.close_competition(
        "TEAM42", principal(1, role="admin"), command_id="close-room-1"
    )["version"] == closed["version"]
    with pytest.raises(CompetitionError) as repeated:
        service.close_competition(
            "TEAM42", principal(1, role="admin"), command_id="close-room-2"
        )
    assert repeated.value.code == "ROOM_CLOSE_UNAVAILABLE"
    with service.database.transaction() as db:
        events = db.execute(
            "SELECT event_type FROM competition_events WHERE competition_id = ? ORDER BY sequence",
            (closed["id"],),
        ).fetchall()
    assert events[-1]["event_type"] == "competition.closed"


def test_room_cannot_be_closed_after_draw_begins(service: CompetitionService) -> None:
    create_room(service)
    players = fill_room(service)
    service.set_ready("TEAM42", players[0], ready=True, command_id="ready-draw-yellow")
    service.set_ready("TEAM42", players[3], ready=True, command_id="ready-draw-white")
    with pytest.raises(CompetitionError) as captured:
        service.close_competition(
            "TEAM42", principal(1, role="admin"), command_id="close-after-draw"
        )
    assert captured.value.code == "ROOM_CLOSE_UNAVAILABLE"
    assert service.snapshot("TEAM42", principal(1, role="admin"))["status"] == CompetitionStatus.DRAW.value


def test_six_players_fill_room_and_captains_ready(service: CompetitionService) -> None:
    create_room(service)
    players = fill_room(service)

    full = service.snapshot("TEAM42", players[0])
    assert full["status"] == CompetitionStatus.READY_CHECK.value
    assert len(full["seats"]) == 6
    assert full["me"]["is_captain"] is True

    yellow_ready = service.set_ready(
        "TEAM42",
        players[0],
        ready=True,
        command_id="ready-yellow-1",
    )
    assert yellow_ready["teams"]["yellow"]["ready"] is True
    assert yellow_ready["status"] == CompetitionStatus.READY_CHECK.value
    assert yellow_ready["me"]["can_claim_seat"] is True

    finished = service.set_ready(
        "TEAM42",
        players[3],
        ready=True,
        command_id="ready-white-1",
    )
    assert finished["teams"]["white"]["ready"] is True
    assert finished["status"] == CompetitionStatus.DRAW.value

    retried = service.set_ready(
        "TEAM42",
        players[3],
        ready=True,
        command_id="ready-white-1",
    )
    assert retried["status"] == CompetitionStatus.DRAW.value


def test_non_captain_cannot_ready(service: CompetitionService) -> None:
    create_room(service)
    players = fill_room(service)
    with pytest.raises(CompetitionError) as captured:
        service.set_ready(
            "TEAM42",
            players[1],
            ready=True,
            command_id="ready-player-2",
        )
    assert captured.value.code == "CAPTAIN_REQUIRED"


def test_leaving_full_room_returns_to_seating(service: CompetitionService) -> None:
    create_room(service)
    players = fill_room(service)
    result = service.leave_seat(
        "TEAM42", players[5], command_id="leave-player-15"
    )
    assert result["status"] == CompetitionStatus.SEATING.value
    assert len(result["seats"]) == 5


def test_leaving_revokes_own_team_ready(service: CompetitionService) -> None:
    create_room(service)
    players = fill_room(service)
    service.set_ready(
        "TEAM42", players[0], ready=True, command_id="ready-yellow-lock"
    )
    result=service.leave_seat("TEAM42", players[1], command_id="leave-after-ready")
    assert not result['teams']['yellow']['ready']


def test_concurrent_claim_has_single_winner(service: CompetitionService) -> None:
    create_room(service)

    def claim(player: Principal, command_id: str):
        try:
            service.claim_seat(
                "TEAM42",
                player,
                side="yellow",
                position=1,
                command_id=command_id,
            )
            return "accepted"
        except CompetitionError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda args: claim(*args),
                [(principal(20), "concurrent-claim-20"), (principal(21), "concurrent-claim-21")],
            )
        )
    assert sorted(results) == ["SEAT_TAKEN", "accepted"]
    snapshot = service.snapshot("TEAM42", principal(20))
    assert len(snapshot["seats"]) == 1


def test_event_sequence_is_contiguous(service: CompetitionService) -> None:
    create_room(service)
    fill_room(service)
    with service.database.transaction() as db:
        rows = db.execute(
            "SELECT sequence FROM competition_events ORDER BY sequence"
        ).fetchall()
    sequences = [int(row["sequence"]) for row in rows]
    assert sequences == list(range(1, len(sequences) + 1))
