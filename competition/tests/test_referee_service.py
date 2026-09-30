from __future__ import annotations

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import CompetitionStatus
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService
from competition.tests.test_match_service import (
    force_one_move_completion,
    player,
    prepare_game_a,
    ready_and_start,
)


@pytest.fixture()
def service(tmp_path) -> CompetitionService:
    value = CompetitionService(
        CompetitionDatabase(tmp_path / "referee.sqlite3"),
        draw_reveal_seconds=1,
        draft_turn_seconds=5,
        c_draw_reveal_seconds=1,
        lineup_seconds=5,
        team_clock_seconds=60,
        test_project_target_tile=4,
    )
    value.initialize()
    return value


def test_issue_reports_are_deduplicated_and_projected_by_role(service) -> None:
    players = prepare_game_a(service)
    first = service.report_issue(
        "MATCH5",
        players[1],
        category="network_device",
        details="网络延迟严重",
        command_id="issue-report-first",
    )
    assert len(first["issues"]) == 1
    service.report_issue(
        "MATCH5",
        players[1],
        category="network_device",
        details="仍然存在网络延迟",
        command_id="issue-report-second",
    )
    assert service.snapshot("MATCH5", players[1])["issues"][0]["details"] == "网络延迟严重"
    assert service.snapshot("MATCH5", players[0])["issues"] == []

    official = player(1, role="admin")
    official_view = service.snapshot("MATCH5", official)
    assert len(official_view["issues"]) == 1
    issue_id = official_view["issues"][0]["id"]
    resolved = service.resolve_issue(
        "MATCH5",
        official,
        issue_id=issue_id,
        status="resolved",
        resolution_note="已切换备用网络",
        command_id="issue-resolve-first",
    )
    assert resolved["issues"] == []


def test_referee_pause_requires_both_captains_before_resume(service) -> None:
    players = prepare_game_a(service)
    started = ready_and_start(service, players, "A")
    official = player(1, role="admin")
    paused = service.suspend_match(
        "MATCH5",
        official,
        reason_code="network_device",
        reason_text="白方设备需要重连",
        phase_token=started["match"]["phase_token"],
        command_id="pause-match-first",
    )
    assert paused["match"]["suspension"]["active"] is True
    assert paused["match"]["suspension"]["started_by_display_name"] == "Player 1"
    assert all(clock["state"] == "stopped" for clock in paused["match"]["clocks"].values())

    own = service.snapshot("MATCH5", players[0])["match"]
    with pytest.raises(CompetitionError) as captured:
        service.sync_client_game(
            "MATCH5", players[0],
            instance_id=own["my_session"]["runtime"]["instance_id"], sequence=1,
            phase_token=own["phase_token"],
            payload={"board": [[2, 0], [0, 0]], "score": 0, "move_count": 1},
            checkpoint={"version": 1, "state": {}}, result_value=0, elapsed_ms=10,
            finished=False, outcome=None,
        )
    assert captured.value.code == "MATCH_SUSPENDED"

    token = paused["match"]["phase_token"]
    service.set_suspension_readiness(
        "MATCH5",
        players[0],
        ready=True,
        phase_token=token,
        command_id="resume-yellow-ready",
    )
    with pytest.raises(CompetitionError) as captured:
        service.resume_match(
            "MATCH5",
            official,
            phase_token=token,
            command_id="resume-too-early",
        )
    assert captured.value.code == "RESUME_READINESS_INCOMPLETE"
    service.set_suspension_readiness(
        "MATCH5",
        players[3],
        ready=True,
        phase_token=token,
        command_id="resume-white-ready",
    )
    resumed = service.resume_match(
        "MATCH5",
        official,
        phase_token=token,
        command_id="resume-match-final",
    )
    assert resumed["match"]["suspension"]["active"] is False
    assert all(clock["state"] == "running" for clock in resumed["match"]["clocks"].values())


def test_result_override_invalidates_confirmations_and_force_advance(service) -> None:
    players = prepare_game_a(service)
    ready_and_start(service, players, "A")
    force_one_move_completion(service, players[0], "A", "yellow")
    result_view = force_one_move_completion(service, players[3], "A", "white")
    old_revision = result_view["match"]["current_result"]["result_revision"]
    service.confirm_current_result(
        "MATCH5",
        players[0],
        result_revision=old_revision,
        phase_token=service.snapshot("MATCH5", players[0])["match"]["phase_token"],
        command_id="override-before-confirm",
    )
    official = player(1, role="admin")
    official_view = service.snapshot("MATCH5", official)
    corrected = service.override_current_result(
        "MATCH5",
        official,
        yellow_score=128,
        white_score=64,
        winner_side="yellow",
        reason="裁判复核服务端记录",
        result_revision=old_revision,
        phase_token=official_view["match"]["phase_token"],
        command_id="override-result-first",
    )
    assert corrected["match"]["current_result"]["result_revision"] == 2
    assert corrected["match"]["current_result"]["corrected"] is True
    assert corrected["match"]["confirmations"] == {"yellow": False, "white": False}
    assert corrected["match"]["series_score"]["yellow"] == 1

    with pytest.raises(CompetitionError) as captured:
        service.confirm_current_result(
            "MATCH5",
            players[0],
            result_revision=old_revision,
            phase_token=service.snapshot("MATCH5", players[0])["match"]["phase_token"],
            command_id="confirm-stale-result",
        )
    assert captured.value.code == "STALE_RESULT"

    token = service.snapshot("MATCH5", official)["match"]["phase_token"]
    with service.database.transaction(immediate=True) as db:
        db.execute("UPDATE competition_game_results SET published_at='2020-01-01T00:00:00+00:00'")
    advanced = service.force_advance_current_result(
        "MATCH5",
        official,
        reason="双方同意由裁判直接推进",
        phase_token=token,
        command_id="force-advance-first",
    )
    assert advanced["status"] == CompetitionStatus.GAME_B_READY.value
    assert advanced["match"]["current_game_key"] == "B"


def test_force_finish_closes_match_even_while_suspended(service) -> None:
    players = prepare_game_a(service)
    started = ready_and_start(service, players, "A")
    official = player(1, role="admin")
    paused = service.suspend_match(
        "MATCH5",
        official,
        reason_code="rules",
        reason_text="规则争议等待裁决",
        phase_token=started["match"]["phase_token"],
        command_id="force-finish-pause",
    )
    finished = service.force_finish_match(
        "MATCH5",
        official,
        winner_side="white",
        reason="黄方退出比赛，裁判判白方获胜",
        phase_token=paused["match"]["phase_token"],
        command_id="force-finish-final",
    )
    assert finished["status"] == CompetitionStatus.FINISHED.value
    assert finished["match"]["winner_side"] == "white"
    assert finished["match"]["suspension"]["active"] is False
    assert len(finished["match"]["results"]) == 3
