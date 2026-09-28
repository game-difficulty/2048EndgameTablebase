from __future__ import annotations

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.practice_leaderboard import PracticeLeaderboard


@pytest.fixture
def board(tmp_path):
    database = CompetitionDatabase(tmp_path / "practice.sqlite3")
    database.initialize()
    return PracticeLeaderboard(database)


def result(board, project, user, value, elapsed=1000, outcome="no_moves"):
    return board.submit(
        project, Principal(user, f"选手{user}"), score=value, board_sum=value,
        elapsed_ms=elapsed, outcome=outcome,
    )


def test_score_best_is_one_row_per_player_and_guest_can_read(board):
    project = "tournament-spawn4-50-3x3"
    assert result(board, project, 1, 100)["improved"] is True
    assert result(board, project, 1, 90)["improved"] is False
    assert result(board, project, 1, 100, elapsed=800)["improved"] is True
    result(board, project, 2, 200)
    guest = board.list(project)
    assert guest["signed_in"] is False
    assert guest["my_best"] is None
    assert [row["user_id"] for row in guest["top"]] == [2, 1]
    assert board.list(project, Principal(1, "选手1"))["my_best"]["elapsed_ms"] == 800


def test_race_requires_target_and_sorts_fastest_first(board):
    project = "tournament-pure2-full-race-3x3"
    with pytest.raises(CompetitionError):
        result(board, project, 1, 0)
    result(board, project, 1, 0, 5400, "target_reached")
    result(board, project, 2, 0, 3200, "target_reached")
    assert [row["user_id"] for row in board.list(project)["top"]] == [2, 1]
    assert result(board, project, 1, 0, 6000, "target_reached")["improved"] is False


def test_dice_uses_board_sum_and_unknown_projects_are_rejected(board):
    project = "tournament-dice-wall-3x3"
    board.submit(project, Principal(1, "选手1"), score=900, board_sum=100, elapsed_ms=1000, outcome="no_moves")
    board.submit(project, Principal(2, "选手2"), score=100, board_sum=200, elapsed_ms=1000, outcome="no_moves")
    assert board.list(project)["top"][0]["user_id"] == 2
    with pytest.raises(CompetitionError):
        board.list("not-a-project")
