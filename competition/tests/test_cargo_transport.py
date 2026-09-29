from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from competition.backend.domain import CompetitionStatus
from competition.backend.projects.cargo_transport import (
    CargoTransportAdapter,
    CargoTransportAdapterV2,
    FIRST_CARGO_MOVE,
    LIMIT_MS,
    SHAPES,
    _can_shift,
    _has_cargo_move,
    _move_cargo_board,
)
from competition.tests.test_client_runtime import setup_game, packet


SEED = "31" * 32
EMPTY = tuple((0, 0, 0, 0) for _ in range(4))


def test_v2_cargo_shape_cursor_is_separate_from_numeric_spawns() -> None:
    adapter = CargoTransportAdapterV2()
    state = adapter.initial_state(seed=SEED)
    assert state.extra["shape_counter"] == 0
    _board, numeric_counter, _spawn = adapter._spawn_number(EMPTY, None, SEED, state.rng_counter)
    assert numeric_counter == state.rng_counter + 1
    cargo_a, shape_counter = adapter._next_cargo(SEED, state.extra["shape_counter"])
    cargo_b, _ = adapter._next_cargo(SEED, state.extra["shape_counter"])
    assert cargo_a == cargo_b and shape_counter == 1
    full = tuple((2, 2, 2, 2) for _ in range(4))
    _board, numeric_counter, spawn = adapter._spawn_number(full, None, SEED, numeric_counter)
    assert spawn is None and numeric_counter == state.rng_counter + 2


def test_cargo_shapes_use_two_column_ports_and_do_not_return_to_entrance() -> None:
    assert len(SHAPES) == 5
    for shape in range(5):
        cargo = {"id": 0, "shape": shape, "row": -2, "col": 1}
        blocked = [0] * 16
        blocked[13], blocked[14] = 2, 4
        _board, entering, _movement, shifted = _move_cargo_board(
            tuple(tuple(blocked[row * 4:(row + 1) * 4]) for row in range(4)), cargo, "down"
        )
        assert shifted and entering["row"] == 1
        assert not _can_shift({**cargo, "row": -1}, "up", [0] * 16)
        _board, outside, _movement, shifted = _move_cargo_board(EMPTY, cargo, "down")
        assert shifted and outside["row"] == 4
        _board, leftmost, _movement, shifted = _move_cargo_board(
            EMPTY, {**cargo, "row": 0, "col": 2}, "left"
        )
        assert shifted and leftmost["col"] == 0
        _board, topmost, _movement, shifted = _move_cargo_board(
            EMPTY, {**cargo, "row": 2}, "up"
        )
        assert shifted and topmost["row"] == 0


def test_partly_exited_l_can_shift_right_to_align_with_outlet() -> None:
    raw = [2, 8, 2, 4, 0, 4, 16, 2, 2, 4, 16, 4, 0, 0, 0, 128]
    board = tuple(tuple(raw[row * 4:(row + 1) * 4]) for row in range(4))
    piece = {"id": 0, "shape": 2, "row": 3, "col": 0}
    aligned_board, aligned, _movements, shifted = _move_cargo_board(board, piece, "right")
    assert shifted and aligned["col"] == 1
    assert aligned_board[3][3] == 128
    _board, _piece, _movements, shifted_up = _move_cargo_board(aligned_board, aligned, "up")
    assert not shifted_up
    _board, delivered, _movements, shifted_down = _move_cargo_board(aligned_board, aligned, "down")
    assert shifted_down and delivered["row"] == 4
    assert _has_cargo_move(board, piece)


def test_delivery_scores_once_and_stages_a_new_shape() -> None:
    adapter = CargoTransportAdapter()
    state = adapter.initial_state(seed=SEED)
    state = replace(
        state,
        board=EMPTY,
        extra={**state.extra, "cargo": {"id": 50, "shape": 0, "row": 3, "col": 1}},
    )
    next_state = adapter.apply_move(state, "down")
    assert next_state.score == 1
    assert next_state.extra["cargo"]["row"] == -2
    assert next_state.extra["last_transition"]["cargoExit"]["row"] == 4
    assert adapter.result_value(next_state) == 1
    assert adapter.public_view(next_state).view_protocol == "cargo-transport-v1"
    assert adapter.public_payload(next_state)["remaining_ms"] == LIMIT_MS


def test_first_cargo_appears_after_ten_effective_opening_moves() -> None:
    adapter = CargoTransportAdapter()
    state = adapter.initial_state(seed=SEED)
    assert state.extra["cargo"] is None
    assert state.rng_counter == 2
    for step in range(1, FIRST_CARGO_MOVE + 1):
        for direction in ("down", "left", "up", "right"):
            try:
                next_state = adapter.apply_move(state, direction)
                break
            except ValueError:
                continue
        else:
            raise AssertionError("expected an effective opening move")
        state = next_state
        assert state.move_count == step
        assert (state.extra["cargo"] is None) == (step < FIRST_CARGO_MOVE)
    assert state.extra["cargo"]["row"] == -2
    assert state.score == 0


def test_opening_can_end_before_first_cargo_when_board_is_dead() -> None:
    adapter = CargoTransportAdapter()
    state = adapter.initial_state(seed=SEED)
    raw = [8, 32, 32, 16, 4, 16, 0, 4, 16, 32, 4, 16, 32, 2, 32, 4]
    board = tuple(tuple(raw[index:index + 4]) for index in range(0, 16, 4))
    state = replace(state, board=board, move_count=8)
    dead = adapter.apply_move(state, "down")
    assert dead.move_count == 9
    assert dead.extra["cargo"] is None
    assert dead.finished and dead.outcome == "no_moves"


def test_server_limit_ends_run_with_current_delivery_count() -> None:
    adapter = CargoTransportAdapter()
    state = adapter.initial_state(seed=SEED)
    timed = adapter.apply_move(replace(state, elapsed_ms=LIMIT_MS), "down")
    assert timed.finished and timed.outcome == "time_limit"
    assert timed.score == state.score


def test_death_requires_no_numeric_or_cargo_move_and_ends_session() -> None:
    full = tuple(tuple(4 if (row + col) % 2 else 2 for col in range(4)) for row in range(4))
    staged = {"id": 0, "shape": 0, "row": -2, "col": 1}
    assert not _has_cargo_move(full, staged)
    assert _has_cargo_move(EMPTY, staged)

    adapter = CargoTransportAdapter()
    state = adapter.initial_state(seed=SEED)
    raw = [8, 32, 32, 16, 4, 16, 0, 4, 16, 32, 4, 16, 32, 2, 32, 4]
    board = tuple(tuple(raw[index:index + 4]) for index in range(0, 16, 4))
    state = replace(state, board=board, elapsed_ms=123_456,
                    extra={**state.extra, "cargo": staged})
    dead = adapter.apply_move(state, "down")
    assert dead.finished and dead.outcome == "no_moves"
    assert dead.move_count == state.move_count + 1
    assert adapter.public_payload(dead)["remaining_ms"] == LIMIT_MS - 123_456


def test_match_settles_cargo_only_after_client_reports_time_limit(tmp_path) -> None:
    service, players = setup_game(tmp_path, CargoTransportAdapter.project_id)
    moment = datetime.now(timezone.utc)
    with service.database.transaction(immediate=True) as db:
        db.execute(
            "UPDATE competition_team_clocks SET running_since = ?",
            ((moment - timedelta(milliseconds=LIMIT_MS + 10)).isoformat(),),
        )
    # Project timeout is a local rule, not a server deadline.
    assert not service.settle_deadline("MATCH5", now=moment)
    assert service.snapshot("MATCH5", players[0])["status"] == CompetitionStatus.GAME_A_PLAYING.value
    for participant, delivered in ((players[0], 3), (players[3], 2)):
        final = packet(service, participant, finished=True, value=delivered,
                       elapsed=LIMIT_MS, outcome="time_limit")
        final["payload"].update(board=[list(row) for row in EMPTY],
                                delivered=delivered, remaining_ms=0)
        service.sync_client_game("MATCH5", participant, **final)
    snapshot = service.snapshot("MATCH5", players[0])
    assert snapshot["status"] == CompetitionStatus.GAME_A_RESULT.value
    assert snapshot["match"]["current_result"]["winner_side"] == "yellow"
    assert snapshot["match"]["current_result"]["reason"] == "delivered_cargo"
    assert all(item["finished"] for item in snapshot["match"]["sessions"].values())
    assert all(item["public_view"]["view_protocol"] == "cargo-transport-v1"
               for item in snapshot["match"]["sessions"].values())


def test_match_death_stops_only_the_dead_side_clock(tmp_path) -> None:
    service, players = setup_game(tmp_path, CargoTransportAdapter.project_id)
    final = packet(service, players[0], finished=True, value=2,
                   elapsed=123_456, outcome="no_moves")
    final["payload"].update(board=[list(row) for row in EMPTY],
                            delivered=2, remaining_ms=LIMIT_MS - 123_456)
    result = service.sync_client_game("MATCH5", players[0], **final)["competition"]
    assert result["match"]["clocks"]["yellow"]["state"] == "stopped"
    assert result["match"]["clocks"]["white"]["state"] == "running"
    session = result["match"]["sessions"]["yellow"]
    assert session["finished"] is True
    assert session["project_clock"]["mode"] == "countdown"
    assert session["project_clock"]["limit_ms"] == LIMIT_MS
    assert session["project_clock"]["running"] is False
    assert session["public_view"]["payload"]["outcome"] == "no_moves"
    assert session["public_view"]["payload"]["remaining_ms"] == LIMIT_MS - 123_456
    frozen = session["public_view"]["payload"]["remaining_ms"]
    assert service.snapshot("MATCH5", players[0])["match"]["sessions"]["yellow"]["public_view"]["payload"]["remaining_ms"] == frozen
