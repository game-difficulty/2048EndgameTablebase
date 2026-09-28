from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json

from competition.backend.db import CompetitionDatabase
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
from competition.backend.service import CompetitionService
from competition.tests.test_match_service import prepare_game_a, ready_and_start


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


def test_match_settles_both_cargo_sessions_at_ten_active_minutes(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "cargo-match.sqlite3"),
        draw_reveal_seconds=1, draft_turn_seconds=5,
        c_draw_reveal_seconds=1, lineup_seconds=5,
        team_clock_seconds=1200,
    )
    service.initialize()
    players = prepare_game_a(service, "CARGO7")
    ready_and_start(service, players, "A", "CARGO7")
    adapter = CargoTransportAdapter()
    moment = datetime.now(timezone.utc)
    with service.database.transaction(immediate=True) as db:
        room = db.execute("SELECT id FROM competitions WHERE room_code = ?", ("CARGO7",)).fetchone()
        competition_id = str(room["id"])
        for side in ("yellow", "white"):
            state = adapter.initial_state(seed=SEED)
            extra = {**state.extra, "project_clock_start_ms": 1_200_000}
            db.execute(
                """
                UPDATE competition_game_sessions
                SET project_ref = ?, rules_version = ?, board_json = ?,
                    adapter_state_json = ?, rng_counter = ?, score = ?
                WHERE competition_id = ? AND game_key = 'A' AND side = ?
                """,
                (adapter.project_id, adapter.rules_version,
                 json.dumps(state.board), json.dumps(extra), state.rng_counter,
                 3 if side == "yellow" else 2, competition_id, side),
            )
        db.execute(
            """
            UPDATE competition_team_clocks
            SET running_since = ? WHERE competition_id = ?
            """,
            ((moment - timedelta(milliseconds=LIMIT_MS + 10)).isoformat(), competition_id),
        )
    assert service.settle_deadline("CARGO7", now=moment)
    snapshot = service.snapshot("CARGO7", players[0])
    assert snapshot["status"] == CompetitionStatus.GAME_A_RESULT.value
    assert snapshot["match"]["current_result"]["winner_side"] == "yellow"
    assert all(item["finished"] for item in snapshot["match"]["sessions"].values())
    assert all(item["public_view"]["view_protocol"] == "cargo-transport-v1"
               for item in snapshot["match"]["sessions"].values())


def test_match_death_stops_only_the_dead_side_clock(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "cargo-death.sqlite3"),
        draw_reveal_seconds=1, draft_turn_seconds=5,
        c_draw_reveal_seconds=1, lineup_seconds=5,
        team_clock_seconds=1200,
    )
    service.initialize()
    players = prepare_game_a(service, "CARGOD")
    ready_and_start(service, players, "A", "CARGOD")
    adapter = CargoTransportAdapter()
    state = adapter.initial_state(seed=SEED)
    raw = [8, 32, 32, 16, 4, 16, 0, 4, 16, 32, 4, 16, 32, 2, 32, 4]
    board = tuple(tuple(raw[index:index + 4]) for index in range(0, 16, 4))
    with service.database.transaction(immediate=True) as db:
        room = db.execute("SELECT id FROM competitions WHERE room_code = ?", ("CARGOD",)).fetchone()
        db.execute(
            """
            UPDATE competition_game_sessions
            SET project_ref = ?, rules_version = ?, seed_hex = ?, board_json = ?,
                adapter_state_json = ?, rng_counter = ?
            WHERE competition_id = ? AND game_key = 'A' AND side = 'yellow'
            """,
            (adapter.project_id, adapter.rules_version, SEED, json.dumps(board),
             json.dumps({**state.extra, "cargo": {"id": 0, "shape": 0, "row": -2, "col": 1},
                         "project_clock_start_ms": 1_200_000}), state.rng_counter, room["id"]),
        )
    token = service.snapshot("CARGOD", players[0])["match"]["phase_token"]
    result = service.move_current_game(
        "CARGOD", players[0], direction="down", phase_token=token,
        command_id="cargo-death-move",
    )
    assert result["match"]["clocks"]["yellow"]["state"] == "stopped"
    assert result["match"]["clocks"]["white"]["state"] == "running"
    session = result["match"]["sessions"]["yellow"]
    assert session["finished"] is True
    assert session["public_view"]["payload"]["outcome"] == "no_moves"
    assert 0 < session["public_view"]["payload"]["remaining_ms"] < LIMIT_MS
    frozen = session["public_view"]["payload"]["remaining_ms"]
    assert service.snapshot("CARGOD", players[0])["match"]["sessions"]["yellow"]["public_view"]["payload"]["remaining_ms"] == frozen
