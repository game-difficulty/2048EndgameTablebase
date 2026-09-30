from __future__ import annotations

import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.projects.contracts import ProjectState
from competition.backend.projects.practice_variants import GrowingTilesAdapter, GrowingTilesAdapterV3, HundredStepSealAdapter, _poly_move
from competition.backend.projects.tournament_variants import (
    TOURNAMENT_ADAPTER_FACTORIES,
    TOURNAMENT_RULES,
    Tournament2048Adapter,
    Tournament2048AdapterV2,
    TOURNAMENT_V3_ADAPTER_FACTORIES,
    ISLAND,
    WALL,
    _max_playable_rectangle,
    _move,
    tournament_project_catalog,
)
from competition.backend.service import CompetitionService
from competition.tests.test_match_service import prepare_game_a, ready_and_start


SEED = "31" * 32


def adapter(project_ref: str) -> Tournament2048Adapter:
    rules = next(item for item in TOURNAMENT_RULES if item.project_ref == project_ref)
    return Tournament2048Adapter(rules)


def adapter_v2(project_ref: str) -> Tournament2048AdapterV2:
    rules = next(item for item in TOURNAMENT_RULES if item.project_ref == project_ref)
    return Tournament2048AdapterV2(rules)


def test_v2_uses_shared_spawn_seed_but_side_specific_dice() -> None:
    item = adapter_v2("tournament-dice-wall-3x4")
    yellow = item.initial_state_for_side(seed=SEED, side="yellow")
    white = item.initial_state_for_side(seed=SEED, side="white")
    assert yellow.seed == white.seed == SEED
    assert yellow.rng_counter == white.rng_counter == 2
    assert yellow.extra["side"] == "yellow"
    assert white.extra["side"] == "white"
    empty = tuple((0, 0, 0, 0) for _ in range(3))
    assert item._spawn(empty, yellow.seed, 2) == item._spawn(empty, white.seed, 2)


def test_v2_island_uses_one_numeric_ticket_even_with_no_empty_cell() -> None:
    item = adapter_v2("tournament-isolated-island-hard-4x4")
    state = item.initial_state(seed=SEED)
    empty = tuple((0, 0, 0, 0) for _ in range(4))
    _board, counter, _spawn = item._spawn(empty, SEED, state.rng_counter)
    assert counter == state.rng_counter + 1
    full = tuple((2, 2, 2, 2) for _ in range(4))
    _board, counter, spawn = item._spawn(full, SEED, counter)
    assert spawn is None and counter == state.rng_counter + 2


def test_v2_undo_keeps_numeric_cursor() -> None:
    item = adapter_v2("tournament-grand-full-undo-race-3x3")
    initial = item.initial_state(seed=SEED)
    direction = next(
        candidate for candidate in ("left", "right", "up", "down")
        if _move(initial.board, candidate, mirror=False, unmergeable=None)[0] != initial.board
    )
    moved = item.apply_move(initial, direction)
    restored = item.apply_action(moved, {"type": "undo"})
    assert restored.board == initial.board
    assert restored.rng_counter == moved.rng_counter == initial.rng_counter + 1


def test_v2_shape_setup_is_independent_of_numeric_cursor() -> None:
    item = adapter_v2("tournament-shape-shifter-hard-12")
    initial = item.initial_state(seed=SEED)
    board_a, _ = item._empty_board(SEED, 1)
    item._spawn(initial.board, SEED, initial.rng_counter)
    board_b, _ = item._empty_board(SEED, 1)
    assert board_a == board_b


def test_v2_pure2_race_accepts_at_least_1022_without_changing_grand_race() -> None:
    pure2 = adapter_v2("tournament-pure2-full-race-3x3")
    reached = ((1024, 0, 0), (0, 0, 0), (0, 0, 0))
    below = ((512, 0, 0), (0, 0, 0), (0, 0, 0))
    assert pure2._outcome(reached) == "target_reached"
    assert pure2._outcome(below) is None
    grand = adapter_v2("tournament-grand-full-undo-race-3x3")
    assert grand._outcome(((2048, 0, 0), (0, 0, 0), (0, 0, 0))) is None


def test_formal_catalog_can_be_frozen_into_a_competition(tmp_path) -> None:
    service = CompetitionService(
        CompetitionDatabase(tmp_path / "formal-projects.sqlite3"),
        bootstrap_organizer_ids=frozenset({1}),
    )
    service.initialize()
    organizer = Principal(user_id=1, display_name="Organizer")
    room = service.create_competition(
        organizer,
        name="Tournament projects",
        room_code="RULES7",
        projects=tournament_project_catalog(),
    )
    assert [project["project_ref"] for project in room["projects"]] == [
        "tournament-cargo-transport-4x4",
        *(rules.project_ref for rules in TOURNAMENT_RULES),
        "practice-hundred-step-seal-4x4",
        "practice-growing-tiles-4x4",
        "practice-pair-bond-4x4",
        "practice-chemical-reaction-4x4",
        "practice-timed-bomb-4x4",
        "practice-full-load-4x4",
        "practice-heavy-tiles-4x4",
        "practice-fission-4x4",
    ]
    assert {project["adapter"]["view_protocol"] for project in room["projects"]} == {
        "2048-board-v2", "cargo-transport-v1", "polyomino-board-v1"
    }
    assert {project["project_ref"]: project["rules_version"] for project in room["projects"]}[
        "practice-growing-tiles-4x4"
    ] == "tournament-v3"
    assert {project["project_ref"]: project["rules_version"] for project in room["projects"]}[
        "tournament-shape-shifter-hard-12"
    ] == "tournament-v3"
    dice = next(project for project in room["projects"] if project["project_ref"] == "tournament-dice-wall-3x4")
    assert dice["name"] == "骰子障碍（3×4）"
    assert dice["rules_version"] == "tournament-v2"
    assert all(project["rules_version"] == "tournament-v4" for project in room["projects"][-6:])


def test_seal_adapter_keeps_independent_shared_seal_and_spawn_streams() -> None:
    item = HundredStepSealAdapter()
    yellow = item.initial_state(seed=SEED)
    white = item.initial_state_for_side(seed=SEED, side="white")
    assert yellow.board == white.board
    assert yellow.extra["sealed_cells"] == white.extra["sealed_cells"]
    assert len(yellow.extra["sealed_cells"]) == 3
    assert all(yellow.board[index // 4][index % 4] == 0 for index in yellow.extra["sealed_cells"])
    assert yellow.rng_counter == 2
    assert item.public_payload(yellow)["next_seal_in"] == 100

    rotated, counter = item._draw_seals(SEED, yellow.extra["sealed_cells"], yellow.extra["seal_counter"])
    assert not set(rotated) & set(yellow.extra["sealed_cells"])
    assert counter > yellow.extra["seal_counter"]
    assert item._spawn_unsealed(yellow.board, SEED, 2, rotated) == item._spawn_unsealed(white.board, SEED, 2, rotated)

    near_rotation = replace(yellow, move_count=99)
    direction = next(direction for direction in ("up", "right", "down", "left")
                     if item._move_with_seals(near_rotation.board, direction, yellow.extra["sealed_cells"])[0]
                     != near_rotation.board)
    after = item.apply_move(near_rotation, direction)
    assert after.move_count == 100
    assert after.rng_counter == near_rotation.rng_counter + 1
    assert after.extra["seal_counter"] > near_rotation.extra["seal_counter"]
    assert not set(after.extra["sealed_cells"]) & set(yellow.extra["sealed_cells"])
    assert after.extra["last_transition"]["seals"]["released"] == yellow.extra["sealed_cells"]
    assert after.extra["last_transition"]["before"] == [value for row in yellow.board for value in row]


def test_growing_tiles_adapter_exposes_rigid_tiles_and_shared_numeric_seed() -> None:
    item = GrowingTilesAdapter()
    yellow = item.initial_state(seed=SEED)
    white = item.initial_state(seed=SEED)
    assert yellow.board == white.board
    assert yellow.rng_counter == white.rng_counter == 2
    assert yellow.extra["tiles"] == white.extra["tiles"]
    assert item.public_view(yellow).view_protocol == "polyomino-board-v1"
    direction = next(direction for direction in ("up", "right", "down", "left")
                     if _poly_move(yellow.extra["tiles"], direction)[-1])
    moved = item.apply_move(yellow, direction)
    assert moved.rng_counter == 3
    assert moved.extra["last_transition"]["before"] == yellow.extra["tiles"]


def test_growing_tiles_merge_geometry_matches_practice_rules() -> None:
    sideways = [
        {"id": "a", "value": 64, "cells": [4]},
        {"id": "b", "value": 64, "cells": [5]},
    ]
    tiles, score, _movements, _merges, changed = _poly_move(sideways, "left")
    assert changed and score == 128
    assert tiles == [{"id": "merge-a-b", "value": 128, "cells": [4, 5]}]

    offset = [
        {"id": "a", "value": 128, "cells": [5, 6]},
        {"id": "b", "value": 128, "cells": [10, 11]},
    ]
    tiles, score, _movements, _merges, changed = _poly_move(offset, "up")
    assert changed and score == 256
    assert tiles == [{"id": "merge-a-b", "value": 256, "cells": [1, 2, 3]}]


def test_restartable_races_remain_operable_after_death() -> None:
    board = ((2, 4, 8), (16, 32, 64), (128, 256, 128))
    for project_ref in ("tournament-pure2-full-race-3x3", "tournament-grand-full-undo-race-3x3"):
        item = adapter_v2(project_ref)
        assert item._outcome(board) is None
        state = item.initial_state(seed=SEED)
        dead = item.public_payload(replace(state, board=board))
        assert dead["no_moves"] is True


@pytest.mark.parametrize("project_ref,rules_version", [
    (item["project_ref"], item["rules_version"]) for item in tournament_project_catalog()
])
def test_every_project_starts_as_two_live_match_sessions(
    tmp_path, project_ref: str, rules_version: str
) -> None:
    service = CompetitionService(CompetitionDatabase(tmp_path / "embedded.sqlite3"))
    service.initialize()
    players = prepare_game_a(service, "EMBED7")
    descriptor = service.project_registry.snapshot(project_ref, rules_version)
    protocol = descriptor["view_protocol"]
    with service.database.transaction(immediate=True) as db:
        competition_id = db.execute(
            "SELECT id FROM competitions WHERE room_code = 'EMBED7'"
        ).fetchone()["id"]
        project_key = db.execute(
            "SELECT project_a FROM competition_drafts WHERE competition_id = ?",
            (competition_id,),
        ).fetchone()["project_a"]
        db.execute(
            """UPDATE competition_projects
               SET project_ref = ?, adapter_rules_version = ?, rules_version = ?,
                   adapter_snapshot_json = ?
               WHERE competition_id = ? AND project_key = ?""",
            (project_ref, rules_version, rules_version, json.dumps(descriptor),
             competition_id, project_key),
        )
    started = ready_and_start(service, players, "A", "EMBED7")
    sessions = started["match"]["sessions"]
    assert sessions["yellow"]["public_view"]["view_protocol"] == protocol
    assert sessions["white"]["public_view"]["view_protocol"] == protocol
    if project_ref == "practice-growing-tiles-4x4":
        assert (sessions["yellow"]["public_view"]["payload"]["rows"],
                sessions["yellow"]["public_view"]["payload"]["cols"]) == (4, 5)
    if project_ref == "tournament-shape-shifter-hard-12":
        assert sessions["yellow"]["public_view"]["payload"]["cols"] <= 6
    with service.database.transaction() as db:
        rows = db.execute(
            "SELECT side, seed_hex, rng_counter FROM competition_game_sessions WHERE competition_id = ?",
            (competition_id,),
        ).fetchall()
    assert len(rows) == 2
    assert rows[0]["seed_hex"] == rows[1]["seed_hex"]
    assert rows[0]["rng_counter"] == rows[1]["rng_counter"] == 0
    for participant, side in ((players[0], "yellow"), (players[3], "white")):
        runtime = service.snapshot("EMBED7", participant)["match"]["my_session"]["runtime"]
        assert runtime["seed"] == rows[0]["seed_hex"]
        assert runtime["protocol"] == "client-runtime-v1"
        assert runtime["sequence"] == 0
        assert runtime["checkpoint"] is None
        assert sessions[side]["public_view"]["payload"]["awaiting_client"]
        assert all(value == 0 for row in sessions[side]["public_view"]["payload"]["board"] for value in row)


def test_all_nine_projects_are_registered_with_public_v2_views() -> None:
    adapters = [factory() for factory in TOURNAMENT_ADAPTER_FACTORIES]
    assert len(adapters) == 9
    assert len({item.project_id for item in adapters}) == 9
    assert {item.descriptor.view_protocol for item in adapters} == {"2048-board-v2"}


@pytest.mark.parametrize("factory", TOURNAMENT_ADAPTER_FACTORIES)
def test_initial_state_matches_declared_board_and_keeps_private_state_private(factory) -> None:
    item = factory()
    state = item.initial_state(seed=SEED)
    if item.rules.shape_playable_cells is None:
        assert len(state.board) == item.rules.rows
        assert all(len(row) == item.rules.cols for row in state.board)
    else:
        assert sum(value != WALL for row in state.board for value in row) == 12
    assert sum(value > 0 for row in state.board for value in row) == 2
    payload = item.public_payload(state)
    assert "seed" not in payload
    assert "history" not in payload
    assert payload["rows"] == len(state.board)
    assert payload["cols"] == len(state.board[0])
    assert payload["evil_spawn"] is item.rules.evil_spawn
    assert "powerups" not in payload


def test_mirror_paths_wrap_at_outer_portal_and_stop_at_center_wall() -> None:
    board = (
        (2, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    moved, _score, movements = _move(board, "left", mirror=True, unmergeable=64)
    assert moved[0] == (0, 0, 2, 0)
    assert movements[0]["from"] == 0
    assert movements[0]["to"] == 2


def test_unmergeable_64_and_256_tiles_remain_separate() -> None:
    mirror_board = (
        (0, 0, 64, 64),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
    )
    mirror_moved, mirror_score, _ = _move(
        mirror_board, "left", mirror=True, unmergeable=64
    )
    assert mirror_moved[0] == mirror_board[0]
    assert mirror_score == 0

    large_board = (
        (256, 256, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0),
    )
    large_moved, large_score, _ = _move(
        large_board, "left", mirror=False, unmergeable=256
    )
    assert large_moved[0][:2] == (256, 256)
    assert large_score == 0


def test_islands_merge_with_their_own_kind_without_scoring() -> None:
    board = ((ISLAND, ISLAND, ISLAND, 2, 0),)
    moved, score, movements = _move(board, "left", mirror=False, unmergeable=None)
    assert moved == ((ISLAND, 2, 0, 0, 0),)
    assert score == 0
    assert len([item for item in movements if item["value"] == ISLAND]) == 3


def test_hard_island_spawn_uses_five_percent_base_rate(monkeypatch) -> None:
    monkeypatch.setattr(
        "competition.backend.projects.tournament_variants._digest",
        lambda _seed, _label: bytes(32),
    )
    item = adapter("tournament-isolated-island-hard-4x4")
    empty = tuple(tuple(0 for _column in range(4)) for _row in range(4))
    spawned, _counter, metadata = item._spawn(empty, SEED, 2)
    assert metadata == {"index": 0, "value": ISLAND}
    assert spawned[0][0] == ISLAND


def test_hard_shape_shifter_has_twelve_connected_playable_cells() -> None:
    item = adapter("tournament-shape-shifter-hard-12")
    first_board = None
    for sample in range(200):
        seed = f"{sample + 1:064x}"
        board, extra = item._empty_board(seed, 0)
        rows, cols = len(board), len(board[0])
        playable = {
            (row, column)
            for row, values in enumerate(board)
            for column, value in enumerate(values)
            if value != WALL
        }
        assert len(playable) == 12
        assert cols >= rows
        assert rows <= 7 and cols <= 7
        assert extra["shape_shifter"] is True
        assert 4 <= _max_playable_rectangle([list(row) for row in board]) <= 8

        reached = {next(iter(playable))}
        frontier = list(reached)
        while frontier:
            row, column = frontier.pop()
            for neighbor in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
                if neighbor in playable and neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        assert reached == playable
        if sample == 0:
            first_board = board
    assert item._empty_board(f"{1:064x}", 0)[0] == first_board


def test_v3_shape_uses_six_by_six_source_and_crops_to_bounding_rectangle() -> None:
    item = TOURNAMENT_V3_ADAPTER_FACTORIES[0]()
    assert item.descriptor.rules_version == "tournament-v3"
    for sample in range(200):
        board, _extra = item._empty_board(f"{sample + 1:064x}", 0)
        rows, cols = len(board), len(board[0])
        assert rows <= 6 and cols <= 6 and cols >= rows
        assert sum(value != WALL for line in board for value in line) == 12
        assert 4 <= _max_playable_rectangle([list(line) for line in board]) <= 8
        assert all(any(value != WALL for value in edge) for edge in (
            board[0], board[-1],
            tuple(line[0] for line in board),
            tuple(line[-1] for line in board),
        ))


def test_v3_growing_tiles_has_five_columns_and_moves_across_them() -> None:
    item = GrowingTilesAdapterV3()
    state = item.initial_state(seed=SEED)
    assert len(state.board) == 4 and all(len(line) == 5 for line in state.board)
    assert item.public_payload(state)["cols"] == 5
    tiles, gained, _movements, _merges, changed = _poly_move(
        [{"id": "a", "value": 2, "cells": [4]}], "left", 4, 5,
    )
    assert changed and gained == 0 and tiles[0]["cells"] == [0]


def test_dice_wall_uses_the_correct_position_class() -> None:
    item = adapter("tournament-dice-wall-3x4")
    corners = {0, 3, 8, 11}
    edges = {1, 2, 4, 7, 9, 10}
    centers = {5, 6}
    seen_dice, seen_centers = set(), set()
    for sample in range(256):
        state = item.initial_state(seed=f"{sample:064x}")
        die = state.extra["dice"]
        wall = state.extra["wall_index"]
        assert len(state.board) == 3 and all(len(row) == 4 for row in state.board)
        assert state.board[wall // 4][wall % 4] == WALL
        assert sum(value == WALL for row in state.board for value in row) == 1
        assert wall in (corners if die <= 3 else edges if die <= 5 else centers)
        seen_dice.add(die)
        if die == 6:
            seen_centers.add(wall)
    assert seen_dice == {1, 2, 3, 4, 5, 6}
    assert seen_centers == centers


def test_undo_restores_board_score_and_move_count_without_rewinding_rng() -> None:
    item = adapter("tournament-grand-full-undo-race-3x3")
    initial = item.initial_state(seed=SEED)
    direction = next(
        candidate
        for candidate in ("left", "right", "up", "down")
        if _move(initial.board, candidate, mirror=False, unmergeable=None)[0] != initial.board
    )
    moved = item.apply_move(initial, direction)
    restored = item.apply_action(moved, {"type": "undo"})
    assert restored.board == initial.board
    assert restored.score == initial.score
    assert restored.move_count == initial.move_count
    assert restored.rng_counter == moved.rng_counter
    assert restored.rng_counter > initial.rng_counter
    assert restored.extra["revision"] > moved.extra["revision"]

    retried = item.apply_move(restored, direction)
    assert retried.rng_counter > moved.rng_counter
    assert retried.extra["last_transition"]["spawn"] != moved.extra["last_transition"]["spawn"]


def test_undo_race_no_move_board_stays_playing_and_can_recover() -> None:
    item = adapter("tournament-grand-full-undo-race-3x3")
    previous_board = ((2, 4, 8), (16, 32, 64), (128, 256, 256))
    dead_board = ((2, 4, 8), (16, 32, 64), (128, 256, 512))
    state = ProjectState(
        board=dead_board,
        score=4096,
        elapsed_ms=12_345,
        finished=False,
        outcome=None,
        seed=SEED,
        move_count=100,
        rng_counter=50,
        extra={
            "revision": 100,
            "history": [{
                "board": [list(row) for row in previous_board],
                "score": 3584,
                "move_count": 99,
            }],
        },
    )

    assert item._outcome(dead_board) is None
    assert item.public_payload(state)["can_undo"] is True
    restored = item.apply_action(state, {"type": "undo"})
    assert restored.finished is False
    assert restored.outcome is None
    assert restored.board == previous_board


@pytest.mark.parametrize(
    "empty_count,tile_value,expected_depth",
    [(1, 16, 7), (2, 16, 6), (3, 16, 5), (5, 16, 5), (6, 16, 4), (1, 4, 5), (1, 8, 7)],
)
def test_evil_spawn_uses_adaptive_depth_and_low_sum_cap(
    monkeypatch, empty_count: int, tile_value: int, expected_depth: int
) -> None:
    calls: list[int] = []
    board_values = [tile_value] * 16
    for index in range(empty_count):
        board_values[index] = 0
    board = tuple(tuple(board_values[row * 4:(row + 1) * 4]) for row in range(4))

    class FakeEvilGen:
        def __init__(self, _board: int):
            pass

        def gen_new_num_seeded(self, depth: int, _seed: int):
            calls.append(depth)
            return 0, 0, 1

    module = ModuleType("native_core.ai_core")
    module.EvilGen = FakeEvilGen
    monkeypatch.setitem(sys.modules, "native_core.ai_core", module)
    item = adapter("tournament-evil-spawn-4x4")
    spawned, _counter, metadata = item._spawn(board, SEED, 2)
    assert calls == [expected_depth]
    assert metadata == {"index": 0, "value": 2}
    assert spawned[0][0] == 2


def test_race_target_resolves_immediately_against_unfinished_opponent() -> None:
    item = adapter("tournament-pure2-full-race-3x3")
    target = ProjectState(
        board=((512, 256, 128), (64, 32, 16), (8, 4, 2)),
        score=0,
        elapsed_ms=12_345,
        finished=True,
        outcome="target_reached",
        seed=SEED,
    )
    opponent = ProjectState(
        board=((2, 0, 0), (0, 0, 0), (0, 0, 0)),
        score=0,
        elapsed_ms=12_345,
        finished=True,
        outcome="opponent_finished",
        seed=SEED,
    )
    assert item.resolve_winner(target, opponent) == ("yellow", "race_target")
