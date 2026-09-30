from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass, replace
from typing import Any

from .contracts import ProjectDescriptor, ProjectState, PublicProjectView


DIRECTIONS = {"up", "down", "left", "right"}
WALL = -1
ISLAND = -3


@dataclass(frozen=True)
class VariantRules:
    project_ref: str
    display_name: str
    rows: int
    cols: int
    spawn4_rate: float = 0.1
    evil_spawn: bool = False
    dice_wall: bool = False
    mirror_portals: bool = False
    unmergeable: int | None = None
    target_sum: int | None = None
    target_tile_count: tuple[int, int] | None = None
    allow_restart: bool = False
    allow_undo: bool = False
    result_metric: str = "score"
    estimated_minutes: str = ""
    isolated_island: bool = False
    shape_playable_cells: int | None = None
    shape_generation_size: int = 7

    @property
    def race(self) -> bool:
        return self.target_sum is not None or self.target_tile_count is not None


TOURNAMENT_RULES = (
    VariantRules(
        "tournament-spawn4-50-3x3", "50%出4（3×3）", 3, 3,
        spawn4_rate=0.5, estimated_minutes="5–20",
    ),
    VariantRules(
        "tournament-evil-spawn-4x4", "寸步难行（4×4）", 4, 4,
        evil_spawn=True, estimated_minutes="1–10",
    ),
    VariantRules(
        "tournament-pure2-full-race-3x3", "极限速通（3×3）", 3, 3,
        spawn4_rate=0, target_sum=1022, allow_restart=True,
        result_metric="race", estimated_minutes="3–10",
    ),
    VariantRules(
        "tournament-grand-full-undo-race-3x3", "大满盘撤销竞速（3×3）", 3, 3,
        target_sum=2044, allow_restart=True, allow_undo=True,
        result_metric="race", estimated_minutes="5–15",
    ),
    VariantRules(
        "tournament-dice-wall-3x4", "骰子障碍（3×4）", 3, 4,
        dice_wall=True, result_metric="board_sum", estimated_minutes="3–5",
    ),
    VariantRules(
        "tournament-mirror-64x10-race-4x4", "镜面领域（4×4）", 4, 4,
        mirror_portals=True, unmergeable=64, target_tile_count=(64, 10),
        allow_restart=True, result_metric="race", estimated_minutes="2–15",
    ),
    VariantRules(
        "tournament-256-brick-5x5", "256砖（5×5）", 5, 5,
        unmergeable=256, estimated_minutes="10–20",
    ),
    VariantRules(
        "tournament-isolated-island-hard-4x4", "孤岛（4×4）", 4, 4,
        isolated_island=True, estimated_minutes="待测",
    ),
    VariantRules(
        "tournament-shape-shifter-hard-12", "随机形状", 7, 7,
        shape_playable_cells=12, estimated_minutes="待测",
    ),
)


def tournament_project_catalog() -> list[dict[str, Any]]:
    return [{
        "key": "project-01",
        "name": "真·华容道（4×4）",
        "description": "前10次有效移动先整理棋盘，随后从上方出现首个特殊块；特殊块整块滑到尽头，从下方中央送出且不能退回入口。无路可走时结束，送出数量多者胜。",
        "project_ref": "tournament-cargo-transport-4x4",
        "adapter_rules_version": "tournament-v2",
        "rules_version": "tournament-v2",
    }, *[
        {
            "key": f"project-{index:02d}",
            "name": rules.display_name,
            "description": _description(rules),
            "project_ref": rules.project_ref,
            "adapter_rules_version": "tournament-v3" if rules.shape_playable_cells is not None else "tournament-v2",
            "rules_version": "tournament-v3" if rules.shape_playable_cells is not None else "tournament-v2",
        }
        for index, rules in enumerate(TOURNAMENT_RULES, start=2)
    ], {
        "key": "project-11",
        "name": "百步封锁（4×4）",
        "description": "开局封住3格，此后每100步轮换；双方死亡后按得分结算。",
        "project_ref": "practice-hundred-step-seal-4x4",
        "adapter_rules_version": "tournament-v2",
        "rules_version": "tournament-v2",
    }, {
        "key": "project-12",
        "name": "越来越大（4×5）",
        "description": "64合成双格128；128相撞合成双格或三格256；双方死亡后按得分结算。",
        "project_ref": "practice-growing-tiles-4x4",
        "adapter_rules_version": "tournament-v3",
        "rules_version": "tournament-v3",
    }, *[
        {
            "key": f"project-{index:02d}",
            "name": name,
            "description": description,
            "project_ref": project_ref,
            "adapter_rules_version": "tournament-v6" if index in (18, 20) else "tournament-v5" if index == 19 else "tournament-v4",
            "rules_version": "tournament-v6" if index in (18, 20) else "tournament-v5" if index == 19 else "tournament-v4",
        }
        for index, project_ref, name, description in (
            (13, "practice-pair-bond-4x4", "出双入对（4×4）", "特殊块相邻可粘合成双格，新双格形成时旧双格消失；双方死亡后按得分结算。"),
            (14, "practice-chemical-reaction-4x4", "化学反应（4×4）", "同色特殊块碰撞消失、异色碰撞生成一格墙；双方死亡后按得分结算。"),
            (15, "practice-timed-bomb-4x4", "定时炸弹（4×4）", "炸弹移动后倒计时减少，归零变墙；双方死亡后按得分结算。"),
            (16, "practice-full-load-4x4", "满载（4×4）", "棋盘数字块超过12个立即结束；按得分结算。"),
            (17, "practice-heavy-tiles-4x4", "越来越重（4×4）", "256只能横移、512只能竖移、1024不能移动；双方死亡后按得分结算。"),
            (18, "practice-fission-4x4", "裂变（4×4）", "1024及以上数字块会分裂成两块并替代该步出数；双方死亡后比较盘面和。"),
            (19, "practice-aftershock-4x4", "余震（4×4）", "一次操作中，只要合出256或更大的数字块，就触发一次余震：从初始四行、四列中随机选一条，向随机方向平移一格。双方死亡后按得分结算。"),
            (20, "practice-look-back-3x4", "回头看看（3×4）", "有效移动有机会改为撤销并出两个数；可重开，先合出2048者获胜。"),
        )
    ]]


def _description(rules: VariantRules) -> str:
    details = {
        "tournament-spawn4-50-3x3": "50%概率生成4；双方死亡后按得分结算。",
        "tournament-evil-spawn-4x4": "AI对抗出数；双方死亡后按得分结算。",
        "tournament-pure2-full-race-3x3": "只生成2，可重开；盘面和达到或超过1022即获胜。",
        "tournament-grand-full-undo-race-3x3": "可重开、可撤销；先达到盘面和2044获胜。",
        "tournament-dice-wall-3x4": "开局掷骰放置固定墙；双方死亡后按盘面和结算。",
        "tournament-mirror-64x10-race-4x4": "中央为墙、外沿传送；64不可合并，先同时拥有10个64获胜。",
        "tournament-256-brick-5x5": "256不可合并；双方死亡后按得分结算。",
        "tournament-isolated-island-hard-4x4": "孤岛只与同类合并且不计分；双方死亡后按得分结算。",
        "tournament-shape-shifter-hard-12": "每局使用随机12格棋盘；棋盘外区域不可进入，双方死亡后按得分结算。",
    }
    return details[rules.project_ref]


def _digest(seed: str, label: str) -> bytes:
    return hmac.new(bytes.fromhex(seed), label.encode("utf-8"), hashlib.sha256).digest()


def _paths(rows: int, cols: int, direction: str, mirror: bool) -> list[list[int]]:
    if mirror:
        order = [2, 3, 0, 1] if direction in {"left", "up"} else [1, 0, 3, 2]
        if direction in {"left", "right"}:
            return [[row * cols + column for column in order] for row in range(rows)]
        return [[row * cols + column for row in order] for column in range(cols)]
    if direction == "left":
        return [[row * cols + column for column in range(cols)] for row in range(rows)]
    if direction == "right":
        return [[row * cols + column for column in reversed(range(cols))] for row in range(rows)]
    if direction == "up":
        return [[row * cols + column for row in range(rows)] for column in range(cols)]
    return [[row * cols + column for row in reversed(range(rows))] for column in range(cols)]


def _segments(path: list[int], flat: list[int]) -> list[list[int]]:
    result: list[list[int]] = []
    current: list[int] = []
    for index in path:
        if flat[index] == WALL:
            if current:
                result.append(current)
                current = []
        else:
            current.append(index)
    if current:
        result.append(current)
    return result


def _largest_rectangle_area(heights: list[int]) -> int:
    stack: list[int] = []
    largest = 0
    for index, height in enumerate([*heights, 0]):
        while stack and height < heights[stack[-1]]:
            popped = heights[stack.pop()]
            width = index - stack[-1] - 1 if stack else index
            largest = max(largest, popped * width)
        stack.append(index)
    return largest


def _max_playable_rectangle(board: list[list[int]]) -> int:
    if not board or not board[0]:
        return 0
    heights = [0] * len(board[0])
    largest = 0
    for row in board:
        for column, value in enumerate(row):
            heights[column] = heights[column] + 1 if value == 0 else 0
        largest = max(largest, _largest_rectangle_area(heights))
    return largest


class _ShapeRandom:
    def __init__(self, seed: str, label: str):
        self.seed = seed
        self.label = label
        self.counter = 0

    def random(self) -> float:
        digest = _digest(self.seed, f"{self.label}:{self.counter}")
        self.counter += 1
        return int.from_bytes(digest[:8], "big") / 2**64

    def index(self, length: int) -> int:
        return min(length - 1, int(self.random() * length))


def _connected_shape(size: int, count: int, rng: _ShapeRandom) -> list[list[int]]:
    board = [[WALL] * size for _ in range(size)]
    visited: set[tuple[int, int]] = set()
    remaining: dict[tuple[int, int], None] = {}
    stack = [(rng.index(size), rng.index(size))]
    while len(visited) < count:
        if not stack:
            choices = list(remaining)
            chosen = choices[rng.index(len(choices))]
            remaining.pop(chosen, None)
            stack = [chosen]
        row, column = stack.pop()
        if not (0 <= row < size and 0 <= column < size) or (row, column) in visited:
            continue
        visited.add((row, column))
        board[row][column] = 0
        neighbors = [
            (row - 1, column), (row + 1, column),
            (row, column - 1), (row, column + 1),
        ]
        randomized: list[tuple[int, int]] = []
        while neighbors:
            randomized.append(neighbors.pop(rng.index(len(neighbors))))
        for candidate in randomized:
            if (
                not (0 <= candidate[0] < size and 0 <= candidate[1] < size)
                or candidate in visited
            ):
                continue
            if rng.random() > 0.5:
                stack.append(candidate)
            else:
                remaining.setdefault(candidate, None)
    return board


def _crop_and_orient_shape(board: list[list[int]]) -> list[list[int]]:
    playable = [
        (row, column)
        for row, values in enumerate(board)
        for column, value in enumerate(values)
        if value == 0
    ]
    min_row = min(row for row, _column in playable)
    max_row = max(row for row, _column in playable)
    min_column = min(column for _row, column in playable)
    max_column = max(column for _row, column in playable)
    cropped = [row[min_column:max_column + 1] for row in board[min_row:max_row + 1]]
    if len(cropped[0]) < len(cropped):
        return [list(column) for column in zip(*cropped)]
    return cropped


def _hard_shape(seed: str, restart_count: int, playable_cells: int, generation_size: int = 7) -> tuple[tuple[int, ...], ...]:
    # Tournament variant: 12 playable cells generated inside a bounded source,
    # filtered to a largest full rectangle area of 4-8 before cropping.
    rng = _ShapeRandom(seed, f"shape:{restart_count}")
    for _attempt in range(4096):
        board = _connected_shape(generation_size, playable_cells, rng)
        rectangle = _max_playable_rectangle(board)
        if 4 <= rectangle <= 8:
            cropped = _crop_and_orient_shape(board)
            return tuple(tuple(row) for row in cropped)
    raise RuntimeError("could not generate a valid hard shape")


def _move(
    board: tuple[tuple[int, ...], ...], direction: str, *, mirror: bool, unmergeable: int | None
) -> tuple[tuple[tuple[int, ...], ...], int, list[dict[str, Any]]]:
    rows, cols = len(board), len(board[0])
    before = [value for row in board for value in row]
    after = [WALL if value == WALL else 0 for value in before]
    gained = 0
    movements: list[dict[str, Any]] = []
    for path in _paths(rows, cols, direction, mirror):
        for segment in _segments(path, before):
            entries = [
                (index, before[index])
                for index in segment
                if before[index] > 0 or before[index] == ISLAND
            ]
            source = 0
            target = 0
            while source < len(entries):
                from_index, value = entries[source]
                destination = segment[target]
                if value == ISLAND:
                    group_end = source + 1
                    while group_end < len(entries) and entries[group_end][1] == ISLAND:
                        group_end += 1
                    after[destination] = ISLAND
                    merged = group_end - source > 1
                    movements.extend(
                        {"from": index, "to": destination, "value": ISLAND, "merged": merged}
                        for index, _value in entries[source:group_end]
                    )
                    source = group_end
                    target += 1
                    continue
                merge = (
                    source + 1 < len(entries)
                    and entries[source + 1][1] == value
                    and value != unmergeable
                )
                if merge:
                    other_index = entries[source + 1][0]
                    after[destination] = value * 2
                    gained += value * 2
                    movements.extend((
                        {"from": from_index, "to": destination, "value": value, "merged": True},
                        {"from": other_index, "to": destination, "value": value, "merged": True},
                    ))
                    source += 2
                else:
                    after[destination] = value
                    movements.append(
                        {"from": from_index, "to": destination, "value": value, "merged": False}
                    )
                    source += 1
                target += 1
    nested = tuple(tuple(after[row * cols:(row + 1) * cols]) for row in range(rows))
    return nested, gained, movements


def _has_moves(board: tuple[tuple[int, ...], ...], rules: VariantRules) -> bool:
    if any(value == 0 for row in board for value in row):
        return True
    return any(_move(board, direction, mirror=rules.mirror_portals, unmergeable=rules.unmergeable)[0] != board for direction in DIRECTIONS)


def _board_sum(board: tuple[tuple[int, ...], ...]) -> int:
    return sum(value for row in board for value in row if value > 0)


def _packed_native_board(board: tuple[tuple[int, ...], ...]) -> int:
    encoded = 0
    for index, value in enumerate(value for row in board for value in row):
        exponent = 0 if value <= 0 else min(15, int(math.log2(value)))
        encoded |= (exponent & 0xF) << ((15 - index) * 4)
    return encoded


class Tournament2048Adapter:
    rules_version = "tournament-v1"

    def __init__(self, rules: VariantRules):
        self.rules = rules
        self.project_id = rules.project_ref

    @property
    def descriptor(self) -> ProjectDescriptor:
        return ProjectDescriptor(
            project_ref=self.project_id,
            rules_version=self.rules_version,
            display_name=self.rules.display_name,
            view_kind="2048-board",
            view_protocol="2048-board-v2",
        )

    def seed_for_side(self, shared_seed: str, side: str) -> str:
        if not self.rules.dice_wall:
            return shared_seed
        return hmac.new(
            bytes.fromhex(shared_seed), f"side:{side}".encode("ascii"), hashlib.sha256
        ).hexdigest()

    def _spawn(
        self, board: tuple[tuple[int, ...], ...], seed: str, counter: int, *, allow_special: bool = True
    ) -> tuple[tuple[tuple[int, ...], ...], int, dict[str, int] | None]:
        flat = [value for row in board for value in row]
        empty = [index for index, value in enumerate(flat) if value == 0]
        if not empty:
            return board, counter, None
        digest = _digest(seed, f"spawn:{counter}")
        if self.rules.isolated_island and allow_special:
            island_count = flat.count(ISLAND)
            island_chance = 0.05 - island_count * 0.02
            island_roll = int.from_bytes(digest[:8], "big") / 2**64
            index = empty[int.from_bytes(digest[8:16], "big") % len(empty)]
            if island_roll < island_chance:
                value = ISLAND
            else:
                value = 4 if digest[16] / 255 < self.rules.spawn4_rate else 2
        elif self.rules.evil_spawn:
            adaptive_depth = (
                7 if len(empty) <= 1 else
                6 if len(empty) <= 2 else
                5 if len(empty) <= 5 else 4
            )
            depth = min(adaptive_depth, 5) if _board_sum(board) < 120 else adaptive_depth
            try:
                from native_core.ai_core import EvilGen

                generator = EvilGen(_packed_native_board(board))
                result = generator.gen_new_num_seeded(
                    depth, int.from_bytes(digest[:4], "big")
                )
                index, exponent = int(result[1]), int(result[2])
                if index not in empty or exponent not in (1, 2):
                    raise ValueError("invalid EvilGen spawn")
                value = 2**exponent
            except (ImportError, ValueError, RuntimeError):
                # Development environments without the native module stay deterministic.
                # Production deploys the same EvilGen core as the browser WASM build.
                index = empty[int.from_bytes(digest[:8], "big") % len(empty)]
                value = 4 if digest[8] % 10 == 0 else 2
        else:
            index = empty[int.from_bytes(digest[:8], "big") % len(empty)]
            value = 4 if digest[8] / 255 < self.rules.spawn4_rate else 2
        flat[index] = value
        rows, cols = len(board), len(board[0])
        nested = tuple(tuple(flat[row * cols:(row + 1) * cols]) for row in range(rows))
        return nested, counter + 1, {"index": index, "value": value}

    def _empty_board(self, seed: str, restart_count: int, side: str | None = None) -> tuple[tuple[tuple[int, ...], ...], dict[str, Any]]:
        if self.rules.shape_playable_cells is not None:
            board = _hard_shape(seed, restart_count, self.rules.shape_playable_cells,
                                self.rules.shape_generation_size)
            return board, {"restart_count": restart_count, "shape_shifter": True}
        flat = [0] * (self.rules.rows * self.rules.cols)
        extra: dict[str, Any] = {"restart_count": restart_count}
        if self.rules.dice_wall:
            dice_label = f"dice:{side}:{restart_count}" if side is not None else f"dice:{restart_count}"
            digest = _digest(seed, dice_label)
            die = digest[0] % 6 + 1
            rows, cols = self.rules.rows, self.rules.cols
            corners = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
            edges = [
                index for index in range(rows * cols)
                if index not in corners
                and (index < cols or index >= (rows - 1) * cols or index % cols in {0, cols - 1})
            ]
            centers = [
                row * cols + col
                for row in range((rows - 1) // 2, rows // 2 + 1)
                for col in range((cols - 1) // 2, cols // 2 + 1)
            ]
            candidates = corners if die <= 3 else edges if die <= 5 else centers
            wall_index = candidates[digest[1] % len(candidates)]
            flat[wall_index] = WALL
            extra.update({"dice": die, "wall_index": wall_index})
        return (
            tuple(
                tuple(flat[row * self.rules.cols:(row + 1) * self.rules.cols])
                for row in range(self.rules.rows)
            ),
            extra,
        )

    def initial_state(self, *, seed: str) -> ProjectState:
        board, extra = self._empty_board(seed, 0)
        board, counter, first = self._spawn(board, seed, 0, allow_special=False)
        board, counter, second = self._spawn(board, seed, counter, allow_special=False)
        extra.update({
            "revision": 0,
            "history": [],
            "last_transition": {"kind": "initial", "spawns": [first, second]},
        })
        return ProjectState(board=board, score=0, elapsed_ms=0, finished=False, seed=seed, rng_counter=counter, extra=extra)

    def _outcome(self, board: tuple[tuple[int, ...], ...]) -> str | None:
        if self.rules.target_sum is not None and _board_sum(board) == self.rules.target_sum:
            return "target_reached"
        if self.rules.target_tile_count is not None:
            value, count = self.rules.target_tile_count
            if sum(cell == value for row in board for cell in row) >= count:
                return "target_reached"
        if not (self.rules.allow_undo or self.rules.allow_restart) and not _has_moves(board, self.rules):
            return "no_moves"
        return None

    @staticmethod
    def _history_entry(state: ProjectState) -> dict[str, Any]:
        return {
            "board": [list(row) for row in state.board],
            "score": state.score,
            "move_count": state.move_count,
        }

    def apply_move(self, state: ProjectState, move: str) -> ProjectState:
        direction = str(move).lower()
        if direction not in DIRECTIONS:
            raise ValueError("invalid direction")
        board, gained, movements = _move(
            state.board,
            direction,
            mirror=self.rules.mirror_portals,
            unmergeable=self.rules.unmergeable,
        )
        if board == state.board:
            raise ValueError("move does not change board")
        history = list(state.extra.get("history") or [])
        if self.rules.allow_undo:
            history = (history + [self._history_entry(state)])[-64:]
        board, counter, spawn = self._spawn(board, state.seed, state.rng_counter)
        outcome = self._outcome(board)
        extra = {
            **state.extra,
            "revision": int(state.extra.get("revision", state.move_count)) + 1,
            "history": history,
            "last_transition": {
                "kind": "move",
                "direction": direction,
                "before": [value for row in state.board for value in row],
                "movements": movements,
                "spawn": spawn,
            },
        }
        return replace(
            state,
            board=board,
            score=state.score + gained,
            finished=outcome is not None,
            outcome=outcome,
            move_count=state.move_count + 1,
            rng_counter=counter,
            extra=extra,
        )

    def apply_action(self, state: ProjectState, action: dict[str, Any]) -> ProjectState:
        kind = str(action.get("type") or "").lower()
        if kind == "move":
            return self.apply_move(state, str(action.get("direction") or ""))
        if kind == "restart" and self.rules.allow_restart:
            restart_count = int(state.extra.get("restart_count", 0)) + 1
            board, setup_extra = self._empty_board(state.seed, restart_count)
            board, counter, first = self._spawn(
                board, state.seed, state.rng_counter, allow_special=False
            )
            board, counter, second = self._spawn(
                board, state.seed, counter, allow_special=False
            )
            extra = {
                **setup_extra,
                "revision": int(state.extra.get("revision", state.move_count)) + 1,
                "history": [],
                "last_transition": {"kind": "restart", "spawns": [first, second]},
            }
            return replace(
                state, board=board, score=0, finished=False, outcome=None,
                move_count=0, rng_counter=counter, extra=extra,
            )
        if kind == "undo" and self.rules.allow_undo:
            history = list(state.extra.get("history") or [])
            if not history:
                raise ValueError("nothing to undo")
            previous = history.pop()
            board = tuple(tuple(int(value) for value in row) for row in previous["board"])
            extra = {
                **state.extra,
                "revision": int(state.extra.get("revision", state.move_count)) + 1,
                "history": history,
                "last_transition": {"kind": "undo"},
            }
            return replace(
                state, board=board, score=int(previous["score"]), finished=False,
                outcome=None, move_count=int(previous["move_count"]),
                # An undo restores gameplay state but deliberately keeps the live
                # RNG cursor, so retrying a move consumes a fresh spawn.
                rng_counter=state.rng_counter, extra=extra,
            )
        raise ValueError("unsupported project action")

    def result_value(self, state: ProjectState) -> int:
        return _board_sum(state.board) if self.rules.result_metric == "board_sum" else state.score

    def resolve_winner(self, yellow: ProjectState, white: ProjectState) -> tuple[str, str]:
        yellow_target = yellow.outcome == "target_reached"
        white_target = white.outcome == "target_reached"
        if self.rules.race and yellow_target != white_target:
            return ("yellow" if yellow_target else "white"), "race_target"
        yellow_value, white_value = self.result_value(yellow), self.result_value(white)
        winner = "yellow" if yellow_value > white_value else "white" if white_value > yellow_value else "draw"
        return winner, "board_sum" if self.rules.result_metric == "board_sum" else "score"

    def public_payload(self, state: ProjectState) -> dict[str, Any]:
        tile_count = None
        if self.rules.target_tile_count:
            value, _target = self.rules.target_tile_count
            tile_count = sum(cell == value for row in state.board for cell in row)
        return {
            "board": [list(row) for row in state.board],
            "rows": len(state.board),
            "cols": len(state.board[0]),
            "score": state.score,
            "board_sum": _board_sum(state.board),
            "move_count": state.move_count,
            "finished": state.finished,
            "outcome": state.outcome,
            "elapsed_ms": state.elapsed_ms,
            "allow_restart": self.rules.allow_restart,
            "allow_undo": self.rules.allow_undo,
            "evil_spawn": self.rules.evil_spawn,
            "can_undo": bool(state.extra.get("history")),
            "target_sum": self.rules.target_sum,
            "target_tile": self.rules.target_tile_count[0] if self.rules.target_tile_count else None,
            "target_count": self.rules.target_tile_count[1] if self.rules.target_tile_count else None,
            "current_target_count": tile_count,
            "no_moves": not _has_moves(state.board, self.rules),
            "unmergeable_value": self.rules.unmergeable,
            "mirror_portals": self.rules.mirror_portals,
            "dice": state.extra.get("dice"),
            "wall_index": state.extra.get("wall_index"),
            "result_metric": self.rules.result_metric,
            "isolated_island": self.rules.isolated_island,
            "shape_shifter": self.rules.shape_playable_cells is not None,
            "last_transition": state.extra.get("last_transition"),
        }

    def public_view(self, state: ProjectState, *, generation: int = 1) -> PublicProjectView:
        return PublicProjectView(
            view_kind=self.descriptor.view_kind,
            view_protocol=self.descriptor.view_protocol,
            generation=max(1, int(generation)),
            sequence=int(state.extra.get("revision", state.move_count)),
            payload=self.public_payload(state),
        )

    def verify_result(self, record: bytes, claimed: ProjectState) -> bool:
        return bool(record) and claimed.finished


TOURNAMENT_ADAPTER_FACTORIES = tuple(
    (lambda selected=rules: Tournament2048Adapter(selected)) for rules in TOURNAMENT_RULES
)


class Tournament2048AdapterV2(Tournament2048Adapter):
    """Shared numeric stream; side-specific dice and independent setup streams."""

    rules_version = "tournament-v2"

    def seed_for_side(self, shared_seed: str, side: str) -> str:
        return shared_seed

    def _outcome(self, board: tuple[tuple[int, ...], ...]) -> str | None:
        if self.project_id == "tournament-pure2-full-race-3x3" and _board_sum(board) >= 1022:
            return "target_reached"
        return super()._outcome(board)

    def _spawn(
        self, board: tuple[tuple[int, ...], ...], seed: str, counter: int, *, allow_special: bool = True
    ) -> tuple[tuple[tuple[int, ...], ...], int, dict[str, int] | None]:
        spawned, next_counter, event = super()._spawn(board, seed, counter, allow_special=allow_special)
        return spawned, max(counter + 1, next_counter), event

    def initial_state(self, *, seed: str) -> ProjectState:
        return self.initial_state_for_side(seed=seed, side="solo")

    def initial_state_for_side(self, *, seed: str, side: str) -> ProjectState:
        board, extra = self._empty_board(seed, 0, side)
        board, counter, first = self._spawn(board, seed, 0, allow_special=False)
        board, counter, second = self._spawn(board, seed, counter, allow_special=False)
        extra.update({
            "side": side,
            "revision": 0,
            "history": [],
            "last_transition": {"kind": "initial", "spawns": [first, second]},
        })
        return ProjectState(board=board, score=0, elapsed_ms=0, finished=False,
                            seed=seed, rng_counter=counter, extra=extra)

    def apply_action(self, state: ProjectState, action: dict[str, Any]) -> ProjectState:
        if str(action.get("type") or "").lower() != "restart" or not self.rules.allow_restart:
            return super().apply_action(state, action)
        restart_count = int(state.extra.get("restart_count", 0)) + 1
        board, setup_extra = self._empty_board(state.seed, restart_count, str(state.extra.get("side", "solo")))
        board, counter, first = self._spawn(board, state.seed, state.rng_counter, allow_special=False)
        board, counter, second = self._spawn(board, state.seed, counter, allow_special=False)
        extra = {
            **setup_extra,
            "side": state.extra.get("side", "solo"),
            "revision": int(state.extra.get("revision", state.move_count)) + 1,
            "history": [],
            "last_transition": {"kind": "restart", "spawns": [first, second]},
        }
        return replace(state, board=board, score=0, finished=False, outcome=None,
                       move_count=0, rng_counter=counter, extra=extra)


TOURNAMENT_V2_ADAPTER_FACTORIES = tuple(
    (lambda selected=rules: Tournament2048AdapterV2(selected)) for rules in TOURNAMENT_RULES
)


class Tournament2048AdapterV3(Tournament2048AdapterV2):
    rules_version = "tournament-v3"


TOURNAMENT_V3_ADAPTER_FACTORIES = (
    lambda: Tournament2048AdapterV3(replace(
        next(rules for rules in TOURNAMENT_RULES if rules.shape_playable_cells is not None),
        rows=6, cols=6, shape_generation_size=6,
    )),
)
