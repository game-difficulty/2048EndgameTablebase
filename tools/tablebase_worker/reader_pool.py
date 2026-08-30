from __future__ import annotations

import asyncio
import math
import logging
import math
import random
import re
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .config import TABLE_FILE_SUFFIXES, TableConfig, WorkerConfig, table_path_status


logger = logging.getLogger("tablebase_worker.reader")


class TableUnavailable(RuntimeError):
    pass


class BattleRouteGenerationError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class BattleRouteResult:
    route_blob: bytes
    step_count: int
    certainty_step: int | None
    termination_reason: str
    initial_board: int
    available_layers: int


TRAINER_ROUTE_RECORD = struct.Struct("<B4I")
TRAINER_ROUTE_RATE_SCALE = 4_000_000_000
BATTLE_ROUTE_DIRECTIONS = ("up", "down", "left", "right")
BATTLE_ROUTE_MOVE_CODES = {"up": 3, "down": 4, "left": 1, "right": 2}
BATTLE_ROUTE_RANDOM_START_ATTEMPTS = 16
BATTLE_ROUTE_MAX_STEPS = 9_999
BATTLE_ROUTE_TIE_ORDER = ("left", "right", "down", "up")


def _count_available_layers(table: TableConfig) -> int:
    suffixes = "|".join(re.escape(suffix) for suffix in TABLE_FILE_SUFFIXES)
    layer_re = re.compile(
        rf"^{re.escape(table.table_id)}_(\d+)(?:{suffixes}|b)$",
        re.IGNORECASE,
    )
    layers: set[int] = set()
    try:
        for item in table.path.iterdir():
            match = layer_re.fullmatch(item.name)
            if match and (item.is_file() or item.is_dir()):
                layers.add(int(match.group(1)))
    except OSError as exc:
        raise BattleRouteGenerationError(
            "TABLE_PATH_UNREADABLE", "Table layer metadata is unavailable"
        ) from exc
    return len(layers)


def _normalized_rate(value: Any, dtype: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    try:
        from Config import DTYPE_CONFIG

        _, _, _, zero_value = DTYPE_CONFIG.get(dtype, DTYPE_CONFIG["uint32"])
        if float(zero_value) < 0:
            numeric += abs(float(zero_value))
    except Exception:
        pass
    return max(0.0, min(1.0, numeric))


def _initial_route_record(board: int) -> bytes:
    value = int(board) & ((1 << 64) - 1)
    return TRAINER_ROUTE_RECORD.pack(
        0,
        value & 0xFFFF,
        (value >> 16) & 0xFFFF,
        (value >> 32) & 0xFFFF,
        (value >> 48) & 0xFFFF,
    )


def _route_step_record(
    direction: str,
    spawn_index: int,
    spawn_exponent: int,
    rates: tuple[float, float, float, float],
) -> bytes:
    direction_code = BATTLE_ROUTE_DIRECTIONS.index(direction)
    change = (
        direction_code
        | ((int(spawn_index) & 0xF) << 2)
        | (((int(spawn_exponent) - 1) & 0x1) << 6)
    )
    encoded_rates = tuple(
        max(0, min(TRAINER_ROUTE_RATE_SCALE, round(rate * TRAINER_ROUTE_RATE_SCALE)))
        for rate in rates
    )
    return TRAINER_ROUTE_RECORD.pack(change, *encoded_rates)


def _board_contains_target(board: int, target: int) -> bool:
    if target < 2 or target & (target - 1):
        return False
    target_exponent = target.bit_length() - 1
    value = int(board)
    return any(((value >> (index * 4)) & 0xF) == target_exponent for index in range(16))


def _spawn_deterministically(
    board: int,
    *,
    spawn_rate: float,
    rng: random.Random,
) -> tuple[int, int, int]:
    import numpy as np

    from engine_core.VBoardMover import decode_board, encode_board

    decoded = decode_board(np.uint64(board)).copy()
    empty_positions = [
        index for index, value in enumerate(decoded.reshape(-1).tolist()) if int(value) == 0
    ]
    if not empty_positions:
        raise BattleRouteGenerationError("NO_SPAWN_CELL", "Moved board has no empty cell")
    spawn_index = empty_positions[rng.randrange(len(empty_positions))]
    spawn_exponent = 2 if rng.random() < spawn_rate else 1
    row, column = divmod(spawn_index, 4)
    decoded[row, column] = 2**spawn_exponent
    return int(encode_board(decoded)), spawn_index, spawn_exponent


def _generate_route_once(
    reader: ReaderAdapter,
    table: TableConfig,
    *,
    initial_board: int,
    route_limit: int,
    spawn_rate: float,
    seed: int,
    available_layers: int,
) -> BattleRouteResult:
    import numpy as np

    from Config import category_info
    from engine_core.BoardMover import s_move_board as standard_move_board
    from engine_core.VBoardMover import s_move_board as variant_move_board

    use_variant = table.pattern in category_info.get("variant", [])
    move_board = variant_move_board if use_variant else standard_move_board
    target_value = int(table.target)
    board = int(initial_board) & ((1 << 64) - 1)
    route = bytearray(_initial_route_record(board))
    rng = random.Random(seed)
    certainty_step: int | None = None
    termination_reason = "max_steps"

    if _board_contains_target(board, target_value):
        return BattleRouteResult(
            bytes(route), 0, None, "target_reached", board, available_layers
        )

    for step_index in range(route_limit):
        raw_results, dtype = reader.lookup(
            board,
            use_variant=use_variant,
            board_is_lookup=False,
        )
        rates = tuple(
            _normalized_rate(raw_results.get(direction), str(dtype or "uint32"))
            for direction in BATTLE_ROUTE_DIRECTIONS
        )
        moved_boards: dict[str, int] = {}
        for direction in BATTLE_ROUTE_DIRECTIONS:
            moved, _score = move_board(
                np.uint64(board), BATTLE_ROUTE_MOVE_CODES[direction]
            )
            moved_value = int(moved)
            if moved_value != board:
                moved_boards[direction] = moved_value
        if not moved_boards:
            termination_reason = "no_legal_move"
            break

        best_direction: str | None = None
        best_rate = -1.0
        rates_by_direction = dict(zip(BATTLE_ROUTE_DIRECTIONS, rates))
        for direction in BATTLE_ROUTE_TIE_ORDER:
            rate = rates_by_direction[direction]
            if direction in moved_boards and rate > best_rate:
                best_direction = direction
                best_rate = rate
        if best_direction is None or best_rate <= 0:
            termination_reason = "zero_success"
            break
        if certainty_step is None and best_rate >= 1.0:
            certainty_step = step_index

        next_board, spawn_index, spawn_exponent = _spawn_deterministically(
            moved_boards[best_direction],
            spawn_rate=spawn_rate,
            rng=rng,
        )
        route.extend(
            _route_step_record(
                best_direction,
                spawn_index,
                spawn_exponent,
                rates,
            )
        )
        board = next_board
        if _board_contains_target(board, target_value):
            termination_reason = "target_reached"
            break

    step_count = len(route) // TRAINER_ROUTE_RECORD.size - 1
    return BattleRouteResult(
        route_blob=bytes(route),
        step_count=step_count,
        certainty_step=certainty_step,
        termination_reason=termination_reason,
        initial_board=int(initial_board),
        available_layers=available_layers,
    )


class ReaderAdapter(Protocol):
    def lookup(
        self,
        board: int,
        *,
        use_variant: bool,
        board_is_lookup: bool,
    ) -> tuple[dict[str, Any], str]: ...

    def random_state(self) -> int: ...


class ExistingBookReaderAdapter:
    """Lazy adapter around the project's existing BookReaderDispatcher."""

    def __init__(self, table: TableConfig):
        import numpy as np

        from engine_core.BookReader import BookReaderDispatcher

        self._np = np
        self._table = table
        self._reader = BookReaderDispatcher()
        self._reader.dispatch(table.path_list, table.pattern, table.target)

    def lookup(
        self,
        board: int,
        *,
        use_variant: bool,
        board_is_lookup: bool,
    ) -> tuple[dict[str, Any], str]:
        from Config import pattern_32k_tiles_map
        from backend.trainer_helpers import replace_board_for_lookup
        from engine_core.VBoardMover import decode_board

        lookup_board = self._np.uint64(board)
        if not board_is_lookup:
            n_large_tiles = pattern_32k_tiles_map.get(self._table.pattern, [0])[0]
            lookup_board = self._np.uint64(
                replace_board_for_lookup(
                    lookup_board,
                    self._table.pattern,
                    n_large_tiles,
                    self._table.target,
                    use_variant,
                )
            )
        return self._reader.move_on_dic(
            decode_board(lookup_board),
            self._table.pattern,
            self._table.target,
            self._table.table_id,
        )

    def random_state(self) -> int:
        from Config import SingletonConfig

        SingletonConfig().config["4_spawn_rate"] = self._table.spawn_rate
        return int(self._reader.get_random_state(self._table.path_list, self._table.table_id))


@dataclass
class TableRuntime:
    config: TableConfig
    ready: bool
    error_code: str | None
    initialized: bool = False
    active: int = 0
    completed: int = 0
    failed: int = 0
    reader: ReaderAdapter | None = None


ReaderFactory = Callable[[TableConfig], ReaderAdapter]


class ReaderPool:
    def __init__(
        self,
        config: WorkerConfig,
        *,
        reader_factory: ReaderFactory = ExistingBookReaderAdapter,
        path_checker: Callable[[TableConfig], tuple[bool, str | None]] = table_path_status,
    ):
        self.config = config
        self._reader_factory = reader_factory
        self._path_checker = path_checker
        self._group_semaphores = {
            name: asyncio.Semaphore(group.concurrency)
            for name, group in config.resource_groups.items()
        }
        self._table_semaphores = {
            table_id: asyncio.Semaphore(table.concurrency)
            for table_id, table in config.tables.items()
        }
        self._executors = {
            name: ThreadPoolExecutor(
                max_workers=group.concurrency,
                thread_name_prefix=f"tablebase-{name}",
            )
            for name, group in config.resource_groups.items()
        }
        self._init_locks = {table_id: threading.Lock() for table_id in config.tables}
        self._tables: dict[str, TableRuntime] = {}
        for table_id, table in config.tables.items():
            ready, error_code = path_checker(table)
            self._tables[table_id] = TableRuntime(table, ready, error_code)
            if ready:
                logger.info("Table path is available: %s", table_id)
            else:
                logger.error("Table is unavailable: %s (%s)", table_id, error_code)
        self._closed = False

    @property
    def allowed_tables(self) -> frozenset[str]:
        return frozenset(self._tables)

    @property
    def ready_tables(self) -> frozenset[str]:
        return frozenset(
            table_id for table_id, runtime in self._tables.items() if runtime.ready
        )

    @property
    def table_metadata(self) -> dict[str, tuple[str, str]]:
        return {
            table_id: (runtime.config.pattern, runtime.config.target)
            for table_id, runtime in self._tables.items()
        }

    def hello_tables(self) -> list[dict[str, Any]]:
        return [
            {"full_pattern": table_id, "ready": runtime.ready}
            for table_id, runtime in sorted(self._tables.items())
        ]

    def refresh_readiness(self) -> list[dict[str, Any]]:
        for table_id, runtime in self._tables.items():
            ready, error_code = self._path_checker(runtime.config)
            if ready != runtime.ready or error_code != runtime.error_code:
                logger.warning(
                    "Table readiness changed: %s (%s -> %s)",
                    table_id,
                    "ready" if runtime.ready else runtime.error_code,
                    "ready" if ready else error_code,
                )
            runtime.ready = ready
            runtime.error_code = error_code
        return self.hello_tables()

    def status(self) -> dict[str, Any]:
        return {
            table_id: {
                "ready": runtime.ready,
                "initialized": runtime.initialized,
                "active": runtime.active,
                "completed": runtime.completed,
                "failed": runtime.failed,
                "error_code": runtime.error_code,
            }
            for table_id, runtime in sorted(self._tables.items())
        }

    def _runtime(self, table_id: str) -> TableRuntime:
        runtime = self._tables.get(table_id)
        if runtime is None:
            raise TableUnavailable("TABLE_NOT_ALLOWED")
        if not runtime.ready:
            raise TableUnavailable(runtime.error_code or "TABLE_UNAVAILABLE")
        return runtime

    def _reader(self, runtime: TableRuntime) -> ReaderAdapter:
        if runtime.reader is not None:
            return runtime.reader
        lock = self._init_locks[runtime.config.table_id]
        with lock:
            if runtime.reader is None:
                logger.info("Initializing table reader: %s", runtime.config.table_id)
                runtime.reader = self._reader_factory(runtime.config)
                runtime.initialized = True
        return runtime.reader

    async def _run(self, table_id: str, operation: Callable[[ReaderAdapter], Any]) -> Any:
        if self._closed:
            raise RuntimeError("Reader pool is closed")
        runtime = self._runtime(table_id)
        group = runtime.config.resource_group
        async with self._group_semaphores[group], self._table_semaphores[table_id]:
            runtime.active += 1
            loop = asyncio.get_running_loop()

            def execute():
                return operation(self._reader(runtime))

            future = loop.run_in_executor(self._executors[group], execute)
            try:
                result = await asyncio.shield(future)
                runtime.completed += 1
                return result
            except asyncio.CancelledError:
                # Native reads cannot be interrupted safely. Hold the resource-group
                # slot until the underlying call exits, then suppress its response.
                try:
                    await future
                except Exception:
                    runtime.failed += 1
                raise
            except Exception:
                runtime.failed += 1
                raise
            finally:
                runtime.active -= 1

    async def lookup(
        self,
        table_id: str,
        board: int,
        *,
        use_variant: bool,
        board_is_lookup: bool,
    ) -> tuple[dict[str, Any], str]:
        return await self._run(
            table_id,
            lambda reader: reader.lookup(
                board,
                use_variant=use_variant,
                board_is_lookup=board_is_lookup,
            ),
        )

    async def lookup_batch(
        self,
        table_id: str,
        boards: tuple[int, ...],
        *,
        use_variant: bool,
        board_is_lookup: bool,
    ) -> list[tuple[dict[str, Any], str]]:
        def execute(reader: ReaderAdapter):
            return [
                reader.lookup(
                    board,
                    use_variant=use_variant,
                    board_is_lookup=board_is_lookup,
                )
                for board in boards
            ]

        return await self._run(table_id, execute)

    async def random_state(self, table_id: str) -> int:
        return await self._run(table_id, lambda reader: reader.random_state())

    async def generate_battle_route(
        self,
        table_id: str,
        *,
        initial_board: int | None,
        max_steps: int | None,
        min_steps: int,
        spawn_rate: float,
        seed_hex: str,
    ) -> BattleRouteResult:
        runtime = self._runtime(table_id)

        def execute(reader: ReaderAdapter) -> BattleRouteResult:
            available_layers = _count_available_layers(runtime.config)
            if available_layers <= 0:
                raise BattleRouteGenerationError(
                    "LAYERS_UNAVAILABLE", "No queryable table layers were found"
                )
            route_limit = (
                int(max_steps)
                if max_steps is not None
                else min(BATTLE_ROUTE_MAX_STEPS, available_layers)
            )
            effective_min_steps = int(min_steps)
            if effective_min_steps <= 0:
                effective_min_steps = int(
                    math.ceil(min(int(runtime.config.target) // 2, available_layers) * 0.6)
                )
            if effective_min_steps > route_limit:
                raise BattleRouteGenerationError(
                    "ROUTE_TOO_SHORT",
                    "The minimum route length exceeds the available route limit",
                )

            base_seed = int(seed_hex, 16)
            attempt_count = 1 if initial_board is not None else BATTLE_ROUTE_RANDOM_START_ATTEMPTS
            best_result: BattleRouteResult | None = None
            for attempt in range(attempt_count):
                candidate_board = (
                    int(initial_board)
                    if initial_board is not None
                    else int(reader.random_state())
                )
                result = _generate_route_once(
                    reader,
                    runtime.config,
                    initial_board=candidate_board,
                    route_limit=route_limit,
                    spawn_rate=float(spawn_rate),
                    seed=(base_seed + attempt) & ((1 << 128) - 1),
                    available_layers=available_layers,
                )
                if best_result is None or result.step_count > best_result.step_count:
                    best_result = result
                if result.step_count >= effective_min_steps:
                    return result

            raise BattleRouteGenerationError(
                "ROUTE_TOO_SHORT",
                (
                    "Generated route does not meet the minimum length"
                    if best_result is None
                    else "Generated route is shorter than the requested minimum"
                ),
            )

        return await self._run(table_id, execute)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
