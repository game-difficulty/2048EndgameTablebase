from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .config import TableConfig, WorkerConfig, table_path_status


logger = logging.getLogger("tablebase_worker.reader")


class TableUnavailable(RuntimeError):
    pass


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

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for executor in self._executors.values():
            executor.shutdown(wait=True, cancel_futures=False)
