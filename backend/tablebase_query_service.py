from __future__ import annotations

import asyncio
import heapq
import itertools
import math
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from Config import pattern_32k_tiles_map
from engine_core.VBoardMover import decode_board

from .session import safe_hex, u64
from .trainer_helpers import replace_board_for_lookup
from .remote_workers.registry import remote_worker_registry


QUERY_WORKERS = 4
MAX_PREFETCH_IN_FLIGHT = QUERY_WORKERS - 1
MAX_PREFETCH_CHILDREN = 8
SERVER_RESULT_CACHE_SIZE = 4096
SERVER_RESULT_CACHE_TTL_SECONDS = 300.0
REGULAR_QUEUE_LIMIT = 64
SUPPORTER_QUEUE_LIMIT = 256
REGULAR_WAIT_LIMIT_SECONDS = 1.5
SUPPORTER_WAIT_LIMIT_SECONDS = 5.0
QueryKey = tuple[str, str, int] | tuple[str, str, int, str]


class TablebaseQuerySuperseded(RuntimeError):
    pass


class TablebaseQueryOverloaded(RuntimeError):
    def __init__(self, estimated_wait_seconds: float):
        self.estimated_wait_seconds = max(0.0, float(estimated_wait_seconds))
        super().__init__("Tablebase query service is busy.")

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "code": "TABLEBASE_BUSY",
            "message": "Tablebase query service is busy. Please retry shortly.",
            "retry_after_ms": max(250, int(math.ceil(self.estimated_wait_seconds * 1000))),
        }


@dataclass(frozen=True)
class TablebaseLookupSpec:
    board_encoded: int
    pattern: str
    target: str
    full_pattern: str
    use_variant: bool
    book_reader: Any = field(compare=False, repr=False)
    provider_kind: str = "local"
    catalog_version: str = ""

    @property
    def query_key(self) -> QueryKey:
        return (
            str(self.catalog_version or ""),
            str(self.full_pattern),
            u64(self.board_encoded),
        )


@dataclass(frozen=True)
class TablebaseLookupResult:
    board_encoded: int
    full_pattern: str
    results: dict[str, Any]
    dtype: str
    best_move: str | None
    legal_moves_mask: int | None = None

    @property
    def board_hex(self) -> str:
        return safe_hex(self.board_encoded)

    @property
    def found(self) -> bool:
        return any(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in self.results.values()
        )


@dataclass
class _Subscriber:
    subscriber_id: str
    stream_key: str
    generation: int
    cancelled: bool = False
    authorized: bool = True


@dataclass
class _QueryJob:
    spec: TablebaseLookupSpec
    lane: int
    supporter: bool
    future: asyncio.Future
    priority: tuple[int, float, int]
    queue_version: int = 1
    status: str = "pending"
    subscribers: dict[str, _Subscriber] = field(default_factory=dict)


class TablebaseQueryHandle:
    def __init__(
        self,
        scheduler: "TablebaseQueryScheduler",
        job: _QueryJob | None,
        subscriber: _Subscriber,
        *,
        cached_result: TablebaseLookupResult | None = None,
    ):
        self._scheduler = scheduler
        self._job = job
        self._subscriber = subscriber
        self._cached_result = cached_result

    @property
    def generation(self) -> int:
        return self._subscriber.generation

    @property
    def stream_key(self) -> str:
        return self._subscriber.stream_key

    @property
    def is_current(self) -> bool:
        return self._scheduler.is_subscriber_current(self._subscriber)

    def cancel(self) -> None:
        self._subscriber.cancelled = True

    async def activate(self) -> None:
        self._subscriber.authorized = True
        condition = self._scheduler._condition
        if condition is not None:
            async with condition:
                condition.notify_all()

    async def wait(self) -> TablebaseLookupResult:
        if not self.is_current:
            raise TablebaseQuerySuperseded()
        if self._cached_result is not None:
            return self._cached_result
        if self._job is None:
            raise TablebaseQuerySuperseded()
        result = await asyncio.shield(self._job.future)
        if not self.is_current:
            raise TablebaseQuerySuperseded()
        return result


def _sanitize_results(raw_results: Any) -> dict[str, Any]:
    if not isinstance(raw_results, dict):
        return {}
    sanitized: dict[str, Any] = {}
    for key, value in raw_results.items():
        normalized_key = str(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            sanitized[normalized_key] = numeric if math.isfinite(numeric) else None
        elif value in ("", None):
            sanitized[normalized_key] = None
        else:
            sanitized[normalized_key] = str(value)
    return sanitized


def _best_move(results: dict[str, Any]) -> str | None:
    for direction, value in results.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return direction
    return None


def execute_tablebase_lookup(spec: TablebaseLookupSpec) -> TablebaseLookupResult:
    n_large_tiles = pattern_32k_tiles_map.get(spec.pattern, [0])[0]
    lookup_board = replace_board_for_lookup(
        np.uint64(u64(spec.board_encoded)),
        spec.pattern,
        n_large_tiles,
        spec.target,
        spec.use_variant,
    )
    raw_results, dtype = spec.book_reader.move_on_dic(
        decode_board(np.uint64(u64(lookup_board))),
        spec.pattern,
        spec.target,
        spec.full_pattern,
    )
    results = _sanitize_results(raw_results)
    return TablebaseLookupResult(
        board_encoded=u64(spec.board_encoded),
        full_pattern=spec.full_pattern,
        results=results,
        dtype=str(dtype or "?"),
        best_move=_best_move(results),
        legal_moves_mask=getattr(raw_results, 'legal_moves_mask', None),
    )


async def execute_tablebase_lookup_async(
    spec: TablebaseLookupSpec,
    *,
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
) -> TablebaseLookupResult:
    if spec.provider_kind != "remote":
        return await loop.run_in_executor(executor, execute_tablebase_lookup, spec)
    response = await remote_worker_registry.lookup(
        full_pattern=spec.full_pattern,
        pattern=spec.pattern,
        target=spec.target,
        board=safe_hex(spec.board_encoded),
        use_variant=spec.use_variant,
    )
    results = _sanitize_results(response.get("results"))
    return TablebaseLookupResult(
        board_encoded=u64(spec.board_encoded),
        full_pattern=spec.full_pattern,
        results=results,
        dtype=str(response.get("dtype") or "?"),
        best_move=_best_move(results),
        legal_moves_mask=response.get('legal_moves_mask'),
    )


class TablebaseQueryScheduler:
    def __init__(self, *, worker_count: int = QUERY_WORKERS):
        self.worker_count = max(1, int(worker_count))
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="tablebase-query",
        )
        self._condition: asyncio.Condition | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._workers: list[asyncio.Task] = []
        self._heap: list[tuple[tuple[int, float, int], int, QueryKey]] = []
        self._jobs: dict[QueryKey, _QueryJob] = {}
        self._latest_generation: dict[str, int] = {}
        self._result_cache: OrderedDict[
            QueryKey, tuple[float, TablebaseLookupResult]
        ] = OrderedDict()
        self._sequence = itertools.count()
        self._active = 0
        self._prefetch_active = 0
        self._service_time_ewma = 0.05

    def _ensure_started(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._workers:
                raise RuntimeError("Tablebase scheduler cannot move between event loops")
            self._loop = loop
            self._condition = asyncio.Condition()
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="tablebase-query",
            )
        if not self._workers:
            self._workers = [
                loop.create_task(self._worker(index))
                for index in range(self.worker_count)
            ]

    def _cache_get(self, key: QueryKey) -> TablebaseLookupResult | None:
        cached = self._result_cache.get(key)
        if cached is None:
            return None
        expires_at, result = cached
        if expires_at <= time.monotonic():
            self._result_cache.pop(key, None)
            return None
        self._result_cache.move_to_end(key)
        return result

    def get_cached_result(
        self,
        *,
        catalog_version: str,
        full_pattern: str,
        board_encoded: int,
    ) -> TablebaseLookupResult | None:
        return self._cache_get(
            (str(catalog_version or ""), str(full_pattern), u64(board_encoded))
        )

    def _cache_set(self, key, result: TablebaseLookupResult) -> None:
        self._result_cache[key] = (
            time.monotonic() + SERVER_RESULT_CACHE_TTL_SECONDS,
            result,
        )
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > SERVER_RESULT_CACHE_SIZE:
            self._result_cache.popitem(last=False)

    def clear_cache(self) -> None:
        self._result_cache.clear()

    def cache_stream_result(self, *, catalog_version, full_pattern, board_encoded, results, dtype, legal_moves_mask=None):
        sanitized = _sanitize_results(results)
        result = TablebaseLookupResult(board_encoded, full_pattern, sanitized, str(dtype), _best_move(sanitized), legal_moves_mask)
        self._cache_set((catalog_version, full_pattern, u64(board_encoded)), result)
        return result

    def current_generation(self, stream_key: str) -> int:
        return int(self._latest_generation.get(str(stream_key), 0))

    def is_subscriber_current(self, subscriber: _Subscriber) -> bool:
        return (
            not subscriber.cancelled
            and self.current_generation(subscriber.stream_key) == subscriber.generation
        )

    def _job_has_current_subscriber(self, job: _QueryJob) -> bool:
        return any(self.is_subscriber_current(item) for item in job.subscribers.values())

    def _job_has_authorized_subscriber(self, job: _QueryJob) -> bool:
        return any(
            item.authorized and self.is_subscriber_current(item)
            for item in job.subscribers.values()
        )

    def _discard_unsubscribed_pending_jobs(
        self,
        *,
        preserve_key: QueryKey | None = None,
    ) -> None:
        for key, job in list(self._jobs.items()):
            if key == preserve_key:
                continue
            if job.status != "pending" or self._job_has_current_subscriber(job):
                continue
            job.status = "done"
            self._jobs.pop(key, None)
            if not job.future.done():
                job.future.set_exception(TablebaseQuerySuperseded())

    def _live_pending_count(self, *, lane: int | None = None) -> int:
        count = 0
        for job in self._jobs.values():
            if job.status != "pending" or not self._job_has_current_subscriber(job):
                continue
            if lane is None or job.lane == lane:
                count += 1
        return count

    def estimated_wait_seconds(self) -> float:
        queued = self._live_pending_count(lane=0)
        work_ahead = max(0, queued + self._active - self.worker_count + 1)
        waves = math.ceil(work_ahead / self.worker_count)
        return waves * self._service_time_ewma

    def stats(self) -> dict[str, Any]:
        return {
            "workers": self.worker_count,
            "active": self._active,
            "prefetch_active": self._prefetch_active,
            "foreground_pending": self._live_pending_count(lane=0),
            "prefetch_pending": self._live_pending_count(lane=1),
            "estimated_wait_seconds": self.estimated_wait_seconds(),
            "service_time_ewma": self._service_time_ewma,
            "cache_entries": len(self._result_cache),
        }

    def _priority(self, *, lane: int, supporter: bool) -> tuple[int, float, int]:
        now = time.monotonic()
        if lane == 0:
            deadline = now + (0.2 if supporter else 0.8)
        else:
            deadline = now + (2.0 if supporter else 3.0)
        return lane, deadline, next(self._sequence)

    def _check_admission(self, *, lane: int, supporter: bool) -> None:
        if lane != 0:
            return
        pending = self._live_pending_count(lane=0)
        estimated_wait = self.estimated_wait_seconds()
        queue_limit = SUPPORTER_QUEUE_LIMIT if supporter else REGULAR_QUEUE_LIMIT
        wait_limit = (
            SUPPORTER_WAIT_LIMIT_SECONDS if supporter else REGULAR_WAIT_LIMIT_SECONDS
        )
        if pending >= queue_limit or estimated_wait > wait_limit:
            raise TablebaseQueryOverloaded(max(estimated_wait, self._service_time_ewma))

    async def submit(
        self,
        spec: TablebaseLookupSpec,
        *,
        stream_key: str,
        supporter: bool,
        lane: str = "foreground",
        supersede: bool = True,
        generation: int | None = None,
        allow_overload: bool = False,
        activate: bool = True,
    ) -> TablebaseQueryHandle:
        self._ensure_started()
        normalized_stream = str(stream_key)
        lane_value = 0 if lane == "foreground" else 1
        if supersede:
            generation = self.current_generation(normalized_stream) + 1
            self._latest_generation[normalized_stream] = generation
            self._discard_unsubscribed_pending_jobs(preserve_key=spec.query_key)
        elif generation is None:
            generation = self.current_generation(normalized_stream)
        generation = int(generation or 0)
        subscriber = _Subscriber(
            uuid.uuid4().hex,
            normalized_stream,
            generation,
            authorized=bool(activate),
        )

        cached = self._cache_get(spec.query_key)
        if cached is not None:
            return TablebaseQueryHandle(self, None, subscriber, cached_result=cached)

        job = self._jobs.get(spec.query_key)
        if not allow_overload and (job is None or job.status == "done"):
            self._check_admission(lane=lane_value, supporter=bool(supporter))
        priority = self._priority(lane=lane_value, supporter=bool(supporter))
        if job is None or job.status == "done":
            job = _QueryJob(
                spec=spec,
                lane=lane_value,
                supporter=bool(supporter),
                future=self._loop.create_future(),
                priority=priority,
            )
            job.future.add_done_callback(
                lambda completed: completed.exception()
                if not completed.cancelled()
                else None
            )
            self._jobs[spec.query_key] = job
            heapq.heappush(
                self._heap,
                (job.priority, job.queue_version, spec.query_key),
            )
        elif job.status == "pending" and priority < job.priority:
            job.priority = priority
            job.lane = min(job.lane, lane_value)
            job.supporter = job.supporter or bool(supporter)
            job.queue_version += 1
            heapq.heappush(
                self._heap,
                (job.priority, job.queue_version, spec.query_key),
            )
        job.subscribers[subscriber.subscriber_id] = subscriber
        async with self._condition:
            self._condition.notify_all()
        return TablebaseQueryHandle(self, job, subscriber)

    def _next_job(self) -> _QueryJob | None:
        deferred: list[tuple[tuple[int, float, int], int, QueryKey]] = []
        selected = None
        while self._heap:
            item = heapq.heappop(self._heap)
            _priority, queue_version, key = item
            job = self._jobs.get(key)
            if (
                job is None
                or job.status != "pending"
                or job.queue_version != queue_version
            ):
                continue
            if not self._job_has_current_subscriber(job):
                job.status = "done"
                self._jobs.pop(key, None)
                if not job.future.done():
                    job.future.set_exception(TablebaseQuerySuperseded())
                continue
            if not self._job_has_authorized_subscriber(job):
                deferred.append(item)
                continue
            if job.lane == 1 and self._prefetch_active >= min(
                MAX_PREFETCH_IN_FLIGHT,
                self.worker_count - 1 if self.worker_count > 1 else 1,
            ):
                deferred.append(item)
                continue
            selected = job
            break
        for item in deferred:
            heapq.heappush(self._heap, item)
        return selected

    async def _worker(self, _worker_index: int) -> None:
        while True:
            async with self._condition:
                job = self._next_job()
                while job is None:
                    await self._condition.wait()
                    job = self._next_job()
                job.status = "running"
                self._active += 1
                if job.lane == 1:
                    self._prefetch_active += 1

            started = time.monotonic()
            try:
                result = await execute_tablebase_lookup_async(
                    job.spec,
                    loop=self._loop,
                    executor=self._executor,
                )
                self._cache_set(job.spec.query_key, result)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                if not job.future.done():
                    job.future.set_exception(exc)
            finally:
                elapsed = max(0.001, time.monotonic() - started)
                self._service_time_ewma = (
                    self._service_time_ewma * 0.8 + elapsed * 0.2
                )
                async with self._condition:
                    job.status = "done"
                    self._jobs.pop(job.spec.query_key, None)
                    self._active = max(0, self._active - 1)
                    if job.lane == 1:
                        self._prefetch_active = max(0, self._prefetch_active - 1)
                    self._condition.notify_all()

    async def close(self) -> None:
        condition = self._condition
        if condition is not None:
            async with condition:
                for job in list(self._jobs.values()):
                    if not job.future.done():
                        job.future.set_exception(TablebaseQuerySuperseded())
                self._jobs.clear()
                self._heap.clear()
                condition.notify_all()
        workers = list(self._workers)
        self._workers = []
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        executor = self._executor
        self._executor = None
        if executor is not None:
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=True,
            )
        self._loop = None
        self._condition = None


tablebase_query_scheduler = TablebaseQueryScheduler()
