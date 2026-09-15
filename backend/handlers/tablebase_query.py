from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from fastapi import WebSocket
from engine_core.BoardMover import s_move_board as r_move_board
from engine_core.VBoardMover import decode_board, encode_board, s_move_board as v_move_board

from ..actions import Action, Message
from ..trainer_default_lookup import default_lookup_for
from ..gamer_ranked.prng import Xoshiro128StarStar
from ..quota.errors import InsufficientTokens
from ..quota.config import MULTIPLIER_UNIT, table_multiplier_units
from ..auth.guest_service import (
    GuestLimitError,
    GuestQueryReservation,
    finalize_guest_query,
    guest_query_allowance,
    reserve_guest_query,
)
from ..quota.service import (
    cancel_reservation,
    finalize_reservation,
    get_token_balance,
    has_numeric_result,
    reserve_operation_tokens,
)
from ..remote_workers.errors import RemoteTablebaseError
from ..serialization import sanitize_config
from ..session import GameSession, np_u64, safe_hex, u64
from ..tablebase_catalog import get_catalog_version, is_guest_tablebase_available
from ..tablebase_query_service import (
    MAX_PREFETCH_CHILDREN,
    TablebaseLookupResult,
    TablebaseLookupSpec,
    TablebaseQueryOverloaded,
    TablebaseQuerySuperseded,
    tablebase_query_scheduler,
)
from ..tester import _tester_append_post_lookup_logs


_TABLEBASE_QUERY_TASKS: set[asyncio.Task] = set()
_DIRECTION_MAP = {"left": 1, "right": 2, "up": 3, "down": 4}
_PREFETCH_RNG_VERSION = 1


def _prefetch_limit(full_pattern: str) -> int:
    # Use table cost, not promotional pricing, to bound speculative disk reads.
    if table_multiplier_units(full_pattern) >= 50 * MULTIPLIER_UNIT:
        return MAX_PREFETCH_CHILDREN // 2
    return MAX_PREFETCH_CHILDREN


@dataclass(frozen=True)
class _PrefetchRngContext:
    state: tuple[int, int, int, int]
    turn: int
    spawn_rate_4: float


@dataclass(frozen=True)
class _PrefetchNode:
    board_encoded: int
    rng: _PrefetchRngContext
    depth: int
    direction: str
    order: int


def _track_task(task: asyncio.Task) -> asyncio.Task:
    _TABLEBASE_QUERY_TASKS.add(task)

    def done(completed: asyncio.Task) -> None:
        _TABLEBASE_QUERY_TASKS.discard(completed)
        try:
            completed.result()
        except (asyncio.CancelledError, Exception):
            pass

    task.add_done_callback(done)
    return task


async def drain_tablebase_query_tasks() -> None:
    while _TABLEBASE_QUERY_TASKS:
        await asyncio.gather(
            *list(_TABLEBASE_QUERY_TASKS),
            return_exceptions=True,
        )


def _session_query_context(session: GameSession, page: str) -> dict[str, Any] | None:
    if page == "trainer":
        full_pattern = str(session.current_pattern or "")
        pattern, target = [str(value) for value in session.pattern_settings[:2]]
        provider_kind = str(getattr(session, "tablebase_provider_kind", "") or "local")
        table_available = bool(
            full_pattern
            and (
                provider_kind == "remote"
                or (provider_kind == "local" and session.book_reader is not None)
            )
        )
    elif page == "tester":
        full_pattern = str(session.tester_full_pattern or "")
        pattern, target = [str(value) for value in session.tester_pattern[:2]]
        provider_kind = str(
            getattr(session, "tester_tablebase_provider_kind", "") or "local"
        )
        table_available = bool(session.tester_table_found and full_pattern)
    else:
        return None
    if not table_available or not pattern or not target:
        return None
    return {
        "full_pattern": full_pattern,
        "pattern": pattern,
        "target": target,
        "use_variant": bool(session.use_variant),
        "book_reader": (
            session.ensure_book_reader() if provider_kind == "local" else None
        ),
        "provider_kind": provider_kind,
    }


def _session_is_supporter(session: GameSession) -> bool:
    return str(getattr(session, "user_entitlement_tier", "free")) == "supporter"


def _session_matches(
    session: GameSession,
    *,
    page: str,
    board_encoded: int,
    full_pattern: str,
) -> bool:
    selected_pattern = (
        session.current_pattern if page == "trainer" else session.tester_full_pattern
    )
    return (
        str(selected_pattern or "") == str(full_pattern)
        and u64(session.board_encoded) == u64(board_encoded)
    )


def _apply_session_result(
    session: GameSession,
    *,
    page: str,
    result: TablebaseLookupResult,
) -> None:
    if page == "trainer":
        session.trainer_results = dict(result.results)
        session.trainer_results_board = np_u64(result.board_encoded)
        if result.dtype != "?":
            session.success_rate_dtype = result.dtype
        return
    session.tester_results = dict(result.results)
    session.tester_result_dtype = str(result.dtype or "?")
    session.tester_best_move = result.best_move
    session.tester_results_board = np_u64(result.board_encoded)
    session.tester_lookup_pending = False
    if result.dtype != "?":
        session.success_rate_dtype = result.dtype


def _prefetch_boards(
    board_encoded: int,
    *,
    best_move: str | None,
    use_variant: bool,
) -> list[int]:
    direction = _DIRECTION_MAP.get(str(best_move or ""))
    if direction is None:
        return []
    move_fn = v_move_board if use_variant else r_move_board
    moved_board, _score = move_fn(np_u64(board_encoded), direction)
    moved_board = np_u64(moved_board)
    if moved_board == np_u64(board_encoded):
        return []
    board = decode_board(moved_board).copy()
    empty_positions = [
        index for index, value in enumerate(board.reshape(-1).tolist()) if int(value) == 0
    ]
    children: list[int] = []
    for spawn_value in (2, 4):
        for index in empty_positions:
            child = board.copy()
            row, col = divmod(index, 4)
            child[row, col] = spawn_value
            children.append(u64(encode_board(child)))
            if len(children) >= MAX_PREFETCH_CHILDREN:
                return children
    return children


def _parse_prefetch_rng(
    payload: dict[str, Any],
    page: str,
) -> _PrefetchRngContext | None:
    if page not in {"tester", "trainer"}:
        return None
    raw = payload.get("prefetch_rng")
    if not isinstance(raw, dict) or raw.get("version") != _PREFETCH_RNG_VERSION:
        return None
    raw_state = raw.get("state")
    if not isinstance(raw_state, list) or len(raw_state) != 4:
        return None
    state = []
    for value in raw_state:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 0xFFFFFFFF
        ):
            return None
        state.append(value)
    if not any(state):
        return None
    try:
        turn = int(raw.get("turn", 0))
        spawn_rate_4 = float(raw.get("spawn_rate_4", 0.1))
    except (TypeError, ValueError):
        return None
    if not 0 <= turn <= 1_000_000 or not math.isfinite(spawn_rate_4):
        return None
    if not 0.0 <= spawn_rate_4 <= 1.0:
        return None
    return _PrefetchRngContext(
        state=tuple(state),
        turn=turn,
        spawn_rate_4=spawn_rate_4,
    )


def _direction_order(result: TablebaseLookupResult) -> list[str]:
    directions = []
    for direction in (
        result.best_move,
        *result.results.keys(),
        *_DIRECTION_MAP.keys(),
    ):
        normalized = str(direction).lower()
        if normalized in _DIRECTION_MAP and normalized not in directions:
            directions.append(normalized)
    return directions


async def wait_for_tester_query_result(session: GameSession) -> bool:
    board_encoded = np_u64(session.board_encoded)
    if (
        np_u64(getattr(session, "tester_results_board", 0)) == board_encoded
        and bool(session.tester_results)
    ):
        return True

    task = getattr(session, "tester_query_task", None)
    if task is None:
        return False
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        return False
    except Exception:
        return False
    return (
        np_u64(getattr(session, "tester_results_board", 0)) == board_encoded
        and bool(session.tester_results)
    )


async def wait_for_trainer_query_result(session: GameSession) -> bool:
    board_encoded = np_u64(session.board_encoded)
    if (
        np_u64(getattr(session, "trainer_results_board", 0)) == board_encoded
        and bool(session.trainer_results)
    ):
        return True

    task = getattr(session, "trainer_query_task", None)
    if task is None:
        return False
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        return False
    except Exception:
        return False
    return (
        np_u64(getattr(session, "trainer_results_board", 0)) == board_encoded
        and bool(session.trainer_results)
    )


def _deterministic_prefetch_child(
    board_encoded: int,
    *,
    direction: str,
    use_variant: bool,
    rng_context: _PrefetchRngContext,
) -> tuple[int, _PrefetchRngContext] | None:
    direction_code = _DIRECTION_MAP.get(str(direction or "").lower())
    if direction_code is None:
        return None
    move_fn = v_move_board if use_variant else r_move_board
    moved_board, _score = move_fn(np_u64(board_encoded), direction_code)
    moved_board = np_u64(moved_board)
    if moved_board == np_u64(board_encoded):
        return None

    board = decode_board(moved_board).copy()
    empty_positions = [
        index for index, value in enumerate(board.reshape(-1).tolist()) if int(value) == 0
    ]
    if not empty_positions:
        return None
    rng = Xoshiro128StarStar(list(rng_context.state))
    position_roll = rng.next_float()
    spawn_index = empty_positions[
        min(len(empty_positions) - 1, int(position_roll * len(empty_positions)))
    ]
    spawn_value = 4 if rng.next_float() < rng_context.spawn_rate_4 else 2
    row, col = divmod(spawn_index, 4)
    board[row, col] = spawn_value
    return (
        u64(encode_board(board)),
        _PrefetchRngContext(
            state=tuple(rng.state),
            turn=rng_context.turn + 1,
            spawn_rate_4=rng_context.spawn_rate_4,
        ),
    )


def _deterministic_prefetch_nodes(
    board_encoded: int,
    *,
    directions: list[str],
    use_variant: bool,
    rng_context: _PrefetchRngContext,
    depth: int,
    seen_boards: set[int],
) -> list[_PrefetchNode]:
    nodes = []
    for order, direction in enumerate(directions):
        child = _deterministic_prefetch_child(
            board_encoded,
            direction=direction,
            use_variant=use_variant,
            rng_context=rng_context,
        )
        if child is None:
            continue
        child_board, child_rng = child
        if child_board in seen_boards:
            continue
        seen_boards.add(child_board)
        nodes.append(
            _PrefetchNode(
                board_encoded=child_board,
                rng=child_rng,
                depth=depth,
                direction=direction,
                order=order,
            )
        )
    return nodes


async def _send_query_result(
    websocket: WebSocket,
    *,
    page: str,
    query_id: str,
    catalog_version: str,
    result: TablebaseLookupResult,
    token_balance: dict[str, Any] | None,
    extra_data: dict[str, Any] | None = None,
) -> None:
    await websocket.send_json(
        {
            "action": Message.TABLEBASE_QUERY_RESULT,
            "data": {
                "page": page,
                "query_id": query_id,
                "catalog_version": catalog_version,
                "full_pattern": result.full_pattern,
                "board_hex": result.board_hex,
                "results": sanitize_config(result.results),
                "dtype": result.dtype,
                "best_move": result.best_move,
                "found": result.found,
                "token_balance": token_balance,
                **(extra_data or {}),
            },
        }
    )


async def _run_prefetch(
    session: GameSession,
    websocket: WebSocket,
    *,
    page: str,
    query_id: str,
    stream_key: str,
    generation: int,
    catalog_version: str,
    parent_spec: TablebaseLookupSpec,
    parent_result: TablebaseLookupResult,
    supporter: bool,
    prefetch_rng: _PrefetchRngContext | None,
) -> None:
    if prefetch_rng is not None:
        await _run_deterministic_prefetch(
            websocket,
            page=page,
            query_id=query_id,
            stream_key=stream_key,
            generation=generation,
            catalog_version=catalog_version,
            parent_spec=parent_spec,
            parent_result=parent_result,
            supporter=supporter,
            prefetch_rng=prefetch_rng,
        )
        return

    child_boards = _prefetch_boards(
        parent_spec.board_encoded,
        best_move=parent_result.best_move,
        use_variant=parent_spec.use_variant,
    )
    child_boards = child_boards[:_prefetch_limit(parent_spec.full_pattern)]
    if not child_boards:
        return
    handles = []
    for child_board in child_boards:
        child_spec = TablebaseLookupSpec(
            board_encoded=child_board,
            pattern=parent_spec.pattern,
            target=parent_spec.target,
            full_pattern=parent_spec.full_pattern,
            use_variant=parent_spec.use_variant,
            book_reader=parent_spec.book_reader,
            provider_kind=parent_spec.provider_kind,
            catalog_version=catalog_version,
        )
        handle = await tablebase_query_scheduler.submit(
            child_spec,
            stream_key=stream_key,
            supporter=supporter,
            lane="prefetch",
            supersede=False,
            generation=generation,
            allow_overload=True,
        )
        handles.append(handle)

    entries = []
    try:
        for handle in handles:
            result = await handle.wait()
            entries.append(
                {
                    "board_hex": result.board_hex,
                    "results": sanitize_config(result.results),
                    "dtype": result.dtype,
                    "best_move": result.best_move,
                    "found": result.found,
                }
            )
    except TablebaseQuerySuperseded:
        for handle in handles:
            handle.cancel()
        return
    except Exception:
        return
    if tablebase_query_scheduler.current_generation(stream_key) != generation:
        return
    try:
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_PREFETCH,
                "data": {
                    "page": page,
                    "query_id": query_id,
                    "catalog_version": catalog_version,
                    "full_pattern": parent_spec.full_pattern,
                    "parent_board_hex": safe_hex(parent_spec.board_encoded),
                    "entries": entries,
                },
            }
        )
    except Exception:
        return


async def _run_deterministic_prefetch(
    websocket: WebSocket,
    *,
    page: str,
    query_id: str,
    stream_key: str,
    generation: int,
    catalog_version: str,
    parent_spec: TablebaseLookupSpec,
    parent_result: TablebaseLookupResult,
    supporter: bool,
    prefetch_rng: _PrefetchRngContext,
) -> None:
    seen_boards = {u64(parent_spec.board_encoded)}
    budget = _prefetch_limit(parent_spec.full_pattern)
    frontier = _deterministic_prefetch_nodes(
        parent_spec.board_encoded,
        directions=_direction_order(parent_result),
        use_variant=parent_spec.use_variant,
        rng_context=prefetch_rng,
        depth=1,
        seen_boards=seen_boards,
    )
    scheduled_count = 0

    async def wait_for_node(handle, node):
        return node, await handle.wait()

    while frontier and scheduled_count < budget:
        wave = frontier[: budget - scheduled_count]
        scheduled_count += len(wave)
        waiters = []
        handles = []
        for node in wave:
            child_spec = TablebaseLookupSpec(
                board_encoded=node.board_encoded,
                pattern=parent_spec.pattern,
                target=parent_spec.target,
                full_pattern=parent_spec.full_pattern,
                use_variant=parent_spec.use_variant,
                book_reader=parent_spec.book_reader,
                provider_kind=parent_spec.provider_kind,
                catalog_version=catalog_version,
            )
            handle = await tablebase_query_scheduler.submit(
                child_spec,
                stream_key=stream_key,
                supporter=supporter,
                lane="prefetch",
                supersede=False,
                generation=generation,
                allow_overload=True,
            )
            handles.append(handle)
            waiters.append(asyncio.create_task(wait_for_node(handle, node)))

        completed = []
        superseded = False
        for waiter in asyncio.as_completed(waiters):
            try:
                node, result = await waiter
            except TablebaseQuerySuperseded:
                superseded = True
                break
            except Exception:
                continue
            completed.append((node, result))
            if tablebase_query_scheduler.current_generation(stream_key) != generation:
                superseded = True
                break
            try:
                await websocket.send_json(
                    {
                        "action": Message.TABLEBASE_PREFETCH,
                        "data": {
                            "page": page,
                            "query_id": query_id,
                            "catalog_version": catalog_version,
                            "full_pattern": parent_spec.full_pattern,
                            "parent_board_hex": safe_hex(parent_spec.board_encoded),
                            "entries": [
                                {
                                    "board_hex": result.board_hex,
                                    "results": sanitize_config(result.results),
                                    "dtype": result.dtype,
                                    "best_move": result.best_move,
                                    "found": result.found,
                                    "depth": node.depth,
                                    "via_direction": node.direction,
                                }
                            ],
                        },
                    }
                )
            except Exception:
                superseded = True
                break

        if superseded:
            for handle in handles:
                handle.cancel()
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)
            return
        await asyncio.gather(*waiters, return_exceptions=True)

        frontier = []
        for node, result in sorted(completed, key=lambda item: item[0].order):
            if scheduled_count + len(frontier) >= budget:
                break
            if not result.best_move:
                continue
            frontier.extend(
                _deterministic_prefetch_nodes(
                    result.board_encoded,
                    directions=[result.best_move],
                    use_variant=parent_spec.use_variant,
                    rng_context=node.rng,
                    depth=node.depth + 1,
                    seen_boards=seen_boards,
                )
            )


async def _finish_query(
    session: GameSession,
    websocket: WebSocket,
    *,
    page: str,
    query_id: str,
    stream_key: str,
    catalog_version: str,
    spec: TablebaseLookupSpec,
    handle,
    reservation,
    supporter: bool,
    prefetch_rng: _PrefetchRngContext | None = None,
    client_local_board: bool = False,
    allow_prefetch: bool = True,
) -> None:
    guest_reservation = isinstance(reservation, GuestQueryReservation)

    def guest_reservation_is_latest() -> bool:
        current = getattr(session, "guest_query_reservation_ids", {}).get(page)
        return current == reservation.event_id

    def clear_guest_reservation() -> None:
        reservations = getattr(session, "guest_query_reservation_ids", {})
        if reservations.get(page) == reservation.event_id:
            reservations.pop(page, None)

    try:
        result = await handle.wait()
    except TablebaseQuerySuperseded:
        if guest_reservation:
            # A retry with the same request_id shares this reservation. The
            # newest handle owns finalization; only refund when another query
            # has replaced the reservation entirely.
            if not guest_reservation_is_latest():
                finalize_guest_query(reservation, consume=False)
        else:
            cancel_reservation(
                reservation,
                reason="superseded_before_result",
                metadata={"page": page, "query_id": query_id},
            )
        return
    except RemoteTablebaseError as exc:
        guest_allowance = None
        if guest_reservation:
            guest_allowance = finalize_guest_query(reservation, consume=False)
            clear_guest_reservation()
            token_balance = None
        else:
            token_balance = cancel_reservation(
                reservation,
                reason=exc.code.lower(),
                metadata={"page": page, "query_id": query_id},
            )
        if handle.is_current:
            try:
                await websocket.send_json(
                    {
                        "action": Message.TABLEBASE_QUERY_RESULT,
                        "data": {
                            "page": page,
                            "query_id": query_id,
                            "catalog_version": get_catalog_version(),
                            "full_pattern": spec.full_pattern,
                            "board_hex": safe_hex(spec.board_encoded),
                            "results": {},
                            "dtype": "?",
                            "found": False,
                            **exc.payload,
                            "token_balance": token_balance,
                            "guest_allowance": guest_allowance,
                        },
                    }
                )
            except Exception:
                pass
        return
    except Exception as exc:
        guest_allowance = None
        if guest_reservation:
            guest_allowance = finalize_guest_query(reservation, consume=False)
            clear_guest_reservation()
            token_balance = None
        else:
            token_balance = finalize_reservation(
                reservation,
                actual_operation_key=f"{page}_lookup_miss",
                metadata={
                    "board_hex": safe_hex(spec.board_encoded),
                    "query_id": query_id,
                    "error": type(exc).__name__,
                },
            )
        if handle.is_current:
            try:
                await websocket.send_json(
                    {
                        "action": Message.TABLEBASE_QUERY_RESULT,
                        "data": {
                            "page": page,
                            "query_id": query_id,
                            "catalog_version": catalog_version,
                            "full_pattern": spec.full_pattern,
                            "board_hex": safe_hex(spec.board_encoded),
                            "results": {},
                            "dtype": "?",
                            "found": False,
                            "code": "TABLEBASE_QUERY_FAILED",
                            "token_balance": token_balance,
                            "guest_allowance": guest_allowance,
                        },
                    }
                )
            except Exception:
                pass
        return

    if not handle.is_current:
        if guest_reservation:
            if not guest_reservation_is_latest():
                finalize_guest_query(reservation, consume=False)
        return
    selected_pattern = (
        session.current_pattern if page == "trainer" else session.tester_full_pattern
    )
    selection_matches = str(selected_pattern or "") == str(result.full_pattern)
    session_matches = selection_matches and (
        client_local_board
        or _session_matches(
            session,
            page=page,
            board_encoded=result.board_encoded,
            full_pattern=result.full_pattern,
        )
    )
    token_balance = None
    extra_data = None
    if session_matches and not client_local_board:
        _apply_session_result(session, page=page, result=result)
        if page == "tester":
            context = getattr(session, "tester_post_lookup_context", None)
            if (
                isinstance(context, dict)
                and u64(context.get("board_encoded", 0)) == u64(result.board_encoded)
            ):
                logs_since = max(0, int(context.get("logs_since", len(session.tester_logs))))
                _tester_append_post_lookup_logs(session)
                session.tester_post_lookup_context = None
                extra_data = {
                    "logs_delta": session.tester_logs[logs_since:],
                    "logs_total": len(session.tester_logs),
                }
    if session_matches and not guest_reservation and allow_prefetch:
        _track_task(
            asyncio.create_task(
                _run_prefetch(
                    session,
                    websocket,
                    page=page,
                    query_id=query_id,
                    stream_key=stream_key,
                    generation=handle.generation,
                    catalog_version=catalog_version,
                    parent_spec=spec,
                    parent_result=result,
                    supporter=supporter,
                    prefetch_rng=prefetch_rng,
                )
            )
        )
        # Let the best child enter the scheduler before the client can submit it
        # as the next foreground query.
        await asyncio.sleep(0)
    try:
        if guest_reservation:
            guest_allowance = finalize_guest_query(reservation, consume=True)
            clear_guest_reservation()
            token_balance = None
            extra_data = {**(extra_data or {}), "guest_allowance": guest_allowance}
        else:
            token_balance = finalize_reservation(
                reservation,
                actual_operation_key=(
                    f"{page}_lookup_hit"
                    if has_numeric_result(result.results)
                    else f"{page}_lookup_miss"
                ),
                metadata={
                    "board_hex": result.board_hex,
                    "query_id": query_id,
                    "source": "tablebase_query_service",
                },
            )
            if token_balance is None and session.user_id is not None:
                token_balance = get_token_balance(session.user_id)
        await _send_query_result(
            websocket,
            page=page,
            query_id=query_id,
            catalog_version=catalog_version,
            result=result,
            token_balance=token_balance,
            extra_data=extra_data,
        )
    except Exception:
        return


async def handle_tablebase_query_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action != Action.TABLEBASE_QUERY:
        return False
    page = str(payload.get("page") or "").strip().lower()
    is_guest = session.user_id is None and bool(getattr(session, "guest_id", None))
    if is_guest and page != "trainer":
        await websocket.send_json(
            {
                "action": Message.AUTH_REQUIRED,
                "data": {
                    "code": "AUTH_REQUIRED",
                    "message": "Sign in to use this feature.",
                },
            }
        )
        return True
    context = _session_query_context(session, page)
    if context is None:
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_QUERY_RESULT,
                "data": {
                    "page": page,
                    "query_id": str(payload.get("query_id") or ""),
                    "code": "TABLEBASE_NOT_READY",
                    "results": {},
                    "dtype": "?",
                },
            }
        )
        return True

    requested_pattern = str(payload.get("full_pattern") or "").strip()
    client_local_board = bool(payload.get("client_local_board"))
    try:
        requested_board = np_u64(int(str(payload.get("board_hex") or ""), 16))
    except (TypeError, ValueError):
        requested_board = np_u64(0)
    if (
        requested_pattern != context["full_pattern"]
        or (
            not client_local_board
            and requested_board != np_u64(session.board_encoded)
        )
    ):
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_QUERY_RESULT,
                "data": {
                    "page": page,
                    "query_id": str(payload.get("query_id") or ""),
                    "code": "STALE_TABLEBASE_QUERY",
                    "full_pattern": context["full_pattern"],
                    "board_hex": safe_hex(
                        requested_board if client_local_board else session.board_encoded
                    ),
                    "results": {},
                    "dtype": "?",
                },
            }
        )
        return True

    query_id = str(payload.get("query_id") or "")[:160]
    if is_guest and not is_guest_tablebase_available(context["full_pattern"]):
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_QUERY_RESULT,
                "data": {
                    "page": page,
                    "query_id": query_id,
                    "full_pattern": context["full_pattern"],
                    "board_hex": safe_hex(requested_board),
                    "results": {},
                    "dtype": "?",
                    "found": False,
                    "code": "GUEST_TABLE_LOGIN_REQUIRED",
                    "guest_allowance": guest_query_allowance(str(session.guest_id)),
                },
            }
        )
        return True
    catalog_version = get_catalog_version()
    spec = TablebaseLookupSpec(
        board_encoded=u64(requested_board),
        pattern=context["pattern"],
        target=context["target"],
        full_pattern=context["full_pattern"],
        use_variant=context["use_variant"],
        book_reader=context["book_reader"],
        provider_kind=context["provider_kind"],
        catalog_version=catalog_version,
    )
    supporter = _session_is_supporter(session)
    prefetch_rng = _parse_prefetch_rng(payload, page)
    stream_key = f"{session.actor_key or session.user_id}:{session.client_id}:{page}"
    try:
        handle = await tablebase_query_scheduler.submit(
            spec,
            stream_key=stream_key,
            supporter=supporter,
            lane="foreground",
            supersede=True,
            activate=False,
        )
    except TablebaseQueryOverloaded as exc:
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_BUSY,
                "data": {
                    "page": page,
                    "query_id": query_id,
                    "full_pattern": context["full_pattern"],
                    "board_hex": safe_hex(requested_board),
                    **exc.payload,
                },
            }
        )
        return True

    is_default, free_default = (False, False)
    if page == "trainer" and client_local_board:
        is_default, free_default = default_lookup_for(session).consume(
            str(payload.get("default_query_ticket") or ""),
            context["full_pattern"], requested_board,
        )
    try:
        if free_default and not is_guest:
            reservation = None
        elif is_guest:
            reservation = reserve_guest_query(
                guest_id=str(session.guest_id),
                request_id=query_id,
                full_pattern=context["full_pattern"],
                ip_address=str(getattr(session, "guest_ip_address", "")),
            )
            session.guest_query_reservation_ids[page] = reservation.event_id
        else:
            reservation = reserve_operation_tokens(
                user_id=session.user_id,
                session_id=session.auth_session_id,
                operation_key=f"{page}_lookup_hit",
                full_pattern=context["full_pattern"],
            )
    except GuestLimitError as exc:
        handle.cancel()
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_QUERY_RESULT,
                "data": {
                    "page": page,
                    "query_id": query_id,
                    "full_pattern": context["full_pattern"],
                    "board_hex": safe_hex(requested_board),
                    "results": {},
                    "dtype": "?",
                    "found": False,
                    "code": exc.code,
                    "message": str(exc),
                    "guest_allowance": guest_query_allowance(str(session.guest_id)),
                },
            }
        )
        return True
    except InsufficientTokens:
        handle.cancel()
        raise
    await handle.activate()

    if not client_local_board:
        if page == "tester":
            session.tester_lookup_pending = True
            session.tester_query_handle = handle
        else:
            session.trainer_query_handle = handle
    query_task = _track_task(
        asyncio.create_task(
            _finish_query(
                session,
                websocket,
                page=page,
                query_id=query_id,
                stream_key=stream_key,
                catalog_version=catalog_version,
                spec=spec,
                handle=handle,
                reservation=reservation,
                supporter=supporter,
                prefetch_rng=prefetch_rng,
                client_local_board=client_local_board,
                allow_prefetch=not is_default,
            )
        )
    )
    if not client_local_board:
        if page == "tester":
            session.tester_query_task = query_task
        else:
            session.trainer_query_task = query_task
    return True
