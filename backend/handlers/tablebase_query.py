from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from fastapi import WebSocket
from engine_core.BoardMover import s_move_board as r_move_board
from engine_core.VBoardMover import decode_board, encode_board, s_move_board as v_move_board

from ..actions import Action, Message
from ..quota.errors import InsufficientTokens
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
from ..tablebase_catalog import get_catalog_version
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
) -> None:
    child_boards = _prefetch_boards(
        parent_spec.board_encoded,
        best_move=parent_result.best_move,
        use_variant=parent_spec.use_variant,
    )
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
) -> None:
    try:
        result = await handle.wait()
    except TablebaseQuerySuperseded:
        cancel_reservation(
            reservation,
            reason="superseded_before_result",
            metadata={"page": page, "query_id": query_id},
        )
        return
    except RemoteTablebaseError as exc:
        cancel_reservation(
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
                            "token_balance": (
                                get_token_balance(session.user_id)
                                if session.user_id is not None
                                else None
                            ),
                        },
                    }
                )
            except Exception:
                pass
        return
    except Exception as exc:
        finalize_reservation(
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
                            "token_balance": (
                                get_token_balance(session.user_id)
                                if session.user_id is not None
                                else None
                            ),
                        },
                    }
                )
            except Exception:
                pass
        return

    finalize_reservation(
        reservation,
        actual_operation_key=(
            f"{page}_lookup_hit" if has_numeric_result(result.results) else f"{page}_lookup_miss"
        ),
        metadata={
            "board_hex": result.board_hex,
            "query_id": query_id,
            "source": "tablebase_query_service",
        },
    )
    if not handle.is_current:
        return
    extra_data = None
    if _session_matches(
        session,
        page=page,
        board_encoded=result.board_encoded,
        full_pattern=result.full_pattern,
    ):
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
    try:
        await _send_query_result(
            websocket,
            page=page,
            query_id=query_id,
            catalog_version=catalog_version,
            result=result,
            token_balance=(
                get_token_balance(session.user_id) if session.user_id is not None else None
            ),
            extra_data=extra_data,
        )
    except Exception:
        return

    if not _session_matches(
        session,
        page=page,
        board_encoded=result.board_encoded,
        full_pattern=result.full_pattern,
    ):
        return
    await _run_prefetch(
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
    )


async def handle_tablebase_query_action(
    action: str,
    payload: dict[str, Any],
    session: GameSession,
    websocket: WebSocket,
) -> bool:
    if action != Action.TABLEBASE_QUERY:
        return False
    page = str(payload.get("page") or "").strip().lower()
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
    try:
        requested_board = np_u64(int(str(payload.get("board_hex") or ""), 16))
    except (TypeError, ValueError):
        requested_board = np_u64(0)
    if (
        requested_pattern != context["full_pattern"]
        or requested_board != np_u64(session.board_encoded)
    ):
        await websocket.send_json(
            {
                "action": Message.TABLEBASE_QUERY_RESULT,
                "data": {
                    "page": page,
                    "query_id": str(payload.get("query_id") or ""),
                    "code": "STALE_TABLEBASE_QUERY",
                    "full_pattern": context["full_pattern"],
                    "board_hex": safe_hex(session.board_encoded),
                    "results": {},
                    "dtype": "?",
                },
            }
        )
        return True

    query_id = str(payload.get("query_id") or "")[:160]
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
    stream_key = f"{session.user_id}:{session.client_id}:{page}"
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

    try:
        reservation = reserve_operation_tokens(
            user_id=session.user_id,
            session_id=session.auth_session_id,
            operation_key=f"{page}_lookup_hit",
            full_pattern=context["full_pattern"],
        )
    except InsufficientTokens:
        handle.cancel()
        raise
    await handle.activate()

    if page == "tester":
        session.tester_lookup_pending = True
        session.tester_query_handle = handle
    else:
        session.trainer_query_handle = handle
    _track_task(
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
            )
        )
    )
    return True
