from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, StrictInt

from backend.auth.dependencies import require_user
from backend.auth.db import auth_db
from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.gamer_ranked.rules import simulate_move, random_spawn
from backend.quota.errors import InsufficientTokens
from backend.quota.config import (apply_pricing_multipliers, operation_cost_units,
                                  resolve_pricing_snapshot, table_multiplier_units, token_to_units)
from backend.quota.service import consume_operation_tokens_once, get_token_balance, has_numeric_result
from backend.tablebase_catalog import ai_table_metadata, get_catalog_version, resolve_tablebase
from backend.tablebase_query_service import TablebaseLookupSpec, tablebase_query_scheduler
from engine_core.BookReader import BookReaderDispatcher
from engine_core.VBoardMover import encode_board

router = APIRouter(prefix="/api/gamer/tablebase", tags=["gamer"])
_active_users = Counter()


class RouteRequest(BaseModel):
    request_id: str = Field(min_length=16, max_length=80, pattern=r"^[a-zA-Z0-9-]+$")
    catalog_version: str = Field(max_length=80)
    full_pattern: str = Field(min_length=1, max_length=80)
    board_codes: list[StrictInt] = Field(min_length=16, max_length=16)
    rng_state: list[StrictInt] = Field(min_length=4, max_length=4)
    spawn_rate4: float = Field(ge=0, le=1)
    difficulty: int = Field(ge=0, le=100)
    random_only: bool = False
    steps: int = Field(default=4, ge=1, le=4)
    advance_first: bool = False


def masked_board(values: list[int], count: int) -> int:
    board = np.asarray(values, dtype=np.int64).copy()
    board[np.argpartition(board, -count)[-count:]] = 32768
    if int(board.sum()) - count * 32768 < 24:
        board = np.array([32768,32768,32768,32768,0,32768,32768,0,
                          0,32768,32768,0,32768,32768,32768,32768])
    return int(encode_board(board.reshape(4, 4)))


def predict_next(values, direction, rng, request, *, random_only=None):
    moved, _ = simulate_move(values, direction)
    if moved == values or 0 not in moved:
        return None
    if not (request.random_only if random_only is None else random_only):
        branch = rng.next_float()
        if request.difficulty >= 100 or (request.difficulty > 0 and branch < request.difficulty / 100):
            return None
    index, exp = random_spawn(moved, rng, request.spawn_rate4)
    moved[index] = 2 ** exp
    return moved


def check_query_budget(user_id, request, fingerprint, index):
    with auth_db() as db:
        paid = db.execute('SELECT user_id,operation_key FROM token_operation_requests WHERE request_id=?',
            (f'gamer-ai:{request.request_id}:{index}',)).fetchone()
    if paid is not None:
        if paid['user_id'] != user_id or paid['operation_key'] != f'gamer-ai:{fingerprint}:{index}':
            raise HTTPException(409, 'REQUEST_ID_CONFLICT')
        return
    required = apply_pricing_multipliers(operation_cost_units('trainer_lookup_hit'),
        table_multiplier_units(request.full_pattern), resolve_pricing_snapshot().global_multiplier_units)
    balance = token_to_units(get_token_balance(user_id)['total'])
    if balance < required:
        raise InsufficientTokens(required_units=required,balance_units=balance)


@router.post("/route")
async def route(request: RouteRequest, user: dict = Depends(require_user)):
    if any(not 0 <= value <= 31 for value in request.board_codes):
        raise HTTPException(400, "Invalid board codes")
    if any(not 0 <= value <= 0xFFFFFFFF for value in request.rng_state) or not any(request.rng_state):
        raise HTTPException(400, "Invalid RNG state")
    if request.catalog_version != get_catalog_version():
        raise HTTPException(409, "CATALOG_CHANGED")
    descriptor = resolve_tablebase(request.full_pattern)
    if (descriptor is None or not ai_table_metadata(descriptor)["compatible"]
            or abs(float(descriptor.get("spawn_rate", .1)) - request.spawn_rate4) >= .01):
        raise HTTPException(409, "TABLEBASE_UNAVAILABLE")
    user_id = int(user["id"])
    if _active_users[user_id] >= 2:
        raise HTTPException(429, "TABLEBASE_BUSY")
    # Bind retries to the complete request, not just a client-selected ID.
    fingerprint = hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True).encode()).hexdigest()
    try:
        check_query_budget(user_id, request, fingerprint, 0)
    except InsufficientTokens as exc:
        raise HTTPException(402, exc.payload) from exc
    async def generate():
        if _active_users[user_id] >= 2:
            yield json.dumps({"type":"error","status":429,"detail":"TABLEBASE_BUSY"}) + "\n"
            return
        _active_users[user_id] += 1
        reader = None
        try:
            if descriptor["_provider"] == "local":
                reader = BookReaderDispatcher()
                await asyncio.to_thread(reader.dispatch,
                    [(descriptor["_absolute_path"], descriptor.get("dtype", "uint32"))],
                    descriptor["pattern"], str(descriptor["target"]))
            values = [0 if code == 0 else 2 ** code for code in request.board_codes]
            rng = Xoshiro128StarStar(request.rng_state.copy())
            for index in range(-int(request.advance_first), request.steps):
                if index >= 0:
                    check_query_budget(user_id, request, fingerprint, index)
                lookup_board = masked_board(values, ai_table_metadata(descriptor)["large_tiles"])
                spec = TablebaseLookupSpec(lookup_board, descriptor["pattern"], str(descriptor["target"]),
                    request.full_pattern, False, reader, descriptor["_provider"], request.catalog_version)
                handle = await tablebase_query_scheduler.submit(spec,
                    stream_key=f"gamer-ai:{user_id}:{request.request_id}",
                    supporter=user.get("entitlements", {}).get("tier") == "supporter",
                    lane="foreground" if index == 0 else "prefetch", supersede=False)
                try:
                    result = await handle.wait()
                finally:
                    handle.cancel()
                if index < 0:
                    values = predict_next(values, result.best_move, rng, request) if result.best_move else None
                    if values is None:
                        break
                    continue
                consume_operation_tokens_once(request_id=f"gamer-ai:{request.request_id}:{index}",
                    user_id=user_id, session_id=user.get("session_id"),
                    operation_key="trainer_lookup_hit" if has_numeric_result(result.results) else "trainer_lookup_miss",
                    full_pattern=request.full_pattern, idempotency_scope=f"gamer-ai:{fingerprint}:{index}",
                    metadata={"source": "gamer_ai", "board_hex": result.board_hex, "route_index": index})
                yield json.dumps({"type": "result", "board_codes": [0 if v == 0 else v.bit_length() - 1 for v in values],
                    "rng_state": rng.state.copy(), "lookup_board": result.board_hex,
                    "full_pattern": request.full_pattern, "catalog_version": request.catalog_version,
                    "random_only": request.random_only and index == 0 and not request.advance_first,
                    "results": result.results, "dtype": result.dtype,
                    "token_balance": get_token_balance(user_id)}, separators=(",", ":")) + "\n"
                if not result.best_move:
                    break
                best_success = result.results[result.best_move] + (1 if result.dtype.startswith('1-') else 0)
                if best_success <= 0:
                    break
                values = predict_next(values, result.best_move, rng, request,
                    random_only=request.random_only and index == 0 and not request.advance_first)
                if values is None:
                    break
        except asyncio.CancelledError:
            raise
        except InsufficientTokens as exc:
            yield json.dumps({"type": "error", "status": 402, "detail": exc.payload}) + "\n"
        except HTTPException as exc:
            yield json.dumps({"type": "error", "status": exc.status_code, "detail": exc.detail}) + "\n"
        except Exception:
            yield json.dumps({"type": "error", "status": 503, "detail": {"code": "TABLEBASE_UNAVAILABLE"}}) + "\n"
        finally:
            _active_users[user_id] -= 1
            if not _active_users[user_id]:
                del _active_users[user_id]

    return StreamingResponse(generate(), media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
