"""Gamer stream request contract and idempotent query budget checks."""
from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, Field, StrictInt

from backend.auth.db import auth_db
from backend.quota.config import (apply_pricing_multipliers, operation_cost_units,
                                  resolve_pricing_snapshot, table_multiplier_units, token_to_units)
from backend.quota.errors import InsufficientTokens
from backend.quota.service import get_token_balance


class RouteRequest(BaseModel):
    request_id: str = Field(min_length=16, max_length=80, pattern=r"^[a-zA-Z0-9-]+$")
    catalog_version: str = Field(max_length=80)
    full_pattern: str = Field(min_length=1, max_length=80)
    board_codes: list[StrictInt] = Field(min_length=16, max_length=16)
    rng_state: list[StrictInt] = Field(min_length=4, max_length=4)
    spawn_rate4: float = Field(ge=0, le=1)
    difficulty: int = Field(ge=0, le=100)
    random_only: bool = False
    steps: int = Field(default=1, ge=1, le=1)
    advance_first: bool = False


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
        raise InsufficientTokens(required_units=required, balance_units=balance)
