from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.auth.dependencies import require_identity
from backend.human_play.routes import same_origin
from .assistance_evidence import record_ai

router = APIRouter(prefix='/api/assistance')


class AiEvidence(BaseModel):
    event_id: str = Field(min_length=16, max_length=160)
    board_codes: list[int] = Field(min_length=16, max_length=16)
    queried_at_ms: int
    direction: Literal['left', 'right', 'up', 'down']


@router.post('/ai-setboard', status_code=204)
def ai_setboard(payload: AiEvidence, request: Request):
    same_origin(request)
    user = require_identity(request)
    try:
        record_ai(user['id'], payload.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
