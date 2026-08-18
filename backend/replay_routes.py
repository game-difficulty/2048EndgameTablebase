from __future__ import annotations

from pathlib import Path
from uuid import UUID

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from backend.auth.dependencies import client_ip, require_user
from backend.auth.service import record_usage
from backend.cloud_files import build_bytes_download_response, get_max_upload_bytes_for_kind
from backend.quota.config import MULTIPLIER_UNIT
from backend.quota.errors import InsufficientTokens
from backend.quota.service import consume_operation_tokens_once, get_token_balance
from backend.tester import get_latest_tester_replay_for_identity
from engine_core.replay_utils import REPLAY_DTYPE, replay_sentinel


router = APIRouter(prefix="/api/replay", tags=["replay"])


class ReplayLoadRequest(BaseModel):
    request_id: str
    filename: str = ""
    size: int = 0


class ReplayLatestRequest(BaseModel):
    request_id: str


def _validated_request_id(value: str) -> str:
    try:
        return str(UUID(str(value or "").strip()))
    except (ValueError, AttributeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid request_id.") from exc


def _consume_replay_load(
    *,
    request_id: str,
    user: dict,
    metadata: dict,
    idempotency_scope: str,
) -> bool:
    try:
        return consume_operation_tokens_once(
            request_id=request_id,
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
            operation_key="replay_load",
            multiplier_override_units=MULTIPLIER_UNIT,
            metadata=metadata,
            idempotency_scope=idempotency_scope,
        )
    except InsufficientTokens as exc:
        raise HTTPException(status_code=402, detail=exc.payload) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/load-authorization")
async def authorize_local_replay_load(
    payload: ReplayLoadRequest,
    request: Request,
    user: dict = Depends(require_user),
):
    request_id = _validated_request_id(payload.request_id)
    filename = Path(str(payload.filename or "")).name
    size = int(payload.size or 0)
    if Path(filename).suffix.lower() != ".rpl":
        raise HTTPException(status_code=400, detail="Only .rpl replay files are supported.")
    if size <= 0 or size > get_max_upload_bytes_for_kind("replay"):
        raise HTTPException(status_code=400, detail="Replay file size is invalid.")

    consumed = _consume_replay_load(
        request_id=request_id,
        user=user,
        metadata={"source": "browser", "filename": filename, "size": size},
        idempotency_scope="replay_load:browser",
    )
    if consumed:
        record_usage(
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
            event_type="replay_load",
            quota_key="replay_load",
            cost=0,
            metadata={"source": "browser", "filename": filename, "size": size},
            ip_address=client_ip(request),
        )
    return {
        "authorized": True,
        "request_id": request_id,
        "token_balance": get_token_balance(int(user["id"])),
    }


@router.post("/latest")
async def load_latest_tester_replay(
    payload: ReplayLatestRequest,
    request: Request,
    user: dict = Depends(require_user),
):
    request_id = _validated_request_id(payload.request_id)
    latest = get_latest_tester_replay_for_identity(
        user_id=int(user["id"]),
        session_id=int(user["session_id"]),
    )
    record = latest.get("record")
    if record is None or len(record) == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": "NO_LATEST_REPLAY", "message": "No tester replay available yet."},
        )

    serialized = np.empty(len(record) + 1, dtype=REPLAY_DTYPE)
    serialized[:-1] = record
    serialized[-1] = replay_sentinel(latest.get("terminal_board") or 0)
    content = serialized.tobytes()
    if len(content) > get_max_upload_bytes_for_kind("replay"):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "REPLAY_TOO_LARGE",
                "message": "Latest tester replay exceeds the size limit.",
            },
        )
    pattern = str(latest.get("pattern") or "")
    filename = f"{pattern or 'tester'}_latest.rpl"

    consumed = _consume_replay_load(
        request_id=request_id,
        user=user,
        metadata={"source": "tester_latest", "pattern": pattern, "size": len(content)},
        idempotency_scope="replay_load:latest",
    )
    if consumed:
        record_usage(
            user_id=int(user["id"]),
            session_id=int(user["session_id"]),
            event_type="replay_load",
            quota_key="replay_load",
            cost=0,
            metadata={"source": "tester_latest", "pattern": pattern, "size": len(content)},
            ip_address=client_ip(request),
        )

    balance = get_token_balance(int(user["id"]))
    response = build_bytes_download_response(
        content,
        filename=filename,
        media_type="application/octet-stream",
    )
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Replay-Pattern"] = pattern
    response.headers["X-Replay-Variant"] = "1" if latest.get("use_variant") else "0"
    response.headers["X-Replay-Source"] = "Tester session"
    response.headers["X-Token-Bonus"] = str(balance["bonus"])
    response.headers["X-Token-Paid"] = str(balance["paid"])
    response.headers["X-Token-Total"] = str(balance["total"])
    return response
