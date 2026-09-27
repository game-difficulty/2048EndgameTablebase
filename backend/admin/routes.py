from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.auth.db import auth_db
from backend.auth.dependencies import require_user
from backend.auth.service import iso, normalize_email, set_account_status_for_admin, utcnow
from backend.quota.service import adjust_paid_tokens_for_admin
from backend.remote_workers.registry import remote_worker_registry


router = APIRouter(prefix="/api/admin", tags=["admin"])

DEFAULT_ALLOWED_IDENTITIES = ("user0", "assweeass@163.com")


@router.get('/profile-reviews')
def profile_reviews(
    request: Request,
    page: int = Query(1, ge=1),
    status: str = Query('pending', pattern='^(pending|reviewed|revoked|superseded|all)$'),
    change_type: str = Query('all', pattern='^(avatar|display_name|all)$'),
    q: str = Query('', max_length=120),
):
    _require_admin(request)
    from backend.profile.reviews import list_reviews
    return list_reviews(page, status, change_type, q)


class ProfileReviewAction(BaseModel):
    action: str = Field(pattern='^(keep|revoke)$')


@router.post('/profile-reviews/{event_id}')
def review_profile(event_id: int, payload: ProfileReviewAction, request: Request):
    user = _require_admin(request)
    from backend.live.routes import same_origin
    same_origin(request.headers)
    from backend.profile.reviews import decide_review
    try:
        return decide_review(event_id, payload.action, int(user['id']))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get('/live')
async def admin_live_status(request: Request):
    _require_admin(request)
    from backend.live.routes import hub
    return hub.control_status()


@router.post('/live')
async def admin_live_control(request: Request, payload: dict = Body(...)):
    _require_admin(request)
    from backend.live.routes import hub, same_origin
    same_origin(request.headers)
    if type(payload.get('enabled')) is not bool:
        raise HTTPException(422, 'enabled must be a boolean')
    if hub.producer_ready and not hub.control_supported:
        raise HTTPException(409, 'Please update the livestream runner first.')
    return await hub.set_enabled(payload['enabled'])


@router.post('/live/batch/void')
async def admin_live_void(request: Request, payload: dict = Body(...)):
    _require_admin(request)
    from backend.live.routes import hub, same_origin
    same_origin(request.headers)
    async with hub.activity_lock:
        batch = getattr(hub.content, 'batch', None)
        if not batch or payload.get('batch_id') != batch['id']:
            raise HTTPException(409, 'batch_changed')
        if hub.control['enabled']:
            raise HTTPException(409, 'pause_before_void')
        await hub.content.void_batch()
    return dict(ok=True, batch=hub.content.batch)


@router.get("/tablebase-workers")
async def admin_tablebase_workers(request: Request):
    _require_admin(request)
    return {
        "availability_epoch": remote_worker_registry.availability_epoch,
        "workers": remote_worker_registry.status(),
    }


def _allowed_identities() -> set[str]:
    raw = os.getenv("ADMIN_ALLOWED_IDENTITIES", "")
    values = [
        item.strip().lower()
        for item in raw.split(",")
        if item.strip()
    ]
    return set(values or DEFAULT_ALLOWED_IDENTITIES)


def _require_admin(request: Request) -> dict[str, Any]:
    user = require_user(request)
    allowed = _allowed_identities()
    email = normalize_email(str(user.get("email") or ""))
    display_name = str(user.get("display_name") or "").strip().lower()
    if email in allowed or display_name in allowed:
        return user
    raise HTTPException(status_code=403, detail="Admin access required.")


class VerseDecision(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=1000)


class VerseAction(BaseModel):
    note: str = Field(default="", max_length=1000)


class ArchiveDecision(BaseModel):
    approved: bool
    note: str = Field(min_length=1, max_length=1000)


@router.get("/verse-claims")
def verse_claims(request: Request, user_id: int | None = Query(None, ge=1)):
    _require_admin(request)
    from backend.human_play.verse_history import pending_claims
    return {"claims": pending_claims(user_id)}


@router.post("/verse-claims/{claim_id}/decision")
def verse_claim_decision(claim_id: int, payload: VerseDecision, request: Request):
    from backend.human_play.routes import same_origin
    from backend.human_play import verse_history
    from backend.human_play.service import RunError
    user = _require_admin(request)
    same_origin(request)
    try:
        claim = verse_history.decide_claim(claim_id, user["id"], payload.approved, payload.note)
    except RunError as exc:
        raise HTTPException(exc.status, exc.code) from exc
    if payload.approved:
        verse_history.start_worker()
    return {"claim": claim}


@router.post("/verse-claims/{claim_id}/retry")
def verse_claim_retry(claim_id: int, payload: VerseAction, request: Request):
    from backend.human_play.routes import same_origin
    from backend.human_play import verse_history
    from backend.human_play.service import RunError
    user = _require_admin(request)
    same_origin(request)
    try:
        claim = verse_history.retry_claim(claim_id, user["id"], payload.note)
    except RunError as exc:
        raise HTTPException(exc.status, exc.code) from exc
    verse_history.start_worker()
    return {"claim": claim}


@router.post("/verse-claims/{claim_id}/revoke")
def verse_claim_revoke(claim_id: int, payload: VerseAction, request: Request):
    from backend.human_play.routes import same_origin
    from backend.human_play import verse_history
    from backend.human_play.service import RunError
    user = _require_admin(request)
    same_origin(request)
    try:
        return {"claim": verse_history.revoke_claim(claim_id, user["id"], payload.note)}
    except RunError as exc:
        raise HTTPException(exc.status, exc.code) from exc


@router.get('/archive-applications')
def archive_applications(request: Request, user_id: int | None = Query(None, ge=1)):
    _require_admin(request)
    from backend.human_play.manual_archive import pending
    return {"applications": pending(user_id)}


def _approval_identity_matches(query: str) -> list[int]:
    normalized = str(query or "").strip().lower()
    if not normalized:
        return []
    like = f"%{normalized}%"
    with auth_db() as db:
        return [int(row["id"]) for row in db.execute(
            """SELECT id FROM users
               WHERE lower(email) LIKE ? OR lower(COALESCE(display_name,'')) LIKE ?
               ORDER BY id DESC LIMIT 200""",
            (like, like),
        ).fetchall()]


def _approval_users(user_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not user_ids:
        return {}
    placeholders = ",".join("?" for _ in user_ids)
    with auth_db() as db:
        rows = db.execute(
            f"SELECT id,email,display_name,status FROM users WHERE id IN ({placeholders})",
            tuple(sorted(user_ids)),
        ).fetchall()
    return {int(row["id"]): dict(row) for row in rows}


@router.get('/approval-transactions')
def approval_transactions(
    request: Request,
    q: str = Query('', max_length=120),
    kind: str = Query('all', max_length=20),
    stage: str = Query('all', max_length=20),
    page: int = Query(1, ge=1),
    page_size: int = Query(30, ge=1, le=100),
):
    _require_admin(request)
    from backend.human_play.approval_transactions import list_transactions
    result = list_transactions(
        query=q,
        kind=kind,
        stage=stage,
        page=page,
        page_size=page_size,
        identity_user_ids=_approval_identity_matches(q),
    )
    identities = _approval_users({int(item['user_id']) for item in result['transactions']})
    operators = _approval_users({int(item['operator_id']) for item in result['transactions'] if item['operator_id']})
    for item in result['transactions']:
        item['user'] = identities.get(int(item['user_id']))
        item['operator'] = operators.get(int(item['operator_id'])) if item['operator_id'] else None
    return result


@router.post('/archive-applications/{application_id}/decision')
def archive_application_decision(application_id: int, payload: ArchiveDecision, request: Request):
    from backend.human_play.routes import same_origin
    from backend.human_play import manual_archive
    from backend.human_play.service import RunError
    user = _require_admin(request)
    same_origin(request)
    try:
        return {"application": manual_archive.decide(
            application_id, user['id'], payload.approved, payload.note)}
    except RunError as exc:
        raise HTTPException(exc.status, exc.code) from exc


@router.post('/archive-applications/{application_id}/revoke')
def archive_application_revoke(application_id: int, payload: VerseAction, request: Request):
    from backend.human_play.routes import same_origin
    from backend.human_play import manual_archive
    from backend.human_play.service import RunError
    user = _require_admin(request)
    same_origin(request)
    try:
        return {"application": manual_archive.revoke(
            application_id, user['id'], payload.note)}
    except RunError as exc:
        raise HTTPException(exc.status, exc.code) from exc


def _token_value(units: int | float | None) -> float:
    return round(float(units or 0) / 1000, 3)


def _scalar(db, query: str, params: tuple[Any, ...] = ()) -> int:
    row = db.execute(query, params).fetchone()
    if row is None:
        return 0
    return int(row[0] or 0)


def _daily_token_activity(db, days: int) -> list[dict[str, Any]]:
    today = utcnow().date()
    first_day = today - timedelta(days=days - 1)
    cutoff = f"{first_day.isoformat()}T00:00:00"
    rows_by_day = {
        str(row["day"]): row
        for row in db.execute(
            """
            SELECT
              substr(created_at, 1, 10) AS day,
              SUM(final_cost_units) AS units,
              COUNT(DISTINCT user_id) AS spending_users
            FROM token_ledger
            WHERE created_at >= ?
              AND final_cost_units > 0
              AND event_type IN ('finalize', 'consume')
            GROUP BY day
            """,
            (cutoff,),
        ).fetchall()
    }

    rows = []
    for offset in range(days):
        day = (first_day + timedelta(days=offset)).isoformat()
        row = rows_by_day.get(day)
        rows.append(
            {
                "date": day,
                "tokens_spent": _token_value(row["units"]) if row else 0,
                "spending_users": int(row["spending_users"] or 0) if row else 0,
            }
        )
    return rows


def _user_payload(row, *, pending_approval: bool = False) -> dict[str, Any]:
    bonus_units = int(row["bonus_balance_units"] or 0)
    paid_units = int(row["paid_balance_units"] or 0)
    entitlement_tier = str(row["entitlement_tier"] or "free")
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"] or "",
        "role": row["role"],
        "status": row["status"],
        "pending_approval": bool(pending_approval),
        "registered_with_invite": bool(row["registered_with_invite"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
        "entitlements": {
            "tier": entitlement_tier,
            "is_supporter": entitlement_tier == "supporter",
            "supporter_since": row["supporter_since"],
            "supporter_until": row["supporter_until"],
            "show_supporter_badge": bool(row["show_supporter_badge"]),
            "can_upload_avatar": bool(row["can_upload_avatar"]),
        },
        "sessions": int(row["session_count"] or 0),
        "usage_events": int(row["usage_count"] or 0),
        "token_balance": {
            "bonus": _token_value(bonus_units),
            "paid": _token_value(paid_units),
            "total": _token_value(bonus_units + paid_units),
        },
    }


def _query_users(
    db,
    q: str,
    *,
    page: int,
    page_size: int,
    tier: str = "all",
    pending_approval_user_ids: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized_query = str(q or "").strip().lower()
    normalized_tier = str(tier or "all").strip().lower()
    if normalized_tier not in {"all", "supporter", "free", "pending"}:
        normalized_tier = "all"
    approval_ids = {int(item) for item in (pending_approval_user_ids or set())}
    params: list[Any] = []
    where_parts: list[str] = []
    if normalized_query:
        where_parts.append(
            """
            (
              lower(users.email) LIKE ?
              OR lower(COALESCE(users.display_name, '')) LIKE ?
              OR CAST(users.id AS TEXT) = ?
            )
            """
        )
        like = f"%{normalized_query}%"
        params.extend([like, like, normalized_query])
    if normalized_tier in {"supporter", "free"}:
        where_parts.append("COALESCE(user_entitlements.tier, 'free') = ?")
        params.append(normalized_tier)
    elif normalized_tier == "pending":
        if approval_ids:
            placeholders = ",".join("?" for _item in approval_ids)
            where_parts.append(f"users.id IN ({placeholders})")
            params.extend(sorted(approval_ids))
        else:
            where_parts.append("1 = 0")
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
    total = _scalar(
        db,
        f"""
        SELECT COUNT(*)
        FROM users
        LEFT JOIN user_entitlements ON user_entitlements.user_id = users.id
        {where}
        """,
        tuple(params),
    )
    resolved_page_size = max(1, min(100, int(page_size)))
    page_count = max(1, (total + resolved_page_size - 1) // resolved_page_size)
    resolved_page = max(1, min(int(page), page_count))
    offset = (resolved_page - 1) * resolved_page_size
    query_params = [*params, resolved_page_size, offset]

    rows = db.execute(
        f"""
        SELECT
          users.id,
          users.email,
          users.display_name,
          users.role,
          users.status,
          users.registered_with_invite,
          users.created_at,
          users.last_login_at,
          COALESCE(token_accounts.bonus_balance_units, 0) AS bonus_balance_units,
          COALESCE(token_accounts.paid_balance_units, 0) AS paid_balance_units,
          COALESCE(user_entitlements.tier, 'free') AS entitlement_tier,
          user_entitlements.supporter_since,
          user_entitlements.supporter_until,
          COALESCE(user_entitlements.show_supporter_badge, 1) AS show_supporter_badge,
          COALESCE(user_entitlements.can_upload_avatar, 0) AS can_upload_avatar,
          COUNT(DISTINCT sessions.id) AS session_count,
          COUNT(DISTINCT usage_events.id) AS usage_count
        FROM users
        LEFT JOIN token_accounts ON token_accounts.user_id = users.id
        LEFT JOIN user_entitlements ON user_entitlements.user_id = users.id
        LEFT JOIN sessions ON sessions.user_id = users.id
        LEFT JOIN usage_events ON usage_events.user_id = users.id
        {where}
        GROUP BY users.id
        ORDER BY users.created_at DESC
        LIMIT ? OFFSET ?
        """,
        tuple(query_params),
    ).fetchall()
    return [_user_payload(row, pending_approval=int(row["id"]) in approval_ids) for row in rows], {
        "page": resolved_page,
        "page_size": resolved_page_size,
        "total": total,
        "page_count": page_count,
    }


def _get_user_payload_by_id(db, user_id: int) -> dict[str, Any] | None:
    rows = db.execute(
        """
        SELECT
          users.id,
          users.email,
          users.display_name,
          users.role,
          users.status,
          users.registered_with_invite,
          users.created_at,
          users.last_login_at,
          COALESCE(token_accounts.bonus_balance_units, 0) AS bonus_balance_units,
          COALESCE(token_accounts.paid_balance_units, 0) AS paid_balance_units,
          COALESCE(user_entitlements.tier, 'free') AS entitlement_tier,
          user_entitlements.supporter_since,
          user_entitlements.supporter_until,
          COALESCE(user_entitlements.show_supporter_badge, 1) AS show_supporter_badge,
          COALESCE(user_entitlements.can_upload_avatar, 0) AS can_upload_avatar,
          COUNT(DISTINCT sessions.id) AS session_count,
          COUNT(DISTINCT usage_events.id) AS usage_count
        FROM users
        LEFT JOIN token_accounts ON token_accounts.user_id = users.id
        LEFT JOIN user_entitlements ON user_entitlements.user_id = users.id
        LEFT JOIN sessions ON sessions.user_id = users.id
        LEFT JOIN usage_events ON usage_events.user_id = users.id
        WHERE users.id = ?
        GROUP BY users.id
        LIMIT 1
        """,
        (int(user_id),),
    ).fetchall()
    return _user_payload(rows[0]) if rows else None


@router.get("/overview")
async def admin_overview(
    request: Request,
    q: str = Query("", max_length=120),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tier: str = Query("all", max_length=20),
):
    _require_admin(request)
    now = utcnow()
    cutoff_24h = iso(now - timedelta(days=1))
    cutoff_7d = iso(now - timedelta(days=7))
    active_session_cutoff = iso(now)
    from backend.human_play.verse_history import pending_approval_user_ids
    from backend.human_play.manual_archive import pending_approval_user_ids as archive_pending_user_ids
    approval_user_ids = pending_approval_user_ids() | archive_pending_user_ids()

    with auth_db() as db:
        summary = {
            "users_total": _scalar(db, "SELECT COUNT(*) FROM users"),
            "users_active": _scalar(db, "SELECT COUNT(*) FROM users WHERE status = 'active'"),
            "users_disabled": _scalar(db, "SELECT COUNT(*) FROM users WHERE status != 'active'"),
            "users_verified": _scalar(db, "SELECT COUNT(*) FROM users WHERE email_verified_at IS NOT NULL"),
            "users_invited": _scalar(db, "SELECT COUNT(*) FROM users WHERE registered_with_invite = 1"),
            "users_without_invite": _scalar(db, "SELECT COUNT(*) FROM users WHERE registered_with_invite = 0"),
            "new_users_24h": _scalar(db, "SELECT COUNT(*) FROM users WHERE created_at >= ?", (cutoff_24h,)),
            "sessions_24h": _scalar(db, "SELECT COUNT(*) FROM sessions WHERE created_at >= ?", (cutoff_24h,)),
            "active_sessions": _scalar(
                db,
                "SELECT COUNT(*) FROM sessions WHERE revoked_at IS NULL AND expires_at > ?",
                (active_session_cutoff,),
            ),
            "usage_events_24h": _scalar(db, "SELECT COUNT(*) FROM usage_events WHERE created_at >= ?", (cutoff_24h,)),
            "usage_events_7d": _scalar(db, "SELECT COUNT(*) FROM usage_events WHERE created_at >= ?", (cutoff_7d,)),
            "uploads_24h": _scalar(db, "SELECT COUNT(*) FROM uploads WHERE created_at >= ?", (cutoff_24h,)),
            "analysis_jobs_24h": _scalar(db, "SELECT COUNT(*) FROM analysis_jobs WHERE created_at >= ?", (cutoff_24h,)),
            "tokens_spent_24h": _token_value(
                _scalar(
                    db,
                    """
                    SELECT COALESCE(SUM(final_cost_units), 0)
                    FROM token_ledger
                    WHERE final_cost_units > 0
                      AND event_type IN ('finalize', 'consume')
                      AND created_at >= ?
                    """,
                    (cutoff_24h,),
                )
            ),
        }
        recent_users, _recent_users_page = _query_users(
            db, "", page=1, page_size=8, pending_approval_user_ids=approval_user_ids)
        users, users_page = _query_users(
            db, q, page=page, page_size=page_size, tier=tier,
            pending_approval_user_ids=approval_user_ids)
        token_activity = _daily_token_activity(db, 14)

    return {
        "summary": summary,
        "token_activity": token_activity,
        "recent_users": recent_users,
        "users": users,
        "users_page": users_page,
        "query": q,
        "tier": tier,
    }


@router.post("/users/{user_id}/tokens")
async def admin_adjust_user_tokens(
    user_id: int,
    request: Request,
    payload: dict = Body(...),
):
    admin_user = _require_admin(request)
    try:
        adjustment = adjust_paid_tokens_for_admin(
            target_user_id=int(user_id),
            admin_user=admin_user,
            mode=str(payload.get("mode") or ""),
            tokens=payload.get("tokens"),
            reason=str(payload.get("reason") or ""),
            payment_amount_cny=payload.get("payment_amount_cny"),
            payment_channel=str(payload.get("payment_channel") or ""),
            set_supporter=bool(payload.get("set_supporter", False)),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="User not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with auth_db() as db:
        user = _get_user_payload_by_id(db, int(user_id))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user": user, "adjustment": adjustment}


class UserStatusUpdate(BaseModel):
    status: str = Field(pattern="^(active|disabled)$")


@router.post("/users/{user_id}/status")
def admin_update_user_status(user_id: int, payload: UserStatusUpdate, request: Request):
    admin_user = _require_admin(request)
    if int(admin_user["id"]) == int(user_id):
        raise HTTPException(status_code=400, detail="You cannot change your own account status.")
    try:
        set_account_status_for_admin(user_id=int(user_id), status=payload.status)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="User not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    with auth_db() as db:
        user = _get_user_payload_by_id(db, int(user_id))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return {"user": user}
