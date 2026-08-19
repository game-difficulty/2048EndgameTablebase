from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from backend.auth.db import auth_db
from backend.auth.dependencies import require_user
from backend.auth.service import iso, normalize_email, utcnow
from backend.quota.service import adjust_paid_tokens_for_admin
from backend.remote_workers.registry import remote_worker_registry


router = APIRouter(prefix="/api/admin", tags=["admin"])

DEFAULT_ALLOWED_IDENTITIES = ("user0", "assweeass@163.com")


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
            WHERE created_at >= ? AND final_cost_units > 0
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


def _user_payload(row) -> dict[str, Any]:
    bonus_units = int(row["bonus_balance_units"] or 0)
    paid_units = int(row["paid_balance_units"] or 0)
    entitlement_tier = str(row["entitlement_tier"] or "free")
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"] or "",
        "role": row["role"],
        "status": row["status"],
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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    normalized_query = str(q or "").strip().lower()
    normalized_tier = str(tier or "all").strip().lower()
    if normalized_tier not in {"all", "supporter", "free"}:
        normalized_tier = "all"
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
    if normalized_tier != "all":
        where_parts.append("COALESCE(user_entitlements.tier, 'free') = ?")
        params.append(normalized_tier)
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
    return [_user_payload(row) for row in rows], {
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
                    "SELECT COALESCE(SUM(final_cost_units), 0) FROM token_ledger WHERE final_cost_units > 0 AND created_at >= ?",
                    (cutoff_24h,),
                )
            ),
        }
        recent_users, _recent_users_page = _query_users(db, "", page=1, page_size=8)
        users, users_page = _query_users(db, q, page=page, page_size=page_size, tier=tier)
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
