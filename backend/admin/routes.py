from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from backend.auth.db import auth_db
from backend.auth.dependencies import require_user
from backend.auth.service import iso, normalize_email, utcnow


router = APIRouter(prefix="/api/admin", tags=["admin"])

DEFAULT_ALLOWED_IDENTITIES = ("user0", "assweeass@163.com")


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


def _daily_counts(db, table: str, cutoff: str) -> dict[str, int]:
    rows = db.execute(
        f"""
        SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count
        FROM {table}
        WHERE created_at >= ?
        GROUP BY day
        """,
        (cutoff,),
    ).fetchall()
    return {str(row["day"]): int(row["count"] or 0) for row in rows}


def _daily_token_costs(db, cutoff: str) -> dict[str, float]:
    rows = db.execute(
        """
        SELECT substr(created_at, 1, 10) AS day, SUM(final_cost_units) AS units
        FROM token_ledger
        WHERE created_at >= ? AND final_cost_units > 0
        GROUP BY day
        """,
        (cutoff,),
    ).fetchall()
    return {str(row["day"]): _token_value(row["units"]) for row in rows}


def _daily_traffic(db, days: int) -> list[dict[str, Any]]:
    today = utcnow().date()
    first_day = today - timedelta(days=days - 1)
    cutoff = f"{first_day.isoformat()}T00:00:00"
    sessions = _daily_counts(db, "sessions", cutoff)
    usage = _daily_counts(db, "usage_events", cutoff)
    token_events = _daily_counts(db, "token_ledger", cutoff)
    uploads = _daily_counts(db, "uploads", cutoff)
    analysis_jobs = _daily_counts(db, "analysis_jobs", cutoff)
    token_costs = _daily_token_costs(db, cutoff)

    rows = []
    for offset in range(days):
        day = (first_day + timedelta(days=offset)).isoformat()
        rows.append(
            {
                "date": day,
                "sessions": sessions.get(day, 0),
                "usage_events": usage.get(day, 0),
                "token_events": token_events.get(day, 0),
                "tokens_spent": token_costs.get(day, 0),
                "uploads": uploads.get(day, 0),
                "analysis_jobs": analysis_jobs.get(day, 0),
            }
        )
    return rows


def _user_payload(row) -> dict[str, Any]:
    bonus_units = int(row["bonus_balance_units"] or 0)
    paid_units = int(row["paid_balance_units"] or 0)
    return {
        "id": int(row["id"]),
        "email": row["email"],
        "display_name": row["display_name"] or "",
        "role": row["role"],
        "status": row["status"],
        "registered_with_invite": bool(row["registered_with_invite"]),
        "created_at": row["created_at"],
        "last_login_at": row["last_login_at"],
        "sessions": int(row["session_count"] or 0),
        "usage_events": int(row["usage_count"] or 0),
        "token_balance": {
            "bonus": _token_value(bonus_units),
            "paid": _token_value(paid_units),
            "total": _token_value(bonus_units + paid_units),
        },
    }


def _query_users(db, q: str, limit: int) -> list[dict[str, Any]]:
    normalized_query = str(q or "").strip().lower()
    params: list[Any] = []
    where = ""
    if normalized_query:
        where = """
        WHERE lower(users.email) LIKE ?
           OR lower(COALESCE(users.display_name, '')) LIKE ?
           OR CAST(users.id AS TEXT) = ?
        """
        like = f"%{normalized_query}%"
        params.extend([like, like, normalized_query])
    params.append(max(1, min(100, int(limit))))

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
          COUNT(DISTINCT sessions.id) AS session_count,
          COUNT(DISTINCT usage_events.id) AS usage_count
        FROM users
        LEFT JOIN token_accounts ON token_accounts.user_id = users.id
        LEFT JOIN sessions ON sessions.user_id = users.id
        LEFT JOIN usage_events ON usage_events.user_id = users.id
        {where}
        GROUP BY users.id
        ORDER BY users.created_at DESC
        LIMIT ?
        """,
        tuple(params),
    ).fetchall()
    return [_user_payload(row) for row in rows]


@router.get("/overview")
async def admin_overview(
    request: Request,
    q: str = Query("", max_length=120),
    limit: int = Query(20, ge=1, le=100),
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
        recent_users = _query_users(db, "", 8)
        users = _query_users(db, q, limit)
        traffic = _daily_traffic(db, 14)
        operation_rows = db.execute(
            """
            SELECT event_type, COUNT(*) AS count
            FROM usage_events
            WHERE created_at >= ?
            GROUP BY event_type
            ORDER BY count DESC, event_type ASC
            LIMIT 12
            """,
            (cutoff_7d,),
        ).fetchall()
        operations = [
            {"event_type": row["event_type"] or "unknown", "count": int(row["count"] or 0)}
            for row in operation_rows
        ]

    return {
        "summary": summary,
        "traffic": traffic,
        "operations": operations,
        "recent_users": recent_users,
        "users": users,
        "query": q,
    }
