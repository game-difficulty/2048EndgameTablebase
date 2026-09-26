"""Play-side ownership checks and leases for optional player broadcasting."""
from __future__ import annotations

import hmac
import json
import os

from fastapi import HTTPException

from backend.auth.db import auth_db
from backend.live import human_rooms

from .store import database
from .service import owned


def _owned_active(user_id: int, run_id: str, *, browser: str, writer: str, epoch: int):
    with database() as db:
        run = owned(db, run_id, int(user_id), browser)
    if (run["status"] != "active" or run["eligibility"] != "eligible"
            or run["writer"] != writer or int(run["epoch"]) != int(epoch)
            or run["source"] != "native"):
        raise HTTPException(409, "live_run_not_current")
    return run


def _result(room_data: dict, run, *, browser: str, writer: str) -> dict:
    lease, expires = human_rooms.issue_lease(
        room_data, seed=run["seed"], writer=writer, epoch=run["epoch"], browser=browser)
    return {**human_rooms.public_payload(room_data), "lease": lease, "lease_expires_at": expires}


def start(user: dict, run_id: str, *, browser: str, writer: str, epoch: int) -> dict:
    run = _owned_active(user["id"], run_id, browser=browser, writer=writer, epoch=epoch)
    # Human Play authenticates this endpoint with the deliberately small
    # identity lookup, so do one bounded profile read when a room is opened.
    # This keeps ordinary heartbeats free of profile/billing queries while the
    # public room still uses the player's real current name and avatar.
    with auth_db() as db:
        owner = db.execute("""SELECT users.display_name,user_profiles.avatar_key
            FROM users LEFT JOIN user_profiles ON user_profiles.user_id=users.id
            WHERE users.id=? AND users.status='active'""", (int(user["id"]),)).fetchone()
    if not owner:
        raise HTTPException(401, "live_owner_unavailable")
    avatar_url = f"/media/avatars/{owner['avatar_key']}" if owner["avatar_key"] else None
    room_data = human_rooms.start(
        owner_user_id=user["id"], run_id=run_id, variant=run["variant"],
        display_name=str(owner["display_name"] or "Player")[:80],
        avatar_url=avatar_url,
    )
    return _result(room_data, run, browser=browser, writer=writer)


def renew(user: dict, room_id: str, *, browser: str, writer: str, epoch: int) -> dict:
    room_data = human_rooms.current(user["id"])
    if not room_data or room_data["room_id"] != room_id:
        raise HTTPException(404, "live_room_not_found")
    run = _owned_active(user["id"], room_data["run_id"], browser=browser,
                        writer=writer, epoch=epoch)
    return _result(room_data, run, browser=browser, writer=writer)


def current(user_id: int) -> dict:
    room_data = human_rooms.current(user_id)
    return {"room": human_rooms.public_payload(room_data) if room_data else None}


def stop(user_id: int, room_id: str) -> dict:
    if not human_rooms.stop(user_id, room_id):
        raise HTTPException(404, "live_room_not_found")
    return {"ok": True}


def verified_milestone(run_id: str, milestone: int, supplied_secret: str) -> dict:
    secret = os.environ.get("HUMAN_LIVE_SIGNING_KEY", "")
    if len(secret) < 32 or not hmac.compare_digest(supplied_secret, secret):
        raise HTTPException(403, "invalid_internal_credential")
    if milestone not in (32768, 65536):
        raise HTTPException(400, "invalid_milestone")
    with database() as db:
        row = db.execute("SELECT state,status,eligibility FROM human_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise HTTPException(404, "run_not_found")
    state = json.loads(row["state"])
    return {
        "verified": row["eligibility"] == "eligible" and str(milestone) in state.get("nodes", {}),
        "seq": int(state.get("seq") or 0), "status": row["status"],
        "eligibility": row["eligibility"],
    }
