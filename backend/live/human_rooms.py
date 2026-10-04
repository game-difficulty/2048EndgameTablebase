"""Persistent directory and short-lived publisher leases for player streams."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time

from fastapi import HTTPException

from backend.auth.db import auth_db


ACTIVE = "active"
LEASE_SECONDS = 75
ROOM_PREFIX = "h-"


def init_schema() -> None:
    with auth_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS live_human_rooms (
            room_id TEXT PRIMARY KEY,
            owner_user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            run_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            generation INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL,
            display_name TEXT NOT NULL,
            avatar_url TEXT,
            started_at REAL NOT NULL,
            last_publisher_at REAL,
            stopped_at REAL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS live_human_one_active_owner
          ON live_human_rooms(owner_user_id) WHERE status='active';
        CREATE INDEX IF NOT EXISTS live_human_active_time
          ON live_human_rooms(status,last_publisher_at,started_at);
        """)


def _row(room_id: str):
    init_schema()
    with auth_db() as db:
        return db.execute("SELECT * FROM live_human_rooms WHERE room_id=?", (room_id,)).fetchone()


def room(room_id: str, *, active_only: bool = True) -> dict | None:
    row = _row(room_id)
    if not row or (active_only and row["status"] != ACTIVE):
        return None
    return dict(row)


def current(owner_user_id: int) -> dict | None:
    init_schema()
    with auth_db() as db:
        row = db.execute("""SELECT * FROM live_human_rooms
            WHERE owner_user_id=? AND status='active'""", (int(owner_user_id),)).fetchone()
    return dict(row) if row else None


def active_rooms() -> list[dict]:
    init_schema()
    with auth_db() as db:
        rows = db.execute("""SELECT * FROM live_human_rooms
            WHERE status='active' ORDER BY started_at,room_id""").fetchall()
    return [dict(row) for row in rows]


def expire_stale(*, stale_after: float = 90, now: float | None = None) -> list[str]:
    """End rooms which did not establish or refresh a publisher in the recovery window."""
    init_schema()
    now = time.time() if now is None else float(now)
    cutoff = now - max(30, float(stale_after))
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute("""SELECT room_id FROM live_human_rooms
            WHERE status='active' AND COALESCE(last_publisher_at,started_at)<?""",
            (cutoff,)).fetchall()
        room_ids = [str(row["room_id"]) for row in rows]
        if room_ids:
            placeholders = ",".join("?" for _ in room_ids)
            db.execute(f"""UPDATE live_human_rooms
                SET status='ended',generation=generation+1,stopped_at=?
                WHERE status='active' AND room_id IN ({placeholders})""",
                (now, *room_ids))
    return room_ids


def start(*, owner_user_id: int, run_id: str, variant: str,
          display_name: str, avatar_url: str | None = None, now: float | None = None) -> dict:
    """Create a session or explicitly switch the owner's active session to a run."""
    init_schema()
    now = time.time() if now is None else float(now)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("""SELECT * FROM live_human_rooms
            WHERE owner_user_id=? AND status='active'""", (int(owner_user_id),)).fetchone()
        if existing:
            if existing["run_id"] != run_id or existing["variant"] != variant:
                db.execute("""UPDATE live_human_rooms SET run_id=?,variant=?,generation=generation+1,
                    display_name=?,avatar_url=?,last_publisher_at=? WHERE room_id=?""",
                    (run_id, variant, display_name, avatar_url, now, existing["room_id"]))
            elif existing["display_name"] != display_name or existing["avatar_url"] != avatar_url:
                db.execute("UPDATE live_human_rooms SET display_name=?,avatar_url=? WHERE room_id=?",
                           (display_name, avatar_url, existing["room_id"]))
            row = db.execute("SELECT * FROM live_human_rooms WHERE room_id=?",
                             (existing["room_id"],)).fetchone()
            return dict(row)
        maximum = max(1, min(int(os.environ.get("LIVE_MAX_HUMAN_ROOMS", "32")), 256))
        active = int(db.execute("SELECT count(*) FROM live_human_rooms WHERE status='active'").fetchone()[0])
        if active >= maximum:
            raise HTTPException(503, "live_room_capacity")
        for _ in range(5):
            room_id = ROOM_PREFIX + secrets.token_hex(12)
            try:
                db.execute("""INSERT INTO live_human_rooms
                    (room_id,owner_user_id,run_id,variant,generation,status,display_name,
                     avatar_url,started_at) VALUES(?,?,?,?,1,'active',?,?,?)""",
                    (room_id, int(owner_user_id), run_id, variant, display_name, avatar_url, now))
                return dict(db.execute("SELECT * FROM live_human_rooms WHERE room_id=?", (room_id,)).fetchone())
            except Exception as exc:
                if "UNIQUE constraint failed: live_human_rooms.room_id" not in str(exc):
                    raise
        raise HTTPException(503, "live_room_id_unavailable")


def stop(owner_user_id: int, room_id: str, now: float | None = None) -> bool:
    init_schema()
    now = time.time() if now is None else float(now)
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        changed = db.execute("""UPDATE live_human_rooms
            SET status='ended',generation=generation+1,stopped_at=?
            WHERE room_id=? AND owner_user_id=? AND status='active'""",
            (now, room_id, int(owner_user_id))).rowcount
    return bool(changed)


def publisher_seen(room_id: str, generation: int, now: float | None = None) -> bool:
    init_schema()
    with auth_db() as db:
        return bool(db.execute("""UPDATE live_human_rooms SET last_publisher_at=?
            WHERE room_id=? AND generation=? AND status='active'""",
            (time.time() if now is None else float(now), room_id, int(generation))).rowcount)


def _secret() -> bytes:
    value = os.environ.get("HUMAN_LIVE_SIGNING_KEY", "")
    if len(value) < 32:
        raise HTTPException(503, "human_live_not_configured")
    return value.encode()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_lease(room_data: dict, *, seed: str, writer: str, epoch: int,
                browser: str, now: float | None = None) -> tuple[str, float]:
    now = time.time() if now is None else float(now)
    expires = now + LEASE_SECONDS
    payload = {
        "v": 1, "room": room_data["room_id"], "owner": int(room_data["owner_user_id"]),
        "run": room_data["run_id"], "variant": room_data["variant"],
        "generation": int(room_data["generation"]), "seed": seed,
        "writer": writer, "epoch": int(epoch), "browser": browser,
        "iat": int(now), "exp": int(expires),
    }
    encoded = _b64(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode())
    signature = _b64(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
    return encoded + "." + signature, expires


def verify_lease(token: str, *, room_id: str, owner_user_id: int,
                 now: float | None = None) -> dict:
    try:
        encoded, signature = token.split(".", 1)
        expected = _b64(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError()
        payload = json.loads(_unb64(encoded))
        now = time.time() if now is None else float(now)
        if (payload.get("v") != 1 or payload.get("room") != room_id
                or int(payload.get("owner", -1)) != int(owner_user_id)
                or float(payload.get("exp", 0)) < now):
            raise ValueError()
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        raise HTTPException(403, "invalid_live_lease")
    current_room = room(room_id)
    if (not current_room or int(current_room["owner_user_id"]) != int(owner_user_id)
            or int(current_room["generation"]) != int(payload["generation"])
            or current_room["run_id"] != payload["run"]
            or current_room["variant"] != payload["variant"]):
        raise HTTPException(403, "live_lease_revoked")
    return payload


def public_origin() -> str:
    return os.environ.get("HUMAN_LIVE_PUBLIC_ORIGIN", "https://live.2048tables.online").rstrip("/")


def public_payload(room_data: dict) -> dict:
    origin = public_origin()
    path = "/rooms/" + room_data["room_id"]
    return {
        "room_id": room_data["room_id"], "run_id": room_data["run_id"],
        "variant": room_data["variant"], "generation": int(room_data["generation"]),
        "status": room_data["status"], "started_at": float(room_data["started_at"]),
        "url": origin + path, "path": path,
        "resume_supported": True,
        "publish_url": origin.replace("https://", "wss://").replace("http://", "ws://")
            + "/api/live/rooms/" + room_data["room_id"] + "/human-publish",
    }
