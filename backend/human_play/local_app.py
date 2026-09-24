"""Loopback-only preview. No production databases, remote calls or native AI startup."""
from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get("HUMAN_LOCAL_DIR") or ROOT / "tmp" / "human-local")
os.environ["CLOUD_AUTH_DB"] = str(LOCAL / "auth.sqlite3")
os.environ["HUMAN_PLAY_DB"] = str(LOCAL / "human.sqlite3")
os.environ["AUTH_COOKIE_SECURE"] = "0"
os.environ["AUTH_SHARED_COOKIE_DOMAIN"] = ""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import RedirectResponse
from backend.http_compression import DisplayCompression, CacheControlledStaticFiles
from backend.auth.db import init_auth_db, auth_db
from backend.auth.routes import router as auth_router, _set_session_cookie
from backend.auth.service import create_session, public_user, iso
from .store import init_db
from .routes import router, same_origin


@asynccontextmanager
async def lifespan(app):
    init_auth_db()
    init_db()
    yield


app = FastAPI(title="Human 2048 — local preview", lifespan=lifespan)
app.add_middleware(DisplayCompression)


@app.middleware("http")
async def loopback_only(request: Request, call_next):
    if (not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}
            or request.url.hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}):
        return Response("Local preview only", status_code=403)
    return await call_next(request)


@app.get("/api/human/local-preview")
def preview():
    return {"local_preview": True, "isolated_database": True}


@app.post("/api/human/local-session")
def local_session(request: Request, response: Response):
    same_origin(request)
    # This endpoint is only in the loopback app; production only includes routes.router.
    with auth_db() as db:
        now = iso()
        db.execute("""INSERT OR IGNORE INTO users
            (email,email_identity,password_hash,display_name,display_name_key,created_at,updated_at)
            VALUES ('human-preview@localhost.invalid','human-preview@localhost.invalid','!disabled',
                    '本地体验玩家','本地体验玩家',?,?)""", (now, now))
        row = db.execute("SELECT * FROM users WHERE email='human-preview@localhost.invalid'").fetchone()
        token, _, expiry = create_session(db, row["id"], user_agent="human-local", ip_address="127.0.0.1")
        user = public_user(row, db=db)
    _set_session_cookie(response, token, expiry, request)
    return {"authenticated": True, "user": user, "device_session_token": token, "expires_at": expiry}


app.include_router(auth_router)
app.include_router(router)


@app.get("/")
def home():
    return RedirectResponse("/human/")


app.mount("/", CacheControlledStaticFiles(directory=ROOT / "frontend" / "dist", html=True, check_dir=False), name="human-preview")
