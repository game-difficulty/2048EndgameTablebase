"""Read-only adapter to the site's sessions. No password/account duplication."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import sqlite3

from fastapi import Request

from .errors import ForumError


@dataclass(frozen=True)
class Principal:
    id: int
    display_name: str


def current_user(request: Request) -> Principal | None:
    settings = request.app.state.settings
    raw = request.headers.get("x-forum-dev-user", "")
    if settings.allow_dev_auth and raw:
        parts = raw.split(":", 1)
        if parts[0].isdigit() and 0 < int(parts[0]) < 2**53 and len(raw) <= 110:
            return Principal(int(parts[0]), (parts[1] if len(parts) > 1 else f"测试用户 {parts[0]}").strip() or "测试用户")
        raise ForumError("INVALID_DEV_USER", "本地测试身份无效。")
    tokens = [request.cookies.get("tb_shared_session"), request.cookies.get("tb_session")]
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() == "bearer":
        tokens.append(token.strip())
    tokens = [x for x in tokens if x and len(x) <= 512]
    if not tokens:
        return None
    if not settings.auth_db or not settings.auth_db.is_file():
        raise ForumError("AUTH_UNAVAILABLE", "账号服务暂不可用，请稍后重试。", 503)
    # mode=ro never creates/migrates the site's database or writes billing/activity.
    try:
        db = sqlite3.connect(settings.auth_db.as_uri() + "?mode=ro", uri=True, timeout=3)
        db.row_factory = sqlite3.Row
        try:
            for token in tokens:
                row = db.execute("""SELECT u.id,u.display_name,u.status,s.expires_at,s.revoked_at
                    FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.session_token_hash=?""",
                    (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
                if row and not row["revoked_at"] and row["status"] == "active":
                    expiry = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
                    if expiry.tzinfo and expiry > datetime.now(timezone.utc):
                        return Principal(int(row["id"]), row["display_name"] or f"玩家 {row['id']}")
        finally:
            db.close()
    except (sqlite3.Error, ValueError, OSError) as exc:
        raise ForumError("AUTH_UNAVAILABLE", "账号服务暂不可用，请稍后重试。", 503) from exc
    return None


def require_user(request: Request) -> Principal:
    user = current_user(request)
    if user is None:
        raise ForumError("AUTH_REQUIRED", "请先登录。", 401)
    return user
