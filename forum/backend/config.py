from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_REVISION = "0001_forum"


@dataclass(frozen=True)
class Settings:
    database_url: str
    public_origin: str = "http://127.0.0.1:5175"
    environment: str = "production"
    allow_dev_auth: bool = False
    admin_ids: frozenset[int] = frozenset()
    auth_db: Path | None = None
    frontend_dist: Path = ROOT / "frontend" / "dist"

    def __post_init__(self):
        if make_url(self.database_url).drivername != "postgresql+psycopg":
            raise ValueError("FORUM_DATABASE_URL must use postgresql+psycopg; SQLite is not supported.")
        origin = urlsplit(self.public_origin)
        if origin.scheme not in {"http", "https"} or not origin.netloc or origin.path or origin.query or origin.fragment or origin.username:
            raise ValueError("FORUM_PUBLIC_ORIGIN must be an origin without path or credentials.")
        if self.environment not in {"production", "development", "test"}:
            raise ValueError("Unknown FORUM_ENV.")
        if self.environment == "production" and (self.allow_dev_auth or origin.scheme != "https"):
            raise ValueError("Production requires HTTPS and forbids development authentication.")


def load_settings() -> Settings:
    url = os.environ.get("FORUM_DATABASE_URL", "")
    if not url:
        raise ValueError("FORUM_DATABASE_URL is required; run migrations before starting the forum.")
    auth_path = os.environ.get("CLOUD_AUTH_DB", "").strip()
    return Settings(
        database_url=url,
        public_origin=os.environ.get("FORUM_PUBLIC_ORIGIN", "https://forum.2048tables.online").rstrip("/"),
        environment=os.environ.get("FORUM_ENV", "production"),
        allow_dev_auth=os.environ.get("FORUM_ALLOW_DEV_AUTH", "0") == "1",
        admin_ids=frozenset(int(x.strip()) for x in os.environ.get("FORUM_ADMIN_IDS", "").split(",") if x.strip()),
        auth_db=Path(auth_path).resolve() if auth_path else None,
    )
