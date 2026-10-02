from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from .config import SCHEMA_REVISION, load_settings
from .db import make_engine, one
from .errors import ForumError
from .routes import router
from .service import ForumService

logger = logging.getLogger(__name__)


class RequestBoundary:
    """Bound chunked bodies too; never buffer an unbounded client upload."""
    def __init__(self, app, origin):
        self.app, self.origin = app, origin

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        if scope["method"] in {"POST", "PUT", "PATCH", "DELETE"}:
            origin = headers.get(b"origin", b"").decode("latin-1")
            # Non-browser clients use explicit Bearer credentials; cookies and dev
            # headers require the exact configured browser origin, including port.
            bearer = headers.get(b"authorization", b"").lower().startswith(b"bearer ")
            if (origin and origin != self.origin) or (not origin and (not bearer or b"cookie" in headers)):
                return await JSONResponse({"detail": {"code": "ORIGIN_REJECTED", "message": "请求来源不受信任。"}}, status_code=403)(scope, receive, send)
            chunks, size = [], 0
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                data = message.get("body", b"")
                size += len(data)
                limit = 5 * 1024 * 1024 if scope["path"] == "/api/forum/v1/media" and scope["method"] == "POST" else 131072
                if size > limit:
                    return await JSONResponse({"detail": {"code": "BODY_TOO_LARGE", "message": "提交内容过大。"}}, status_code=413)(scope, receive, send)
                chunks.append(data)
                if not message.get("more_body"):
                    break
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": b"".join(chunks), "more_body": False}
                return await receive()
            return await self.app(scope, bounded_receive, send)
        return await self.app(scope, receive, send)


def create_app(settings=None):
    settings = settings or load_settings()
    engine = make_engine(settings.database_url)

    def readiness():
        with engine.connect() as conn:
            if one(conn, "SELECT version_num FROM alembic_version")["version_num"] != SCHEMA_REVISION:
                raise RuntimeError("Forum schema mismatch; run the explicit Alembic migration first.")

    @asynccontextmanager
    async def lifespan(app):
        try:
            await run_in_threadpool(readiness)
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="2048 Forum", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.forum = ForumService(engine, settings)
    app.add_middleware(RequestBoundary, origin=settings.public_origin)

    @app.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "private, no-store"
        else:
            response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; worker-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.exception_handler(ForumError)
    async def forum_error(request, exc):
        headers = {"Retry-After": "60"} if exc.status == 429 else None
        return JSONResponse({"detail": {"code": exc.code, "message": exc.message}}, status_code=exc.status, headers=headers)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request, exc):
        # Never log SQL parameters, account data, drafts or connection URLs.
        logger.error("Forum database failure (%s)", type(exc).__name__)
        return JSONResponse({"detail": {"code": "DATABASE_UNAVAILABLE", "message": "服务暂时繁忙，请保留内容并稍后重试。"}}, status_code=503)

    @app.get("/health/live")
    def live():
        return {"status": "ok", "service": "forum"}

    @app.get("/health/ready")
    def ready():
        readiness()
        return {"status": "ok", "database": "postgresql", "revision": SCHEMA_REVISION}

    app.include_router(router)
    dist = settings.frontend_dist.resolve()
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="forum-assets")
    if (dist / "index.html").is_file():
        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            if path == "favicon.svg":
                return FileResponse(dist / "favicon.svg", media_type="image/svg+xml")
            if path.startswith(("api/", "health/", "assets/")):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            return FileResponse(dist / "index.html")
    return app
