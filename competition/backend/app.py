from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .auth import principal_from_socket_auth, principal_from_websocket
from .config import load_settings
from .db import CompetitionDatabase
from .errors import CompetitionError
from .hub import RoomHub
from .practice_leaderboard import PracticeLeaderboard
from .routes import router
from .service import CompetitionService


def create_app() -> FastAPI:
    settings = load_settings()
    database = CompetitionDatabase(settings.database_path)
    service = CompetitionService(
        database,
        bootstrap_organizer_ids=settings.bootstrap_organizer_ids,
        room_creator_ids=settings.room_creator_ids,
        draw_reveal_seconds=settings.draw_reveal_seconds,
        draft_turn_seconds=settings.draft_turn_seconds,
        c_draw_reveal_seconds=settings.c_draw_reveal_seconds,
        lineup_seconds=settings.lineup_seconds,
        team_clock_seconds=settings.team_clock_seconds,
        test_project_target_tile=settings.test_project_target_tile,
        live_result_retention_seconds=settings.live_result_retention_seconds,
    )
    hub = RoomHub()

    async def deadline_worker() -> None:
        while True:
            try:
                changed_codes = await asyncio.to_thread(service.process_due_rooms)
                for code in changed_codes:
                    await hub.broadcast(code, lambda viewer, value=code: service.snapshot(value, viewer))
            except asyncio.CancelledError:
                raise
            except Exception:
                # The next scan and request-side catch-up use the same idempotent path.
                pass
            await asyncio.sleep(0.5)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await asyncio.to_thread(service.initialize)
        worker = asyncio.create_task(deadline_worker())
        try:
            yield
        finally:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="2048 Competition Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.competition_settings = settings
    app.state.competition_service = service
    app.state.practice_leaderboard = PracticeLeaderboard(database)
    app.state.competition_hub = hub
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Competition-Dev-User"],
    )
    app.include_router(router)

    @app.exception_handler(CompetitionError)
    async def competition_error_handler(_request: Request, exc: CompetitionError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.websocket("/ws/rooms/{room_code}")
    async def room_socket(websocket: WebSocket, room_code: str) -> None:
        await websocket.accept()
        principal = principal_from_websocket(websocket, settings)
        if principal is None:
            try:
                message = await asyncio.wait_for(websocket.receive_json(), timeout=8.0)
            except (TimeoutError, ValueError, WebSocketDisconnect):
                await websocket.send_json(
                    {
                        "type": "error",
                        "error": {
                            "code": "AUTH_REQUIRED",
                            "message": "Authentication required.",
                        },
                    }
                )
                await websocket.close(code=4401)
                return
            if message.get("type") != "authenticate":
                await websocket.close(code=4401)
                return
            principal = principal_from_socket_auth(message.get("data") or {}, settings)
        if principal is None:
            await websocket.close(code=4401)
            return
        normalized_code = str(room_code).strip().upper()
        try:
            snapshot = await asyncio.to_thread(service.snapshot, normalized_code, principal)
        except CompetitionError as exc:
            await websocket.send_json({"type": "error", "error": exc.detail})
            await websocket.close(code=4404)
            return
        await hub.connect(normalized_code, websocket, principal)
        await websocket.send_json({"type": "room.snapshot", "data": snapshot})
        try:
            while True:
                message = await websocket.receive_json()
                message_type = str(message.get("type") or "")
                if message_type == "ping":
                    await websocket.send_json({"type": "pong"})
                elif message_type == "room.resync":
                    fresh = await asyncio.to_thread(
                        service.snapshot, normalized_code, principal
                    )
                    await websocket.send_json({"type": "room.snapshot", "data": fresh})
        except WebSocketDisconnect:
            pass
        finally:
            await hub.disconnect(normalized_code, websocket)

    frontend_dist = settings.frontend_dist
    assets = frontend_dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="competition-assets")

    if (frontend_dist / "index.html").is_file():
        @app.get("/{path:path}", include_in_schema=False)
        async def frontend_entry(path: str):
            candidate = frontend_dist / path
            if path and candidate.is_file() and frontend_dist in candidate.resolve().parents:
                return FileResponse(candidate)
            return FileResponse(frontend_dist / "index.html")

    return app


app = create_app()
