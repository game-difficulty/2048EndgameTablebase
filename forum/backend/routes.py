from uuid import UUID
from typing import Literal
import asyncio
import json
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi import APIRouter, Depends, Header, Query, Request

from .auth import current_user, require_user
from .schemas import (
    DraftInput,
    EditPost,
    ModerationInput,
    NewPost,
    NewTopic,
    ReportInput,
)
from .schemas import (
    ReasonInput,
    PreferenceInput,
    TopicInput,
    AdminUserInput,
    BoardInput,
    ReplyDraftInput,
    ReadingInput,
    AppealInput,
    AppealDecision,
)
from . import assets
from .errors import ForumError
from .db import all_rows

router = APIRouter(prefix="/api/forum/v1")


def service(request: Request):
    return request.app.state.forum


@router.get("/session")
def session(request: Request, user=Depends(current_user)):
    svc = service(request)
    with svc.engine.connect() as conn:
        from .db import one

        moderator = bool(
            user
            and (
                svc.is_admin(user)
                or one(conn, "SELECT 1 FROM forum_roles WHERE user_id=:id", id=user.id)
            )
        )
    return {
        "user": {"id": user.id, "display_name": user.display_name} if user else None,
        "is_admin": svc.is_admin(user),
        "can_moderate": moderator,
        "development_auth": request.app.state.settings.allow_dev_auth,
    }


@router.get("/boards")
def boards(request: Request, user=Depends(current_user)):
    return {"items": service(request).boards(user)}


@router.get("/topics")
def topics(
    request: Request,
    user=Depends(current_user),
    board: str = Query("", max_length=40),
    cursor: str = Query("", max_length=2048),
    q: str = Query("", max_length=80),
    saved: bool = False,
    limit: int = Query(20, ge=1, le=50),
    view: Literal[
        "activity", "newest", "unanswered", "featured", "following", "unread"
    ] = "activity",
    tag: str = Query("", max_length=24),
    author: int | None = Query(None, ge=1, le=9007199254740991),
    topic_id: int | None = Query(None, ge=1, le=9007199254740991),
    since: str | None = Query(None, max_length=40),
    until: str | None = Query(None, max_length=40),
    kind: Literal["", "discussion", "question", "poll"] = "",
):
    return service(request).list_topics(
        user,
        board,
        cursor,
        q.strip(),
        saved,
        limit,
        view=view,
        tag=tag,
        author=author,
        topic_id=topic_id,
        since=since,
        until=until,
        kind=kind,
    )


@router.post("/topics", status_code=201)
def create_topic(
    request: Request,
    payload: NewTopic,
    user=Depends(require_user),
    idempotency_key: UUID = Header(...),
):
    return service(request).create_topic(user, idempotency_key, payload.model_dump())


@router.get("/topics/{topic_id}")
def detail(
    request: Request,
    topic_id: int,
    user=Depends(current_user),
    after: int = Query(0, ge=0, le=2147483647),
    limit: int = Query(40, ge=1, le=100),
    focus_post: int | None = Query(None, ge=1, le=9007199254740991),
    author_only: bool = False,
):
    return service(request).detail(
        topic_id, user, after, limit, focus_post, author_only
    )


@router.post("/topics/{topic_id}/posts", status_code=201)
def reply(
    request: Request,
    topic_id: int,
    payload: NewPost,
    user=Depends(require_user),
    idempotency_key: UUID = Header(...),
):
    return service(request).reply(topic_id, user, idempotency_key, payload.model_dump())


@router.patch("/posts/{post_id}")
def edit(request: Request, post_id: int, payload: EditPost, user=Depends(require_user)):
    return service(request).edit_post(post_id, user, payload.model_dump())


@router.delete("/posts/{post_id}")
def delete(
    request: Request,
    post_id: int,
    revision: int = Query(..., ge=1),
    user=Depends(require_user),
):
    return service(request).edit_post(
        post_id, user, {"revision": revision}, delete=True
    )


@router.put("/posts/{post_id}/like")
@router.delete("/posts/{post_id}/like")
def like(request: Request, post_id: int, user=Depends(require_user)):
    return service(request).toggle("reaction", post_id, user, request.method == "PUT")


@router.put("/topics/{topic_id}/bookmark")
@router.delete("/topics/{topic_id}/bookmark")
def bookmark(request: Request, topic_id: int, user=Depends(require_user)):
    return service(request).toggle("bookmark", topic_id, user, request.method == "PUT")


@router.get("/drafts")
def drafts(request: Request, user=Depends(require_user)):
    return {"items": service(request).drafts(user)}


@router.put("/drafts/{draft_id}")
def save_draft(
    request: Request, draft_id: UUID, payload: DraftInput, user=Depends(require_user)
):
    return service(request).save_draft(draft_id, user, payload.model_dump())


@router.delete("/drafts/{draft_id}")
def delete_draft(
    request: Request,
    draft_id: UUID,
    revision: int = Query(..., ge=1),
    user=Depends(require_user),
):
    return service(request).delete_draft(draft_id, user, revision)


@router.get("/notifications")
def notifications(
    request: Request,
    user=Depends(require_user),
    before: int = Query(9223372036854775807, ge=1, le=9223372036854775807),
    kind: Literal[
        "", "reply", "mention", "subscription", "follow", "moderation", "system"
    ] = "",
):
    return {"items": service(request).notifications(user, before, kind)}


@router.put("/notifications/read")
def read_notifications(
    request: Request, through_id: int = Query(..., ge=1), user=Depends(require_user)
):
    return service(request).read_notifications(user, through_id)


@router.post("/posts/{post_id}/reports", status_code=201)
def report(
    request: Request, post_id: int, payload: ReportInput, user=Depends(require_user)
):
    return service(request).report(post_id, user, payload.reason)


@router.get("/moderation/reports")
def reports(request: Request, user=Depends(require_user)):
    return {"items": service(request).reports(user)}


@router.post("/topics/{topic_id}/moderation")
def moderation(
    request: Request,
    topic_id: int,
    payload: ModerationInput,
    user=Depends(require_user),
):
    return service(request).moderation(topic_id, user, payload.model_dump())


@router.patch("/topics/{topic_id}")
def update_topic(
    request: Request, topic_id: int, payload: TopicInput, user=Depends(require_user)
):
    return service(request).update_topic(topic_id, user, payload.model_dump())


@router.post("/media", status_code=201)
async def upload(
    request: Request, kind: Literal["image", "replay"], user=Depends(require_user)
):
    raw = await request.body()
    if not raw or len(raw) > (
        assets.MAX_IMAGE if kind == "image" else assets.MAX_REPLAY
    ):
        raise ForumError("MEDIA_SIZE", "图片最多 5 MiB，录像最多 2 MiB。", 413)
    return await run_in_threadpool(assets.store, service(request), user, raw, kind)


@router.post("/media/play/{run_id}", status_code=201)
def import_play(request: Request, run_id: UUID, user=Depends(require_user)):
    return assets.import_play(service(request), user, run_id)


@router.get("/media")
def own_media(request: Request, user=Depends(require_user)):
    with service(request).engine.connect() as conn:
        return {
            "items": all_rows(
                conn,
                "SELECT id,kind,size,created_at FROM forum_media WHERE owner_id=:u AND status='active' ORDER BY created_at DESC LIMIT 100",
                u=user.id,
            )
        }


@router.get("/media/{ident}")
def media(request: Request, ident: UUID, user=Depends(current_user)):
    svc = service(request)
    with svc.engine.connect() as conn:
        row = assets.access(conn, ident, user, svc)
    if row["kind"] != "image":
        raise ForumError("MEDIA_KIND", "此附件不是图片。", 404)
    return Response(
        bytes(row["data"]),
        media_type=row["mime"],
        headers={"Content-Disposition": "inline; filename=forum-image.webp"},
    )


@router.get("/media/{ident}/replay")
def replay(request: Request, ident: UUID, user=Depends(current_user)):
    svc = service(request)
    with svc.engine.connect() as conn:
        row = assets.access(conn, ident, user, svc)
    if row["kind"] == "play":
        result = assets.fetch_play(svc.settings, row["metadata"]["run_id"])
        version = assets.replay_version(result)
        if row["metadata"].get("artifact_version") != version:
            raise ForumError(
                "SOURCE_REVISED",
                "Play 来源已修订或旧引用未绑定版本，请重新导入；原步数引用不会自动指向新录像。",
                409,
            )
        return {
            **result,
            "artifact_version": version,
            "verification": "public-source",
            "source_url": f"{svc.settings.play_origin}/api/human/replays/{row['metadata']['run_id']}",
        }
    if row["kind"] != "replay":
        raise ForumError("MEDIA_KIND", "此附件不是录像。", 404)
    return assets.replay_payload(bytes(row["data"]))


@router.delete("/media/{ident}")
def remove_media(
    request: Request, ident: UUID, payload: ReasonInput, user=Depends(require_user)
):
    return service(request).remove_media(user, ident, payload.reason)


@router.get("/subscriptions")
def subscriptions(request: Request, user=Depends(require_user)):
    return {"items": service(request).subscriptions(user)}


@router.put("/subscriptions/{kind}/{ident}")
@router.delete("/subscriptions/{kind}/{ident}")
def subscribe(
    request: Request,
    kind: Literal["topic", "board"],
    ident: int,
    user=Depends(require_user),
):
    return service(request).subscribe(user, kind, ident, request.method == "PUT")


@router.get("/notification-status")
def notification_status(request: Request, user=Depends(require_user)):
    return service(request).notification_status(user)


@router.put("/notification-preferences")
def notification_preference(
    request: Request, payload: PreferenceInput, user=Depends(require_user)
):
    return service(request).notification_preference(user, payload.enabled)


@router.get("/notification-stream")
async def notification_stream(request: Request, user=Depends(require_user)):
    async def updates():
        last = None
        # Periodic reconnect prevents stale sessions from keeping an indefinite stream.
        for _ in range(30):
            if await request.is_disconnected():
                return
            try:
                current = await run_in_threadpool(current_user, request)
                if not current or current.id != user.id:
                    return
                status = await run_in_threadpool(
                    service(request).notification_status, user
                )
            except ForumError:
                return
            value = json.dumps(status)
            if value != last:
                yield f"id: {status['latest']}\nevent: notifications\ndata: {value}\n\n"
                last = value
            else:
                yield ": keepalive\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(
        updates(),
        media_type="text/event-stream",
        headers={"Cache-Control": "private, no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/moderation/overview/{section}")
def management(
    request: Request,
    section: Literal["topics", "reports", "audit", "users", "media", "jobs"],
    before: int = Query(9223372036854775807, ge=1, le=9223372036854775807),
    q: str = Query("", max_length=80),
    user=Depends(require_user),
):
    return {"items": service(request).management(user, section, before, q)}


@router.post("/moderation/reports/{ident}/resolve")
def resolve_report(
    request: Request, ident: int, payload: ReasonInput, user=Depends(require_user)
):
    return service(request).resolve_report(user, ident, payload.reason)


@router.post("/moderation/users/{ident}")
def admin_user(
    request: Request, ident: int, payload: AdminUserInput, user=Depends(require_user)
):
    return service(request).admin_user(user, ident, payload.model_dump())


@router.patch("/moderation/boards/{ident}")
def edit_board(
    request: Request, ident: int, payload: BoardInput, user=Depends(require_user)
):
    return service(request).edit_board(user, ident, payload.model_dump())


@router.get("/profiles")
def profiles(request: Request, q: str = Query("", max_length=80)):
    return {"items": service(request).profiles(q)}


@router.get("/profiles/{ident}")
def profile(
    request: Request,
    ident: int,
    before: int = Query(9223372036854775807, ge=1, le=9223372036854775807),
    before_reply: int = Query(9223372036854775807, ge=1, le=9223372036854775807),
    user=Depends(current_user),
):
    return service(request).profile(ident, user, before, before_reply)


@router.get("/follows")
def follows(request: Request, user=Depends(require_user)):
    return {"items": service(request).follows(user)}


@router.put("/follows/{ident}")
@router.delete("/follows/{ident}")
def follow(request: Request, ident: int, user=Depends(require_user)):
    return service(request).follow(user, ident, request.method == "PUT")


@router.get("/reading")
def reading(request: Request, user=Depends(require_user)):
    return {"items": service(request).reading(user)}


@router.get("/topics/{ident}/reading")
def topic_reading(request: Request, ident: int, user=Depends(require_user)):
    return {"position": service(request).reading(user, ident)}


@router.put("/topics/{ident}/reading")
def mark_reading(
    request: Request, ident: int, payload: ReadingInput, user=Depends(require_user)
):
    return service(request).mark_reading(user, ident, payload.post_id)


@router.get("/topics/{ident}/reply-draft")
def reply_draft(request: Request, ident: int, user=Depends(require_user)):
    return {"draft": service(request).reply_draft(user, ident)}


@router.get("/reply-drafts")
def reply_drafts(request: Request, user=Depends(require_user)):
    return {"items": service(request).reply_drafts(user)}


@router.delete("/reply-drafts/{ident}")
def clear_reply_draft(
    request: Request,
    ident: int,
    revision: int = Query(..., ge=1, le=2147483646),
    user=Depends(require_user),
):
    return service(request).clear_reply_draft(user, ident, revision)


@router.put("/topics/{ident}/reply-draft")
def save_reply_draft(
    request: Request, ident: int, payload: ReplyDraftInput, user=Depends(require_user)
):
    return service(request).save_reply_draft(user, ident, payload.model_dump())


@router.get("/appeal-actions")
def appeal_actions(request: Request, user=Depends(require_user)):
    return {"items": service(request).appeal_actions(user)}


@router.get("/appeals")
def appeals(
    request: Request,
    before: int = Query(9223372036854775807, ge=1, le=9223372036854775807),
    user=Depends(require_user),
):
    return {"items": service(request).appeals(user, before=before)}


@router.post("/appeals", status_code=201)
def create_appeal(request: Request, payload: AppealInput, user=Depends(require_user)):
    return service(request).create_appeal(user, payload.model_dump())


@router.get("/moderation/appeals")
def review_appeals(
    request: Request,
    before: int = Query(9223372036854775807, ge=1, le=9223372036854775807),
    user=Depends(require_user),
):
    return {"items": service(request).appeals(user, reviewing=True, before=before)}


@router.post("/moderation/appeals/{ident}")
def decide_appeal(
    request: Request, ident: int, payload: AppealDecision, user=Depends(require_user)
):
    return service(request).decide_appeal(user, ident, payload.model_dump())
