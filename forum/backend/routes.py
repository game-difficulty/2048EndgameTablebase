from uuid import UUID
from fastapi import APIRouter, Depends, Header, Query, Request

from .auth import current_user, require_user
from .schemas import DraftInput, EditPost, ModerationInput, NewPost, NewTopic, ReportInput

router = APIRouter(prefix="/api/forum/v1")


def service(request: Request):
    return request.app.state.forum


@router.get("/session")
def session(request: Request, user=Depends(current_user)):
    svc = service(request)
    with svc.engine.connect() as conn:
        from .db import one
        moderator = bool(user and (svc.is_admin(user) or one(conn, "SELECT 1 FROM forum_roles WHERE user_id=:id", id=user.id)))
    return {"user": {"id": user.id, "display_name": user.display_name} if user else None,
            "is_admin": svc.is_admin(user), "can_moderate": moderator,
            "development_auth": request.app.state.settings.allow_dev_auth}


@router.get("/boards")
def boards(request: Request, user=Depends(current_user)):
    return {"items": service(request).boards(user)}


@router.get("/topics")
def topics(request: Request, user=Depends(current_user), board: str = Query("", max_length=40),
           cursor: str = Query("", max_length=256), q: str = Query("", max_length=80),
           saved: bool = False, limit: int = Query(20, ge=1, le=50)):
    return service(request).list_topics(user, board, cursor, q.strip(), saved, limit)


@router.post("/topics", status_code=201)
def create_topic(request: Request, payload: NewTopic, user=Depends(require_user),
                 idempotency_key: UUID = Header(...)):
    return service(request).create_topic(user, idempotency_key, payload.model_dump())


@router.get("/topics/{topic_id}")
def detail(request: Request, topic_id: int, user=Depends(current_user),
           after: int = Query(0, ge=0, le=2147483647), limit: int = Query(40, ge=1, le=100),
           focus_post: int | None = Query(None, ge=1, le=9007199254740991)):
    return service(request).detail(topic_id, user, after, limit, focus_post)


@router.post("/topics/{topic_id}/posts", status_code=201)
def reply(request: Request, topic_id: int, payload: NewPost, user=Depends(require_user), idempotency_key: UUID = Header(...)):
    return service(request).reply(topic_id, user, idempotency_key, payload.model_dump())


@router.patch("/posts/{post_id}")
def edit(request: Request, post_id: int, payload: EditPost, user=Depends(require_user)):
    return service(request).edit_post(post_id, user, payload.model_dump())


@router.delete("/posts/{post_id}")
def delete(request: Request, post_id: int, revision: int = Query(..., ge=1), user=Depends(require_user)):
    return service(request).edit_post(post_id, user, {"revision": revision}, delete=True)


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
def save_draft(request: Request, draft_id: UUID, payload: DraftInput, user=Depends(require_user)):
    return service(request).save_draft(draft_id, user, payload.model_dump())


@router.delete("/drafts/{draft_id}")
def delete_draft(request: Request, draft_id: UUID, revision: int = Query(..., ge=1), user=Depends(require_user)):
    return service(request).delete_draft(draft_id, user, revision)


@router.get("/notifications")
def notifications(request: Request, user=Depends(require_user)):
    return {"items": service(request).notifications(user)}


@router.put("/notifications/read")
def read_notifications(request: Request, through_id: int = Query(..., ge=1), user=Depends(require_user)):
    return service(request).read_notifications(user, through_id)


@router.post("/posts/{post_id}/reports", status_code=201)
def report(request: Request, post_id: int, payload: ReportInput, user=Depends(require_user)):
    return service(request).report(post_id, user, payload.reason)


@router.get("/moderation/reports")
def reports(request: Request, user=Depends(require_user)):
    return {"items": service(request).reports(user)}


@router.post("/topics/{topic_id}/moderation")
def moderation(request: Request, topic_id: int, payload: ModerationInput, user=Depends(require_user)):
    return service(request).moderation(topic_id, user, payload.model_dump())
