from typing import Literal
from fastapi import APIRouter, Depends, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from .auth import require_user, current_user
from .schemas import (
    VoteInput,
    ReasonInput,
    QuestionInput,
    MoveTopicInput,
    NotificationSettings,
    AnnouncementInput,
    PrivacyDecision,
    ExternalCardInput,
)
from .db import one, all_rows

router = APIRouter(prefix="/api/forum/v1")


def svc(request):
    return request.app.state.forum


@router.put("/topics/{ident}/vote")
def vote(request: Request, ident: int, payload: VoteInput, user=Depends(require_user)):
    return svc(request).vote(user, ident, payload.choices)


@router.post("/topics/{ident}/poll/close")
def close_poll(
    request: Request, ident: int, payload: ReasonInput, user=Depends(require_user)
):
    return svc(request).close_poll(user, ident, payload.reason)


@router.put("/topics/{ident}/question")
def question(
    request: Request, ident: int, payload: QuestionInput, user=Depends(require_user)
):
    return svc(request).question(user, ident, payload.model_dump())


@router.post("/topics/{ident}/move")
def move(
    request: Request, ident: int, payload: MoveTopicInput, user=Depends(require_user)
):
    return svc(request).move_topic(user, ident, payload.model_dump())


@router.get("/posts/{ident}/revisions")
def revisions(request: Request, ident: int, user=Depends(require_user)):
    return {"items": svc(request).revisions(user, ident)}


@router.get("/notification-settings")
def get_notification_settings(request: Request, user=Depends(require_user)):
    return svc(request).notification_settings(user)


@router.put("/notification-settings")
def put_notification_settings(
    request: Request, payload: NotificationSettings, user=Depends(require_user)
):
    return svc(request).notification_settings(user, payload.model_dump())


@router.get("/blocks")
def blocks(request: Request, user=Depends(require_user)):
    return {"items": svc(request).block_list(user)}


@router.put("/blocks/{kind}/{ident}")
@router.delete("/blocks/{kind}/{ident}")
def block(
    request: Request,
    kind: Literal["user", "board"],
    ident: int,
    user=Depends(require_user),
):
    return svc(request).block(user, kind, ident, request.method == "PUT")


@router.get("/announcements")
def announcements(
    request: Request,
    site: Literal["forum", "main", "play", "competition", "live"] = "forum",
):
    return {"items": svc(request).announcements(None, public=True, site=site)}


@router.get("/moderation/announcements")
def manage_announcements(request: Request, user=Depends(require_user)):
    return {"items": svc(request).announcements(user)}


@router.post("/moderation/announcements", status_code=201)
def create_announcement(
    request: Request, payload: AnnouncementInput, user=Depends(require_user)
):
    return svc(request).save_announcement(user, None, payload.model_dump())


@router.put("/moderation/announcements/{ident}")
def update_announcement(
    request: Request, ident: int, payload: AnnouncementInput, user=Depends(require_user)
):
    return svc(request).save_announcement(user, ident, payload.model_dump())


@router.delete("/moderation/announcements/{ident}")
def retract_announcement(
    request: Request,
    ident: int,
    payload: ReasonInput,
    revision: int = Query(..., ge=1),
    user=Depends(require_user),
):
    return svc(request).retract_announcement(user, ident, revision, payload.reason)


@router.get("/privacy/export")
def export(request: Request, user=Depends(require_user)):
    return JSONResponse(
        jsonable_encoder(svc(request).export_private(user)),
        headers={"Content-Disposition": "attachment; filename=forum-data.json"},
    )


@router.get("/privacy/requests")
def privacy_requests(request: Request, user=Depends(require_user)):
    return {"items": svc(request).privacy_requests(user)}


@router.post("/privacy/requests", status_code=201)
def request_privacy(request: Request, payload: ReasonInput, user=Depends(require_user)):
    return svc(request).request_privacy(user, payload.reason)


@router.get("/moderation/privacy")
def privacy_management(request: Request, user=Depends(require_user)):
    return {"items": svc(request).privacy_requests(user, admin=True)}


@router.post("/moderation/privacy/{ident}")
def privacy_decision(
    request: Request, ident: int, payload: PrivacyDecision, user=Depends(require_user)
):
    return svc(request).decide_privacy(user, ident, payload.model_dump())


@router.get("/external-cards")
def cards(request: Request):
    return {"items": svc(request).external_cards(None)}


@router.get("/moderation/external-cards")
def manage_cards(request: Request, user=Depends(require_user)):
    return {"items": svc(request).external_cards(user, admin=True)}


@router.put("/moderation/external-cards")
def save_card(request: Request, payload: ExternalCardInput, user=Depends(require_user)):
    return svc(request).save_external_card(user, payload.model_dump())


@router.get("/moderation/metrics")
def metrics(request: Request, user=Depends(require_user)):
    svc(request).require_admin(user)
    with svc(request).engine.connect() as conn:
        return {
            "queue": one(
                conn,
                "SELECT count(*) FILTER(WHERE delivered_at IS NULL) AS pending,count(*) FILTER(WHERE attempts>0 AND delivered_at IS NULL) AS retrying,coalesce(extract(epoch FROM now()-min(created_at) FILTER(WHERE delivered_at IS NULL)),0) AS oldest_seconds FROM forum_outbox",
            ),
            "content": one(
                conn,
                "SELECT count(*) AS topics,count(*) FILTER(WHERE status='hidden') AS hidden FROM forum_topics",
            ),
            "media": one(
                conn,
                "SELECT count(*) AS files,coalesce(sum(size),0) AS bytes FROM forum_media WHERE status='active'",
            ),
            "open_reports": one(
                conn, "SELECT count(*) AS count FROM forum_reports WHERE status='open'"
            )["count"],
            "open_appeals": one(
                conn, "SELECT count(*) AS count FROM forum_appeals WHERE status='open'"
            )["count"],
        }
