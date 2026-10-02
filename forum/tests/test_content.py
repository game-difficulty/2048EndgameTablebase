import asyncio
import io
from uuid import uuid4
import pytest
from PIL import Image
from starlette.requests import Request

from backend.human_play import engine, verse_replay
from backend.gamer_ranked.prng import Xoshiro128StarStar
from forum.backend import assets, routes
from forum.backend.auth import Principal
from forum.backend.errors import ForumError
from forum.backend.db import one, execute
from forum.worker import deliver_one
from .conftest import headers, document, topic

API = "/api/forum/v1"


def png():
    out = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(out, format="PNG")
    return out.getvalue()


def upload(client, raw=None, user=2, kind="image"):
    r = client.post(
        API + "/media?kind=" + kind,
        content=png() if raw is None else raw,
        headers=headers(user),
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def replay_fixture(version=2):
    run = {
        "id": str(uuid4()),
        "variant": "4x4",
        "seed": "00000001000000020000000300000004",
        "reason": "stopped",
        "created": 0,
    }
    state = engine.initial(run["id"], run["variant"], run["seed"])
    initial = state["board"][:]
    events = b""
    moves = []
    for _ in range(12):
        for direction in range(4):
            board, _ = engine.move(state["board"], 4, 4, direction)
            if board != state["board"]:
                break
        index, value = engine.spawn(board, Xoshiro128StarStar(state["rng"][:]))
        event = engine.EVENT.pack(
            direction | (index << 2) | (64 if value == 4 else 0), 100
        )
        state = engine.advance(state, run["variant"], event)
        events += event
        moves.append((direction, index, value, 100))
    return (
        engine.replay_bytes(run, events, version),
        state,
        verse_replay.encode_rpl1("4x4", initial, moves),
    )


def test_images_private_until_published_and_hidden_with_topic(client):
    ident = upload(client)
    url = f"{API}/media/{ident}"
    assert client.get(url).status_code == 404
    own = client.get(url, headers=headers(2))
    assert own.status_code == 200 and own.headers["content-type"] == "image/webp"
    image = Image.open(io.BytesIO(own.content))
    assert image.size == (16, 12) and not image.getexif()
    assert client.get(url, headers=headers(3)).status_code == 404
    t = topic(client, body=document(f"![说明]({url})"))
    assert client.get(url).status_code == 200
    client.post(
        f"{API}/topics/{t['topic_id']}/moderation",
        json={"action": "hide", "reason": "暂时隐藏附件"},
        headers=headers(1),
    )
    assert client.get(url).status_code == 404
    assert client.get(url, headers=headers(2)).status_code == 200
    assert client.get(API + "/media", headers=headers(3)).json()["items"] == []


def test_no_foreign_attachment_rebinding_and_revision_removal(client):
    ident = upload(client)
    url = f"{API}/media/{ident}"
    t = topic(client, body=document(f"![说明]({url})"))
    r = client.post(
        API + "/topics",
        json={
            "title": "窃取附件尝试",
            "board_slug": "general",
            "body": document(f"![图片]({url})"),
        },
        headers=headers(3, uuid4()),
    )
    assert r.status_code == 403
    r = client.patch(
        f"{API}/posts/{t['post_id']}",
        json={"revision": 1, "body": document("已撤回图片")},
        headers=headers(2),
    )
    assert r.status_code == 200
    assert client.get(url).status_code == 404


def test_attachment_examples_in_code_or_escapes_are_not_bound(client):
    missing = str(uuid4())
    source = f"`[[replay:{missing}]]`\n\n\\[[replay:{missing}]]\n\n```text\n![example]({API}/media/{missing})\n```"
    topic(client, body=document(source))


@pytest.mark.parametrize(
    "raw", [b'<svg onload="alert(1)"></svg>', b"not an image", b"GIF89a"]
)
def test_image_decoder_rejects_active_and_invalid_content(client, raw):
    assert (
        client.post(
            API + "/media?kind=image", content=raw, headers=headers(2)
        ).status_code
        == 400
    )


def test_upload_limits_and_owner_withdrawal(client):
    assert (
        client.post(
            API + "/media?kind=image",
            content=b"x" * (5 * 1024 * 1024 + 1),
            headers=headers(2),
        ).status_code
        == 413
    )
    ident = upload(client)
    assert (
        client.request(
            "DELETE",
            f"{API}/media/{ident}",
            json={"reason": "越权删除附件"},
            headers=headers(3),
        ).status_code
        == 404
    )
    assert (
        client.request(
            "DELETE",
            f"{API}/media/{ident}",
            json={"reason": "撤回个人附件"},
            headers=headers(2),
        ).status_code
        == 200
    )
    assert client.get(f"{API}/media/{ident}", headers=headers(2)).status_code == 404


@pytest.mark.parametrize("version", [1, 2])
def test_play_replay_validation_and_roundtrip(client, version):
    raw, state, _ = replay_fixture(version)
    ident = upload(client, raw, kind="replay")
    assert client.get(f"{API}/media/{ident}/replay").status_code == 404
    t = topic(client, body=document(f"[[replay:{ident}@6]]"))
    result = client.get(f"{API}/media/{ident}/replay").json()
    assert (
        result["score"] == state["score"]
        and result["steps"] == 12
        and result["verification"] == "rules-only"
    )
    assert len(result["moves"]) == 12
    assert client.get(f"{API}/media/{ident}").status_code == 404
    corrupt = bytearray(raw)
    corrupt[-1] ^= 255
    # A changed duration is valid; corrupt a direction/spawn byte instead.
    import struct

    start = 8 + struct.unpack_from("<I", raw, 4)[0]
    corrupt[start] = 255
    assert (
        client.post(
            API + "/media?kind=replay", content=bytes(corrupt), headers=headers(2)
        ).status_code
        == 400
    )


def test_rpl1_validation_and_crc(client):
    _, state, rpl = replay_fixture()
    ident = upload(client, rpl, kind="replay")
    assert (
        client.get(f"{API}/media/{ident}/replay", headers=headers(2)).json()["score"]
        == state["score"]
    )
    assert (
        client.post(
            API + "/media?kind=replay", content=rpl[:-1] + b"x", headers=headers(2)
        ).status_code
        == 400
    )


def test_public_play_source_rechecked_after_withdrawal(client, monkeypatch):
    raw, _, _ = replay_fixture()
    checked = assets.replay_payload(raw)
    calls = []

    def source(settings, run):
        calls.append(str(run))
        return checked

    monkeypatch.setattr(assets, "fetch_play", source)
    run = str(uuid4())
    r = client.post(f"{API}/media/play/{run}", headers=headers(2))
    assert r.status_code == 201
    ident = r.json()["id"]
    topic(client, body=document(f"[[replay:{ident}]]"))
    assert client.get(f"{API}/media/{ident}/replay").status_code == 200
    assert calls == [run, run]

    def removed(*_):
        raise ForumError("PLAY_UNAVAILABLE", "已撤回", 404)

    monkeypatch.setattr(assets, "fetch_play", removed)
    assert client.get(f"{API}/media/{ident}/replay").status_code == 404
    assert (
        client.post(API + "/media/play/evil.invalid", headers=headers(2)).status_code
        == 422
    )


def drain(app):
    for _ in range(100):
        if not deliver_one(app.state.forum.engine):
            return
    raise AssertionError("outbox did not drain")


def test_subscription_worker_idempotent_and_notification_preference(client, app):
    t = topic(client)
    for who in (3, 4):
        assert (
            client.put(
                f"{API}/subscriptions/topic/{t['topic_id']}", headers=headers(who)
            ).status_code
            == 200
        )
    client.put(
        API + "/notification-preferences", json={"enabled": False}, headers=headers(4)
    )
    client.post(
        f"{API}/topics/{t['topic_id']}/posts",
        json={"body": document()},
        headers=headers(5, uuid4()),
    )
    drain(app)
    drain(app)
    assert (
        len(client.get(API + "/notifications", headers=headers(3)).json()["items"]) == 1
    )
    assert (
        client.get(API + "/notification-status", headers=headers(3)).json()["unread"]
        == 1
    )
    assert client.get(API + "/notifications", headers=headers(4)).json()["items"] == []
    assert (
        len(client.get(API + "/subscriptions", headers=headers(3)).json()["items"]) == 1
    )
    assert client.get(API + "/subscriptions", headers=headers(5)).json()["items"] == []
    client.delete(f"{API}/subscriptions/topic/{t['topic_id']}", headers=headers(3))
    client.post(
        f"{API}/topics/{t['topic_id']}/posts",
        json={"body": document()},
        headers=headers(5, uuid4()),
    )
    drain(app)
    assert (
        len(client.get(API + "/notifications", headers=headers(3)).json()["items"]) == 1
    )


def test_board_subscription_and_hidden_events(client, app):
    boards = client.get(API + "/boards").json()["items"]
    board = next(b for b in boards if b["slug"] == "general")
    client.put(f"{API}/subscriptions/board/{board['id']}", headers=headers(3))
    t = topic(client)
    hidden = topic(client)
    client.post(
        f"{API}/topics/{hidden['topic_id']}/moderation",
        json={"action": "hide", "reason": "隐藏测试主题"},
        headers=headers(1),
    )
    drain(app)
    items = client.get(API + "/notifications", headers=headers(3)).json()["items"]
    assert len(items) == 1 and items[0]["topic_id"] == t["topic_id"]


def test_admin_grants_scoped_roles_mutes_and_audits(client):
    t = topic(client, user=2)
    topic(client, user=3)
    boards = client.get(API + "/boards").json()["items"]
    board = next(b for b in boards if b["slug"] == "general")
    payload = {"action": "grant", "board_id": board["id"], "reason": "授予综合区版主"}
    assert (
        client.post(
            API + "/moderation/users/3", json=payload, headers=headers(2)
        ).status_code
        == 403
    )
    assert (
        client.post(
            API + "/moderation/users/3", json=payload, headers=headers(1)
        ).status_code
        == 200
    )
    assert client.get(API + "/session", headers=headers(3)).json()["can_moderate"]
    assert (
        client.get(API + "/moderation/overview/users", headers=headers(3)).status_code
        == 403
    )
    assert (
        client.post(
            f"{API}/topics/{t['topic_id']}/moderation",
            json={"action": "hide", "reason": "版主隐藏测试"},
            headers=headers(3),
        ).status_code
        == 200
    )
    assert (
        len(
            client.get(API + "/moderation/overview/topics", headers=headers(3)).json()[
                "items"
            ]
        )
        == 2
    )
    assert (
        client.post(
            API + "/moderation/users/2",
            json={"action": "mute", "hours": 2, "reason": "测试限时禁言"},
            headers=headers(1),
        ).status_code
        == 200
    )
    assert (
        client.post(
            API + "/topics",
            json={"title": "禁言后发帖", "board_slug": "general", "body": document()},
            headers=headers(2, uuid4()),
        ).status_code
        == 403
    )
    assert (
        client.post(
            API + "/moderation/users/2",
            json={"action": "unmute", "reason": "解除限时禁言"},
            headers=headers(1),
        ).status_code
        == 200
    )
    assert (
        client.post(
            API + "/moderation/users/1",
            json={"action": "mute", "reason": "尝试禁言管理员"},
            headers=headers(1),
        ).status_code
        == 409
    )
    audits = client.get(API + "/moderation/overview/audit", headers=headers(1)).json()[
        "items"
    ]
    assert {"grant", "mute", "unmute", "hide"}.issubset({a["action"] for a in audits})


def test_report_resolution_and_board_settings_and_pinned_pagination(client):
    first = topic(client)
    second = topic(client)
    r = client.post(
        f"{API}/posts/{first['post_id']}/reports",
        json={"reason": "请求检查正文"},
        headers=headers(3),
    ).json()
    assert (
        client.post(
            f"{API}/moderation/reports/{r['id']}/resolve",
            json={"reason": "核验内容正常"},
            headers=headers(1),
        ).status_code
        == 200
    )
    assert (
        client.get(API + "/moderation/reports", headers=headers(1)).json()["items"]
        == []
    )
    client.post(
        f"{API}/topics/{first['topic_id']}/moderation",
        json={"action": "pin", "reason": "置顶重要讨论"},
        headers=headers(1),
    )
    page = client.get(API + "/topics?limit=1").json()
    assert page["items"][0]["id"] == first["topic_id"]
    assert (
        client.get(API + "/topics", params={"cursor": page["next_cursor"]}).json()[
            "items"
        ][0]["id"]
        == second["topic_id"]
    )
    board = client.get(API + "/boards").json()["items"][1]
    data = {
        "name": "综合讨论区",
        "description": "更新板块说明",
        "position": 25,
        "staff_only": False,
        "reason": "调整板块文案",
    }
    assert (
        client.patch(
            f"{API}/moderation/boards/{board['id']}", json=data, headers=headers(2)
        ).status_code
        == 403
    )
    assert (
        client.patch(
            f"{API}/moderation/boards/{board['id']}", json=data, headers=headers(1)
        ).status_code
        == 200
    )
    assert (
        client.patch(
            f"{API}/topics/{second['topic_id']}",
            json={"title": "更新主题标题", "tags": ["复盘"], "revision": 1},
            headers=headers(2),
        ).status_code
        == 200
    )


def test_sse_snapshot_only_for_authenticated_recipient_and_revocation(app, monkeypatch):
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": API + "/notification-stream",
            "headers": [],
            "app": app,
        }
    )

    async def disconnected():
        return False

    request.is_disconnected = disconnected
    calls = iter([Principal(2, "Alice"), None])
    monkeypatch.setattr(routes, "current_user", lambda _: next(calls))

    async def no_sleep(_):
        pass

    monkeypatch.setattr(routes.asyncio, "sleep", no_sleep)

    async def run():
        response = await routes.notification_stream(request, Principal(2, "Alice"))
        return [part async for part in response.body_iterator]

    parts = asyncio.run(run())
    assert (
        len(parts) == 1
        and "event: notifications" in parts[0]
        and '"unread": 0' in parts[0]
    )
