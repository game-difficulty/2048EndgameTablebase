from datetime import datetime, timezone, timedelta
from uuid import uuid4
from .conftest import headers, document, topic
from .test_social import reply, moderate
from forum.backend.db import execute, one
from forum.scheduler import tick
from forum.worker import deliver_one
import json
import pytest
from forum.backend import assets
from forum.backend.errors import ForumError

API = "/api/forum/v1"


def write(client, method, route, user=2, **body):
    return client.request(
        method.upper(), API + route, headers=headers(user), json=body or None
    )


def deadline(days=1):
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def test_poll_votes_results_deadline_and_immutable_choices(client, app):
    t = topic(
        client,
        kind="poll",
        poll={
            "options": ["向左", "向右", "向上"],
            "max_choices": 2,
            "closes_at": deadline(),
            "results": "voted",
        },
    )
    ident = t["topic_id"]
    assert client.get(f"{API}/topics/{ident}").json()["poll"]["counts"] is None
    assert (
        write(
            client, "put", f"/topics/{ident}/vote", user=3, choices=[0, 1]
        ).status_code
        == 200
    )
    assert (
        write(client, "put", f"/topics/{ident}/vote", user=3, choices=[1]).json()[
            "voters"
        ]
        == 1
    )
    assert client.get(f"{API}/topics/{ident}", headers=headers(3)).json()["poll"][
        "counts"
    ] == [0, 1, 0]
    assert (
        write(client, "put", f"/topics/{ident}/vote", choices=[0, 1, 2]).status_code
        == 400
    )
    assert (
        write(client, "put", f"/topics/{ident}/vote", choices=[99]).status_code == 400
    )
    assert (
        client.patch(
            f"{API}/topics/{ident}",
            headers=headers(),
            json={
                "title": "试图替换投票选项",
                "revision": 1,
                "options": ["偷偷改选项"],
            },
        ).status_code
        == 422
    )
    assert (
        write(
            client,
            "post",
            f"/topics/{ident}/poll/close",
            user=3,
            reason="普通投票者不能关闭",
        ).status_code
        == 403
    )
    assert (
        write(
            client, "post", f"/topics/{ident}/poll/close", reason="作者提前结束投票"
        ).status_code
        == 200
    )
    assert client.get(f"{API}/topics/{ident}").json()["poll"]["counts"] == [0, 1, 0]
    assert write(client, "put", f"/topics/{ident}/vote", choices=[0]).status_code == 409


def test_question_accept_revision_scope_and_deleted_answer(client):
    t = topic(client, kind="question")
    r = reply(client, t["topic_id"])
    other = topic(client)
    path = f'/topics/{t["topic_id"]}/question'
    body = {
        "status": "solved",
        "accepted_post_id": r["post_id"],
        "revision": 1,
        "reason": "回答解决了问题",
    }
    assert write(client, "put", path, user=4, **body).status_code == 403
    assert (
        write(
            client, "put", path, **{**body, "accepted_post_id": other["post_id"]}
        ).status_code
        == 400
    )
    assert write(client, "put", path, **body).status_code == 200
    assert write(client, "put", path, **body).status_code == 409
    assert (
        client.delete(
            f'{API}/posts/{r["post_id"]}?revision=1', headers=headers(3)
        ).status_code
        == 200
    )
    result = client.get(f'{API}/topics/{t["topic_id"]}').json()["topic"]
    assert result["accepted_post_id"] is None and result["question_status"] == "open"


def test_discovery_chinese_short_terms_filters_and_cursor_context(client):
    a = topic(client, title="残局研究甲", tags=["32K"], kind="question")
    b = topic(
        client, title="普通讨论主题", body=document("这段讨论涉及残局乙"), tags=["65K"]
    )
    topic(client, user=3, title="另一个讨论丙", body=document("普通正文"))
    for term in ["残", "残局", "残局乙"]:
        items = client.get(API + "/topics", params={"q": term}).json()["items"]
        assert len(items) == (1 if term == "残局乙" else 2)
    assert (
        client.get(API + "/topics?tag=32K&kind=question").json()["items"][0]["id"]
        == a["topic_id"]
    )
    assert len(client.get(API + "/topics?author=2").json()["items"]) == 2
    page = client.get(API + "/topics?view=newest&limit=1").json()
    assert page["next_cursor"]
    assert (
        len(
            client.get(
                API + "/topics",
                params={"view": "newest", "limit": 1, "cursor": page["next_cursor"]},
            ).json()["items"]
        )
        == 1
    )
    assert (
        client.get(
            API + "/topics", params={"view": "activity", "cursor": page["next_cursor"]}
        ).status_code
        == 400
    )
    assert client.get(API + "/topics?view=following").status_code == 401
    write(client, "put", "/follows/2", user=3)
    assert (
        len(
            client.get(API + "/topics?view=following", headers=headers(3)).json()[
                "items"
            ]
        )
        == 2
    )
    write(
        client, "put", f'/topics/{a["topic_id"]}/reading', user=3, post_id=a["post_id"]
    )
    assert a["topic_id"] not in [
        x["id"]
        for x in client.get(API + "/topics?view=unread", headers=headers(3)).json()[
            "items"
        ]
    ]
    moderate(client, b["topic_id"], "hide")
    assert len(client.get(API + "/topics?q=残局").json()["items"]) == 1


def test_blocks_hide_feed_reply_and_notifications_and_preferences(client, app):
    a = topic(client)
    topic(client, user=3)
    write(client, "put", "/follows/2", user=3)
    write(client, "put", "/blocks/user/2", user=3)
    assert a["topic_id"] not in [
        t["id"] for t in client.get(API + "/topics", headers=headers(3)).json()["items"]
    ]
    post = client.get(f'{API}/topics/{a["topic_id"]}', headers=headers(3)).json()[
        "posts"
    ][0]
    assert post["blocked"] and post["body"] is None
    topic(client, body=document("测试 <@3> 屏蔽"))
    while deliver_one(app.state.forum.engine):
        pass
    assert client.get(API + "/notifications", headers=headers(3)).json()["items"] == []
    write(client, "delete", "/blocks/user/2", user=3)
    write(
        client,
        "put",
        "/notification-settings",
        user=3,
        enabled=True,
        categories={"mention": False, "follow": False},
    )
    topic(client, body=document("再次 <@3> 测试分类设置"))
    while deliver_one(app.state.forum.engine):
        pass
    assert client.get(API + "/notifications", headers=headers(3)).json()["items"] == []
    assert client.get(API + "/blocks", headers=headers()).json()["items"] == []


def test_announcement_scheduling_expiration_and_idempotence(client, app):
    topic(client, user=3)
    payload = {
        "topic": {
            "board_slug": "announcements",
            "title": "管理员定时公告",
            "body": document("公告测试正文"),
        },
        "sites": ["forum", "play"],
        "publish_at": deadline(-1),
        "expires_at": deadline(),
        "notify_all": True,
        "schedule": True,
        "allow_replies": False,
        "revision": 0,
    }
    assert (
        write(client, "post", "/moderation/announcements", **payload).status_code == 403
    )
    r = write(client, "post", "/moderation/announcements", user=1, **payload)
    assert r.status_code == 201, r.text
    assert client.get(API + "/announcements").json()["items"] == []
    assert tick(app.state.forum)
    assert not tick(app.state.forum)
    public = client.get(API + "/announcements").json()["items"]
    assert len(public) == 1
    assert client.get(API + "/announcements?site=main").json()["items"] == []
    t = client.get(f'{API}/topics/{public[0]["topic_id"]}').json()["topic"]
    assert t["locked"] and t["pinned"]
    assert (
        client.get(API + "/notifications?kind=system", headers=headers(3)).json()[
            "items"
        ][0]["topic_id"]
        == t["id"]
    )
    assert (
        len(client.get(API + "/notifications", headers=headers(3)).json()["items"]) == 1
    )
    with app.state.forum.engine.begin() as conn:
        execute(
            conn, "UPDATE forum_announcements SET expires_at=now()-interval '1 second'"
        )
    assert tick(app.state.forum)
    assert client.get(API + "/announcements").json()["items"] == []
    assert not client.get(f'{API}/topics/{t["id"]}').json()["topic"]["pinned"]
    r = client.request(
        "DELETE",
        f'{API}/moderation/announcements/{public[0]["id"]}?revision=3',
        headers=headers(1),
        json={"reason": "撤回已到期公告"},
    )
    assert r.status_code == 200, r.text
    assert client.get(f'{API}/topics/{t["id"]}').status_code == 404


def test_privacy_export_requests_and_system_inbox(client):
    a = topic(client)
    topic(client, user=3)
    result = client.get(API + "/privacy/export", headers=headers()).json()
    assert [t["id"] for t in result["topics"]] == [a["topic_id"]]
    r = write(
        client, "post", "/privacy/requests", reason="申请核对并删除指定的隐私内容"
    )
    assert r.status_code == 201
    assert (
        write(client, "post", "/privacy/requests", reason="重复请求测试").status_code
        == 409
    )
    assert (
        client.get(API + "/privacy/requests", headers=headers(3)).json()["items"] == []
    )
    ident = r.json()["id"]
    assert (
        write(
            client,
            "post",
            f"/moderation/privacy/{ident}",
            user=3,
            decision="completed",
            reason="普通用户不能处理",
        ).status_code
        == 403
    )
    assert (
        write(
            client,
            "post",
            f"/moderation/privacy/{ident}",
            user=1,
            decision="rejected",
            reason="请补充具体内容链接后核查",
        ).status_code
        == 200
    )
    notice = client.get(
        API + "/notifications?kind=moderation", headers=headers()
    ).json()["items"][0]
    assert (
        notice["path"] == "/settings" and notice["body"] == "请补充具体内容链接后核查"
    )
    assert (
        client.get(API + "/notification-status", headers=headers()).json()["unread"]
        == 1
    )
    client.put(f'{API}/notifications/read?through_id={notice["id"]}', headers=headers())
    assert (
        client.get(API + "/notification-status", headers=headers()).json()["unread"]
        == 0
    )


def test_external_card_revisions_withdrawal_and_fixed_source_paths(client):
    p = {
        "source": "competition",
        "source_id": "result-1",
        "revision": 1,
        "title": "官方赛果来源动态",
        "summary": "管理员核对的摘要",
        "path": "/events/1",
        "status": "published",
        "reason": "核查来源记录",
    }
    assert write(client, "put", "/moderation/external-cards", **p).status_code == 403
    assert (
        write(client, "put", "/moderation/external-cards", user=1, **p).status_code
        == 200
    )
    assert (
        write(client, "put", "/moderation/external-cards", user=1, **p).status_code
        == 409
    )
    assert (
        write(
            client,
            "put",
            "/moderation/external-cards",
            user=1,
            **{**p, "path": "//evil.test", "revision": 2},
        ).status_code
        == 400
    )
    assert len(client.get(API + "/external-cards").json()["items"]) == 1
    assert (
        write(
            client,
            "put",
            "/moderation/external-cards",
            user=1,
            **{**p, "status": "withdrawn", "revision": 2},
        ).status_code
        == 200
    )
    assert client.get(API + "/external-cards").json()["items"] == []


def test_move_needs_both_board_scopes_and_history_private(client):
    t = topic(client)
    boards = client.get(API + "/boards").json()["items"]
    strategy = next(b["id"] for b in boards if b["slug"] == "strategy")
    assert (
        write(
            client,
            "post",
            f'/topics/{t["topic_id"]}/move',
            board_id=strategy,
            revision=1,
            reason="作者不可自行移帖",
        ).status_code
        == 403
    )
    assert (
        write(
            client,
            "post",
            f'/topics/{t["topic_id"]}/move',
            user=1,
            board_id=strategy,
            revision=1,
            reason="移动至攻略讨论",
        ).status_code
        == 200
    )
    assert (
        client.get(f'{API}/topics/{t["topic_id"]}').json()["topic"]["board_id"]
        == strategy
    )
    write(
        client,
        "patch",
        f'/posts/{t["post_id"]}',
        body=document("修订后的内容"),
        revision=1,
    )
    assert (
        client.get(
            f'{API}/posts/{t["post_id"]}/revisions', headers=headers(3)
        ).status_code
        == 403
    )
    assert (
        len(
            client.get(
                f'{API}/posts/{t["post_id"]}/revisions', headers=headers()
            ).json()["items"]
        )
        == 1
    )


def test_hypothetical_branch_records_spawns_and_never_claims_official():
    p = {
        "variant": "2x4",
        "initial": [2, 2, 0, 0, 0, 0, 0, 0],
        "moves": [[3, 1, 2, 0]],
        "origin": "example@1",
    }
    result = assets.replay_payload(b"FBR1" + json.dumps(p).encode())
    assert result["verification"] == "hypothetical" and result["score"] == 4
    for invalid in [[[3, 0, 2, 0]], [[3, 1, 8, 0]], [[3, 1, 2, 30]], [[True, 1, 2, 0]]]:
        with pytest.raises(ForumError):
            assets.replay_payload(
                b"FBR1" + json.dumps({**p, "moves": invalid}).encode()
            )
    with pytest.raises(ForumError):
        assets.replay_payload(
            b"FBR1" + json.dumps({**p, "verification": "official"}).encode()
        )


def test_public_metadata_and_sitemap_hide_private_titles(client, app):
    t = topic(client, title=r"标题 <script>\g<evil>", body=document("不执行脚本的摘要"))
    response = client.get("/t/" + str(t["topic_id"]))
    assert (
        response.status_code == 200
        and "&lt;script&gt;" in response.text
        and "<script>\\g" not in response.text
    )
    assert f'/t/{t["topic_id"]}' in client.get("/sitemap.xml").text
    moderate(client, t["topic_id"], "hide")
    response = client.get("/t/" + str(t["topic_id"]))
    assert (
        response.status_code == 404
        and "noindex" in response.text
        and "不执行脚本的摘要" not in response.text
    )
    assert f'/t/{t["topic_id"]}' not in client.get("/sitemap.xml").text


def test_peer_ip_boundary_and_shared_bucket(app, settings):
    from dataclasses import replace
    from starlette.requests import Request
    from forum.backend.rate_limit import client_ip, allow

    request = Request(
        {
            "type": "http",
            "client": ("127.0.0.1", 1234),
            "headers": [
                (b"x-real-ip", b"192.0.2.3"),
                (b"x-forwarded-for", b"198.51.100.1"),
            ],
        }
    )
    assert client_ip(request, settings) == "127.0.0.1"
    assert (
        client_ip(request, replace(settings, trusted_proxies=("127.0.0.1/32",)))
        == "192.0.2.3"
    )
    assert allow(app.state.forum.engine, "192.0.2.3", True, "test-key")
    with app.state.forum.engine.begin() as conn:
        execute(conn, "UPDATE forum_ip_windows SET count=120")
    assert not allow(app.state.forum.engine, "192.0.2.3", True, "test-key")
    assert allow(app.state.forum.engine, "192.0.2.4", True, "test-key")


def test_bad_scheduled_announcement_does_not_poison_queue(client, app):
    payload = {
        "topic": {
            "board_slug": "announcements",
            "title": "失效的定时公告",
            "body": document("初始正文"),
        },
        "sites": ["forum"],
        "publish_at": deadline(-1),
        "schedule": True,
        "revision": 0,
    }
    first = write(
        client, "post", "/moderation/announcements", user=1, **payload
    ).json()["id"]
    second = write(
        client, "post", "/moderation/announcements", user=1, **payload
    ).json()["id"]
    with app.state.forum.engine.begin() as conn:
        execute(
            conn,
            "UPDATE forum_announcements SET payload=jsonb_set(payload,'{body}',CAST(:body AS jsonb)) WHERE id=:id",
            body=json.dumps(document(f"[[replay:{uuid4()}]]")),
            id=first,
        )
    assert tick(app.state.forum)
    assert tick(app.state.forum)
    assert not tick(app.state.forum)
    with app.state.forum.engine.connect() as conn:
        assert (
            one(conn, "SELECT status FROM forum_announcements WHERE id=:id", id=first)[
                "status"
            ]
            == "draft"
        )
        assert (
            one(conn, "SELECT status FROM forum_announcements WHERE id=:id", id=second)[
                "status"
            ]
            == "published"
        )
        assert one(conn, "SELECT count(*) AS n FROM forum_topics")["n"] == 1


def test_announcement_rechecks_changed_schedule_after_claim(client, app, monkeypatch):
    import forum.scheduler as scheduler

    payload = {
        "topic": {
            "board_slug": "announcements",
            "title": "改期的定时公告",
            "body": document("公告内容"),
        },
        "sites": ["forum"],
        "publish_at": deadline(-1),
        "schedule": True,
        "revision": 0,
    }
    ident = write(
        client, "post", "/moderation/announcements", user=1, **payload
    ).json()["id"]
    original = scheduler.one

    def race(conn, sql, **args):
        result = original(conn, sql, **args)
        if sql.startswith("SELECT id,author_id"):
            with app.state.forum.engine.begin() as other:
                execute(
                    other,
                    "UPDATE forum_announcements SET publish_at=now()+interval '1 day' WHERE id=:id",
                    id=ident,
                )
        return result

    monkeypatch.setattr(scheduler, "one", race)
    assert tick(app.state.forum)
    assert client.get(API + "/announcements").json()["items"] == []


def test_public_reply_history_excludes_hidden_and_deleted(client):
    t = topic(client)
    r = reply(client, t["topic_id"])
    assert client.get(API + "/profiles/3").json()["replies"][0]["id"] == r["post_id"]
    moderate(client, t["topic_id"], "hide")
    assert client.get(API + "/profiles/3").json()["replies"] == []


def test_staff_can_correct_locked_announcement_without_opening_replies(client):
    t = topic(client)
    moderate(client, t["topic_id"], "lock")
    path = f'/posts/{t["post_id"]}'
    assert (
        write(
            client, "patch", path, body=document("作者不能越过锁定"), revision=1
        ).status_code
        == 409
    )
    assert (
        write(
            client, "patch", path, user=1, body=document("版主修订公告正文"), revision=1
        ).status_code
        == 200
    )
    assert client.get(f'{API}/topics/{t["topic_id"]}').json()["topic"]["locked"]


def test_play_revised_source_cannot_retarget_step_anchor(client, monkeypatch):
    from .test_content import replay_fixture

    raw, _, _ = replay_fixture()
    checked = assets.replay_payload(raw)
    monkeypatch.setattr(assets, "fetch_play", lambda *_: checked)
    media = client.post(f"{API}/media/play/{uuid4()}", headers=headers()).json()
    topic(client, body=document(f'[[replay:{media["id"]}@1]]'))
    assert client.get(f'{API}/media/{media["id"]}/replay').status_code == 200
    checked["moves"] = checked["moves"][:-1]
    result = client.get(f'{API}/media/{media["id"]}/replay')
    assert (
        result.status_code == 409
        and result.json()["detail"]["code"] == "SOURCE_REVISED"
    )
