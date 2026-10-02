from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from fastapi.testclient import TestClient
from .conftest import headers, document, topic
from forum.backend.mentions import recipients
from forum.worker import deliver_one

ROOT = "/api/forum/v1"


def write(client, method, path, user=2, **body):
    return client.request(
        method.upper(), ROOT + path, headers=headers(user), json=body or None
    )


def reply(client, ident, user=3, text="回复讨论内容"):
    r = client.post(
        f"{ROOT}/topics/{ident}/posts",
        headers=headers(user, uuid4()),
        json={"body": document(text)},
    )
    assert r.status_code == 201, r.text
    return r.json()


def moderate(client, ident, action):
    r = write(
        client,
        "post",
        f"/topics/{ident}/moderation",
        user=1,
        action=action,
        reason="管理测试说明",
    )
    assert r.status_code == 200, r.text


def test_mentions_syntax():
    assert recipients(
        document(
            r"<@2> **<@3>** `<@4>` \<@5> [<@6>](https://example.org) ![<@7>](/image)"
        )
    ) == {2, 3}
    assert (
        recipients(document("```\n<@2>\n```\n\n    <@3>\n\n<@0> <@1234567890123456>"))
        == set()
    )


def test_mentions_unique_edits_preferences_and_limit(client, app):
    topic(client, user=3)
    t = topic(client, body=document("邀请 <@3> 与 <@3>"))
    notifications = client.get(ROOT + "/notifications", headers=headers(3)).json()[
        "items"
    ]
    assert len(notifications) == 1 and notifications[0]["kind"] == "mention"
    write(
        client,
        "patch",
        f'/posts/{t["post_id"]}',
        body=document("暂时去掉提及"),
        revision=1,
    )
    write(
        client,
        "patch",
        f'/posts/{t["post_id"]}',
        body=document("重新 <@3>"),
        revision=2,
    )
    assert (
        len(client.get(ROOT + "/notifications", headers=headers(3)).json()["items"])
        == 1
    )
    write(client, "put", "/notification-preferences", user=3, enabled=False)
    topic(client, body=document("新主题 <@3>"))
    assert (
        len(client.get(ROOT + "/notifications", headers=headers(3)).json()["items"])
        == 1
    )
    response = client.post(
        ROOT + "/topics",
        headers=headers(2, uuid4()),
        json={
            "board_slug": "general",
            "title": "过多用户提及",
            "body": document(" ".join(f"<@{x}>" for x in range(1, 12))),
        },
    )
    assert (
        response.status_code == 400
        and response.json()["detail"]["code"] == "MENTION_LIMIT"
    )


def test_follow_delivery_is_private_and_not_retroactive(client, app):
    old = topic(client)
    assert write(client, "put", "/follows/2", user=3).status_code == 200
    assert write(client, "put", "/follows/3", user=3).status_code == 400
    assert client.get(ROOT + "/follows").status_code == 401
    assert client.get(ROOT + "/follows", headers=headers(2)).json()["items"] == []
    new = topic(client)
    while deliver_one(app.state.forum.engine):
        pass
    n = client.get(ROOT + "/notifications", headers=headers(3)).json()["items"]
    assert [(x["post_id"], x["kind"]) for x in n] == [(new["post_id"], "follow")]
    assert client.get(ROOT + "/profiles/2", headers=headers(3)).json()["following"]
    moderate(client, new["topic_id"], "hide")
    public = client.get(ROOT + "/profiles/2").json()
    assert [t["id"] for t in public["topics"]] == [old["topic_id"]]
    write(client, "delete", "/follows/2", user=3)
    topic(client)
    while deliver_one(app.state.forum.engine):
        pass
    assert (
        len(client.get(ROOT + "/notifications", headers=headers(3)).json()["items"])
        == 1
    )


def test_reply_and_mentions_subscriptions_deduplicate(client, app):
    t = topic(client)
    write(client, "put", f'/subscriptions/topic/{t["topic_id"]}')
    r = reply(client, t["topic_id"], text="这里 <@2> 继续讨论")
    while deliver_one(app.state.forum.engine):
        pass
    n = client.get(ROOT + "/notifications", headers=headers()).json()["items"]
    assert len(n) == 1 and n[0]["kind"] == "mention" and n[0]["post_id"] == r["post_id"]


def test_reading_monotonic_private_validates_topic_and_visibility(client):
    t = topic(client)
    r = reply(client, t["topic_id"])
    path = f'/topics/{t["topic_id"]}/reading'
    assert write(client, "put", path, post_id=r["post_id"]).status_code == 200
    assert write(client, "put", path, post_id=t["post_id"]).status_code == 200
    assert (
        client.get(ROOT + path, headers=headers()).json()["position"]["post_number"]
        == 2
    )
    assert client.get(ROOT + path, headers=headers(3)).json()["position"] is None
    other = topic(client)
    assert write(client, "put", path, post_id=other["post_id"]).status_code == 404
    moderate(client, t["topic_id"], "hide")
    assert client.get(ROOT + "/reading", headers=headers()).json()["items"] == []
    assert client.get(ROOT + path, headers=headers(3)).status_code == 404


def test_reply_draft_conflicts_clear_and_private(client):
    t = topic(client)
    path = f'/topics/{t["topic_id"]}/reply-draft'
    assert (
        write(
            client, "put", path, text="草稿第一版", revision=0, reply_to=t["post_id"]
        ).json()["revision"]
        == 1
    )
    assert client.get(ROOT + path, headers=headers(3)).json()["draft"] is None
    assert (
        write(client, "put", path, text="过期设备内容", revision=0).status_code == 409
    )
    assert write(client, "put", path, text="", revision=1).json()["revision"] == 2
    assert (
        write(client, "put", path, text="旧设备不能复活内容", revision=0).status_code
        == 409
    )
    other = topic(client)
    assert (
        write(
            client,
            "put",
            path,
            text="跨主题指定回复",
            revision=2,
            reply_to=other["post_id"],
        ).status_code
        == 400
    )


def test_muted_user_can_appeal_and_save_draft_admin_can_reverse(client):
    t = topic(client)
    assert (
        write(
            client,
            "post",
            "/moderation/users/2",
            user=1,
            action="mute",
            reason="禁言测试原因",
        ).status_code
        == 200
    )
    actions = client.get(ROOT + "/appeal-actions", headers=headers()).json()["items"]
    assert len(actions) == 1 and actions[0]["action"] == "mute"
    assert (
        write(
            client,
            "put",
            f'/topics/{t["topic_id"]}/reply-draft',
            text="禁言期间暂存",
            revision=0,
        ).status_code
        == 200
    )
    a = write(
        client, "post", "/appeals", action_id=actions[0]["id"], reason="请求重新核查"
    )
    assert a.status_code == 201, a.text
    assert (
        write(
            client,
            "post",
            "/appeals",
            action_id=actions[0]["id"],
            reason="重复提交申诉",
        ).status_code
        == 409
    )
    assert client.get(ROOT + "/appeals", headers=headers(3)).json()["items"] == []
    assert (
        write(
            client,
            "post",
            f'/moderation/appeals/{a.json()["id"]}',
            user=3,
            decision="accepted",
            reason="越权处理尝试",
        ).status_code
        == 403
    )
    accepted = write(
        client,
        "post",
        f'/moderation/appeals/{a.json()["id"]}',
        user=1,
        decision="accepted",
        reason="核查后解除禁言",
    )
    assert accepted.status_code == 200, accepted.text
    reply(client, t["topic_id"], user=2)
    assert (
        client.get(ROOT + "/appeals", headers=headers()).json()["items"][0]["status"]
        == "accepted"
    )
    assert (
        write(
            client,
            "post",
            f'/moderation/appeals/{a.json()["id"]}',
            user=1,
            decision="rejected",
            reason="重复处理尝试",
        ).status_code
        == 409
    )


def test_appeal_scoped_and_stale_decision_never_overwrites_new_moderation(client):
    t = topic(client)
    topic(client, user=4)
    # Obtain the actual general board ID rather than relying on seed ordering.
    board = next(
        b["id"]
        for b in client.get(ROOT + "/boards").json()["items"]
        if b["slug"] == "general"
    )
    write(
        client,
        "post",
        "/moderation/users/4",
        user=1,
        action="grant",
        board_id=board,
        reason="授予版主测试",
    )
    moderate(client, t["topic_id"], "hide")
    action = client.get(ROOT + "/appeal-actions", headers=headers()).json()["items"][0]
    assert (
        write(
            client,
            "post",
            "/appeals",
            user=3,
            action_id=action["id"],
            reason="不是主题作者",
        ).status_code
        == 404
    )
    a = write(
        client, "post", "/appeals", action_id=action["id"], reason="请核查隐藏原因"
    ).json()["id"]
    assert (
        client.get(ROOT + "/moderation/appeals", headers=headers(4)).json()["items"][0][
            "id"
        ]
        == a
    )
    moderate(client, t["topic_id"], "hide")
    assert (
        write(
            client,
            "post",
            f"/moderation/appeals/{a}",
            user=4,
            decision="accepted",
            reason="不能覆盖新决定",
        ).status_code
        == 409
    )
    assert (
        write(
            client,
            "post",
            f"/moderation/appeals/{a}",
            user=4,
            decision="rejected",
            reason="已有新处置请另行申诉",
        ).status_code
        == 200
    )
    assert client.get(f'{ROOT}/topics/{t["topic_id"]}').status_code == 404


def test_hidden_and_locked_can_be_appealed_independently(client):
    t = topic(client)
    moderate(client, t["topic_id"], "hide")
    moderate(client, t["topic_id"], "lock")
    actions = client.get(ROOT + "/appeal-actions", headers=headers()).json()["items"]
    assert {a["action"] for a in actions} == {"hide", "lock"}
    for action in actions:
        a = write(
            client, "post", "/appeals", action_id=action["id"], reason="申请重新核查"
        ).json()["id"]
        assert (
            write(
                client,
                "post",
                f"/moderation/appeals/{a}",
                user=1,
                decision="accepted",
                reason="通过申诉恢复状态",
            ).status_code
            == 200
        )
    public = client.get(f'{ROOT}/topics/{t["topic_id"]}').json()["topic"]
    assert public["status"] == "published" and not public["locked"]


def test_concurrent_draft_saves_have_one_winner(client, app):
    t = topic(client)
    path = f'/topics/{t["topic_id"]}/reply-draft'
    barrier = Barrier(2)

    def save(value):
        with TestClient(app) as second:
            barrier.wait(timeout=5)
            return write(second, "put", path, text=value, revision=0).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(save, ["设备甲的草稿", "设备乙的草稿"]))
    assert sorted(results) == [200, 409]
    draft = client.get(ROOT + path, headers=headers()).json()["draft"]
    assert draft["revision"] == 1 and draft["text"] in ["设备甲的草稿", "设备乙的草稿"]


def test_appeals_cannot_cross_board_or_self_review(client):
    t = topic(client, user=4)
    topic(client, user=3)
    boards = client.get(ROOT + "/boards").json()["items"]
    for user, slug in [(4, "general"), (3, "strategy")]:
        board = next(b["id"] for b in boards if b["slug"] == slug)
        assert (
            write(
                client,
                "post",
                f"/moderation/users/{user}",
                user=1,
                action="grant",
                board_id=board,
                reason="测试板块授权",
            ).status_code
            == 200
        )
    moderate(client, t["topic_id"], "lock")
    action = client.get(ROOT + "/appeal-actions", headers=headers(4)).json()["items"][0]
    a = write(
        client,
        "post",
        "/appeals",
        user=4,
        action_id=action["id"],
        reason="请求核查主题锁定",
    ).json()["id"]
    assert (
        client.get(ROOT + "/moderation/appeals", headers=headers(3)).json()["items"]
        == []
    )
    assert (
        write(
            client,
            "post",
            f"/moderation/appeals/{a}",
            user=3,
            decision="accepted",
            reason="不能跨板块处理",
        ).status_code
        == 403
    )
    assert (
        write(
            client,
            "post",
            f"/moderation/appeals/{a}",
            user=4,
            decision="accepted",
            reason="不能审核自身申诉",
        ).status_code
        == 403
    )


def test_old_sanction_appeal_cannot_clear_replacement_sanction(client):
    topic(client)
    write(
        client,
        "post",
        "/moderation/users/2",
        user=1,
        action="mute",
        reason="第一条禁言记录",
    )
    action = client.get(ROOT + "/appeal-actions", headers=headers()).json()["items"][0]
    a = write(
        client, "post", "/appeals", action_id=action["id"], reason="对首次禁言申诉"
    ).json()["id"]
    write(
        client,
        "post",
        "/moderation/users/2",
        user=1,
        action="mute",
        reason="后续新的禁言记录",
        hours=48,
    )
    response = write(
        client,
        "post",
        f"/moderation/appeals/{a}",
        user=1,
        decision="accepted",
        reason="不能撤销后续禁言",
    )
    assert response.status_code == 409
    latest = client.get(ROOT + "/appeal-actions", headers=headers()).json()["items"]
    assert len(latest) == 1 and latest[0]["id"] != action["id"]


def test_reply_draft_inventory_and_removal_when_topic_becomes_hidden(client):
    t = topic(client)
    path = f'/topics/{t["topic_id"]}/reply-draft'
    write(client, "put", path, user=3, text="属于我自己的回复草稿", revision=0)
    assert client.get(ROOT + "/reply-drafts", headers=headers()).json()["items"] == []
    moderate(client, t["topic_id"], "hide")
    items = client.get(ROOT + "/reply-drafts", headers=headers(3)).json()["items"]
    assert (
        len(items) == 1
        and not items[0]["available"]
        and items[0]["title"] == "主题已不可见"
    )
    clear = f'/reply-drafts/{t["topic_id"]}?revision=1'
    assert write(client, "delete", clear, user=2).status_code == 409
    assert write(client, "delete", clear, user=3).status_code == 200
    assert client.get(ROOT + "/reply-drafts", headers=headers(3)).json()["items"] == []
    moderate(client, t["topic_id"], "restore")
    assert (
        write(
            client, "put", path, user=3, text="旧设备不能还原已清空草稿", revision=1
        ).status_code
        == 409
    )
