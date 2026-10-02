from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from forum.backend.app import create_app
from forum.backend.config import Settings
from forum.backend.db import all_rows, execute, one
from forum.backend.schemas import Document
from .conftest import document, headers, topic

API = "/api/forum/v1"


def test_ready_and_board_permissions(client):
    assert client.get("/health/ready").json()["database"] == "postgresql"
    boards = client.get(API + "/boards").json()["items"]
    assert len(boards) == 6 and all(not b["can_post"] for b in boards)
    user_boards = client.get(API + "/boards", headers=headers()).json()["items"]
    assert not user_boards[0]["can_post"] and user_boards[1]["can_post"]
    assert client.get(API + "/boards").headers["cache-control"] == "private, no-store"


def test_only_postgres_and_no_production_dev_auth(settings):
    with pytest.raises(ValueError):
        replace(settings, database_url="sqlite://")
    with pytest.raises(ValueError):
        replace(settings, environment="production", public_origin="https://forum.2048tables.online")


def test_anonymous_and_staff_board_protected(client):
    payload={"board_slug":"general","title":"匿名的主题","body":document()}
    assert client.post(API+"/topics",json=payload,headers={"Origin":"http://testserver","Idempotency-Key":str(uuid4())}).status_code==401
    payload["board_slug"]="announcements"
    assert client.post(API+"/topics",json=payload,headers=headers(2,uuid4())).status_code==403
    assert client.post(API+"/topics",json=payload,headers=headers(1,uuid4())).status_code==201


def test_topic_and_reply_idempotency(client, app):
    payload={"board_slug":"general","title":"重试不会重复","body":document()}
    key=uuid4()
    a=client.post(API+"/topics",json=payload,headers=headers(2,key))
    b=client.post(API+"/topics",json=payload,headers=headers(2,key))
    assert a.status_code==b.status_code==201 and a.json()==b.json()
    payload["title"]="不同内容不能重用标识"
    assert client.post(API+"/topics",json=payload,headers=headers(2,key)).status_code==409
    tid=a.json()["topic_id"]
    reply_key=uuid4()
    for _ in range(2):
        assert client.post(f"{API}/topics/{tid}/posts",json={"body":document()},headers=headers(3,reply_key)).status_code==201
    result=client.get(f"{API}/topics/{tid}").json()
    assert len(result["posts"])==2
    assert len(client.get(API+"/notifications",headers=headers(2)).json()["items"])==1
    with app.state.forum.engine.connect() as conn:
        assert one(conn,"SELECT count(*) AS n FROM forum_outbox")["n"]==2


def test_parallel_replies_get_unique_monotone_numbers(client):
    tid=topic(client)["topic_id"]
    def write(i):
        r=client.post(f"{API}/topics/{tid}/posts",json={"body":document(str(i))},headers=headers(100+i,uuid4()))
        assert r.status_code==201,r.text
        return r.json()["post_number"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        numbers=list(pool.map(write,range(8)))
    assert sorted(numbers)==list(range(2,10))


def test_parallel_same_submission_creates_once(client):
    tid=topic(client)["topic_id"];key=uuid4()
    def write(_):
        r=client.post(f"{API}/topics/{tid}/posts",json={"body":document()},headers=headers(5,key))
        assert r.status_code==201,r.text
        return r.json()["post_id"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        ids=list(pool.map(write,range(6)))
    assert len(set(ids))==1


def test_edit_conflict_authorization_and_tombstone(client):
    tid=topic(client)["topic_id"]
    response=client.post(f"{API}/topics/{tid}/posts",json={"body":document("旧正文")},headers=headers(3,uuid4())).json()
    pid=response["post_id"]
    patch={"revision":1,"body":document("新正文")}
    assert client.patch(f"{API}/posts/{pid}",json=patch,headers=headers(4)).status_code==403
    assert client.patch(f"{API}/posts/{pid}",json=patch,headers=headers(3)).status_code==200
    assert client.patch(f"{API}/posts/{pid}",json=patch,headers=headers(3)).status_code==409
    assert client.delete(f"{API}/posts/{pid}?revision=2",headers=headers(3)).status_code==200
    post=client.get(f"{API}/topics/{tid}").json()["posts"][1]
    assert post["body"] is None and post["post_number"]==2
    assert "旧正文" not in str(post) and "新正文" not in str(post)


def test_reply_cannot_reference_other_topic(client):
    a=topic(client);b=topic(client)
    r=client.post(f"{API}/topics/{a['topic_id']}/posts",json={"body":document(),"reply_to":b["post_id"]},headers=headers(3,uuid4()))
    assert r.status_code==400


def test_draft_owner_isolation_and_optimistic_revision(client):
    did=uuid4();payload={"revision":0,"title":"我的私密草稿","text":"不能泄露"}
    a=client.put(f"{API}/drafts/{did}",json=payload,headers=headers(2));assert a.status_code==200
    assert client.get(API+"/drafts",headers=headers(3)).json()["items"]==[]
    assert client.put(f"{API}/drafts/{did}",json=payload,headers=headers(2)).status_code==409
    assert client.delete(f"{API}/drafts/{did}?revision=1",headers=headers(3)).status_code==409
    payload["revision"]=1
    assert client.put(f"{API}/drafts/{did}",json=payload,headers=headers(2)).json()["revision"]==2
    assert client.delete(f"{API}/drafts/{did}?revision=2",headers=headers(2)).status_code==200


def test_likes_bookmarks_are_idempotent_and_private(client):
    t=topic(client)
    assert client.put(f"{API}/posts/{t['post_id']}/like",headers=headers(2)).status_code==400
    for _ in range(2):
        assert client.put(f"{API}/posts/{t['post_id']}/like",headers=headers(3)).status_code==200
        assert client.put(f"{API}/topics/{t['topic_id']}/bookmark",headers=headers(3)).status_code==200
    assert client.get(f"{API}/topics/{t['topic_id']}").json()["posts"][0]["likes"]==1
    assert len(client.get(API+"/topics?saved=true",headers=headers(3)).json()["items"])==1
    assert client.get(API+"/topics?saved=true",headers=headers(2)).json()["items"]==[]
    assert client.get(API+"/topics?saved=true").status_code==401


def test_hidden_content_search_notification_and_audit(client,app):
    t=topic(client,title="不应泄露的秘密标题")
    client.post(f"{API}/topics/{t['topic_id']}/posts",json={"body":document()},headers=headers(3,uuid4()))
    client.post(f"{API}/posts/{t['post_id']}/reports",json={"reason":"包含隐私信息"},headers=headers(3))
    assert client.post(f"{API}/topics/{t['topic_id']}/moderation",json={"action":"hide","reason":"包含隐私信息"},headers=headers(3)).status_code==403
    assert client.post(f"{API}/topics/{t['topic_id']}/moderation",json={"action":"hide","reason":"包含隐私信息"},headers=headers(1)).status_code==200
    assert client.get(f"{API}/topics/{t['topic_id']}").status_code==404
    assert client.get(API+"/topics?q=秘密").json()["items"]==[]
    assert client.get(f"{API}/topics/{t['topic_id']}",headers=headers(2)).status_code==200
    notice=client.get(API+"/notifications",headers=headers(2)).json()["items"][0]
    assert not notice["available"] and notice["title"]=="内容已不可见"
    assert client.get(API+"/moderation/reports",headers=headers(1)).json()["items"]==[]
    with app.state.forum.engine.connect() as conn:
        assert one(conn,"SELECT count(*) AS n FROM forum_moderation_actions")["n"]==1


def test_lock_and_scoped_moderator(client,app):
    a=topic(client);b=topic(client,board_slug="strategy")
    with app.state.forum.engine.begin() as conn:
        execute(conn,"INSERT INTO forum_roles(user_id,board_id,role) SELECT 5,id,'moderator' FROM forum_boards WHERE slug='general'")
    body={"action":"lock","reason":"暂时锁定讨论"}
    assert client.post(f"{API}/topics/{a['topic_id']}/moderation",json=body,headers=headers(5)).status_code==200
    assert client.post(f"{API}/topics/{b['topic_id']}/moderation",json=body,headers=headers(5)).status_code==403
    assert client.post(f"{API}/topics/{a['topic_id']}/posts",json={"body":document()},headers=headers(2,uuid4())).status_code==409


def test_chinese_search_literal_wildcards_and_pagination(client):
    a=topic(client,title="L3 残局复盘")
    topic(client,title="正常标题讨论",body=document("100% 进度"))
    topic(client,title="另一个主题",body=document("另一段交流"))
    for query in ("残局","复盘","L3"):
        assert client.get(API+"/topics",params={"q":query}).json()["items"][0]["id"]==a["topic_id"]
    assert len(client.get(API+"/topics",params={"q":"%"}).json()["items"])==1
    page1=client.get(API+"/topics?limit=2").json()
    page2=client.get(API+"/topics",params={"limit":2,"cursor":page1["next_cursor"]}).json()
    assert len(page1["items"])+len(page2["items"])==3
    assert not set(t["id"] for t in page1["items"])&set(t["id"] for t in page2["items"])
    assert client.get(API+"/topics?cursor=garbage").status_code==400


@pytest.mark.parametrize("method,path",[("post","/topics"),("patch","/posts/1"),("put","/drafts/"+str(uuid4())),("delete","/posts/1?revision=1")])
def test_csrf_covers_all_mutations(client,method,path):
    assert getattr(client,method)(API+path,headers={**headers(2),"Origin":"https://evil.2048tables.online"}).status_code==403
    assert getattr(client,method)(API+path,headers={"X-Forum-Dev-User":"2:User"}).status_code==403


def test_body_limit_and_structured_content(client):
    assert client.post(API+"/topics",content=b"x"*131073,headers=headers(2,uuid4())).status_code==413
    t=topic(client,body=document("<script>alert(1)</script>"))
    assert client.get(f"{API}/topics/{t['topic_id']}").json()["posts"][0]["body"]["blocks"][0]["text"]=="<script>alert(1)</script>"
    with pytest.raises(ValidationError):
        Document.model_validate({"version":1,"blocks":[{"type":"html","html":"<script>"}]})
    with pytest.raises(ValidationError):
        Document.model_validate({"blocks":[{"type":"board","rows":4,"cols":4,"cells":[3]*16}]})
    for rows,cols in [(4,4),(3,4),(3,3),(2,4)]:
        doc={"version":1,"blocks":[{"type":"board","rows":rows,"cols":cols,"cells":[2]*(rows*cols)}]}
        topic(client,body=doc)


def test_rate_limit_is_shared_by_database(client,app):
    with app.state.forum.engine.begin() as conn:
        execute(conn,"INSERT INTO forum_rate_windows(user_id,count) VALUES(2,60)")
    payload={"title":"被限流的主题","board_slug":"general","body":document()}
    r=client.post(API+"/topics",json=payload,headers=headers(2,uuid4()))
    assert r.status_code==429 and r.headers["retry-after"]=="60"


def test_anchor_beyond_first_page_and_wrong_topic(client):
    t=topic(client)
    last=None
    for _ in range(43):
        r=client.post(f"{API}/topics/{t['topic_id']}/posts",json={"body":document()},headers=headers(3,uuid4()))
        assert r.status_code==201
        last=r.json()["post_id"]
    result=client.get(f"{API}/topics/{t['topic_id']}?focus_post={last}").json()
    assert result["start_after"]>0 and any(p["id"]==last for p in result["posts"])
    other=topic(client)
    assert client.get(f"{API}/topics/{other['topic_id']}?focus_post={last}").status_code==404


def test_read_notifications_cannot_modify_another_inbox(client):
    a=topic(client,user=2)
    b=topic(client,user=3)
    for t in (a,b):
        client.post(f"{API}/topics/{t['topic_id']}/posts",json={"body":document()},headers=headers(4,uuid4()))
    notices=client.get(API+"/notifications",headers=headers(3)).json()["items"]
    client.put(f"{API}/notifications/read?through_id={notices[0]['id']}",headers=headers(2))
    assert client.get(API+"/notifications",headers=headers(3)).json()["items"][0]["read_at"] is None


def test_muted_user_and_untrusted_verification_field(client,app):
    with app.state.forum.engine.begin() as conn:
        execute(conn,"INSERT INTO forum_sanctions VALUES(2,'限时禁言',now()+interval '1 hour',1)")
    payload={"title":"被禁言的主题","board_slug":"general","body":document()}
    assert client.post(API+"/topics",json=payload,headers=headers(2,uuid4())).status_code==403
    payload["body"]={"version":1,"blocks":[{"type":"board","rows":4,"cols":4,"cells":[2]*16,"verified":True}]}
    assert client.post(API+"/topics",json=payload,headers=headers(3,uuid4())).status_code==422


def test_board_syntax_source_survives_publish_search_edit_and_reply(client):
    # Rendering is a reader concern; PostgreSQL retains editable source verbatim.
    source = "讨论这个局面\n[[board:4x4:fedc/ba98/7654/3210|下一步]]\n`[[board:3x3:1]]`"
    t = topic(client, body=document(source))
    detail = client.get(f"{API}/topics/{t['topic_id']}").json()
    assert detail["posts"][0]["body"] == document(source)
    assert len(client.get(API+"/topics",params={"q":"下一步"}).json()["items"]) == 1
    edited = source.replace("下一步", "修订说明")
    r = client.patch(f"{API}/posts/{t['post_id']}",json={"revision":1,"body":document(edited)},headers=headers(2))
    assert r.status_code == 200
    r = client.post(f"{API}/topics/{t['topic_id']}/posts",json={"body":document("[[board:2x4:1234/5678]]")},headers=headers(3,uuid4()))
    assert r.status_code == 201
    detail = client.get(f"{API}/topics/{t['topic_id']}").json()
    assert detail["posts"][0]["body"] == document(edited)
    assert detail["posts"][1]["body"] == document("[[board:2x4:1234/5678]]")


def test_real_shared_session_revocation_and_disabled_account(app,settings,tmp_path):
    path=tmp_path/"auth.sqlite3";token="test-session-token"
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE users(id INTEGER PRIMARY KEY,display_name TEXT,status TEXT); CREATE TABLE sessions(user_id INTEGER,session_token_hash TEXT,expires_at TEXT,revoked_at TEXT);")
        conn.execute("INSERT INTO users VALUES (90,'真实适配测试','active')")
        conn.execute("INSERT INTO sessions VALUES (?,?,?,NULL)",(90,hashlib.sha256(token.encode()).hexdigest(),(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()))
    with TestClient(create_app(replace(settings,allow_dev_auth=False,auth_db=path))) as client:
        client.cookies.set("tb_shared_session",token)
        assert client.get(API+"/session").json()["user"]["id"]==90
        with sqlite3.connect(path) as conn:conn.execute("UPDATE users SET status='disabled'")
        assert client.get(API+"/session").json()["user"] is None
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE users SET status='active'")
            conn.execute("UPDATE sessions SET revoked_at='revoked'")
        assert client.get(API+"/session").json()["user"] is None
        # A caller-provided dev header never authenticates when the flag is off.
        assert client.get(API+"/session",headers={"X-Forum-Dev-User":"1:admin"}).json()["user"] is None
