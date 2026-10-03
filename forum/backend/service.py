from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json

from .db import all_rows, execute, one
from .errors import ForumError
from .community import CommunityFeatures
from . import assets, mentions
from .social import SocialFeatures
from .operations import Operations
from .discovery import search_topics
from contextlib import nullcontext


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def document_text(body):
    return "\n".join(b.get("text", b.get("caption", "")) for b in body["blocks"])


def cursor_encode(row):
    raw = encoded(
        [row["last_activity"].isoformat(), row["id"], row.get("pinned", False)]
    )
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def cursor_decode(value):
    try:
        parsed = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
        stamp, ident, pinned = (*parsed, False) if len(parsed) == 2 else parsed
        stamp = datetime.fromisoformat(stamp)
        if (
            not stamp.tzinfo
            or type(ident) is not int
            or not 0 < ident < 2**63
            or type(pinned) is not bool
        ):
            raise ValueError()
        return stamp, ident, pinned
    except (ValueError, TypeError, OverflowError) as exc:
        raise ForumError("INVALID_CURSOR", "分页游标无效。") from exc


class ForumService(CommunityFeatures, SocialFeatures, Operations):
    def __init__(self, engine, settings):
        self.engine, self.settings = engine, settings

    def is_admin(self, user):
        return user is not None and user.id in self.settings.admin_ids

    def moderate(self, conn, user, board_id):
        return self.is_admin(user) or bool(
            user
            and one(
                conn,
                "SELECT 1 FROM forum_roles WHERE user_id=:u AND board_id=:b",
                u=user.id,
                b=board_id,
            )
        )

    def writer(self, conn, user, allow_muted=False):
        # Stable lock order: actor -> topic -> post. Works across API processes.
        execute(conn, "SELECT pg_advisory_xact_lock(:u)", u=user.id)
        if not allow_muted and one(
            conn,
            "SELECT 1 FROM forum_sanctions WHERE user_id=:u AND expires_at>now()",
            u=user.id,
        ):
            raise ForumError("MUTED", "当前账号暂时不能在社区发布内容。", 403)
        rate = one(
            conn,
            """INSERT INTO forum_rate_windows(user_id,count) VALUES (:u,1)
            ON CONFLICT(user_id) DO UPDATE SET
            count=CASE WHEN forum_rate_windows.window_started<now()-interval '1 minute' THEN 1 ELSE forum_rate_windows.count+1 END,
            window_started=CASE WHEN forum_rate_windows.window_started<now()-interval '1 minute' THEN now() ELSE forum_rate_windows.window_started END
            RETURNING count""",
            u=user.id,
        )
        if rate["count"] > 60:
            raise ForumError("RATE_LIMITED", "操作过于频繁，请稍后再试。", 429)
        execute(
            conn,
            """INSERT INTO forum_profiles(user_id,display_name) VALUES (:u,:name)
            ON CONFLICT(user_id) DO UPDATE SET display_name=excluded.display_name,updated_at=now()
            WHERE forum_profiles.display_name<>excluded.display_name""",
            u=user.id,
            name=user.display_name,
        )

    def event(self, conn, kind, ident, payload):
        execute(
            conn,
            """INSERT INTO forum_outbox(kind,aggregate_id,payload)
            VALUES (:kind,:id,CAST(:data AS jsonb))""",
            kind=kind,
            id=ident,
            data=encoded(payload),
        )

    def idempotent(self, conn, user, key, operation, payload, action):
        self.writer(conn, user)
        fingerprint = hashlib.sha256(encoded([operation, payload]).encode()).hexdigest()
        previous = one(
            conn,
            "SELECT * FROM forum_idempotency WHERE user_id=:u AND key=:key",
            u=user.id,
            key=key,
        )
        if previous:
            if previous["fingerprint"] != fingerprint:
                raise ForumError(
                    "IDEMPOTENCY_CONFLICT", "此提交标识已用于不同的内容。", 409
                )
            return previous["result"]
        result = action()
        execute(
            conn,
            """INSERT INTO forum_idempotency(user_id,key,fingerprint,result)
            VALUES (:u,:key,:fp,CAST(:result AS jsonb))""",
            u=user.id,
            key=key,
            fp=fingerprint,
            result=encoded(result),
        )
        return result

    def topic(self, conn, topic_id, user, lock=False):
        row = one(
            conn,
            "SELECT * FROM forum_topics WHERE id=:id" + (" FOR UPDATE" if lock else ""),
            id=topic_id,
        )
        if not row or (
            row["status"] != "published"
            and not (
                user
                and (
                    row["author_id"] == user.id
                    or self.moderate(conn, user, row["board_id"])
                )
            )
        ):
            raise ForumError("TOPIC_NOT_FOUND", "主题不存在或不可见。", 404)
        return row

    def post(self, conn, post_id, user, lock=False):
        target = one(conn, "SELECT topic_id FROM forum_posts WHERE id=:id", id=post_id)
        if not target:
            raise ForumError("POST_NOT_FOUND", "回复不存在。", 404)
        topic = self.topic(conn, target["topic_id"], user, lock)
        post = one(
            conn,
            "SELECT * FROM forum_posts WHERE id=:id" + (" FOR UPDATE" if lock else ""),
            id=post_id,
        )
        return topic, post

    def writable_topic(self, topic):
        if topic["locked"] or topic["status"] != "published":
            raise ForumError("TOPIC_CLOSED", "该主题已锁定或隐藏。", 409)

    def boards(self, user):
        with self.engine.connect() as conn:
            return [
                dict(
                    b,
                    can_post=bool(
                        user
                        and (not b["staff_only"] or self.moderate(conn, user, b["id"]))
                    ),
                )
                for b in all_rows(
                    conn, "SELECT * FROM forum_boards ORDER BY position,id"
                )
            ]

    def list_topics(
        self, user, board="", cursor="", query="", saved=False, limit=20, **filters
    ):
        return search_topics(self, user, board, cursor, query, saved, limit, **filters)

    def create_topic(self, user, key, payload, connection=None):
        with (
            nullcontext(connection) if connection is not None else self.engine.begin()
        ) as conn:

            def action():
                board = one(
                    conn,
                    "SELECT * FROM forum_boards WHERE slug=:slug",
                    slug=payload["board_slug"],
                )
                if not board:
                    raise ForumError("BOARD_NOT_FOUND", "板块不存在。", 404)
                if board["staff_only"] and not self.moderate(conn, user, board["id"]):
                    raise ForumError(
                        "BOARD_RESTRICTED", "此板块仅允许管理人员发布主题。", 403
                    )
                topic = one(
                    conn,
                    """INSERT INTO forum_topics(board_id,author_id,title,tags,kind)
                    VALUES (:b,:u,:title,CAST(:tags AS jsonb),:kind) RETURNING id""",
                    b=board["id"],
                    u=user.id,
                    title=payload["title"],
                    tags=encoded(payload["tags"]),
                    kind=payload.get("kind", "discussion"),
                )
                self.create_poll(conn, topic["id"], payload)
                post = one(
                    conn,
                    """INSERT INTO forum_posts(topic_id,author_id,post_number,body,body_text)
                    VALUES (:t,:u,1,CAST(:body AS jsonb),:txt) RETURNING id""",
                    t=topic["id"],
                    u=user.id,
                    body=encoded(payload["body"]),
                    txt=document_text(payload["body"]),
                )
                self.event(conn, "topic.created", topic["id"], {"post_id": post["id"]})
                assets.bind(conn, post["id"], payload["body"], user)
                mentions.notify(conn, post["id"], payload["body"], user)
                return {"topic_id": topic["id"], "post_id": post["id"]}

            return self.idempotent(conn, user, key, "topic.create", payload, action)

    def detail(
        self, topic_id, user, after=0, limit=40, focus_post=None, author_only=False
    ):
        with self.engine.connect() as conn:
            topic = dict(self.topic(conn, topic_id, user))
            board = one(
                conn,
                "SELECT slug,name FROM forum_boards WHERE id=:id",
                id=topic["board_id"],
            )
            topic.update(
                board_slug=board["slug"],
                board_name=board["name"],
                can_moderate=self.moderate(conn, user, topic["board_id"]),
                bookmarked=bool(
                    user
                    and one(
                        conn,
                        "SELECT 1 FROM forum_bookmarks WHERE user_id=:u AND topic_id=:t",
                        u=user.id,
                        t=topic_id,
                    )
                ),
            )
            if focus_post is not None:
                focus = one(
                    conn,
                    "SELECT post_number FROM forum_posts WHERE id=:p AND topic_id=:t",
                    p=focus_post,
                    t=topic_id,
                )
                if not focus:
                    raise ForumError("POST_NOT_FOUND", "指定的楼层不存在。", 404)
                after = max(0, focus["post_number"] - min(10, limit))
            rows = all_rows(
                conn,
                """SELECT p.id,p.topic_id,p.author_id,p.post_number,p.reply_to,p.revision,p.status,p.created_at,p.edited_at,
                CASE WHEN p.status='published' AND NOT EXISTS(SELECT 1 FROM forum_blocks x WHERE x.user_id=:viewer AND x.kind='user' AND x.target_id=p.author_id) THEN p.body ELSE NULL END AS body,u.display_name,
                EXISTS(SELECT 1 FROM forum_blocks x WHERE x.user_id=:viewer AND x.kind='user' AND x.target_id=p.author_id) AS blocked,
                (SELECT count(*) FROM forum_reactions r WHERE r.post_id=p.id) AS likes,
                EXISTS(SELECT 1 FROM forum_reactions r WHERE r.post_id=p.id AND r.user_id=:viewer) AS liked
                FROM forum_posts p JOIN forum_profiles u ON u.user_id=p.author_id
                WHERE p.topic_id=:t AND p.post_number>:after AND (NOT :author_only OR p.author_id=:author) ORDER BY p.post_number LIMIT :limit""",
                t=topic_id,
                after=after,
                limit=limit + 1,
                viewer=user.id if user else 0,
                author_only=author_only,
                author=topic["author_id"],
            )
            return {
                "topic": topic,
                "poll": self.poll_state(conn, topic_id, user),
                "posts": rows[:limit],
                "start_after": after,
                "next_after": (
                    rows[limit - 1]["post_number"] if len(rows) > limit else None
                ),
            }

    def reply(self, topic_id, user, key, payload):
        with self.engine.begin() as conn:

            def action():
                topic = self.topic(conn, topic_id, user, True)
                self.writable_topic(topic)
                recipient = topic["author_id"]
                if payload["reply_to"]:
                    parent = one(
                        conn,
                        "SELECT * FROM forum_posts WHERE id=:id AND topic_id=:t AND status='published'",
                        id=payload["reply_to"],
                        t=topic_id,
                    )
                    if not parent:
                        raise ForumError(
                            "INVALID_REPLY", "引用的回复不存在或不属于此主题。"
                        )
                    recipient = parent["author_id"]
                post = one(
                    conn,
                    """INSERT INTO forum_posts(topic_id,author_id,post_number,reply_to,body,body_text)
                    VALUES (:t,:u,:n,:parent,CAST(:body AS jsonb),:txt) RETURNING id,post_number""",
                    t=topic_id,
                    u=user.id,
                    n=topic["next_post_number"],
                    parent=payload["reply_to"],
                    body=encoded(payload["body"]),
                    txt=document_text(payload["body"]),
                )
                assets.bind(conn, post["id"], payload["body"], user)
                mentions.notify(conn, post["id"], payload["body"], user)
                execute(
                    conn,
                    "UPDATE forum_topics SET next_post_number=next_post_number+1,last_activity=now() WHERE id=:id",
                    id=topic_id,
                )
                for receiver in {recipient, topic["author_id"]} - {user.id}:
                    execute(
                        conn,
                        """INSERT INTO forum_notifications(recipient_id,actor_id,post_id)
                        SELECT :u,:actor,:p WHERE forum_can_notify(:u,:actor,:board,'reply')
                        ON CONFLICT DO NOTHING""",
                        u=receiver,
                        actor=user.id,
                        p=post["id"],
                        board=topic["board_id"],
                    )
                self.event(conn, "post.created", post["id"], {"topic_id": topic_id})
                return {
                    "topic_id": topic_id,
                    "post_id": post["id"],
                    "post_number": post["post_number"],
                }

            return self.idempotent(
                conn, user, key, f"topic.{topic_id}.reply", payload, action
            )

    def edit_post(self, post_id, user, payload, delete=False):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic, post = self.post(conn, post_id, user, True)
            if post["author_id"] != user.id and not self.moderate(
                conn, user, topic["board_id"]
            ):
                raise ForumError("FORBIDDEN", "不能修改他人的回复。", 403)
            if (
                not self.moderate(conn, user, topic["board_id"])
                or topic["status"] != "published"
            ):
                self.writable_topic(topic)
            if post["status"] != "published":
                raise ForumError("POST_DELETED", "该回复已删除。", 409)
            if post["revision"] != payload["revision"]:
                raise ForumError(
                    "REVISION_CONFLICT", "内容已被修改，请保留你的文本并重新加载。", 409
                )
            if delete and post["post_number"] == 1:
                raise ForumError(
                    "ROOT_POST", "主帖请通过举报或管理操作处理，不能单独删除。"
                )
            execute(
                conn,
                """INSERT INTO forum_post_revisions(post_id,revision,body,editor_id)
                VALUES (:id,:v,CAST(:body AS jsonb),:u)""",
                id=post_id,
                v=post["revision"],
                body=encoded(post["body"]),
                u=user.id,
            )
            if delete:
                execute(
                    conn,
                    "UPDATE forum_posts SET status='deleted',revision=revision+1,edited_at=now() WHERE id=:id",
                    id=post_id,
                )
                execute(
                    conn,
                    "UPDATE forum_topics SET accepted_post_id=NULL,question_status='open',revision=revision+1 WHERE id=:t AND accepted_post_id=:p",
                    t=topic["id"],
                    p=post_id,
                )
            else:
                assets.bind(conn, post_id, payload["body"], user)
                mentions.notify(conn, post_id, payload["body"], user)
                execute(
                    conn,
                    """UPDATE forum_posts SET body=CAST(:body AS jsonb),body_text=:txt,
                    revision=revision+1,edited_at=now() WHERE id=:id""",
                    id=post_id,
                    body=encoded(payload["body"]),
                    txt=document_text(payload["body"]),
                )
            self.event(
                conn,
                "post.deleted" if delete else "post.edited",
                post_id,
                {"topic_id": topic["id"]},
            )
            return {"post_id": post_id, "revision": post["revision"] + 1}

    def toggle(self, kind, target_id, user, enabled):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            if kind == "reaction":
                topic, post = self.post(conn, target_id, user, True)
                self.writable_topic(topic)
                if post["status"] != "published":
                    raise ForumError("POST_DELETED", "该回复已删除。", 409)
                if user.id == post["author_id"]:
                    raise ForumError("SELF_REACTION", "不能给自己的内容点赞。")
                table, column = "forum_reactions", "post_id"
            else:
                self.topic(conn, target_id, user, True)
                table, column = "forum_bookmarks", "topic_id"
            if enabled:
                execute(
                    conn,
                    f"INSERT INTO {table}(user_id,{column}) VALUES (:u,:id) ON CONFLICT DO NOTHING",
                    u=user.id,
                    id=target_id,
                )
            else:
                execute(
                    conn,
                    f"DELETE FROM {table} WHERE user_id=:u AND {column}=:id",
                    u=user.id,
                    id=target_id,
                )
            return {"enabled": enabled}

    def drafts(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                "SELECT * FROM forum_drafts WHERE user_id=:u ORDER BY updated_at DESC LIMIT 50",
                u=user.id,
            )

    def save_draft(self, draft_id, user, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            old = one(
                conn,
                "SELECT revision FROM forum_drafts WHERE user_id=:u AND id=:id FOR UPDATE",
                u=user.id,
                id=draft_id,
            )
            if (old["revision"] if old else 0) != payload["revision"]:
                raise ForumError(
                    "REVISION_CONFLICT", "草稿已在其他窗口更新，请另存或重新打开。", 409
                )
            if (
                not old
                and one(
                    conn,
                    "SELECT count(*) AS n FROM forum_drafts WHERE user_id=:u",
                    u=user.id,
                )["n"]
                >= 50
            ):
                raise ForumError(
                    "DRAFT_LIMIT", "最多保留 50 份草稿，请先清理已发布草稿。", 409
                )
            data = {k: v for k, v in payload.items() if k != "revision"}
            return one(
                conn,
                """INSERT INTO forum_drafts(user_id,id,payload) VALUES (:u,:id,CAST(:data AS jsonb))
                ON CONFLICT(user_id,id) DO UPDATE SET payload=excluded.payload,revision=forum_drafts.revision+1,updated_at=now()
                RETURNING id,revision,updated_at""",
                u=user.id,
                id=draft_id,
                data=encoded(data),
            )

    def delete_draft(self, draft_id, user, revision):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            result = execute(
                conn,
                "DELETE FROM forum_drafts WHERE user_id=:u AND id=:id AND revision=:v",
                u=user.id,
                id=draft_id,
                v=revision,
            )
            if not result.rowcount:
                raise ForumError("REVISION_CONFLICT", "草稿已更新或不存在。", 409)
            return {"deleted": True}

    def notifications(self, user, before=9223372036854775807, kind=""):
        return self.inbox(user, before, kind)

    def read_notifications(self, user, through_id):
        with self.engine.begin() as conn:
            execute(
                conn,
                "UPDATE forum_notifications SET read_at=now() WHERE recipient_id=:u AND id<=:id AND read_at IS NULL",
                u=user.id,
                id=through_id,
            )
            execute(
                conn,
                "UPDATE forum_system_notifications SET read_at=now() WHERE recipient_id=:u AND id<=:id AND read_at IS NULL",
                u=user.id,
                id=through_id,
            )
        return {"ok": True}

    def report(self, post_id, user, reason):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            self.post(conn, post_id, user, True)
            return one(
                conn,
                """INSERT INTO forum_reports(reporter_id,post_id,reason) VALUES (:u,:id,:r)
                ON CONFLICT(reporter_id,post_id) DO UPDATE SET reason=excluded.reason,status='open'
                RETURNING id,status""",
                u=user.id,
                id=post_id,
                r=reason,
            )

    def reports(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT r.*,t.title,t.id AS topic_id,t.status AS topic_status,p.post_number
                FROM forum_reports r JOIN forum_posts p ON p.id=r.post_id JOIN forum_topics t ON t.id=p.topic_id
                WHERE r.status='open' AND (:admin OR EXISTS (SELECT 1 FROM forum_roles f WHERE f.user_id=:u AND f.board_id=t.board_id))
                ORDER BY r.id LIMIT 100""",
                admin=self.is_admin(user),
                u=user.id,
            )

    def moderation(self, topic_id, user, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic = self.topic(conn, topic_id, user, True)
            if not self.moderate(conn, user, topic["board_id"]):
                raise ForumError("FORBIDDEN", "没有此板块的管理权限。", 403)
            action = payload["action"]
            if action in {"hide", "restore"}:
                execute(
                    conn,
                    "UPDATE forum_topics SET status=:s,revision=revision+1 WHERE id=:id",
                    id=topic_id,
                    s="hidden" if action == "hide" else "published",
                )
            elif action in {"pin", "unpin"}:
                execute(
                    conn,
                    "UPDATE forum_topics SET pinned=:v,revision=revision+1 WHERE id=:id",
                    id=topic_id,
                    v=action == "pin",
                )
            else:
                execute(
                    conn,
                    "UPDATE forum_topics SET locked=:v,revision=revision+1 WHERE id=:id",
                    id=topic_id,
                    v=action == "lock",
                )
            execute(
                conn,
                """INSERT INTO forum_moderation_actions(operator_id,topic_id,action,reason,previous)
                VALUES (:u,:t,:a,:r,CAST(:prev AS jsonb))""",
                u=user.id,
                t=topic_id,
                a=action,
                r=payload["reason"],
                prev=encoded(
                    {
                        "status": topic["status"],
                        "locked": topic["locked"],
                        "pinned": topic["pinned"],
                    }
                ),
            )
            execute(
                conn,
                """UPDATE forum_reports SET status='resolved' WHERE post_id IN
                (SELECT id FROM forum_posts WHERE topic_id=:t)""",
                t=topic_id,
            )
            self.event(conn, "topic.moderated", topic_id, payload)
            self.system_notice(
                conn,
                topic["author_id"],
                "主题管理通知",
                payload["reason"],
                f'topic:{topic_id}:{topic["revision"]}',
                f"/t/{topic_id}",
            )
            return {"ok": True}
