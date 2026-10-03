"""Discussion extensions with transactional authorization and private state."""

import json
from datetime import datetime, timezone, timedelta
from uuid import uuid4
from .db import all_rows, execute, one
from .errors import ForumError


def date(value):
    try:
        parsed = datetime.fromisoformat(value)
        if not parsed.tzinfo:
            raise ValueError()
        return parsed
    except (ValueError, TypeError):
        raise ForumError("INVALID_DATE", "时间必须包含时区。")


class Operations:
    def inbox(self, user, before, kind=""):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT * FROM (
                SELECT n.id,n.kind,n.read_at,n.created_at,u.display_name AS actor_name,t.id AS topic_id,p.post_number,p.id AS post_id,
                    CASE WHEN t.status='published' AND p.status='published' THEN t.title ELSE '内容已不可见' END AS title,
                    (t.status='published' AND p.status='published') AS available,''::text AS body,'/t/'||t.id||'#p-'||p.id AS path
                FROM forum_notifications n JOIN forum_posts p ON p.id=n.post_id JOIN forum_topics t ON t.id=p.topic_id
                JOIN forum_profiles u ON u.user_id=n.actor_id WHERE n.recipient_id=:u AND n.id<:before
                AND NOT EXISTS(SELECT 1 FROM forum_blocks b WHERE b.user_id=:u AND ((b.kind='user' AND b.target_id=n.actor_id) OR (b.kind='board' AND b.target_id=t.board_id)))
                UNION ALL SELECT id,kind,read_at,created_at,'社区管理',NULL,NULL,NULL,title,true,body,path
                FROM forum_system_notifications WHERE recipient_id=:u AND id<:before
            ) n WHERE :kind='' OR kind=:kind ORDER BY id DESC LIMIT 100""",
                u=user.id,
                before=before,
                kind=kind,
            )

    def inbox_status(self, user):
        with self.engine.connect() as conn:
            row = one(
                conn,
                """SELECT coalesce(max(id),0) AS latest,count(*) FILTER(WHERE read_at IS NULL) AS unread FROM (
                SELECT n.id,n.read_at FROM forum_notifications n JOIN forum_posts p ON p.id=n.post_id JOIN forum_topics t ON t.id=p.topic_id
                WHERE n.recipient_id=:u AND NOT EXISTS(SELECT 1 FROM forum_blocks b WHERE b.user_id=:u AND
                    ((b.kind='user' AND b.target_id=n.actor_id) OR (b.kind='board' AND b.target_id=t.board_id)))
                UNION ALL SELECT id,read_at FROM forum_system_notifications WHERE recipient_id=:u) n""",
                u=user.id,
            )
            pref = one(
                conn,
                "SELECT enabled FROM forum_notification_preferences WHERE user_id=:u",
                u=user.id,
            )
            return {**row, "enabled": pref["enabled"] if pref else True}

    def require_admin(self, user):
        if not self.is_admin(user):
            raise ForumError("FORBIDDEN", "需要管理员权限。", 403)

    def system_notice(
        self, conn, recipient, title, body, key, path="/community", kind="moderation"
    ):
        execute(
            conn,
            """INSERT INTO forum_system_notifications(recipient_id,kind,title,body,event_key,path)
            SELECT :u,:kind,:title,:body,:key,:path WHERE forum_can_notify(:u,0,0,:kind)
            ON CONFLICT(recipient_id,event_key) DO NOTHING""",
            u=recipient,
            kind=kind,
            title=title,
            body=body,
            key=key,
            path=path,
        )

    def notification_settings(self, user, payload=None):
        with self.engine.begin() as conn:
            if payload is not None:
                self.writer(conn, user, allow_muted=True)
                execute(
                    conn,
                    """INSERT INTO forum_notification_preferences(user_id,enabled,categories) VALUES(:u,:e,CAST(:c AS jsonb))
                    ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,categories=excluded.categories""",
                    u=user.id,
                    e=payload["enabled"],
                    c=json.dumps(payload["categories"]),
                )
            return one(
                conn,
                "SELECT enabled,categories FROM forum_notification_preferences WHERE user_id=:u",
                u=user.id,
            ) or {"enabled": True, "categories": {}}

    def block_list(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT b.*,CASE WHEN b.kind='user' THEN p.display_name ELSE c.name END AS name
                FROM forum_blocks b LEFT JOIN forum_profiles p ON b.kind='user' AND p.user_id=b.target_id
                LEFT JOIN forum_boards c ON b.kind='board' AND c.id=b.target_id WHERE b.user_id=:u ORDER BY b.created_at DESC LIMIT 200""",
                u=user.id,
            )

    def block(self, user, kind, ident, enabled):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            if enabled:
                valid = one(
                    conn,
                    (
                        "SELECT 1 FROM forum_profiles WHERE user_id=:id"
                        if kind == "user"
                        else "SELECT 1 FROM forum_boards WHERE id=:id"
                    ),
                    id=ident,
                )
                if not valid or (kind == "user" and ident == user.id):
                    raise ForumError("INVALID_BLOCK", "请选择其他社区用户或有效板块。")
                existing = one(
                    conn,
                    "SELECT 1 FROM forum_blocks WHERE user_id=:u AND kind=:k AND target_id=:id",
                    u=user.id,
                    k=kind,
                    id=ident,
                )
                if (
                    not existing
                    and one(
                        conn,
                        "SELECT count(*) AS n FROM forum_blocks WHERE user_id=:u",
                        u=user.id,
                    )["n"]
                    >= 200
                ):
                    raise ForumError("BLOCK_LIMIT", "最多屏蔽 200 项。", 409)
                execute(
                    conn,
                    "INSERT INTO forum_blocks(user_id,kind,target_id) VALUES(:u,:k,:id) ON CONFLICT DO NOTHING",
                    u=user.id,
                    k=kind,
                    id=ident,
                )
            else:
                execute(
                    conn,
                    "DELETE FROM forum_blocks WHERE user_id=:u AND kind=:k AND target_id=:id",
                    u=user.id,
                    k=kind,
                    id=ident,
                )
        return {"blocked": enabled}

    def create_poll(self, conn, topic_id, payload):
        if payload.get("kind") != "poll":
            return
        poll = payload["poll"]
        end = date(poll["closes_at"])
        if (
            not datetime.now(timezone.utc)
            < end
            <= datetime.now(timezone.utc) + timedelta(days=366)
        ):
            raise ForumError("INVALID_DEADLINE", "投票截止时间应在未来一年内。")
        execute(
            conn,
            """INSERT INTO forum_polls(topic_id,options,max_choices,closes_at,results)
            VALUES(:t,CAST(:o AS jsonb),:m,:end,:r)""",
            t=topic_id,
            o=json.dumps(poll["options"]),
            m=poll["max_choices"],
            end=end,
            r=poll["results"],
        )

    def poll_state(self, conn, topic_id, user):
        poll = one(
            conn,
            "SELECT *,closed OR closes_at<=now() AS ended FROM forum_polls WHERE topic_id=:t",
            t=topic_id,
        )
        if not poll:
            return None
        chosen = one(
            conn,
            "SELECT choices FROM forum_votes WHERE topic_id=:t AND user_id=:u",
            t=topic_id,
            u=user.id if user else 0,
        )
        visible = (
            poll["results"] == "always"
            or poll["ended"]
            or (poll["results"] == "voted" and bool(chosen))
        )
        counts = None
        if visible:
            counts = [0] * len(poll["options"])
            for row in all_rows(
                conn,
                """SELECT value::int AS choice,count(*) AS n FROM forum_votes,
                jsonb_array_elements_text(choices) WHERE topic_id=:t GROUP BY value""",
                t=topic_id,
            ):
                counts[row["choice"]] = row["n"]
        return {
            **poll,
            "choices": chosen["choices"] if chosen else [],
            "counts": counts,
            "voters": (
                one(
                    conn,
                    "SELECT count(*) AS n FROM forum_votes WHERE topic_id=:t",
                    t=topic_id,
                )["n"]
                if visible
                else None
            ),
        }

    def vote(self, user, ident, choices):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic = self.topic(conn, ident, user, True)
            self.writable_topic(topic)
            poll = self.poll_state(conn, ident, user)
            if not poll:
                raise ForumError("NOT_FOUND", "此主题没有投票。", 404)
            if poll["ended"]:
                raise ForumError("POLL_CLOSED", "投票已结束。", 409)
            choices = sorted(set(choices))
            if (
                not choices
                or len(choices) > poll["max_choices"]
                or any(c < 0 or c >= len(poll["options"]) for c in choices)
            ):
                raise ForumError("INVALID_VOTE", "选项无效或超过可选数量。")
            execute(
                conn,
                """INSERT INTO forum_votes(topic_id,user_id,choices) VALUES(:t,:u,CAST(:c AS jsonb))
                ON CONFLICT(topic_id,user_id) DO UPDATE SET choices=excluded.choices""",
                t=ident,
                u=user.id,
                c=json.dumps(choices),
            )
            return self.poll_state(conn, ident, user)

    def close_poll(self, user, ident, reason):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic = self.topic(conn, ident, user, True)
            if topic["author_id"] != user.id and not self.moderate(
                conn, user, topic["board_id"]
            ):
                raise ForumError("FORBIDDEN", "只有作者或版主可关闭投票。", 403)
            old = one(
                conn,
                "UPDATE forum_polls SET closed=true WHERE topic_id=:t RETURNING topic_id",
                t=ident,
            )
            if not old:
                raise ForumError("NOT_FOUND", "投票不存在。", 404)
            self.audit(conn, user, "topic", ident, "close_poll", reason, {}, ident)
        return {"ok": True}

    def question(self, user, ident, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic = self.topic(conn, ident, user, True)
            if topic["kind"] != "question":
                raise ForumError("NOT_QUESTION", "此主题不是问答。")
            if topic["author_id"] != user.id and not self.moderate(
                conn, user, topic["board_id"]
            ):
                raise ForumError("FORBIDDEN", "只有作者或版主可以更新问答状态。", 403)
            if topic["revision"] != payload["revision"]:
                raise ForumError("REVISION_CONFLICT", "状态已变化，请刷新。", 409)
            if not self.moderate(conn, user, topic["board_id"]):
                self.writable_topic(topic)
            accepted = payload["accepted_post_id"]
            duplicate = payload["duplicate_of"]
            if accepted and (
                payload["status"] != "solved"
                or not one(
                    conn,
                    "SELECT 1 FROM forum_posts WHERE id=:p AND topic_id=:t AND post_number>1 AND status='published'",
                    p=accepted,
                    t=ident,
                )
            ):
                raise ForumError("INVALID_ANSWER", "只能采纳此主题内仍公开的回复。")
            if duplicate and (
                duplicate == ident
                or not one(
                    conn,
                    "SELECT 1 FROM forum_topics WHERE id=:id AND status='published'",
                    id=duplicate,
                )
            ):
                raise ForumError("INVALID_DUPLICATE", "关联主题不存在或不可见。")
            execute(
                conn,
                """UPDATE forum_topics SET question_status=:s,accepted_post_id=:p,duplicate_of=:d,revision=revision+1 WHERE id=:t""",
                s=payload["status"],
                p=accepted,
                d=duplicate,
                t=ident,
            )
            self.audit(
                conn,
                user,
                "topic",
                ident,
                "question",
                payload["reason"],
                dict(topic),
                ident,
            )
        return {"ok": True}

    def move_topic(self, user, ident, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic = self.topic(conn, ident, user, True)
            if not self.moderate(conn, user, topic["board_id"]) or not self.moderate(
                conn, user, payload["board_id"]
            ):
                raise ForumError(
                    "FORBIDDEN", "移帖需要同时拥有来源与目标板块权限。", 403
                )
            if topic["revision"] != payload["revision"]:
                raise ForumError("REVISION_CONFLICT", "主题已修改。", 409)
            if not one(
                conn, "SELECT 1 FROM forum_boards WHERE id=:b", b=payload["board_id"]
            ):
                raise ForumError("NOT_FOUND", "板块不存在。", 404)
            execute(
                conn,
                "UPDATE forum_topics SET board_id=:b,revision=revision+1 WHERE id=:t",
                b=payload["board_id"],
                t=ident,
            )
            self.audit(
                conn,
                user,
                "topic",
                ident,
                "move",
                payload["reason"],
                dict(topic),
                ident,
            )
            self.system_notice(
                conn,
                topic["author_id"],
                "主题已移动",
                payload["reason"],
                f'move:{ident}:{topic["revision"]}',
                f"/t/{ident}",
            )
        return {"ok": True}

    def revisions(self, user, post_id):
        with self.engine.connect() as conn:
            topic, post = self.post(conn, post_id, user)
            if post["author_id"] != user.id and not self.moderate(
                conn, user, topic["board_id"]
            ):
                raise ForumError("FORBIDDEN", "修订记录仅作者及版主可见。", 403)
            return all_rows(
                conn,
                "SELECT revision,body,editor_id,created_at FROM forum_post_revisions WHERE post_id=:p ORDER BY revision DESC LIMIT 100",
                p=post_id,
            )

    def save_announcement(self, user, ident, payload):
        self.require_admin(user)
        topic = payload["topic"]
        if topic["kind"] != "discussion":
            raise ForumError("INVALID_ANNOUNCEMENT", "公告使用普通讨论类型。")
        publish = date(payload["publish_at"]) if payload["publish_at"] else None
        expiry = date(payload["expires_at"]) if payload["expires_at"] else None
        if payload["schedule"] and not publish:
            raise ForumError("INVALID_DATE", "定时公告需要发布时间。")
        if expiry and expiry <= (publish or datetime.now(timezone.utc)):
            raise ForumError("INVALID_DATE", "到期时间应晚于发布时间。")
        with self.engine.begin() as conn:
            self.writer(conn, user)
            if not one(
                conn, "SELECT 1 FROM forum_boards WHERE slug=:s", s=topic["board_slug"]
            ):
                raise ForumError("NOT_FOUND", "板块不存在。", 404)
            old = (
                one(
                    conn,
                    "SELECT * FROM forum_announcements WHERE id=:id FOR UPDATE",
                    id=ident,
                )
                if ident
                else None
            )
            if ident and not old:
                raise ForumError("NOT_FOUND", "公告不存在。", 404)
            if payload["revision"] != (old["revision"] if old else 0):
                raise ForumError("REVISION_CONFLICT", "公告已更新，请刷新。", 409)
            if old and old["status"] not in {"draft", "scheduled"}:
                raise ForumError(
                    "ANNOUNCEMENT_PUBLISHED",
                    "已发布公告请在主题中修改，或撤回公告。",
                    409,
                )
            args = dict(
                u=user.id,
                p=json.dumps(topic),
                sites=json.dumps(list(dict.fromkeys(payload["sites"]))),
                start=publish,
                end=expiry,
                replies=payload["allow_replies"],
                notify=payload["notify_all"],
                s="scheduled" if payload["schedule"] else "draft",
            )
            if old:
                row = one(
                    conn,
                    """UPDATE forum_announcements SET author_id=:u,payload=CAST(:p AS jsonb),sites=CAST(:sites AS jsonb),
                    publish_at=:start,expires_at=:end,allow_replies=:replies,notify_all=:notify,status=:s,revision=revision+1 WHERE id=:id RETURNING id,revision""",
                    id=ident,
                    **args,
                )
            else:
                row = one(
                    conn,
                    """INSERT INTO forum_announcements(author_id,payload,sites,publish_at,expires_at,allow_replies,notify_all,status,publish_key)
                    VALUES(:u,CAST(:p AS jsonb),CAST(:sites AS jsonb),:start,:end,:replies,:notify,:s,:key) RETURNING id,revision""",
                    key=uuid4(),
                    **args,
                )
            self.audit(
                conn,
                user,
                "announcement",
                row["id"],
                "save_announcement",
                "保存公告配置",
                dict(old) if old else {},
            )
            return row

    def announcements(self, user, public=False, site="forum"):
        if not public:
            self.require_admin(user)
        with self.engine.connect() as conn:
            if public:
                return all_rows(
                    conn,
                    """SELECT a.id,a.topic_id,a.revision,a.expires_at,t.title,b.slug AS board_slug
                    FROM forum_announcements a JOIN forum_topics t ON t.id=a.topic_id JOIN forum_boards b ON b.id=t.board_id
                    WHERE a.status='published' AND t.status='published' AND a.sites @> CAST(:sites AS jsonb)
                    AND (a.expires_at IS NULL OR a.expires_at>now()) ORDER BY a.id DESC LIMIT 20""",
                    sites=json.dumps([site]),
                )
            return all_rows(
                conn, "SELECT * FROM forum_announcements ORDER BY id DESC LIMIT 100"
            )

    def retract_announcement(self, user, ident, revision, reason):
        self.require_admin(user)
        with self.engine.begin() as conn:
            self.writer(conn, user)
            row = one(
                conn,
                "SELECT * FROM forum_announcements WHERE id=:id FOR UPDATE",
                id=ident,
            )
            if not row:
                raise ForumError("NOT_FOUND", "公告不存在。", 404)
            if row["revision"] != revision:
                raise ForumError("REVISION_CONFLICT", "公告已变化。", 409)
            if row["topic_id"]:
                execute(
                    conn,
                    "UPDATE forum_topics SET status='hidden',pinned=false,revision=revision+1 WHERE id=:t",
                    t=row["topic_id"],
                )
            execute(
                conn,
                "UPDATE forum_announcements SET status='retracted',revision=revision+1 WHERE id=:id",
                id=ident,
            )
            self.audit(
                conn,
                user,
                "announcement",
                ident,
                "retract_announcement",
                reason,
                dict(row),
                row["topic_id"],
            )
        return {"ok": True}

    def privacy_requests(self, user, admin=False):
        if admin:
            self.require_admin(user)
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                "SELECT * FROM forum_privacy_requests WHERE :admin OR user_id=:u ORDER BY id DESC LIMIT 100",
                admin=admin,
                u=user.id,
            )

    def request_privacy(self, user, reason):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            row = one(
                conn,
                "INSERT INTO forum_privacy_requests(user_id,reason) VALUES(:u,:r) ON CONFLICT(user_id) WHERE status='open' DO NOTHING RETURNING id",
                u=user.id,
                r=reason,
            )
            if not row:
                raise ForumError("REQUEST_EXISTS", "已有待处理请求。", 409)
            return row

    def decide_privacy(self, user, ident, payload):
        self.require_admin(user)
        with self.engine.begin() as conn:
            self.writer(conn, user)
            row = one(
                conn,
                "SELECT * FROM forum_privacy_requests WHERE id=:id FOR UPDATE",
                id=ident,
            )
            if not row:
                raise ForumError("NOT_FOUND", "请求不存在。", 404)
            if row["status"] != "open":
                raise ForumError("REQUEST_CLOSED", "请求已处理。", 409)
            # Completion is recorded after an operator reviews scope with the user; no silent mass erasure.
            execute(
                conn,
                "UPDATE forum_privacy_requests SET status=:s,decision=:r,decided_at=now() WHERE id=:id",
                s=payload["decision"],
                r=payload["reason"],
                id=ident,
            )
            self.audit(
                conn,
                user,
                "privacy",
                ident,
                payload["decision"],
                payload["reason"],
                dict(row),
            )
            self.system_notice(
                conn,
                row["user_id"],
                "隐私请求处理结果",
                payload["reason"],
                f"privacy:{ident}",
                "/settings",
            )
        return {"ok": True}

    def export_private(self, user):
        with self.engine.connect() as conn:
            # Own content and private preferences only; moderation evidence and other users' drafts are excluded.
            tables = {
                "posts": ("forum_posts", "author_id"),
                "topics": ("forum_topics", "author_id"),
                "drafts": ("forum_drafts", "user_id"),
                "reply_drafts": ("forum_reply_drafts", "user_id"),
                "bookmarks": ("forum_bookmarks", "user_id"),
                "subscriptions": ("forum_subscriptions", "user_id"),
                "follows": ("forum_follows", "user_id"),
                "blocks": ("forum_blocks", "user_id"),
                "reading": ("forum_reading", "user_id"),
                "preferences": ("forum_notification_preferences", "user_id"),
                "appeals": ("forum_appeals", "user_id"),
                "privacy_requests": ("forum_privacy_requests", "user_id"),
            }
            result = {
                "schema_version": 1,
                "user_id": user.id,
                "generated_at": datetime.now(timezone.utc),
                "limits": {},
            }
            for name, (table, column) in tables.items():
                rows = all_rows(
                    conn,
                    f"SELECT * FROM {table} WHERE {column}=:u LIMIT 1001",
                    u=user.id,
                )
                result[name] = rows[:1000]
                if len(rows) > 1000:
                    result["limits"][name] = "超过 1000 条，请提交完整导出请求。"
            return result

    def external_cards(self, user, admin=False):
        if admin:
            self.require_admin(user)
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT c.* FROM forum_external_cards c LEFT JOIN forum_topics t ON t.id=c.topic_id
                WHERE :admin OR (c.status='published' AND (c.topic_id IS NULL OR t.status='published')) ORDER BY c.updated_at DESC LIMIT 100""",
                admin=admin,
            )

    def save_external_card(self, user, payload):
        self.require_admin(user)
        if payload["path"].startswith("//"):
            raise ForumError("INVALID_PATH", "来源路径必须为站内路径。")
        with self.engine.begin() as conn:
            self.writer(conn, user)
            if payload["topic_id"]:
                self.topic(conn, payload["topic_id"], user, True)
            old = one(
                conn,
                "SELECT * FROM forum_external_cards WHERE source=:s AND source_id=:id FOR UPDATE",
                s=payload["source"],
                id=payload["source_id"],
            )
            if old and payload["revision"] <= old["revision"]:
                raise ForumError(
                    "REVISION_CONFLICT", "来源版本必须递增，不能重放旧赛果。", 409
                )
            row = one(
                conn,
                """INSERT INTO forum_external_cards(source,source_id,revision,title,summary,path,status,topic_id)
                VALUES(:source,:source_id,:revision,:title,:summary,:path,:status,:topic_id)
                ON CONFLICT(source,source_id) DO UPDATE SET revision=excluded.revision,title=excluded.title,summary=excluded.summary,
                path=excluded.path,status=excluded.status,topic_id=excluded.topic_id,updated_at=now() RETURNING id""",
                **{k: v for k, v in payload.items() if k != "reason"},
            )
            self.audit(
                conn,
                user,
                "external_card",
                row["id"],
                "source_update",
                payload["reason"],
                dict(old) if old else {},
                payload["topic_id"],
            )
            return row
