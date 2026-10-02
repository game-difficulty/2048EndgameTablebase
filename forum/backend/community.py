"""Subscriptions and permission-scoped community administration."""

import json
from .db import all_rows, execute, one
from .errors import ForumError


class CommunityFeatures:
    def manager(self, conn, user):
        if not self.is_admin(user) and not one(
            conn, "SELECT 1 FROM forum_roles WHERE user_id=:u", u=user.id
        ):
            raise ForumError("FORBIDDEN", "需要管理权限。", 403)

    def audit(self, conn, user, kind, ident, action, reason, previous, topic_id=None):
        execute(
            conn,
            """INSERT INTO forum_moderation_actions(operator_id,topic_id,target_type,target_id,action,reason,previous)
            VALUES(:u,:t,:kind,:id,:a,:r,CAST(:prev AS jsonb))""",
            u=user.id,
            t=topic_id,
            kind=kind,
            id=str(ident),
            a=action,
            r=reason,
            prev=json.dumps(previous, default=str),
        )

    def subscriptions(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT s.*,CASE WHEN s.kind='board' THEN b.name
                WHEN t.status='published' THEN t.title ELSE '内容已不可见' END AS title,
                CASE WHEN s.kind='board' THEN '/c/'||b.slug ELSE '/t/'||t.id END AS path
                FROM forum_subscriptions s LEFT JOIN forum_topics t ON s.kind='topic' AND t.id=s.target_id
                LEFT JOIN forum_boards b ON s.kind='board' AND b.id=s.target_id
                WHERE s.user_id=:u ORDER BY s.created_at DESC LIMIT 200""",
                u=user.id,
            )

    def subscribe(self, user, kind, ident, enabled):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            if enabled:
                if kind == "topic":
                    self.topic(conn, ident, user)
                elif not one(conn, "SELECT 1 FROM forum_boards WHERE id=:id", id=ident):
                    raise ForumError("BOARD_NOT_FOUND", "板块不存在。", 404)
                if (
                    one(
                        conn,
                        "SELECT count(*) AS n FROM forum_subscriptions WHERE user_id=:u",
                        u=user.id,
                    )["n"]
                    >= 200
                ):
                    if not one(
                        conn,
                        "SELECT 1 FROM forum_subscriptions WHERE user_id=:u AND kind=:k AND target_id=:id",
                        u=user.id,
                        k=kind,
                        id=ident,
                    ):
                        raise ForumError(
                            "SUBSCRIPTION_LIMIT", "最多订阅 200 个主题或板块。", 409
                        )
                execute(
                    conn,
                    "INSERT INTO forum_subscriptions VALUES(:u,:k,:id,now()) ON CONFLICT DO NOTHING",
                    u=user.id,
                    k=kind,
                    id=ident,
                )
            else:
                execute(
                    conn,
                    "DELETE FROM forum_subscriptions WHERE user_id=:u AND kind=:k AND target_id=:id",
                    u=user.id,
                    k=kind,
                    id=ident,
                )
        return {"enabled": enabled}

    def notification_status(self, user):
        with self.engine.connect() as conn:
            row = dict(
                one(
                    conn,
                    """SELECT coalesce(max(id),0) AS latest,count(*) FILTER(WHERE read_at IS NULL) AS unread
                FROM forum_notifications WHERE recipient_id=:u""",
                    u=user.id,
                )
            )
            pref = one(
                conn,
                "SELECT enabled FROM forum_notification_preferences WHERE user_id=:u",
                u=user.id,
            )
            return {**row, "enabled": pref["enabled"] if pref else True}

    def notification_preference(self, user, enabled):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            execute(
                conn,
                """INSERT INTO forum_notification_preferences VALUES(:u,:v)
                ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled""",
                u=user.id,
                v=enabled,
            )
        return {"enabled": enabled}

    def update_topic(self, ident, user, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            topic = self.topic(conn, ident, user, True)
            self.writable_topic(topic)
            if topic["author_id"] != user.id and not self.moderate(
                conn, user, topic["board_id"]
            ):
                raise ForumError("FORBIDDEN", "不能编辑这个主题。", 403)
            if topic["revision"] != payload["revision"]:
                raise ForumError("REVISION_CONFLICT", "主题已更新，请重新加载。", 409)
            execute(
                conn,
                "UPDATE forum_topics SET title=:title,tags=CAST(:tags AS jsonb),revision=revision+1 WHERE id=:id",
                title=payload["title"],
                tags=json.dumps(payload["tags"]),
                id=ident,
            )
            self.audit(
                conn,
                user,
                "topic",
                ident,
                "edit_metadata",
                "修改主题标题和标签",
                dict(topic),
                ident,
            )
        return {"ok": True}

    def management(self, user, section, before=9223372036854775807, query=""):
        with self.engine.connect() as conn:
            self.manager(conn, user)
            admin = self.is_admin(user)
            scope = "(:admin OR EXISTS(SELECT 1 FROM forum_roles r WHERE r.user_id=:u AND r.board_id=t.board_id))"
            args = {
                "admin": admin,
                "u": user.id,
                "before": before,
                "q": "%"
                + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                + "%",
            }
            if section == "topics":
                return all_rows(
                    conn,
                    "SELECT t.* FROM forum_topics t WHERE t.id<:before AND t.title ILIKE :q AND "
                    + scope
                    + " ORDER BY t.id DESC LIMIT 50",
                    **args
                )
            if section == "audit":
                return all_rows(
                    conn,
                    """SELECT a.* FROM forum_moderation_actions a LEFT JOIN forum_topics t ON t.id=a.topic_id
                    WHERE a.id<:before AND """
                    + scope
                    + " ORDER BY a.id DESC LIMIT 50",
                    **args
                )
            if section == "reports":
                return all_rows(
                    conn,
                    """SELECT r.*,t.title,t.id AS topic_id,p.post_number,p.body FROM forum_reports r
                    JOIN forum_posts p ON p.id=r.post_id JOIN forum_topics t ON t.id=p.topic_id
                    WHERE r.status='open' AND r.id<:before AND """
                    + scope
                    + " ORDER BY r.id DESC LIMIT 50",
                    **args
                )
            if not admin:
                raise ForumError("FORBIDDEN", "此功能仅管理员可用。", 403)
            if section == "users":
                return all_rows(
                    conn,
                    """SELECT p.*,s.reason AS sanction_reason,s.expires_at,
                    (SELECT jsonb_agg(r.board_id) FROM forum_roles r WHERE r.user_id=p.user_id) AS boards
                    FROM forum_profiles p LEFT JOIN forum_sanctions s ON s.user_id=p.user_id AND s.expires_at>now()
                    WHERE p.user_id<:before AND (p.display_name ILIKE :q OR CAST(p.user_id AS text) ILIKE :q)
                    ORDER BY p.user_id DESC LIMIT 50""",
                    **args
                )
            if section == "media":
                return all_rows(
                    conn,
                    "SELECT id,owner_id,kind,size,status,created_at FROM forum_media ORDER BY created_at DESC LIMIT 100",
                )
            if section == "jobs":
                return all_rows(
                    conn,
                    "SELECT id,kind,created_at,delivered_at,attempts,last_error FROM forum_outbox WHERE id<:before ORDER BY id DESC LIMIT 50",
                    **args
                )
            raise ForumError("INVALID_SECTION", "未知管理页面。")

    def resolve_report(self, user, ident, reason):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            report = one(
                conn, "SELECT * FROM forum_reports WHERE id=:id FOR UPDATE", id=ident
            )
            if not report:
                raise ForumError("NOT_FOUND", "举报不存在。", 404)
            topic, _ = self.post(conn, report["post_id"], user)
            if not self.moderate(conn, user, topic["board_id"]):
                raise ForumError("FORBIDDEN", "没有此板块权限。", 403)
            execute(
                conn,
                "UPDATE forum_reports SET status='resolved' WHERE id=:id",
                id=ident,
            )
            self.audit(
                conn,
                user,
                "report",
                ident,
                "resolve",
                reason,
                dict(report),
                topic["id"],
            )
        return {"ok": True}

    def admin_user(self, user, ident, payload):
        if not self.is_admin(user):
            raise ForumError("FORBIDDEN", "仅管理员可以操作。", 403)
        if ident in self.settings.admin_ids:
            raise ForumError(
                "PROTECTED_ADMIN", "管理员账号通过服务配置维护，不能禁言或降权。", 409
            )
        with self.engine.begin() as conn:
            self.writer(conn, user)
            if not one(
                conn, "SELECT 1 FROM forum_profiles WHERE user_id=:id FOR UPDATE", id=ident
            ):
                raise ForumError("NOT_FOUND", "此账号尚未在论坛活动。", 404)
            action = payload["action"]
            if action in {"grant", "revoke"}:
                board = payload["board_id"]
                if not board or not one(
                    conn, "SELECT 1 FROM forum_boards WHERE id=:b", b=board
                ):
                    raise ForumError("INVALID_BOARD", "请选择有效板块。")
                previous = one(
                    conn,
                    "SELECT * FROM forum_roles WHERE user_id=:id AND board_id=:b",
                    id=ident,
                    b=board,
                )
                if action == "grant":
                    execute(
                        conn,
                        "INSERT INTO forum_roles VALUES(:id,:b,'moderator') ON CONFLICT DO NOTHING",
                        id=ident,
                        b=board,
                    )
                else:
                    execute(
                        conn,
                        "DELETE FROM forum_roles WHERE user_id=:id AND board_id=:b",
                        id=ident,
                        b=board,
                    )
            else:
                previous = one(
                    conn, "SELECT * FROM forum_sanctions WHERE user_id=:id", id=ident
                )
                if action == "mute":
                    execute(
                        conn,
                        """INSERT INTO forum_sanctions VALUES(:id,:r,now()+:hours*interval '1 hour',:u)
                        ON CONFLICT(user_id) DO UPDATE SET reason=excluded.reason,expires_at=excluded.expires_at,operator_id=excluded.operator_id""",
                        id=ident,
                        r=payload["reason"],
                        hours=payload["hours"],
                        u=user.id,
                    )
                else:
                    execute(
                        conn, "DELETE FROM forum_sanctions WHERE user_id=:id", id=ident
                    )
            self.audit(
                conn,
                user,
                "user",
                ident,
                action,
                payload["reason"],
                dict(previous) if previous else {},
            )
        return {"ok": True}

    def edit_board(self, user, ident, payload):
        if not self.is_admin(user):
            raise ForumError("FORBIDDEN", "仅管理员可以操作。", 403)
        with self.engine.begin() as conn:
            self.writer(conn, user)
            old = one(
                conn, "SELECT * FROM forum_boards WHERE id=:id FOR UPDATE", id=ident
            )
            if not old:
                raise ForumError("NOT_FOUND", "板块不存在。", 404)
            execute(
                conn,
                "UPDATE forum_boards SET name=:name,description=:description,position=:position,staff_only=:staff_only WHERE id=:id",
                id=ident,
                **{k: v for k, v in payload.items() if k != "reason"}
            )
            self.audit(
                conn, user, "board", ident, "edit_board", payload["reason"], dict(old)
            )
        return {"ok": True}

    def remove_media(self, user, ident, reason):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            row = one(
                conn,
                "SELECT owner_id,status FROM forum_media WHERE id=:id FOR UPDATE",
                id=ident,
            )
            if not row or (row["owner_id"] != user.id and not self.is_admin(user)):
                raise ForumError("NOT_FOUND", "附件不存在或没有权限。", 404)
            execute(
                conn,
                "UPDATE forum_media SET status='removed',data=:empty,size=0 WHERE id=:id",
                id=ident,
                empty=b"",
            )
            self.audit(conn, user, "media", ident, "remove_media", reason, dict(row))
        return {"ok": True}
