"""Private community state and moderation appeals, persisted in PostgreSQL."""

from .db import all_rows, execute, one
from .errors import ForumError


class SocialFeatures:
    def profiles(self, query):
        pattern = (
            "%"
            + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            + "%"
        )
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT user_id,display_name FROM forum_profiles
                WHERE display_name ILIKE :q OR CAST(user_id AS text)=:id
                ORDER BY user_id DESC LIMIT 20""",
                q=pattern,
                id=query,
            )

    def profile(self, ident, user, before, before_reply=9223372036854775807):
        with self.engine.connect() as conn:
            profile = one(
                conn,
                "SELECT user_id,display_name FROM forum_profiles WHERE user_id=:id",
                id=ident,
            )
            if not profile:
                raise ForumError("NOT_FOUND", "用户尚未在社区活动。", 404)
            following = bool(
                user
                and one(
                    conn,
                    "SELECT 1 FROM forum_follows WHERE user_id=:u AND followed_id=:id",
                    u=user.id,
                    id=ident,
                )
            )
            topics = all_rows(
                conn,
                """SELECT id,title,created_at FROM forum_topics
                WHERE author_id=:id AND status='published' AND id<:before ORDER BY id DESC LIMIT 20""",
                id=ident,
                before=before,
            )
            replies = all_rows(
                conn,
                """SELECT p.id,p.topic_id,p.post_number,p.created_at,left(p.body_text,240) AS excerpt,t.title
                FROM forum_posts p JOIN forum_topics t ON t.id=p.topic_id
                WHERE p.author_id=:id AND p.post_number>1 AND p.status='published' AND t.status='published'
                AND p.id<:before ORDER BY p.id DESC LIMIT 20""",
                id=ident,
                before=before_reply,
            )
            return {
                **profile,
                "following": following,
                "topics": topics,
                "replies": replies,
            }

    def follows(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT f.followed_id,p.display_name FROM forum_follows f
                JOIN forum_profiles p ON p.user_id=f.followed_id WHERE f.user_id=:u
                ORDER BY f.created_at DESC LIMIT 200""",
                u=user.id,
            )

    def follow(self, user, ident, enabled):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            if enabled:
                if ident == user.id or not one(
                    conn, "SELECT 1 FROM forum_profiles WHERE user_id=:id", id=ident
                ):
                    raise ForumError("INVALID_FOLLOW", "请选择其他已加入社区的用户。")
                exists = one(
                    conn,
                    "SELECT 1 FROM forum_follows WHERE user_id=:u AND followed_id=:id",
                    u=user.id,
                    id=ident,
                )
                if (
                    not exists
                    and one(
                        conn,
                        "SELECT count(*) AS n FROM forum_follows WHERE user_id=:u",
                        u=user.id,
                    )["n"]
                    >= 200
                ):
                    raise ForumError("FOLLOW_LIMIT", "最多关注 200 位用户。", 409)
                execute(
                    conn,
                    "INSERT INTO forum_follows(user_id,followed_id) VALUES(:u,:id) ON CONFLICT DO NOTHING",
                    u=user.id,
                    id=ident,
                )
            else:
                execute(
                    conn,
                    "DELETE FROM forum_follows WHERE user_id=:u AND followed_id=:id",
                    u=user.id,
                    id=ident,
                )
        return {"following": enabled}

    def reading(self, user, topic_id=None):
        with self.engine.connect() as conn:
            if topic_id is not None:
                self.topic(conn, topic_id, user)
                return one(
                    conn,
                    "SELECT post_id,post_number,updated_at FROM forum_reading WHERE user_id=:u AND topic_id=:t",
                    u=user.id,
                    t=topic_id,
                )
            return all_rows(
                conn,
                """SELECT r.*,t.title FROM forum_reading r JOIN forum_topics t ON t.id=r.topic_id
                WHERE r.user_id=:u AND t.status='published' ORDER BY r.updated_at DESC LIMIT 100""",
                u=user.id,
            )

    def mark_reading(self, user, topic_id, post_id):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            self.topic(conn, topic_id, user)
            post = one(
                conn,
                "SELECT post_number FROM forum_posts WHERE id=:p AND topic_id=:t",
                p=post_id,
                t=topic_id,
            )
            if not post:
                raise ForumError("INVALID_POST", "楼层不属于此主题。", 404)
            execute(
                conn,
                """INSERT INTO forum_reading(user_id,topic_id,post_id,post_number) VALUES(:u,:t,:p,:n)
                ON CONFLICT(user_id,topic_id) DO UPDATE SET post_id=excluded.post_id,
                post_number=excluded.post_number,updated_at=now()
                WHERE forum_reading.post_number<excluded.post_number""",
                u=user.id,
                t=topic_id,
                p=post_id,
                n=post["post_number"],
            )
        return {"ok": True}

    def reply_draft(self, user, topic_id):
        with self.engine.connect() as conn:
            self.topic(conn, topic_id, user)
            return one(
                conn,
                """SELECT d.text,d.reply_to,d.revision,p.post_number FROM forum_reply_drafts d
                LEFT JOIN forum_posts p ON p.id=d.reply_to WHERE d.user_id=:u AND d.topic_id=:t""",
                u=user.id,
                t=topic_id,
            )

    def reply_drafts(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT d.topic_id,d.revision,d.updated_at,left(d.text,160) AS preview,
                CASE WHEN t.status='published' OR t.author_id=:u THEN t.title ELSE '主题已不可见' END AS title,
                (t.status='published' OR t.author_id=:u) AS available
                FROM forum_reply_drafts d JOIN forum_topics t ON t.id=d.topic_id
                WHERE d.user_id=:u AND d.text<>'' ORDER BY d.updated_at DESC LIMIT 50""",
                u=user.id,
            )

    def clear_reply_draft(self, user, topic_id, revision):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            row = one(
                conn,
                """UPDATE forum_reply_drafts SET text='',reply_to=NULL,revision=revision+1,updated_at=now()
                WHERE user_id=:u AND topic_id=:t AND revision=:r RETURNING revision""",
                u=user.id,
                t=topic_id,
                r=revision,
            )
            if not row:
                raise ForumError(
                    "REVISION_CONFLICT", "草稿已变化，请刷新列表后再清空。", 409
                )
            return row

    def save_reply_draft(self, user, topic_id, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            self.topic(conn, topic_id, user)
            if payload["reply_to"] and not one(
                conn,
                "SELECT 1 FROM forum_posts WHERE id=:p AND topic_id=:t AND status='published'",
                p=payload["reply_to"],
                t=topic_id,
            ):
                raise ForumError("INVALID_REPLY", "指定回复已删除或不属于此主题。")
            old = one(
                conn,
                "SELECT revision,text FROM forum_reply_drafts WHERE user_id=:u AND topic_id=:t",
                u=user.id,
                t=topic_id,
            )
            if payload["revision"] != (old["revision"] if old else 0):
                raise ForumError(
                    "REVISION_CONFLICT",
                    "另一处已更新草稿。请保留本地文本，并查看云端版本。",
                    409,
                )
            if (
                payload["text"]
                and (not old or not old["text"])
                and one(
                    conn,
                    "SELECT count(*) AS n FROM forum_reply_drafts WHERE user_id=:u AND text<>''",
                    u=user.id,
                )["n"]
                >= 50
            ):
                raise ForumError("DRAFT_LIMIT", "最多保留 50 份回复草稿。", 409)
            # Keep empty rows as revision tombstones: stale devices must never resurrect deleted drafts.
            return one(
                conn,
                """INSERT INTO forum_reply_drafts(user_id,topic_id,text,reply_to) VALUES(:u,:t,:text,:reply_to)
                ON CONFLICT(user_id,topic_id) DO UPDATE SET text=excluded.text,reply_to=excluded.reply_to,
                revision=forum_reply_drafts.revision+1,updated_at=now() RETURNING revision""",
                u=user.id,
                t=topic_id,
                text=payload["text"],
                reply_to=payload["reply_to"],
            )

    def appeal_actions(self, user):
        with self.engine.connect() as conn:
            return all_rows(
                conn,
                """SELECT a.id,a.topic_id,a.action,a.reason,a.created_at,t.title,
                s.expires_at,ap.id AS appeal_id FROM forum_moderation_actions a
                LEFT JOIN forum_topics t ON t.id=a.topic_id
                LEFT JOIN forum_sanctions s ON a.target_type='user' AND a.target_id=CAST(s.user_id AS text)
                LEFT JOIN forum_appeals ap ON ap.action_id=a.id AND ap.user_id=:u
                WHERE ((t.author_id=:u AND ((a.action='hide' AND t.status='hidden') OR (a.action='lock' AND t.locked)))
                    OR (a.target_type='user' AND a.target_id=CAST(:u AS text) AND a.action='mute' AND s.expires_at>now()))
                AND NOT EXISTS(SELECT 1 FROM forum_moderation_actions n WHERE n.id>a.id AND
                    ((a.action='hide' AND n.topic_id=a.topic_id AND n.action IN ('hide','restore'))
                    OR (a.action='lock' AND n.topic_id=a.topic_id AND n.action IN ('lock','unlock'))
                    OR (a.action='mute' AND n.target_type='user' AND n.target_id=a.target_id AND n.action IN ('mute','unmute'))))
                ORDER BY a.id DESC LIMIT 100""",
                u=user.id,
            )

    def appeal_target(self, conn, action, user, reviewing=False):
        if action["action"] in {"hide", "lock"} and action["topic_id"]:
            target = self.topic(conn, action["topic_id"], user, True)
            allowed = (
                self.moderate(conn, user, target["board_id"])
                if reviewing
                else target["author_id"] == user.id
            )
            current = (
                target["status"] == "hidden"
                if action["action"] == "hide"
                else target["locked"]
            )
            newer = one(
                conn,
                """SELECT 1 FROM forum_moderation_actions WHERE topic_id=:t AND id>:id
                AND ((:action='hide' AND action IN ('hide','restore')) OR (:action='lock' AND action IN ('lock','unlock')))""",
                t=target["id"],
                id=action["id"],
                action=action["action"],
            )
        elif action["action"] == "mute" and action["target_type"] == "user":
            ident = int(action["target_id"])
            allowed = self.is_admin(user) if reviewing else ident == user.id
            if not allowed:
                raise ForumError("FORBIDDEN", "没有此申诉的处理权限。", 403)
            one(
                conn,
                "SELECT user_id FROM forum_profiles WHERE user_id=:u FOR UPDATE",
                u=ident,
            )
            target = one(
                conn,
                "SELECT * FROM forum_sanctions WHERE user_id=:u AND expires_at>now()",
                u=ident,
            )
            current = bool(target)
            newer = one(
                conn,
                """SELECT 1 FROM forum_moderation_actions WHERE target_type='user' AND target_id=:u
                AND id>:id AND action IN ('mute','unmute')""",
                u=str(ident),
                id=action["id"],
            )
        else:
            raise ForumError("INVALID_APPEAL", "这条管理记录不支持申诉。")
        if not allowed:
            raise ForumError("FORBIDDEN", "没有此申诉的处理权限。", 403)
        return target, bool(current and not newer)

    def create_appeal(self, user, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user, allow_muted=True)
            action = one(
                conn,
                "SELECT * FROM forum_moderation_actions WHERE id=:id",
                id=payload["action_id"],
            )
            if not action:
                raise ForumError("NOT_FOUND", "管理记录不存在。", 404)
            _, current = self.appeal_target(conn, action, user)
            if not current:
                raise ForumError(
                    "APPEAL_STALE", "管理状态已变化，请刷新后选择最新记录。", 409
                )
            result = one(
                conn,
                """INSERT INTO forum_appeals(user_id,action_id,reason) VALUES(:u,:a,:r)
                ON CONFLICT(user_id,action_id) DO NOTHING RETURNING id,status""",
                u=user.id,
                a=action["id"],
                r=payload["reason"],
            )
            if not result:
                raise ForumError(
                    "APPEAL_EXISTS", "此管理记录已提交申诉，可在下方查看进度。", 409
                )
            return result

    def appeals(self, user, reviewing=False, before=9223372036854775807):
        with self.engine.connect() as conn:
            if reviewing:
                self.manager(conn, user)
            return all_rows(
                conn,
                """SELECT ap.*,a.action,a.reason AS moderation_reason,a.topic_id,a.target_id,
                a.operator_id,t.title,p.display_name FROM forum_appeals ap
                JOIN forum_moderation_actions a ON a.id=ap.action_id JOIN forum_profiles p ON p.user_id=ap.user_id
                LEFT JOIN forum_topics t ON t.id=a.topic_id WHERE ap.id<:before AND
                ((NOT :review AND ap.user_id=:u) OR (:review AND
                (:admin OR EXISTS(SELECT 1 FROM forum_roles r WHERE r.user_id=:u AND r.board_id=t.board_id))))
                ORDER BY ap.id DESC LIMIT 50""",
                u=user.id,
                review=reviewing,
                admin=self.is_admin(user),
                before=before,
            )

    def decide_appeal(self, user, ident, payload):
        with self.engine.begin() as conn:
            self.writer(conn, user)
            self.manager(conn, user)
            appeal = one(conn, "SELECT * FROM forum_appeals WHERE id=:id", id=ident)
            if not appeal:
                raise ForumError("NOT_FOUND", "申诉不存在。", 404)
            action = one(
                conn,
                "SELECT * FROM forum_moderation_actions WHERE id=:id",
                id=appeal["action_id"],
            )
            target, current = self.appeal_target(conn, action, user, reviewing=True)
            # Same lock order as moderation: actor -> target -> appeal.
            appeal = one(
                conn, "SELECT * FROM forum_appeals WHERE id=:id FOR UPDATE", id=ident
            )
            if appeal["status"] != "open":
                raise ForumError("APPEAL_CLOSED", "申诉已处理，请刷新列表。", 409)
            if user.id == appeal["user_id"]:
                raise ForumError("SELF_REVIEW", "不能处理自己的申诉。", 403)
            if payload["decision"] == "accepted":
                if not current:
                    raise ForumError(
                        "APPEAL_STALE",
                        "状态已被后续管理决定改变，不能自动撤销。请说明情况并结案。",
                        409,
                    )
                if action["action"] == "mute":
                    execute(
                        conn,
                        "DELETE FROM forum_sanctions WHERE user_id=:u",
                        u=int(action["target_id"]),
                    )
                elif action["action"] == "hide":
                    execute(
                        conn,
                        "UPDATE forum_topics SET status='published',revision=revision+1 WHERE id=:t",
                        t=action["topic_id"],
                    )
                else:
                    execute(
                        conn,
                        "UPDATE forum_topics SET locked=false,revision=revision+1 WHERE id=:t",
                        t=action["topic_id"],
                    )
                self.audit(
                    conn,
                    user,
                    action["target_type"],
                    action["target_id"] or action["topic_id"],
                    {"hide": "restore", "lock": "unlock", "mute": "unmute"}[
                        action["action"]
                    ],
                    payload["reason"],
                    dict(target),
                    action["topic_id"],
                )
            execute(
                conn,
                """UPDATE forum_appeals SET status=:s,decision=:r,reviewer_id=:u,decided_at=now()
                WHERE id=:id""",
                s=payload["decision"],
                r=payload["reason"],
                u=user.id,
                id=ident,
            )
            self.audit(
                conn,
                user,
                "appeal",
                ident,
                payload["decision"],
                payload["reason"],
                dict(appeal),
                action["topic_id"],
            )
            self.system_notice(
                conn,
                appeal["user_id"],
                "申诉处理结果",
                payload["reason"],
                f"appeal:{ident}",
            )
        return {"ok": True}
