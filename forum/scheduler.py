"""Durable announcement lifecycle. Called by the existing forum worker."""

from .backend.db import one, execute, all_rows
from .backend.auth import Principal
from .backend.errors import ForumError


def tick(svc):
    # Claim after locking the actor, preserving the API's actor -> object lock order.
    with svc.engine.connect() as conn:
        candidate = one(
            conn,
            "SELECT id,author_id FROM forum_announcements WHERE (status='scheduled' AND publish_at<=now()) OR (status='published' AND expires_at<=now()) ORDER BY id LIMIT 1",
        )
    if not candidate:
        return False
    with svc.engine.begin() as conn:
        execute(conn, "SELECT pg_advisory_xact_lock(:u)", u=candidate["author_id"])
        row = one(
            conn,
            "SELECT * FROM forum_announcements WHERE id=:id AND ((status='scheduled' AND publish_at<=now()) OR (status='published' AND expires_at<=now())) FOR UPDATE SKIP LOCKED",
            id=candidate["id"],
        )
        if not row:
            return True
        if row["status"] == "published":
            execute(
                conn,
                "UPDATE forum_announcements SET status='expired',revision=revision+1 WHERE id=:id AND expires_at<=now()",
                id=row["id"],
            )
            execute(
                conn,
                "UPDATE forum_topics SET pinned=false,revision=revision+1 WHERE id=:id",
                id=row["topic_id"],
            )
            return True
        if row["status"] != "scheduled":
            return True
        if row["author_id"] not in svc.settings.admin_ids or one(
            conn,
            "SELECT 1 FROM forum_sanctions WHERE user_id=:u AND expires_at>now()",
            u=row["author_id"],
        ):
            execute(
                conn,
                "UPDATE forum_announcements SET status='draft',revision=revision+1 WHERE id=:id",
                id=row["id"],
            )
            svc.system_notice(
                conn,
                row["author_id"],
                "公告发布已暂停",
                "作者权限发生变化，请管理员重新安排。",
                f'announcement-paused:{row["id"]}:{row["revision"]}',
            )
            return True
        if one(
            conn,
            "SELECT 1 FROM forum_announcements WHERE id=:id AND expires_at<=now()",
            id=row["id"],
        ):
            execute(
                conn,
                "UPDATE forum_announcements SET status='expired',revision=revision+1 WHERE id=:id",
                id=row["id"],
            )
            return True
        profile = one(
            conn,
            "SELECT display_name FROM forum_profiles WHERE user_id=:u",
            u=row["author_id"],
        )
        user = Principal(row["author_id"], profile["display_name"])
        try:
            # A removed attachment or changed board policy must not poison the queue.
            # Roll back partial topic writes while retaining the announcement lock.
            with conn.begin_nested():
                result = svc.create_topic(
                    user, row["publish_key"], row["payload"], connection=conn
                )
        except ForumError:
            execute(
                conn,
                "UPDATE forum_announcements SET status='draft',revision=revision+1 WHERE id=:id",
                id=row["id"],
            )
            svc.system_notice(
                conn,
                user.id,
                "公告发布未完成",
                "内容或附件已无法发布，请检查公告草稿后重新安排。",
                f'announcement-invalid:{row["id"]}:{row["revision"]}',
            )
            return True
        execute(
            conn,
            "UPDATE forum_topics SET pinned=true,locked=:closed WHERE id=:t",
            t=result["topic_id"],
            closed=not row["allow_replies"],
        )
        execute(
            conn,
            "UPDATE forum_announcements SET status='published',topic_id=:t,revision=revision+1 WHERE id=:id",
            t=result["topic_id"],
            id=row["id"],
        )
        svc.audit(
            conn,
            user,
            "announcement",
            row["id"],
            "publish_announcement",
            "定时公告发布",
            dict(row),
            result["topic_id"],
        )
        if row["notify_all"]:
            # No private body copies: public availability is rechecked when listing notices.
            execute(
                conn,
                """INSERT INTO forum_notifications(recipient_id,actor_id,post_id,kind)
                SELECT user_id,:actor,:post,'system' FROM forum_profiles
                WHERE user_id<>:actor AND forum_can_notify(user_id,:actor,:board,'system') ON CONFLICT DO NOTHING""",
                actor=user.id,
                post=result["post_id"],
                board=one(
                    conn,
                    "SELECT board_id FROM forum_topics WHERE id=:t",
                    t=result["topic_id"],
                )["board_id"],
            )
        return True
