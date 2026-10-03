"""Run with python -m forum.worker. Transactional, idempotent outbox consumer."""

import time
from .backend.config import load_settings
from .backend.db import make_engine, one, execute


def deliver_one(engine):
    with engine.begin() as conn:
        event = one(
            conn,
            """SELECT * FROM forum_outbox WHERE delivered_at IS NULL AND next_attempt_at<=now()
            ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1""",
        )
        if not event:
            return False
        # Savepoint ensures delivery and status commit together, with bounded retry.
        try:
            with conn.begin_nested():
                if event["kind"] in {"post.created", "topic.created"}:
                    post_id = (
                        event["aggregate_id"]
                        if event["kind"] == "post.created"
                        else event["payload"]["post_id"]
                    )
                    execute(
                        conn,
                        """INSERT INTO forum_notifications(recipient_id,actor_id,post_id,kind)
                        SELECT DISTINCT s.user_id,p.author_id,p.id,'subscription' FROM forum_posts p JOIN forum_topics t ON t.id=p.topic_id
                        JOIN forum_subscriptions s ON (s.kind='topic' AND s.target_id=t.id)
                            OR (s.kind='board' AND s.target_id=t.board_id AND p.post_number=1)
                        WHERE p.id=:id AND p.status='published' AND t.status='published' AND s.user_id<>p.author_id
                        AND s.created_at<=:created
                        AND forum_can_notify(s.user_id,p.author_id,t.board_id,'subscription')
                        ON CONFLICT DO NOTHING""",
                        id=post_id,
                        created=event["created_at"],
                    )
                    if event["kind"] == "topic.created":
                        execute(
                            conn,
                            """INSERT INTO forum_notifications(recipient_id,actor_id,post_id,kind)
                            SELECT f.user_id,p.author_id,p.id,'follow' FROM forum_posts p JOIN forum_topics t ON t.id=p.topic_id
                            JOIN forum_follows f ON f.followed_id=p.author_id
                            WHERE p.id=:id AND p.status='published' AND t.status='published' AND f.created_at<=:created
                            AND forum_can_notify(f.user_id,p.author_id,t.board_id,'follow')
                            ON CONFLICT DO NOTHING""",
                            id=post_id,
                            created=event["created_at"],
                        )
                execute(
                    conn,
                    "UPDATE forum_outbox SET delivered_at=now(),last_error=NULL WHERE id=:id",
                    id=event["id"],
                )
        except Exception as exc:
            execute(
                conn,
                """UPDATE forum_outbox SET attempts=attempts+1,last_error=:error,
                next_attempt_at=now()+interval '1 minute' WHERE id=:id""",
                id=event["id"],
                error=type(exc).__name__[:100],
            )
        return True


if __name__ == "__main__":
    from .backend.service import ForumService
    from .scheduler import tick

    settings = load_settings()
    engine = make_engine(settings.database_url)
    svc = ForumService(engine, settings)
    try:
        while True:
            try:
                scheduled = tick(svc)
                if not deliver_one(engine) and not scheduled:
                    time.sleep(1)
            except Exception as exc:
                print(f"Forum worker retry: {type(exc).__name__}", flush=True)
                time.sleep(5)
    finally:
        engine.dispose()
