import base64
import json
from datetime import datetime
from .db import all_rows
from .errors import ForumError


def search_topics(
    svc,
    user,
    board="",
    cursor="",
    query="",
    saved=False,
    limit=20,
    view="activity",
    tag="",
    author=None,
    topic_id=None,
    since=None,
    until=None,
    kind="",
):
    params = {"limit": limit + 1, "viewer": user.id if user else 0}
    where = ["t.status='published'"]
    column = "t.created_at" if view == "newest" else "t.last_activity"
    fingerprint = json.dumps(
        [board, query, saved, view, tag, author, topic_id, since, until, kind],
        sort_keys=True,
    )
    if cursor:
        try:
            decoded = json.loads(
                base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
            )
            if len(decoded) == 4:
                stamp, ident, pinned, context = decoded
                if context != fingerprint:
                    raise ValueError()
            elif len(decoded) in {2, 3} and view == "activity":
                stamp, ident = decoded[:2]
                pinned = decoded[2] if len(decoded) == 3 else False
            else:
                raise ValueError()
            stamp = datetime.fromisoformat(stamp)
            if (
                not stamp.tzinfo
                or type(ident) != int
                or not 0 < ident < 2**63
                or type(pinned) != bool
            ):
                raise ValueError()
            params.update(stamp=stamp, cursor_id=ident, pinned=pinned)
            where.append(f"(t.pinned,{column},t.id)<(:pinned,:stamp,:cursor_id)")
        except (ValueError, TypeError, OverflowError):
            raise ForumError("INVALID_CURSOR", "分页条件已变化，请重新搜索。")
    for key, value, sql in [
        ("board", board, "b.slug=:board"),
        ("tag", tag, "t.tags @> CAST(:tag AS jsonb)"),
        ("author", author, "t.author_id=:author"),
        ("topic_id", topic_id, "t.id=:topic_id"),
        ("kind", kind, "t.kind=:kind"),
    ]:
        if value:
            where.append(sql)
            params[key] = json.dumps([value]) if key == "tag" else value
    for key, value, operator in [("since", since, ">="), ("until", until, "<")]:
        if value:
            try:
                parsed = datetime.fromisoformat(value)
                if not parsed.tzinfo:
                    raise ValueError()
            except ValueError:
                raise ForumError("INVALID_DATE", "筛选时间需要包含时区。")
            params[key] = parsed
            where.append(f"t.created_at{operator}:{key}")
    if query:
        params["query"] = (
            "%"
            + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            + "%"
        )
        # pg_trgm accelerates longer substrings; 1–2 character Chinese terms use a separate GIN index.
        if len(query) <= 2:
            params["short"] = [query.lower()]
            title = "forum_short_terms(t.title) @> CAST(:short AS text[]) AND t.title ILIKE :query"
            body = "forum_short_terms(p.body_text) @> CAST(:short AS text[]) AND p.body_text ILIKE :query"
        else:
            title = "t.title ILIKE :query"
            body = "p.body_text ILIKE :query"
        where.append(
            f"(({title}) OR EXISTS(SELECT 1 FROM forum_posts p WHERE p.topic_id=t.id AND p.status='published' AND {body}))"
        )
    if saved:
        where.append(
            "EXISTS(SELECT 1 FROM forum_bookmarks k WHERE k.topic_id=t.id AND k.user_id=:viewer)"
        )
    if view in {"following", "unread"} or saved:
        if not user:
            raise ForumError("AUTH_REQUIRED", "请登录后查看。", 401)
    if view == "following":
        where.append(
            "EXISTS(SELECT 1 FROM forum_follows f WHERE f.user_id=:viewer AND f.followed_id=t.author_id)"
        )
    if view == "unread":
        where.append(
            "NOT EXISTS(SELECT 1 FROM forum_reading r WHERE r.user_id=:viewer AND r.topic_id=t.id AND r.post_number>=t.next_post_number-1)"
        )
    if view == "unanswered":
        where.append(
            "NOT EXISTS(SELECT 1 FROM forum_posts p WHERE p.topic_id=t.id AND p.status='published' AND p.post_number>1)"
        )
    if view == "featured":
        where.append("t.pinned")
    where.append(
        "NOT EXISTS(SELECT 1 FROM forum_blocks x WHERE x.user_id=:viewer AND ((x.kind='board' AND x.target_id=t.board_id) OR (x.kind='user' AND x.target_id=t.author_id)))"
    )
    with svc.engine.connect() as conn:
        rows = all_rows(
            conn,
            """SELECT t.*,b.slug AS board_slug,b.name AS board_name,u.display_name,
            (SELECT count(*) FROM forum_posts p WHERE p.topic_id=t.id AND p.status='published' AND p.post_number>1) AS replies
            FROM forum_topics t JOIN forum_boards b ON b.id=t.board_id JOIN forum_profiles u ON u.user_id=t.author_id
            WHERE """
            + " AND ".join(where)
            + f" ORDER BY t.pinned DESC,{column} DESC,t.id DESC LIMIT :limit",
            **params,
        )
    next_cursor = ""
    if len(rows) > limit:
        row = rows[limit - 1]
        next_cursor = (
            base64.urlsafe_b64encode(
                json.dumps(
                    [
                        row[column.split(".")[1]].isoformat(),
                        row["id"],
                        row["pinned"],
                        fingerprint,
                    ]
                ).encode()
            )
            .decode()
            .rstrip("=")
        )
    return {"items": rows[:limit], "next_cursor": next_cursor}
