"""Public-only metadata. Hidden titles never enter HTML or sitemap output."""

import html
import re
from .db import one, all_rows


def page_html(engine, template, path, origin):
    match = re.fullmatch(r"t/([1-9][0-9]{0,14})", path)
    title = "2048 社区"
    description = "分享 2048 局面、复盘、攻略与讨论。"
    status = 200
    public = path == "" or path.startswith("c/")
    if match:
        with engine.connect() as conn:
            row = one(
                conn,
                "SELECT t.title,left(p.body_text,180) AS excerpt FROM forum_topics t JOIN forum_posts p ON p.topic_id=t.id AND p.post_number=1 WHERE t.id=:id AND t.status='published' AND p.status='published'",
                id=int(match[1]),
            )
        if row:
            title = row["title"] + " · 2048 社区"
            description = row["excerpt"]
            public = True
        else:
            title = "主题不可见 · 2048 社区"
            description = "该主题不存在或已不可见。"
            status = 404
    canonical = origin + "/" + path
    metadata = f'<meta name="description" content="{html.escape(description,quote=True)}"><meta name="robots" content="{"index,follow" if public else "noindex,nofollow"}"><link rel="canonical" href="{html.escape(canonical,quote=True)}"><meta property="og:title" content="{html.escape(title,quote=True)}"><meta property="og:description" content="{html.escape(description,quote=True)}">'
    result = re.sub(
        r"<title>.*?</title>",
        lambda _: "<title>" + html.escape(title) + "</title>",
        template,
        count=1,
        flags=re.S,
    )
    return result.replace("</head>", metadata + "</head>"), status


def sitemap(engine, origin, before):
    with engine.connect() as conn:
        rows = all_rows(
            conn,
            "SELECT id,last_activity FROM forum_topics WHERE status='published' AND id<:before ORDER BY id DESC LIMIT 1000",
            before=before,
        )
    entries = [
        f'<url><loc>{html.escape(origin)}/t/{r["id"]}</loc><lastmod>{r["last_activity"].date().isoformat()}</lastmod></url>'
        for r in rows
    ]
    return (
        '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        + "".join(entries)
        + "</urlset>"
    )
