"""Bounded attachment decoding. No external URLs supplied by users are fetched."""

import io
import json
import re
import warnings
import struct
from uuid import UUID, uuid4

import httpx
from PIL import Image, ImageOps, UnidentifiedImageError
from markdown_it import MarkdownIt

from backend.human_play import engine, verse_replay
from backend.human_play.codec import gunzip_limited
from .db import execute, one, all_rows
from .errors import ForumError

MEDIA_RE = re.compile(r"/api/forum/v1/media/([0-9a-fA-F-]{36})(?![0-9a-fA-F-])")
REPLAY_RE = re.compile(r"\[\[replay:([0-9a-fA-F-]{36})(?:@\d+)?\]\]", re.IGNORECASE)
MARKDOWN = MarkdownIt("commonmark", {"html": False})
CODE_RUN = re.compile(r"`+|~{3,}")
MAX_IMAGE = 5 * 1024 * 1024
MAX_REPLAY = 2 * 1024 * 1024


def references(body):
    text = "\n".join(b.get("text", "") for b in body["blocks"])
    images = []
    for token in MARKDOWN.parse(text):
        for child in token.children or []:
            if child.type == "image":
                match = MEDIA_RE.fullmatch(child.attrGet("src") or "")
                if match:
                    images.append(match[1])
    # Replay syntax is parsed before Markdown in the reader. Preserve the same
    # explicit escapes and backtick/tilde code protection for binding permission.
    pos = 0
    visible = []
    while pos < len(text):
        marker = CODE_RUN.match(text, pos)
        if marker:
            run = marker.group()
            end = text.find(run, pos + len(run))
            while end != -1 and (
                (end > 0 and text[end - 1] == run[0])
                or (end + len(run) < len(text) and text[end + len(run)] == run[0])
            ):
                end = text.find(run, end + len(run))
            pos = len(text) if end == -1 else end + len(run)
            visible.append(" ")
            continue
        if text[pos : pos + 10].lower().startswith("\\[[replay:"):
            end = text.find("]]", pos + 10)
            pos = len(text) if end == -1 else end + 2
            continue
        visible.append(text[pos])
        pos += 1
    try:
        result = {UUID(x) for x in images + REPLAY_RE.findall("".join(visible))}
    except ValueError as exc:
        raise ForumError("INVALID_MEDIA", "附件标识无效。") from exc
    if len(result) > 20:
        raise ForumError("MEDIA_LIMIT", "每篇正文最多引用 20 个附件。")
    return result


def access(conn, ident, user, svc):
    row = one(
        conn, "SELECT * FROM forum_media WHERE id=:id AND status='active'", id=ident
    )
    if not row:
        raise ForumError("MEDIA_NOT_FOUND", "附件已移除或不可见。", 404)
    if user and (row["owner_id"] == user.id or svc.is_admin(user)):
        return row
    visible = one(
        conn,
        """SELECT 1 FROM forum_post_media m JOIN forum_posts p ON p.id=m.post_id
        JOIN forum_topics t ON t.id=p.topic_id WHERE m.media_id=:id AND p.status='published'
        AND (t.status='published' OR EXISTS(SELECT 1 FROM forum_roles r WHERE r.user_id=:u AND r.board_id=t.board_id)) LIMIT 1""",
        id=ident,
        u=user.id if user else 0,
    )
    if not visible:
        raise ForumError("MEDIA_NOT_FOUND", "附件已移除或不可见。", 404)
    return row


def bind(conn, post_id, body, user):
    ids = references(body)
    # Only owned assets or assets already attached to this exact post can be bound.
    # Reading someone else's public attachment is not authorization to republish it.
    for ident in sorted(ids, key=str):
        row = one(
            conn,
            """SELECT owner_id,status FROM forum_media WHERE id=:id FOR SHARE""",
            id=ident,
        )
        old = one(
            conn,
            "SELECT 1 FROM forum_post_media WHERE post_id=:p AND media_id=:id",
            p=post_id,
            id=ident,
        )
        if (
            not row
            or row["status"] != "active"
            or (row["owner_id"] != user.id and not old)
        ):
            raise ForumError(
                "INVALID_MEDIA", "附件不存在、已移除或不属于当前账号。", 403
            )
    execute(conn, "DELETE FROM forum_post_media WHERE post_id=:p", p=post_id)
    for ident in ids:
        execute(
            conn, "INSERT INTO forum_post_media VALUES(:p,:id)", p=post_id, id=ident
        )


def image_payload(raw):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if (
                    source.format not in {"PNG", "JPEG", "WEBP"}
                    or source.width * source.height > 16_000_000
                    or getattr(source, "n_frames", 1) != 1
                ):
                    raise ValueError("unsupported image")
                source.load()
                clean = ImageOps.exif_transpose(source).convert(
                    "RGBA" if source.mode in {"RGBA", "LA", "P"} else "RGB"
                )
                clean.thumbnail((2560, 2560))
                output = io.BytesIO()
                clean.save(output, format="WEBP", quality=90, method=3)
                return output.getvalue(), {"width": clean.width, "height": clean.height}
    except (
        UnidentifiedImageError,
        ValueError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ForumError(
            "INVALID_IMAGE", "请上传静态 PNG、JPEG 或 WebP 图片，最多 1600 万像素。"
        ) from exc


def replay_payload(raw):
    try:
        if len(raw) > MAX_REPLAY:
            raise ValueError("size")
        if raw.startswith(b"\x1f\x8b"):
            raw = gunzip_limited(raw, MAX_REPLAY)
        if raw.startswith((b"HPR1", b"HPR2")):
            header, events = engine.parse_replay(raw)
            initial = engine.initial(
                str(header["run_id"]), header["variant"], header["seed"]
            )
            state = engine.advance(initial, header["variant"], events)
            moves = [
                (code & 3, (code >> 2) & 15, 4 if code & 64 else 2, delta)
                for code, delta in engine.EVENT.iter_unpack(events)
            ]
            variant, board = header["variant"], initial["board"]
            score = state["score"]
        else:
            checked = verse_replay.inspect_replay(raw)
            variant, board, moves = verse_replay._rpl1(checked["normalized"])
            score = checked["score"]
        return {
            "variant": variant,
            "initial": board,
            "moves": moves,
            "score": score,
            "steps": len(moves),
            "verification": "rules-only",
        }
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        OverflowError,
        AttributeError,
        struct.error,
        RecursionError,
    ) as exc:
        raise ForumError(
            "INVALID_REPLAY",
            "录像无效或不支持；可使用 Play 的 HPR1/HPR2、Verse 或 RPL1 导出文件。",
        ) from exc


def fetch_play(settings, run_id):
    # Fixed administrator-configured origin; no credentials, redirects or proxy env.
    try:
        with httpx.Client(
            timeout=12, follow_redirects=False, trust_env=False
        ) as client:
            with client.stream(
                "GET", f"{settings.play_origin}/api/human/replays/{run_id}"
            ) as response:
                if response.status_code != 200:
                    raise ForumError(
                        "PLAY_UNAVAILABLE",
                        "该 Play 录像未公开、已撤回或暂不可用。",
                        404 if response.status_code in {403, 404} else 503,
                    )
                raw = bytearray()
                # Decode gzip with our bounded decoder, not an unbounded HTTP decoder.
                for chunk in response.iter_raw(chunk_size=65536):
                    raw.extend(chunk)
                    if len(raw) > MAX_REPLAY:
                        raise ForumError("REPLAY_TOO_LARGE", "录像超过大小限制。", 413)
                return replay_payload(bytes(raw))
    except httpx.HTTPError as exc:
        raise ForumError(
            "PLAY_UNAVAILABLE", "Play 录像服务暂不可用，请稍后重试。", 503
        ) from exc


def store(svc, user, raw, kind):
    # Reserve/check quota before expensive decoding; lock serializes per-account uploads.
    with svc.engine.begin() as conn:
        svc.writer(conn, user)
        quota = one(
            conn,
            "SELECT count(*) AS n,coalesce(sum(size),0) AS size FROM forum_media WHERE owner_id=:u AND status='active'",
            u=user.id,
        )
        if quota["n"] >= 100 or quota["size"] + len(raw) > 50 * 1024 * 1024:
            raise ForumError(
                "MEDIA_QUOTA",
                "附件配额已满（100 个 / 50 MiB），请先移除不需要的附件。",
                409,
            )
        if kind == "image":
            data, meta = image_payload(raw)
            mime = "image/webp"
        else:
            meta = replay_payload(raw)
            data, mime = raw, "application/octet-stream"
            # Canonical events are generated when reading, not duplicated in storage.
            meta = {k: v for k, v in meta.items() if k not in {"initial", "moves"}}
        if len(data) > (MAX_IMAGE if kind == "image" else MAX_REPLAY):
            raise ForumError("MEDIA_SIZE", "处理后的附件超过大小限制。", 413)
        if quota["size"] + len(data) > 50 * 1024 * 1024:
            raise ForumError(
                "MEDIA_QUOTA", "处理后的附件超过剩余容量，请先清理附件。", 409
            )
        ident = uuid4()
        execute(
            conn,
            """INSERT INTO forum_media(id,owner_id,kind,mime,data,metadata,size)
            VALUES(:id,:u,:kind,:mime,:data,CAST(:meta AS jsonb),:size)""",
            id=ident,
            u=user.id,
            kind=kind,
            mime=mime,
            data=data,
            meta=json.dumps(meta),
            size=len(data),
        )
        return {"id": str(ident), "kind": kind, "metadata": meta}


def import_play(svc, user, run_id):
    checked = fetch_play(svc.settings, run_id)
    with svc.engine.begin() as conn:
        svc.writer(conn, user)
        if (
            one(
                conn,
                "SELECT count(*) AS n FROM forum_media WHERE owner_id=:u AND status='active'",
                u=user.id,
            )["n"]
            >= 100
        ):
            raise ForumError("MEDIA_QUOTA", "附件数量已达上限。", 409)
        ident = uuid4()
        metadata = {
            "run_id": str(run_id),
            "variant": checked["variant"],
            "verification": "public-source",
        }
        execute(
            conn,
            """INSERT INTO forum_media(id,owner_id,kind,mime,data,metadata,size)
            VALUES(:id,:u,'play','application/json',:data,CAST(:meta AS jsonb),0)""",
            id=ident,
            u=user.id,
            data=b"",
            meta=json.dumps(metadata),
        )
        return {"id": str(ident), "kind": "play", "metadata": metadata}
