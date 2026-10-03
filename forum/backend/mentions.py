"""A Markdown inline rule: escaped text, code and link labels never ping users."""

import re
from markdown_it import MarkdownIt
from .db import one, execute
from .errors import ForumError


def mention_rule(state, silent):
    if state.linkLevel:
        return False
    match = re.match(r"<@([1-9][0-9]{0,14})>", state.src[state.pos :])
    if not match:
        return False
    if not silent:
        token = state.push("mention", "", 0)
        token.content = match[1]
    state.pos += len(match[0])
    return True


parser = MarkdownIt("commonmark", {"html": False})
parser.inline.ruler.before("autolink", "mention", mention_rule)


def recipients(body):
    result = set()
    for block in body["blocks"]:
        if block["type"] == "paragraph":
            for token in parser.parse(block["text"]):
                for child in token.children or []:
                    if child.type == "mention":
                        result.add(int(child.content))
    if len(result) > 10:
        raise ForumError("MENTION_LIMIT", "每条内容最多提及 10 位用户。")
    return result


def notify(conn, post_id, body, actor):
    for target in sorted(recipients(body) - {actor.id}):
        inserted = one(
            conn,
            """INSERT INTO forum_mentions(post_id,user_id)
            SELECT :p,user_id FROM forum_profiles WHERE user_id=:u
            ON CONFLICT DO NOTHING RETURNING user_id""",
            p=post_id,
            u=target,
        )
        if inserted:
            execute(
                conn,
                """INSERT INTO forum_notifications(recipient_id,actor_id,post_id,kind)
                SELECT :u,:a,p.id,'mention' FROM forum_posts p JOIN forum_topics t ON t.id=p.topic_id
                WHERE p.id=:p AND p.status='published' AND t.status='published'
                AND forum_can_notify(:u,:a,t.board_id,'mention')
                ON CONFLICT(recipient_id,post_id) DO UPDATE SET kind='mention' """,
                p=post_id,
                u=target,
                a=actor.id,
            )
