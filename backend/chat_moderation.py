"""Small, editable deny list; intentionally not a comprehensive moderation system."""
import json
import unicodedata
from functools import lru_cache
from pathlib import Path
from backend.auth.db import auth_db


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if not unicodedata.category(c).startswith(('Z', 'C', 'P')))


@lru_cache(maxsize=1)
def terms():
    path = Path(__file__).resolve().parents[1] / 'docs_and_configs/chat_blocklist.json'
    return tuple(normalized(term) for term in json.loads(path.read_text(encoding='utf-8')) if normalized(term))


def blocked(text):
    content = normalized(text)
    return any(term in content for term in terms())


def visible_messages(messages):
    # Only stable user IDs may be used to hide content. Old nickname-only records
    # cannot be safely attributed after rename or nickname reuse.
    user_ids = {m['user_id'] for m in messages if m.get('type') == 'chat' and m.get('user_id')}
    if not user_ids:
        return messages
    placeholders = ','.join('?' for _ in user_ids)
    with auth_db() as db:
        rows = db.execute(f"SELECT id FROM users WHERE id IN ({placeholders}) AND status != 'active'", tuple(user_ids))
        disabled_ids = {row['id'] for row in rows}
    return [m for m in messages if not (m.get('type') == 'chat' and m.get('user_id') in disabled_ids)]
