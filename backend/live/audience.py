"""Small session aggregates; never a per-viewer heartbeat/event log."""
import hashlib
import time
import uuid

from backend.auth.db import auth_db


def init_schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS live_audience_session (
            singleton INTEGER PRIMARY KEY CHECK(singleton=1), session_id TEXT NOT NULL,
            started_at REAL NOT NULL, last_online_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS live_audience_scores (
            session_id TEXT NOT NULL, actor_key TEXT NOT NULL, watch_seconds REAL NOT NULL DEFAULT 0,
            likes INTEGER NOT NULL DEFAULT 0, messages INTEGER NOT NULL DEFAULT 0,
            gift_units INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(session_id,actor_key)
        );
    ''')
    db.execute('INSERT OR IGNORE INTO live_audience_session VALUES(1,?,?,0)', (str(uuid.uuid4()), time.time()))


def session(db):
    return db.execute('SELECT * FROM live_audience_session WHERE singleton=1').fetchone()


def online_tick(watch, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        current = session(db)
        if current['last_online_at'] and now - current['last_online_at'] > 1800:
            db.execute('DELETE FROM live_audience_scores')
            db.execute('UPDATE live_audience_session SET session_id=?,started_at=? WHERE singleton=1', (str(uuid.uuid4()), now))
        db.execute('UPDATE live_audience_session SET last_online_at=? WHERE singleton=1', (now,))
        for key, seconds in watch.items():
            add(db, key, 'watch_seconds', min(10, max(0, seconds)))


def add(db, actor_key, kind, amount=1):
    if kind not in {'watch_seconds', 'likes', 'messages', 'gift_units'}:
        raise ValueError('invalid_contribution')
    current = session(db)
    db.execute('INSERT OR IGNORE INTO live_audience_scores(session_id,actor_key) VALUES(?,?)', (current['session_id'], actor_key))
    value = f'min(10,{kind}+?)' if kind in {'likes', 'messages'} else f'{kind}+?'
    db.execute(f'UPDATE live_audience_scores SET {kind}={value} WHERE session_id=? AND actor_key=?',
               (amount, current['session_id'], actor_key))


def record(actor_key, kind, amount=1):
    with auth_db() as db:
        add(db, actor_key, kind, amount)


def ranking(identities):
    with auth_db() as db:
        current = session(db)
        # Only connected identities are returned, never IPs, emails or private IDs.
        keys = list(identities)
        placeholders = ','.join('?' for _ in keys)
        rows = {row['actor_key']: row for row in db.execute(
            f'SELECT * FROM live_audience_scores WHERE session_id=? AND actor_key IN ({placeholders})',
            [current['session_id'], *keys])} if keys else {}
    result = []
    for key, identity in identities.items():
        row = rows.get(key)
        units = row['gift_units'] + (int(row['watch_seconds'] // 600) + row['likes'] + row['messages']) * 1000 if row else 0
        public_id = hashlib.sha256((current['session_id'] + key).encode()).hexdigest()[:16]
        result.append(dict(identity, id=public_id, contribution_units=units))
    result.sort(key=lambda item: (-(item.get('supporter_level') == 2), -item['contribution_units'], item['id']))
    return dict(session_id=current['session_id'], viewers=[dict(item, rank=index+1) for index, item in enumerate(result)])


class Presence:
    def __init__(self):
        self.connections = {}
        self.started = {}

    def join(self, connection, key, identity, now=None):
        self.connections[connection] = (key, identity)
        self.started.setdefault(key, time.monotonic() if now is None else now)

    def leave(self, connection):
        entry = self.connections.pop(connection, None)
        if entry and entry[0] not in self.identities():
            self.started.pop(entry[0], None)

    def identities(self):
        return {key: identity for key, identity in self.connections.values()}

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        watch = {key: max(0, now-start) for key, start in self.started.items()}
        self.started = dict.fromkeys(self.started, now)
        return watch
