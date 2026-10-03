"""Sparse, authenticated assistance evidence shared by Main and Play."""
from __future__ import annotations

from contextlib import contextmanager
import math
import os
from pathlib import Path
import sqlite3
import time

from backend.auth.db import get_auth_db_path
from backend.lookup_mask import MASK_VERSION, descriptor, lookup_key

DIRECTIONS = ('left', 'right', 'up', 'down')


@contextmanager
def database():
    path = Path(os.environ.get('ASSISTANCE_EVIDENCE_DB') or get_auth_db_path().with_name('assistance-evidence.sqlite3'))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=3)
    db.row_factory = sqlite3.Row
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    with database() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''
        CREATE TABLE IF NOT EXISTS assistance_events (
            id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, event_id TEXT NOT NULL,
            kind TEXT NOT NULL, source TEXT NOT NULL, variant TEXT NOT NULL,
            board_key TEXT NOT NULL, full_pattern TEXT NOT NULL, mask_version INTEGER,
            queried_at_ms INTEGER NOT NULL, received_at_ms INTEGER NOT NULL,
            direction TEXT NOT NULL, result_status TEXT NOT NULL,
            UNIQUE(user_id,event_id)
        );
        CREATE INDEX IF NOT EXISTS assistance_user_time ON assistance_events(user_id,queried_at_ms);
        CREATE INDEX IF NOT EXISTS assistance_user_received ON assistance_events(user_id,received_at_ms);
        CREATE INDEX IF NOT EXISTS assistance_received ON assistance_events(received_at_ms);
        ''')


def insert(user_id, event_id, kind, source, variant, board_key, pattern, mask_version, queried_at_ms, direction, status):
    received = round(time.time() * 1000)
    if type(queried_at_ms) is not int or not received - 86400000 <= queried_at_ms <= received + 300000:
        raise ValueError('invalid_evidence_time')
    with database() as db:
        if db.execute('SELECT 1 FROM assistance_events WHERE user_id=? AND event_id=?', (user_id, event_id)).fetchone():
            return
        # Bound abuse without putting a rate-limit failure on the game's critical path.
        recent = db.execute('SELECT COUNT(*) FROM assistance_events WHERE user_id=? AND received_at_ms>?',
                            (user_id, received - 60000)).fetchone()[0]
        if recent >= 120:
            return
        db.execute('''INSERT OR IGNORE INTO assistance_events
            (user_id,event_id,kind,source,variant,board_key,full_pattern,mask_version,
             queried_at_ms,received_at_ms,direction,result_status) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
            (user_id, event_id, kind, source, variant, board_key, pattern, mask_version,
             queried_at_ms, received, direction, status))


def record_table(user_id, event_id, source, board_encoded, full_pattern, queried_at_ms, results):
    if not user_id or source not in ('setboard', 'palette'):
        return
    numeric = {direction: float(results[direction]) for direction in DIRECTIONS
               if type(results.get(direction)) in (int, float) and math.isfinite(results[direction])}
    if not numeric:
        return
    direction = max(numeric, key=numeric.get)  # DIRECTIONS provides deterministic ties.
    variant = descriptor(full_pattern)[4]
    insert(user_id, event_id, 'table', source, variant, lookup_key(board_encoded, full_pattern),
           full_pattern, MASK_VERSION, queried_at_ms, direction, 'found')


def record_ai(user_id, payload):
    codes = payload['board_codes']
    if len(codes) != 16 or any(type(code) is not int or not 0 <= code <= 31 for code in codes):
        raise ValueError('invalid_evidence_board')
    # Delimit exponents: 15 and 16 must never collapse into the same nibble.
    key = ','.join(map(str, codes))
    insert(user_id, payload['event_id'], 'ai', 'setboard', '4x4', key, '', None,
           payload['queried_at_ms'], payload['direction'], 'found')


def candidates(user_id, start_ms, end_ms):
    with database() as db:
        return [dict(row) for row in db.execute('''SELECT * FROM assistance_events
            WHERE user_id=? AND queried_at_ms BETWEEN ? AND ? ORDER BY queried_at_ms,id''',
            (user_id, start_ms, end_ms))]


def cleanup_expired():
    # Review details contain their own evidence copy; deleting the raw log never
    # deletes a pending or completed review. Bound maintenance transaction size.
    days = max(30, int(os.environ.get('ASSISTANCE_EVIDENCE_RETENTION_DAYS', '90')))
    cutoff = round((time.time() - days * 86400) * 1000)
    with database() as db:
        db.execute('DELETE FROM assistance_events WHERE id IN '
                   '(SELECT id FROM assistance_events WHERE received_at_ms<? LIMIT 1000)', (cutoff,))
