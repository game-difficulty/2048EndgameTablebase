"""Cross-process admission before reading replay bodies, separate from game write locks."""
from contextlib import contextmanager
import sqlite3
import os
import time
import uuid

from fastapi import HTTPException
from .store import db_path

# (per-user concurrency, global concurrency, requests/minute per user).
# Small checkpoints have their own lane so full replays cannot starve online play.
POLICIES = {'large': (1, 2, 12), 'small': (4, 16, 120), 'download': (1, 4, 10)}


@contextmanager
def connection():
    db = sqlite3.connect(db_path().with_suffix('.traffic.sqlite3'), timeout=2)
    # Admission counters are expendable after a machine crash; game evidence is not.
    # Counters do not need FULL synchronous durability. The game DB stays FULL.
    db.execute('PRAGMA synchronous=NORMAL')
    try:
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()


def initialize():
    with connection() as db:
        db.execute('PRAGMA journal_mode=WAL')
        db.executescript('''CREATE TABLE IF NOT EXISTS slots (
            token TEXT PRIMARY KEY, identity TEXT, lane TEXT, expires REAL);
            CREATE INDEX IF NOT EXISTS slot_expiry ON slots(expires);
            CREATE TABLE IF NOT EXISTS rates (
            identity TEXT, lane TEXT, minute INTEGER, count INTEGER,
            PRIMARY KEY(identity,lane));
            CREATE INDEX IF NOT EXISTS rate_expiry ON rates(minute);''')


def acquire(identity, lane):
    defaults = POLICIES[lane]
    per_user, global_limit, frequency = (max(1, int(os.getenv(f'HUMAN_{lane.upper()}_{suffix}', str(default))))
        for suffix, default in zip(('PER_USER', 'GLOBAL', 'PER_MINUTE'), defaults))
    now = time.time(); minute = int(now // 60); token = uuid.uuid4().hex
    with connection() as db:
        db.execute('BEGIN IMMEDIATE')
        db.execute('DELETE FROM slots WHERE expires<?', (now,))
        db.execute('DELETE FROM rates WHERE minute<?', (minute - 2,))
        total, own = db.execute('SELECT count(*),coalesce(sum(identity=?),0) FROM slots WHERE lane=?',
                               (identity, lane)).fetchone()
        row = db.execute('SELECT minute,count FROM rates WHERE identity=? AND lane=?', (identity, lane)).fetchone()
        count = row[1] if row and row[0] == minute else 0
        if count >= frequency or total >= global_limit or own >= per_user:
            wait = max(1, 60 - int(now % 60)) if count >= frequency else 2
            raise HTTPException(429, {'code': 'replay_rate_limit'}, headers={'Retry-After': str(wait)})
        db.execute('INSERT INTO slots VALUES(?,?,?,?)', (token, identity, lane, now + 300))
        db.execute('INSERT OR REPLACE INTO rates VALUES(?,?,?,?)', (identity, lane, minute, count + 1))
    return token


def release(token):
    with connection() as db:
        db.execute('DELETE FROM slots WHERE token=?', (token,))
