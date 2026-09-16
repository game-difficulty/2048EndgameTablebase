"""Persistent FIFO giveaways funded exclusively by paid Tokens."""
import json
import secrets
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from backend.auth.db import auth_db


def init_schema():
    with auth_db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS live_red_envelopes (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                sender_id INTEGER NOT NULL REFERENCES users(id), request_id TEXT NOT NULL,
                actor_json TEXT NOT NULL, amount INTEGER NOT NULL, count INTEGER NOT NULL,
                mode TEXT NOT NULL, shares_json TEXT NOT NULL, remainder INTEGER NOT NULL,
                created_at REAL NOT NULL, started_at REAL, closed_at REAL,
                claimed INTEGER NOT NULL DEFAULT 0, distributed INTEGER NOT NULL DEFAULT 0,
                refunded INTEGER NOT NULL DEFAULT 0, announced INTEGER NOT NULL DEFAULT 0,
                UNIQUE(sender_id,request_id)
            );
            CREATE INDEX IF NOT EXISTS live_red_queue ON live_red_envelopes(sequence) WHERE closed_at IS NULL;
            CREATE INDEX IF NOT EXISTS live_red_sender ON live_red_envelopes(sender_id,created_at);
            CREATE INDEX IF NOT EXISTS live_red_outbox ON live_red_envelopes(sequence) WHERE announced=0;
            CREATE TABLE IF NOT EXISTS live_red_claims (
                envelope_id TEXT NOT NULL REFERENCES live_red_envelopes(id),
                user_id INTEGER NOT NULL REFERENCES users(id), amount INTEGER NOT NULL,
                created_at REAL NOT NULL, PRIMARY KEY(envelope_id,user_id)
            );
        ''')


def split_amount(amount, count, mode):
    if mode == 'equal':
        return [amount // count] * count
    # Uniform positive integer compositions are exchangeable; shuffling also
    # explicitly decouples the allocation from claim order. No client seed.
    cuts = [0, *sorted(secrets.SystemRandom().sample(range(1, amount), count - 1)), amount]
    shares = [right - left for left, right in zip(cuts, cuts[1:])]
    secrets.SystemRandom().shuffle(shares)
    return shares


def _transfer(db, uid, amount, kind, envelope_id, now):
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
    db.execute('INSERT OR IGNORE INTO token_accounts(user_id,created_at,updated_at) VALUES(?,?,?)', (uid, stamp, stamp))
    account = db.execute('SELECT * FROM token_accounts WHERE user_id=?', (uid,)).fetchone()
    units = amount * 1000
    if account['paid_balance_units'] + units < 0:
        raise HTTPException(402, 'red_paid_balance')
    before = account['paid_balance_units'] + account['bonus_balance_units']
    db.execute('UPDATE token_accounts SET paid_balance_units=paid_balance_units+?,updated_at=? WHERE user_id=?', (units, stamp, uid))
    db.execute('''INSERT INTO token_ledger(user_id,event_type,operation_key,paid_delta_units,
        balance_before_units,balance_after_units,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)''',
        (uid, kind, 'live_red:' + envelope_id, units, before, before + units, '{}', stamp))


def _advance(db, now):
    head = db.execute('SELECT * FROM live_red_envelopes WHERE closed_at IS NULL ORDER BY sequence LIMIT 1').fetchone()
    if head and head['started_at'] is not None:
        expired = now >= head['started_at'] + 60
        exhausted = head['claimed'] == head['count'] and now >= head['started_at'] + 15
        if expired or exhausted:
            refund = head['amount'] - head['remainder'] - head['distributed']
            if refund:
                _transfer(db, head['sender_id'], refund, 'live_red_refund', head['id'], now)
            db.execute('UPDATE live_red_envelopes SET closed_at=?,refunded=? WHERE id=?', (now, refund, head['id']))
            head = db.execute('SELECT * FROM live_red_envelopes WHERE closed_at IS NULL ORDER BY sequence LIMIT 1').fetchone()
    if head and head['started_at'] is None:
        db.execute('UPDATE live_red_envelopes SET started_at=? WHERE id=?', (now, head['id']))


def _public(row, now):
    started = row['started_at']
    status = ('closed' if row['closed_at'] is not None else 'queued' if started is None
              else 'exhausted' if row['claimed'] == row['count'] else 'expired' if now >= started + 60 else 'active')
    return dict(id=row['id'], sender_id=row['sender_id'], actor=json.loads(row['actor_json']),
                amount=row['amount'], count=row['count'], mode=row['mode'], remainder=row['remainder'],
                claimed=row['claimed'], distributed=row['distributed'], refunded=row['refunded'],
                created_at=row['created_at'], started_at=started, expires_at=started + 60 if started is not None else None,
                status=status)


def tick(now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        _advance(db, now)
        head = db.execute('SELECT * FROM live_red_envelopes WHERE closed_at IS NULL ORDER BY sequence LIMIT 1').fetchone()
        count = db.execute('SELECT COUNT(*) FROM live_red_envelopes WHERE closed_at IS NULL').fetchone()[0]
        return dict(active=_public(head, now) if head else None, queued=max(0, count - 1))


def detail(envelope_id, uid=None, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        row = db.execute('SELECT * FROM live_red_envelopes WHERE id=?', (envelope_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'red_not_found')
        result = _public(row, now)
        own = db.execute('SELECT amount FROM live_red_claims WHERE envelope_id=? AND user_id=?', (envelope_id, uid)).fetchone()
        result['award'] = own[0] if own else 0
        return result


def create(uid, actor, body, now=None):
    now = time.time() if now is None else now
    amount, count, mode, request_id = (body.get(key) for key in ('amount', 'count', 'mode', 'request_id'))
    try:
        if not isinstance(request_id, str) or str(uuid.UUID(request_id)) != request_id:
            raise ValueError()
    except (ValueError, AttributeError):
        raise HTTPException(400, 'red_invalid')
    if type(amount) is not int or type(count) is not int or not 1000 <= amount <= 50000 or not 5 <= count <= 20 or mode not in ('equal', 'random'):
        raise HTTPException(400, 'red_invalid')
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT * FROM live_red_envelopes WHERE sender_id=? AND request_id=?', (uid, request_id)).fetchone()
        if old:
            if (old['amount'], old['count'], old['mode']) != (amount, count, mode):
                raise HTTPException(409, 'red_request_conflict')
            return _public(old, now)
        recent = db.execute('SELECT MAX(created_at) FROM live_red_envelopes WHERE sender_id=?', (uid,)).fetchone()[0]
        if recent is not None and now < recent + 15:
            raise HTTPException(429, 'red_cooldown')
        envelope_id = str(uuid.uuid4())
        shares = split_amount(amount, count, mode)
        _transfer(db, uid, -amount, 'live_red_send', envelope_id, now)
        db.execute('''INSERT INTO live_red_envelopes
            (id,sender_id,request_id,actor_json,amount,count,mode,shares_json,remainder,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (envelope_id, uid, request_id, json.dumps(actor, ensure_ascii=False), amount, count, mode,
             json.dumps(shares), amount - sum(shares), now))
        _advance(db, now)
        return _public(db.execute('SELECT * FROM live_red_envelopes WHERE id=?', (envelope_id,)).fetchone(), now)


def claim(envelope_id, uid, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM live_red_envelopes WHERE id=?', (envelope_id,)).fetchone()
        if not row:
            raise HTTPException(404, 'red_not_found')
        old = db.execute('SELECT amount FROM live_red_claims WHERE envelope_id=? AND user_id=?', (envelope_id, uid)).fetchone()
        if old:
            return dict(_public(row, now), award=old[0])
        if uid == row['sender_id']:
            raise HTTPException(409, 'red_own')
        if _public(row, now)['status'] != 'active':
            return dict(_public(row, now), award=0)
        award = json.loads(row['shares_json'])[row['claimed']]
        _transfer(db, uid, award, 'live_red_award', envelope_id, now)
        db.execute('INSERT INTO live_red_claims VALUES(?,?,?,?)', (envelope_id, uid, award, now))
        db.execute('UPDATE live_red_envelopes SET claimed=claimed+1,distributed=distributed+? WHERE id=?', (award, envelope_id))
        row = db.execute('SELECT * FROM live_red_envelopes WHERE id=?', (envelope_id,)).fetchone()
        return dict(_public(row, now), award=award)


def events(pending=False):
    with auth_db() as db:
        rows = db.execute('SELECT * FROM live_red_envelopes ' + ('WHERE announced=0 ORDER BY sequence LIMIT 100' if pending else 'ORDER BY sequence DESC LIMIT 100'))
        return [dict(type='red_envelope', id='red:' + row['id'], envelope_id=row['id'], at=row['created_at'],
                     amount=row['amount'], count=row['count'], **json.loads(row['actor_json'])) for row in rows]


def delivered(ids):
    with auth_db() as db:
        db.executemany('UPDATE live_red_envelopes SET announced=1 WHERE id=?', [(value,) for value in ids])
