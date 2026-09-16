"""Milestone giveaways. Drawing and paid-token credits share one transaction."""
import secrets
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from backend.auth.db import auth_db

RULES = {32768: (5_000, 300, 1_000), 65536: (20_000, 1_000, 3_000)}
DRAW_SECONDS = 180
RESULT_SECONDS = 180
MAX_WINNERS = 10


def init_schema():
    with auth_db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS live_lucky_bags (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, milestone INTEGER NOT NULL,
                created_at REAL NOT NULL, draw_at REAL NOT NULL, drawn_at REAL,
                expires_at REAL NOT NULL, pool INTEGER NOT NULL, minimum INTEGER NOT NULL,
                maximum INTEGER NOT NULL, UNIQUE(run_id, milestone)
            );
            CREATE INDEX IF NOT EXISTS live_lucky_due ON live_lucky_bags(draw_at) WHERE drawn_at IS NULL;
            CREATE INDEX IF NOT EXISTS live_lucky_visible ON live_lucky_bags(expires_at);
            CREATE INDEX IF NOT EXISTS live_lucky_ledger_users ON token_ledger(user_id)
                WHERE event_type='live_lucky_award';
            CREATE TABLE IF NOT EXISTS live_lucky_entries (
                bag_id TEXT NOT NULL REFERENCES live_lucky_bags(id),
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                joined_at REAL NOT NULL, award INTEGER NOT NULL DEFAULT 0,
                present INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(bag_id, user_id)
            );
        ''')


def create(run_id, milestone, now=None):
    pool, minimum, maximum = RULES[milestone]
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('''INSERT OR IGNORE INTO live_lucky_bags
            (id,run_id,milestone,created_at,draw_at,expires_at,pool,minimum,maximum)
            VALUES(?,?,?,?,?,?,?,?,?)''',
            (str(uuid.uuid4()), run_id, milestone, now, now + DRAW_SECONDS,
             now + DRAW_SECONDS + RESULT_SECONDS, pool, minimum, maximum))
        return dict(db.execute('SELECT * FROM live_lucky_bags WHERE run_id=? AND milestone=?',
                               (run_id, milestone)).fetchone())


def listing(user_id=None, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        rows = db.execute('''SELECT b.*, COUNT(e.user_id) AS participants,
            SUM(CASE WHEN e.award>0 THEN 1 ELSE 0 END) AS winners,
            COALESCE(SUM(e.award),0) AS distributed
            FROM live_lucky_bags b LEFT JOIN live_lucky_entries e ON e.bag_id=b.id
            WHERE b.expires_at>? OR b.drawn_at IS NULL GROUP BY b.id ORDER BY b.created_at,b.id''', (now,))
        bags = [dict(row, max_winners=MAX_WINNERS) for row in rows]
        if user_id is not None:
            for bag in bags:
                entry = db.execute('SELECT award,present FROM live_lucky_entries WHERE bag_id=? AND user_id=?',
                                   (bag['id'], user_id)).fetchone()
                bag['joined'] = bool(entry)
                bag['award'] = entry['award'] if entry else 0
                bag['present'] = bool(entry['present']) if entry else False
    return bags


def join(bag_id, user_id, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        bag = db.execute('SELECT * FROM live_lucky_bags WHERE id=?', (bag_id,)).fetchone()
        if not bag:
            raise HTTPException(404, 'lucky_bag_not_found')
        # An uncertain response can be retried even after the draw closes.
        if db.execute('SELECT 1 FROM live_lucky_entries WHERE bag_id=? AND user_id=?', (bag_id, user_id)).fetchone():
            return
        if bag['drawn_at'] is not None or now >= bag['draw_at']:
            raise HTTPException(409, 'lucky_bag_closed')
        db.execute('INSERT INTO live_lucky_entries(bag_id,user_id,joined_at) VALUES(?,?,?)', (bag_id, user_id, now))


def amounts(count, pool, minimum, maximum):
    """Reserve minimums, then shuffle so every winner has equal expectation."""
    if count < 0 or minimum < 1 or maximum < minimum or pool < count * minimum:
        raise ValueError("Invalid lucky-bag bounds")
    remaining = pool
    result = []
    for left in range(count - 1, -1, -1):
        low = minimum
        high = min(maximum, remaining - left * minimum)
        value = low + secrets.randbelow(high - low + 1)
        result.append(value)
        remaining -= value
    secrets.SystemRandom().shuffle(result)
    return result


def draw(bag_id, present_users, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        bag = db.execute('SELECT * FROM live_lucky_bags WHERE id=?', (bag_id,)).fetchone()
        if not bag or bag['drawn_at'] is not None or now < bag['draw_at']:
            return False
        entrants = [row[0] for row in db.execute('''SELECT e.user_id FROM live_lucky_entries e
            JOIN users u ON u.id=e.user_id WHERE e.bag_id=? AND u.status='active' ''', (bag_id,))
            if row[0] in present_users]
        db.executemany('UPDATE live_lucky_entries SET present=1 WHERE bag_id=? AND user_id=?',
                       [(bag_id, uid) for uid in entrants])
        winners = secrets.SystemRandom().sample(entrants, min(MAX_WINNERS, len(entrants)))
        prizes = amounts(len(winners), bag['pool'], bag['minimum'], bag['maximum'])
        stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
        for uid, prize in zip(winners, prizes):
            units = prize * 1000
            db.execute('''INSERT OR IGNORE INTO token_accounts
                (user_id,created_at,updated_at) VALUES(?,?,?)''', (uid, stamp, stamp))
            before = db.execute('SELECT bonus_balance_units+paid_balance_units FROM token_accounts WHERE user_id=?', (uid,)).fetchone()[0]
            db.execute('UPDATE token_accounts SET paid_balance_units=paid_balance_units+?,updated_at=? WHERE user_id=?',
                       (units, stamp, uid))
            db.execute('''INSERT INTO token_ledger(user_id,event_type,operation_key,paid_delta_units,
                balance_before_units,balance_after_units,metadata_json,created_at)
                VALUES(?,'live_lucky_award',?,?,?,?,?,?)''',
                (uid, 'live_lucky:' + bag_id, units, before, before + units, '{"milestone":%d}' % bag['milestone'], stamp))
            db.execute('UPDATE live_lucky_entries SET award=? WHERE bag_id=? AND user_id=?', (prize, bag_id, uid))
        db.execute('UPDATE live_lucky_bags SET drawn_at=?,expires_at=? WHERE id=?', (now, now + RESULT_SECONDS, bag_id))
        return True


def cleanup(now=None):
    # Keep the two small run/milestone keys forever to prevent replayed triggers.
    # Award audits remain in token_ledger; participation need not grow forever.
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('''DELETE FROM live_lucky_entries WHERE bag_id IN
            (SELECT id FROM live_lucky_bags WHERE drawn_at IS NOT NULL AND expires_at<?)''', (now - 30 * 86400,))
