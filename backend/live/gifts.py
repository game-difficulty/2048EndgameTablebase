"""Gift orders share the token database transaction; animations are an outbox consumer."""
import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException
from backend.auth.db import auth_db
from backend.quota.config import resolve_pricing_snapshot, operation_cost_units, apply_pricing_multipliers
from backend.quota.service import consume_operation_tokens_once, get_token_balance
from .supporters import public_actor
from . import audience

GIFTS = [
    ('heart', '小心意', 'Little Heart', 0), ('flowers', '鲜花', 'Flowers', 0),
    ('two', '来个 2', 'A Little Two', 0), ('four', '好 4', 'Nice Four', 0),
    ('dealer', '发牌员NB', 'Ace Dealer', 1), ('666', '666', '666', 1),
    ('bug', 'BUG', 'BUG', 0), ('klbm', '卡老播吗', 'KLBM', 1),
    ('2048', '2048！', '2048!', 3), ('crown', '32K 太简单了', '32K? Too Easy', 3),
    ('final', 'final 1k', 'final 1k', 2), ('legend', '65K 传奇', '65K Legend', 4),
    ('knowledge', '知识增加', 'Mind Expanded', 0), ('button', '按钮', 'The Button', 0),
    ('whale', '鲸鱼', 'Whale', 0), ('moai', '什', 'What', 0),
    ('meaning', '何意味', 'What Does It Mean?', 0), ('rip', '寄', 'RIP', 0),
    ('tea', '如喝水', 'Easy as Tea', 0), ('chicken', '幽默唤鸡', 'Chicken Workout', 0),
    ('serious', '严肃唤鸡', 'Serious Splits', 0),
]
GIFT_IDS = {gift[0] for gift in GIFTS}
RETIRED_GIFT_IDS = {'coffee', 'fireworks', 'merge', 'brilliant'}


def init_schema():
    with auth_db() as db:
        audience.init_schema(db)
        db.executescript('''
            CREATE TABLE IF NOT EXISTS live_gift_orders (
                request_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
                gift_id TEXT NOT NULL, quantity INTEGER NOT NULL, cost_units INTEGER NOT NULL,
                fingerprint TEXT NOT NULL, combo_id TEXT NOT NULL, combo_count INTEGER NOT NULL,
                created_at REAL NOT NULL, event_json TEXT, delivered_at REAL
            );
            CREATE INDEX IF NOT EXISTS live_gift_user_time ON live_gift_orders(user_id, created_at);
            CREATE INDEX IF NOT EXISTS live_gift_pending ON live_gift_orders(delivered_at, created_at);
            CREATE INDEX IF NOT EXISTS live_supporter_payments ON token_ledger(user_id)
                WHERE event_type='admin_topup' AND paid_delta_units>0;
            CREATE TABLE IF NOT EXISTS live_gift_preferences (
                user_id INTEGER PRIMARY KEY REFERENCES users(id), daily_limit_units INTEGER,
                entrance_enabled INTEGER NOT NULL DEFAULT 1, last_entrance_at REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS live_gift_daily (
                user_id INTEGER NOT NULL REFERENCES users(id), day TEXT NOT NULL, spent_units INTEGER NOT NULL,
                PRIMARY KEY(user_id, day)
            );
        ''')


def catalogue():
    pricing = resolve_pricing_snapshot()
    prices = {item[0]: operation_cost_units('live_gift_' + item[0]) for item in GIFTS}
    if any(value <= 0 for value in prices.values()):
        raise HTTPException(503, 'gift_catalog_unavailable')
    version = hashlib.sha256(json.dumps([pricing.policy_key, pricing.global_multiplier_units, prices], sort_keys=True).encode()).hexdigest()[:24]
    # Per-quantity totals preserve the existing billing rounding rules exactly.
    items = [dict(id=key, zh=zh, en=en, tier=tier, totals=[
        apply_pricing_multipliers(prices[key] * count, 1000, pricing.global_multiplier_units)
        for count in range(1, 11)], base_units=prices[key], global_multiplier_units=pricing.global_multiplier_units) for key, zh, en, tier in GIFTS]
    return dict(version=version, gifts=items), pricing


def day_key(now=None):
    return datetime.fromtimestamp(now or time.time(), timezone(timedelta(hours=8))).date().isoformat()


def preferences(db, user_id):
    row = db.execute('SELECT * FROM live_gift_preferences WHERE user_id=?', (user_id,)).fetchone()
    return dict(daily_limit_units=row['daily_limit_units'] if row else None,
                entrance_enabled=bool(row['entrance_enabled']) if row else True)


def mine(user_id):
    with auth_db() as db:
        pref = preferences(db, user_id)
        daily = db.execute('SELECT spent_units FROM live_gift_daily WHERE user_id=? AND day=?', (user_id, day_key())).fetchone()
        orders = [dict(row) for row in db.execute('''SELECT request_id,gift_id,quantity,cost_units,created_at
            FROM live_gift_orders WHERE user_id=? ORDER BY created_at DESC LIMIT 50''', (user_id,))]
        return dict(**pref, spent_units=daily[0] if daily else 0, token_balance=get_token_balance(user_id, db=db), orders=orders)


def set_preferences(user_id, daily_limit_units, entrance_enabled):
    if daily_limit_units is not None and (type(daily_limit_units) is not int or not 0 <= daily_limit_units <= 100_000_000_000):
        raise HTTPException(400, 'invalid_budget')
    if type(entrance_enabled) is not bool:
        raise HTTPException(400, 'invalid_preferences')
    with auth_db() as db:
        db.execute('''INSERT INTO live_gift_preferences(user_id,daily_limit_units,entrance_enabled) VALUES(?,?,?)
            ON CONFLICT(user_id) DO UPDATE SET daily_limit_units=excluded.daily_limit_units, entrance_enabled=excluded.entrance_enabled''',
            (user_id, daily_limit_units, int(entrance_enabled)))
    return mine(user_id)


def order(user_id, request_id):
    with auth_db() as db:
        row = db.execute('SELECT * FROM live_gift_orders WHERE user_id=? AND request_id=?', (user_id, request_id)).fetchone()
        if not row:
            raise HTTPException(404, 'gift_not_found')
        return receipt(db, row)


def receipt(db, row):
    return dict(status='sent', request_id=row['request_id'], gift_id=row['gift_id'], quantity=row['quantity'],
                cost_units=row['cost_units'], combo_id=row['combo_id'], combo_count=row['combo_count'],
                token_balance=get_token_balance(row['user_id'], db=db))


def send(user, data, online):
    request_id = data.get('request_id', '')
    try:
        if str(uuid.UUID(request_id)) != request_id:
            raise ValueError()
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, 'invalid_request_id')
    gift_id, quantity = data.get('gift_id'), data.get('quantity')
    if not isinstance(gift_id, str) or gift_id not in GIFT_IDS | RETIRED_GIFT_IDS or type(quantity) is not int or not 1 <= quantity <= 1000:
        raise HTTPException(400, 'invalid_gift')
    expected = data.get('expected_cost_units')
    if type(expected) is not int or expected < 0 or not isinstance(data.get('quote_version'), str):
        raise HTTPException(400, 'invalid_quote')
    fingerprint = json.dumps([gift_id, quantity, expected, data['quote_version']], separators=(',', ':'))
    user_id, now = int(user['id']), time.time()
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        old = db.execute('SELECT * FROM live_gift_orders WHERE request_id=?', (request_id,)).fetchone()
        if old:
            if old['user_id'] != user_id or old['fingerprint'] != fingerprint:
                raise HTTPException(409, 'gift_request_conflict')
            return receipt(db, old)
        if gift_id not in GIFT_IDS:
            raise HTTPException(400, 'invalid_gift')
        if not online:
            raise HTTPException(409, 'stream_offline')
        catalog, pricing = catalogue()
        definition = next(gift for gift in catalog['gifts'] if gift['id'] == gift_id)
        cost = apply_pricing_multipliers(definition['base_units'] * quantity, 1000, pricing.global_multiplier_units)
        if data['quote_version'] != catalog['version'] or expected != cost:
            raise HTTPException(409, 'gift_price_changed')
        pref = preferences(db, user_id)
        day = day_key(now)
        spent = db.execute('SELECT spent_units FROM live_gift_daily WHERE user_id=? AND day=?', (user_id, day)).fetchone()
        if pref['daily_limit_units'] is not None and (spent[0] if spent else 0) + cost > pref['daily_limit_units']:
            raise HTTPException(409, 'gift_daily_budget')
        previous = db.execute('''SELECT * FROM live_gift_orders WHERE user_id=? AND gift_id=?
            ORDER BY created_at DESC LIMIT 1''', (user_id, gift_id)).fetchone()
        continuing = previous and now - previous['created_at'] <= 5
        combo_id = previous['combo_id'] if continuing else request_id
        combo_count = (previous['combo_count'] if continuing else 0) + quantity
        event = dict(type='gift', id=request_id, combo_id=combo_id, combo_count=combo_count,
                     gift_id=gift_id, quantity=quantity, tier=definition['tier'], at=now, actor=public_actor(user, db),
                     bulk_effect=10000 <= definition['base_units'] <= 16000
                     and combo_count - quantity < 100 <= combo_count)
        consumed = consume_operation_tokens_once(request_id=request_id, user_id=user_id,
            session_id=user.get('session_id'), operation_key='live_gift_' + gift_id,
            multiplier_override_units=1000, quantity=quantity, db=db, pricing_snapshot=pricing,
            metadata={'gift_id': gift_id, 'quantity': quantity, 'combo_id': combo_id})
        if not consumed:
            raise HTTPException(409, 'gift_request_conflict')
        audience.add(db, f'u:{user_id}', 'gift_units', cost)
        db.execute('''INSERT INTO live_gift_orders VALUES(?,?,?,?,?,?,?,?,?,?,NULL)''',
            (request_id,user_id,gift_id,quantity,cost,fingerprint,combo_id,combo_count,now,json.dumps(event)))
        db.execute('''INSERT INTO live_gift_daily VALUES(?,?,?) ON CONFLICT(user_id,day)
            DO UPDATE SET spent_units=spent_units+excluded.spent_units''', (user_id,day,cost))
        return receipt(db, db.execute('SELECT * FROM live_gift_orders WHERE request_id=?', (request_id,)).fetchone())


def pending_events():
    with auth_db() as db:
        rows = db.execute('''SELECT request_id,event_json,created_at FROM live_gift_orders
            WHERE delivered_at IS NULL ORDER BY created_at LIMIT 50''').fetchall()
    return [dict(id=row['request_id'], event=json.loads(row['event_json']), fresh=time.time()-row['created_at'] < 30) for row in rows]


def delivered(ids):
    with auth_db() as db:
        db.executemany('UPDATE live_gift_orders SET delivered_at=? WHERE request_id=?', [(time.time(), value) for value in ids])


def recent_events():
    with auth_db() as db:
        rows = db.execute('SELECT event_json FROM live_gift_orders WHERE event_json IS NOT NULL ORDER BY created_at DESC LIMIT 100').fetchall()
    return [json.loads(row[0]) for row in reversed(rows)]


def cleanup():
    # Keep compact billing/idempotency records; old animation/name snapshots are unnecessary.
    with auth_db() as db:
        db.execute('UPDATE live_gift_orders SET event_json=NULL WHERE created_at<? AND delivered_at IS NOT NULL AND event_json IS NOT NULL', (time.time()-30*86400,))


def entrance(user):
    if not user:
        return None
    now = time.time()
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        actor = public_actor(user, db)
        if not actor['supporter']:
            return None
        db.execute('INSERT OR IGNORE INTO live_gift_preferences(user_id) VALUES(?)', (user['id'],))
        row = db.execute('SELECT entrance_enabled,last_entrance_at FROM live_gift_preferences WHERE user_id=?', (user['id'],)).fetchone()
        if not row['entrance_enabled'] or now-row['last_entrance_at'] < 1800:
            return None
        db.execute('UPDATE live_gift_preferences SET last_entrance_at=? WHERE user_id=?', (now,user['id']))
    return dict(type='entrance', id=str(uuid.uuid4()), at=now, actor=actor)
