"""Buy-only fixed-product outcome shares, separate from AI pari-mutuel pools.

Reserves and payouts use integer milli-Tokens. Virtual opening inventory never
credits a user wallet. A winning share redeems one unit, including principal.
"""
import json
import math
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from backend.auth.db import auth_db
from .predictions import _credit

UNIT = 1000
INITIAL = 10_000 * UNIT
LIMIT = 60_000 * UNIT
AMOUNTS = (500, 1000, 5000, 10000)
KINDS = {'winner': ('yellow', 'white'), 'first_two': ('2:0', '1:1', '0:2')}
RULES = 'competition-fpmm-v1'


def init_schema():
    with auth_db() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS competition_prediction_markets (
          id TEXT PRIMARY KEY, room_id TEXT NOT NULL, public_key TEXT NOT NULL,
          generation INTEGER NOT NULL, kind TEXT NOT NULL, rules TEXT NOT NULL,
          options TEXT NOT NULL, reserves TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
          source_sequence INTEGER NOT NULL, opened_at TEXT NOT NULL, minimum_until TEXT NOT NULL,
          status TEXT NOT NULL, winner TEXT, reason TEXT, settled_at REAL,
          UNIQUE(public_key,generation,kind));
        CREATE TABLE IF NOT EXISTS competition_prediction_positions (
          market_id TEXT NOT NULL REFERENCES competition_prediction_markets(id),
          user_id INTEGER NOT NULL REFERENCES users(id), option_id TEXT NOT NULL,
          stake INTEGER NOT NULL CHECK(stake>0 AND stake<=60000000),
          shares INTEGER NOT NULL CHECK(shares>0), payout INTEGER,
          PRIMARY KEY(market_id,user_id));
        CREATE TABLE IF NOT EXISTS competition_prediction_orders (
          user_id INTEGER NOT NULL REFERENCES users(id), request_id TEXT NOT NULL,
          market_id TEXT NOT NULL, option_id TEXT NOT NULL, stake INTEGER NOT NULL,
          shares INTEGER NOT NULL, revision INTEGER NOT NULL, created_at REAL NOT NULL,
          PRIMARY KEY(user_id,request_id));
        CREATE INDEX IF NOT EXISTS competition_prediction_pending
          ON competition_prediction_markets(settled_at,public_key);
        ''')


def buy(reserves, option, units):
    """Split collateral into a complete outcome set, then swap into one outcome.

    R'_i = ceil(product(R) / product(R_j + a, j != i));
    purchased shares = R_i + a - R'_i. Round against the trader to retain K.
    """
    if option not in reserves or type(units) is not int or units <= 0:
        raise ValueError('invalid_buy')
    product = math.prod(reserves.values())
    denominator = math.prod(value + units for key, value in reserves.items() if key != option)
    updated = {key: value + units for key, value in reserves.items()}
    updated[option] = (product + denominator - 1) // denominator
    return reserves[option] + units - updated[option], updated


def _identity(facts, room_id):
    return (isinstance(facts, dict) and room_id == 'competition-' + str(facts.get('match_public_key'))
            and type(facts.get('generation')) is int and type(facts.get('content_sequence')) is int)


def _outcome(facts, kind):
    if facts['phase'] == 'CANCELLED':
        return None, 'cancelled'
    result = facts.get('public_result') or {}
    if kind == 'winner':
        winner = result.get('winner_side')
        return (winner, 'match_winner') if winner in KINDS[kind] else (None, 'draw_or_unresolved')
    games = {item['game_key']: item for item in result.get('games', [])}
    winners = [games.get(game, {}).get('winner_side') for game in ('A', 'B')]
    if any(winner not in ('yellow', 'white') for winner in winners):
        return None, 'draw_or_unresolved'
    return f"{winners.count('yellow')}:{winners.count('white')}", 'first_two_score'


def reconcile(room_id, facts):
    if not _identity(facts, room_id):
        return
    window = facts.get('prediction_window')
    if not window:
        return
    terminal = facts['phase'] in ('FINISHED', 'CANCELLED')
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        for kind, ids in KINDS.items():
            market_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{facts['match_public_key']}:{facts['generation']}:{kind}"))
            options = []
            for option in ids:
                names = [item.get('display_name', '') for item in facts.get('teams', {}).get(option, {}).get('roster', [])]
                options.append({'id': option, 'name': ' / '.join(filter(None, names)) or option})
            db.execute('''INSERT OR IGNORE INTO competition_prediction_markets
                (id,room_id,public_key,generation,kind,rules,options,reserves,source_sequence,opened_at,minimum_until,status)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (market_id, room_id, facts['match_public_key'], facts['generation'], kind, RULES,
                 json.dumps(options), json.dumps(dict.fromkeys(ids, INITIAL)), facts['content_sequence'],
                 window['opened_at'], window['minimum_until'], 'open' if window['open'] else 'closed'))
            market = db.execute('SELECT * FROM competition_prediction_markets WHERE id=?', (market_id,)).fetchone()
            if market['settled_at'] is not None or facts['content_sequence'] < market['source_sequence']:
                continue
            # Closed markets never reopen, even after stale polling or a restart.
            status = 'open' if market['status'] == 'open' and window['open'] else 'closed'
            db.execute('UPDATE competition_prediction_markets SET status=?,source_sequence=? WHERE id=?',
                       (status, facts['content_sequence'], market_id))
            if not terminal:
                continue
            winner, reason = _outcome(facts, kind)
            stamp = datetime.now(timezone.utc).isoformat()
            positions = db.execute('SELECT * FROM competition_prediction_positions WHERE market_id=?', (market_id,)).fetchall()
            for position in positions:
                payout = position['stake'] if winner is None else position['shares'] if position['option_id'] == winner else 0
                if payout:
                    _credit(db, position['user_id'], payout, 'competition_prediction_refund' if winner is None else 'competition_prediction_payout',
                            f"competition-prediction:{market_id}:{position['user_id']}:settle",
                            dict(market_id=market_id, winner=winner, reason=reason, principal_units=position['stake']), stamp)
                db.execute('UPDATE competition_prediction_positions SET payout=? WHERE market_id=? AND user_id=?',
                           (payout, market_id, position['user_id']))
            db.execute('UPDATE competition_prediction_markets SET status=?,winner=?,reason=?,settled_at=? WHERE id=?',
                       ('void' if winner is None else 'settled', winner, reason, time.time(), market_id))


def listing(room_id, user_id=None):
    with auth_db() as db:
        rows = db.execute('SELECT * FROM competition_prediction_markets WHERE room_id=? ORDER BY opened_at DESC,kind', (room_id,)).fetchall()
        markets = []
        generation = max((row['generation'] for row in rows), default=0)
        for row in rows:
            if row['generation'] != generation:
                continue
            reserves = json.loads(row['reserves'])
            options = json.loads(row['options'])
            for option in options:
                option['quotes'] = {str(amount): buy(reserves, option['id'], amount * UNIT)[0] for amount in AMOUNTS}
                option['marginal_odds'] = sum(reserves[option['id']] / value for value in reserves.values())
            market = {key: row[key] for key in ('id', 'kind', 'status', 'revision', 'opened_at', 'minimum_until', 'winner', 'reason')}
            market['options'] = options
            if user_id:
                position = db.execute('SELECT option_id,stake,shares,payout FROM competition_prediction_positions WHERE market_id=? AND user_id=?',
                                      (row['id'], user_id)).fetchone()
                market['mine'] = dict(position) if position else None
            markets.append(market)
        return dict(protocol=RULES, markets=markets, amounts=list(AMOUNTS), limit_units=LIMIT,
                    initial_reserve_units=INITIAL, server_time=time.time())


def place(room_id, user_id, body, source):
    """source performs a fresh authenticated Competition Service read, never cache."""
    if not isinstance(body, dict):
        raise HTTPException(400, 'competition_prediction_invalid')
    try:
        request_id = str(uuid.UUID(body.get('request_id')))
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(400, 'competition_prediction_invalid')
    market_id, option, amount, revision = (body.get(key) for key in ('market_id', 'option_id', 'amount', 'revision'))
    if (type(amount) is not int or amount not in AMOUNTS or type(revision) is not int
            or not isinstance(market_id, str) or not isinstance(option, str)):
        raise HTTPException(400, 'competition_prediction_invalid')
    units = amount * UNIT
    # Confirm uncertain responses even when the match is now closed/offline.
    with auth_db() as db:
        previous = db.execute('SELECT * FROM competition_prediction_orders WHERE user_id=? AND request_id=?', (user_id, request_id)).fetchone()
        if previous:
            return _retry(db, previous, room_id, market_id, option, units)
    facts = source()
    if not _identity(facts, room_id):
        raise HTTPException(503, 'competition_prediction_unavailable')
    reconcile(room_id, facts)
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT * FROM competition_prediction_orders WHERE user_id=? AND request_id=?', (user_id, request_id)).fetchone()
        if previous:
            return _retry(db, previous, room_id, market_id, option, units)
        market = db.execute('SELECT * FROM competition_prediction_markets WHERE id=? AND room_id=?', (market_id, room_id)).fetchone()
        window = facts.get('prediction_window') or {}
        if (not market or market['status'] != 'open' or not window.get('open') or facts.get('suspended')
                or facts['generation'] != market['generation'] or facts['content_sequence'] < market['source_sequence']):
            raise HTTPException(409, 'competition_prediction_closed')
        reserves = json.loads(market['reserves'])
        if option not in reserves:
            raise HTTPException(400, 'competition_prediction_invalid')
        position = db.execute('SELECT * FROM competition_prediction_positions WHERE market_id=? AND user_id=?', (market_id, user_id)).fetchone()
        if position and position['option_id'] != option:
            raise HTTPException(409, 'competition_prediction_cannot_switch')
        if (position['stake'] if position else 0) + units > LIMIT:
            raise HTTPException(409, 'competition_prediction_limit')
        if revision != market['revision']:
            raise HTTPException(409, 'competition_prediction_price_changed')
        shares, reserves = buy(reserves, option, units)
        stamp = datetime.now(timezone.utc).isoformat()
        _credit(db, user_id, -units, 'competition_prediction_stake', f'competition-prediction:{user_id}:{request_id}',
                dict(market_id=market_id, option_id=option, shares=shares, rules=RULES), stamp)
        db.execute('''INSERT INTO competition_prediction_positions(market_id,user_id,option_id,stake,shares) VALUES(?,?,?,?,?)
            ON CONFLICT(market_id,user_id) DO UPDATE SET stake=stake+excluded.stake,shares=shares+excluded.shares''',
            (market_id, user_id, option, units, shares))
        db.execute('INSERT INTO competition_prediction_orders VALUES(?,?,?,?,?,?,?,?)',
                   (user_id, request_id, market_id, option, units, shares, revision, time.time()))
        db.execute('UPDATE competition_prediction_markets SET reserves=?,revision=revision+1 WHERE id=?', (json.dumps(reserves), market_id))
    return dict(accepted=True, shares=shares, stake=units, request_id=request_id)


def _retry(db, order, room_id, market_id, option, units):
    market = db.execute('SELECT room_id FROM competition_prediction_markets WHERE id=?', (order['market_id'],)).fetchone()
    if not market or market['room_id'] != room_id or (order['market_id'], order['option_id'], order['stake']) != (market_id, option, units):
        raise HTTPException(409, 'competition_prediction_request_conflict')
    return dict(accepted=True, duplicate=True, shares=order['shares'], stake=order['stake'], request_id=order['request_id'])


def pending_rooms():
    with auth_db() as db:
        return [dict(row) for row in db.execute('SELECT DISTINCT room_id,public_key FROM competition_prediction_markets WHERE settled_at IS NULL')]
