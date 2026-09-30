"""Permanent-token escrow and deterministic, capped pari-mutuel settlement."""
import json
import time
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException
from backend.auth.db import auth_db

AMOUNTS = (100, 1000, 5000, 10000)
UNIT = 1000
RULE_VERSION = 'matched-pairwise-pools-v3'
TARGET = 65536
TARGET_REWARD = 8  # Total payout multiplier; the side-bet principal is not returned.
BONUS_TARGET = 32768  # Must coexist with TARGET on the same board.
BONUS_REWARD = 50  # Highest tier only, not added to TARGET_REWARD.
TARGET_RULE_VERSION = 'target-no-principal-v2'


def init_schema():
    with auth_db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS room_prediction_markets (
                id TEXT PRIMARY KEY, room_id TEXT NOT NULL, options TEXT NOT NULL,
                started REAL NOT NULL, deadline REAL NOT NULL, status TEXT NOT NULL,
                rules TEXT NOT NULL, target_rules TEXT NOT NULL,
                winners TEXT, reason TEXT, settled REAL);
            CREATE INDEX IF NOT EXISTS prediction_room ON room_prediction_markets(room_id,started DESC);
            CREATE TABLE IF NOT EXISTS room_prediction_stakes (
                market_id TEXT NOT NULL REFERENCES room_prediction_markets(id),
                user_id INTEGER NOT NULL REFERENCES users(id), option_id TEXT NOT NULL,
                units INTEGER NOT NULL CHECK(units>0), PRIMARY KEY(market_id,user_id));
            CREATE TABLE IF NOT EXISTS room_prediction_requests (
                user_id INTEGER NOT NULL REFERENCES users(id), request_id TEXT NOT NULL,
                market_id TEXT NOT NULL, option_id TEXT NOT NULL, units INTEGER NOT NULL,
                PRIMARY KEY(user_id,request_id));
            CREATE TABLE IF NOT EXISTS room_prediction_settlements (
                market_id TEXT NOT NULL, user_id INTEGER NOT NULL REFERENCES users(id),
                principal INTEGER NOT NULL, profit INTEGER NOT NULL, refund INTEGER NOT NULL,
                loss INTEGER NOT NULL, PRIMARY KEY(market_id,user_id));
            CREATE TABLE IF NOT EXISTS room_prediction_target_stakes (
                market_id TEXT NOT NULL REFERENCES room_prediction_markets(id),
                user_id INTEGER NOT NULL REFERENCES users(id), option_id TEXT NOT NULL,
                units INTEGER NOT NULL CHECK(units>0), PRIMARY KEY(market_id,user_id));
            CREATE TABLE IF NOT EXISTS room_prediction_target_requests (
                user_id INTEGER NOT NULL REFERENCES users(id), request_id TEXT NOT NULL,
                market_id TEXT NOT NULL, option_id TEXT NOT NULL, units INTEGER NOT NULL,
                PRIMARY KEY(user_id,request_id));
            CREATE TABLE IF NOT EXISTS room_prediction_target_settlements (
                market_id TEXT NOT NULL, user_id INTEGER NOT NULL REFERENCES users(id),
                principal INTEGER NOT NULL, profit INTEGER NOT NULL, refund INTEGER NOT NULL,
                loss INTEGER NOT NULL, PRIMARY KEY(market_id,user_id));
            CREATE TABLE IF NOT EXISTS room_prediction_target_outcomes (
                market_id TEXT NOT NULL REFERENCES room_prediction_markets(id),
                option_id TEXT NOT NULL, outcome TEXT NOT NULL CHECK(outcome IN ('reached','missed')),
                PRIMARY KEY(market_id,option_id));
            CREATE TABLE IF NOT EXISTS room_prediction_announcements (
                id TEXT PRIMARY KEY, room_id TEXT NOT NULL, market_id TEXT NOT NULL,
                option_id TEXT NOT NULL, tier INTEGER NOT NULL, event_json TEXT NOT NULL,
                announced INTEGER NOT NULL DEFAULT 0,
                UNIQUE(market_id,option_id,tier));
            CREATE INDEX IF NOT EXISTS prediction_announcement_room
                ON room_prediction_announcements(room_id,announced);
        ''')
        # Additive migration retains historical outcomes/settlements unchanged.
        db.execute('BEGIN IMMEDIATE')
        columns = {row['name'] for row in db.execute('PRAGMA table_info(room_prediction_target_outcomes)')}
        if 'combo_reached' not in columns:
            db.execute('ALTER TABLE room_prediction_target_outcomes ADD COLUMN combo_reached INTEGER NOT NULL DEFAULT 0 CHECK(combo_reached IN (0,1))')
        market_columns = {row['name'] for row in db.execute('PRAGMA table_info(room_prediction_markets)')}
        if 'target_rules' not in market_columns:
            db.execute('ALTER TABLE room_prediction_markets ADD COLUMN target_rules TEXT')
        db.execute('UPDATE room_prediction_markets SET target_rules=? WHERE target_rules IS NULL',
                   (TARGET_RULE_VERSION,))


def ensure_market(room_id, batch):
    if batch.get('transition'):
        return
    with auth_db() as db:
        db.execute('''INSERT OR IGNORE INTO room_prediction_markets
            (id,room_id,options,started,deadline,status,rules,target_rules)
            VALUES(?,?,?,?,?,'open',?,?)''',
            (batch['id'], room_id, json.dumps(batch['participants']), batch['started_at'],
             batch['deadline'], RULE_VERSION, TARGET_RULE_VERSION))
        saved = db.execute('SELECT room_id,started FROM room_prediction_markets WHERE id=?', (batch['id'],)).fetchone()
        if saved['room_id'] != room_id or saved['started'] != batch['started_at']:
            raise ValueError('prediction_market_conflict')


def close(room_id, market_id):
    with auth_db() as db:
        db.execute("UPDATE room_prediction_markets SET status='closed' WHERE id=? AND room_id=? AND status='open'",
                   (market_id, room_id))


def record_targets(room_id, market_id, outcomes, now=None):
    """Only trusted, persisted content facts enter this bridge, never client claims."""
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        market = db.execute('SELECT options,target_rules,settled FROM room_prediction_markets WHERE id=? AND room_id=?',
                            (market_id,room_id)).fetchone()
        if not market or market['settled'] is not None:
            return
        if market['target_rules'] != TARGET_RULE_VERSION:
            raise ValueError('unknown_prediction_target_rules')
        options = {p['id'] for p in json.loads(market['options'])}
        for option, outcome in outcomes.items():
            if option not in options or outcome not in ('reached','missed','reached_combo'):
                raise ValueError('invalid_prediction_target')
            base = 'reached' if outcome == 'reached_combo' else outcome
            combo = int(outcome == 'reached_combo')
            old = db.execute('SELECT outcome FROM room_prediction_target_outcomes WHERE market_id=? AND option_id=?',
                             (market_id,option)).fetchone()
            if old and old['outcome'] != base:
                raise ValueError('prediction_target_conflict')
            db.execute('''INSERT INTO room_prediction_target_outcomes(market_id,option_id,outcome,combo_reached)
                VALUES(?,?,?,?) ON CONFLICT(market_id,option_id)
                DO UPDATE SET combo_reached=MAX(combo_reached,excluded.combo_reached)''', (market_id,option,base,combo))
        # Facts, wallet credits, cumulative results and the chat outbox commit together.
        at = time.time() if now is None else now
        for option, outcome in _target_outcomes(db, market_id).items():
            if outcome in ('reached', 'reached_combo'):
                _award_target_tier(db, room_id, market_id, market, option, TARGET_REWARD, at)
                if outcome == 'reached_combo':
                    _award_target_tier(db, room_id, market_id, market, option, BONUS_REWARD, at)


def _award_target_tier(db, room_id, market_id, market, option, reward, now):
    event_id = f'prediction-reward:{market_id}:{option}:{reward}'
    if db.execute('SELECT 1 FROM room_prediction_announcements WHERE id=?', (event_id,)).fetchone():
        return
    # Request rowids preserve the first committed side-bet order.
    stakes = db.execute('''SELECT s.*,u.display_name,
        (SELECT MIN(r.rowid) FROM room_prediction_target_requests r
         WHERE r.market_id=s.market_id AND r.user_id=s.user_id) AS first_request
        FROM room_prediction_target_stakes s JOIN users u ON u.id=s.user_id
        WHERE s.market_id=? AND s.option_id=? ORDER BY first_request,s.user_id''', (market_id,option)).fetchall()
    if not stakes:
        return
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
    for stake in stakes:
        uid, units = stake['user_id'], stake['units']
        previous = db.execute('SELECT principal,profit,refund FROM room_prediction_target_settlements WHERE market_id=? AND user_id=?',
                              (market_id,uid)).fetchone()
        paid = sum(previous) if previous else 0
        delta = units * reward - paid
        if delta > 0:
            _credit(db,uid,delta,'room_prediction_target_settlement',event_id,
                    dict(room_id=room_id,market_id=market_id,option_id=option,reward_multiplier=reward,
                         target_rules=TARGET_RULE_VERSION,
                         principal=0,
                         profit=units*reward,refund=0,loss=0),stamp)
            db.execute('''INSERT INTO room_prediction_target_settlements VALUES(?,?,?,?,0,0)
                ON CONFLICT(market_id,user_id) DO UPDATE SET principal=excluded.principal,profit=excluded.profit,
                refund=0,loss=0''', (market_id,uid,0,units*reward))
    player = next(p for p in json.loads(market['options']) if p['id'] == option)
    event = dict(type='prediction_reward',id=event_id,at=now,market_id=market_id,option_id=option,
                 player_name=player['name'],tier='65k+32k' if reward == BONUS_REWARD else '65k',
                 names=[s['display_name'] for s in stakes[:3]],recipient_count=len(stakes),
                 total_units=sum(s['units'] for s in stakes)*reward)
    db.execute('INSERT INTO room_prediction_announcements VALUES(?,?,?,?,?,?,0)',
               (event_id,room_id,market_id,option,reward,json.dumps(event,ensure_ascii=False)))


def announcement_events(room_id, pending=False):
    with auth_db() as db:
        rows = db.execute('SELECT event_json FROM room_prediction_announcements WHERE room_id=? '
                          + ('AND announced=0 ORDER BY rowid LIMIT 100' if pending else 'ORDER BY rowid DESC LIMIT 100'), (room_id,))
        events = [json.loads(row['event_json']) for row in rows]
        return events if pending else list(reversed(events))


def announcements_delivered(room_id, ids):
    with auth_db() as db:
        db.executemany('UPDATE room_prediction_announcements SET announced=1 WHERE room_id=? AND id=?',
                       [(room_id,event_id) for event_id in ids])


def _target_outcomes(db, market_id):
    return {row['option_id']: 'reached_combo' if row['combo_reached'] else row['outcome']
            for row in db.execute('SELECT option_id,outcome,combo_reached FROM room_prediction_target_outcomes WHERE market_id=?', (market_id,))}


def _proportional(total, stakes):
    """Allocate integer ``total`` by stake with deterministic largest remainders."""
    denominator = sum(s['units'] for s in stakes)
    if total <= 0 or not denominator:
        return {s['user_id']: 0 for s in stakes}
    allocated = {}
    remainders = []
    for stake in stakes:
        units, remainder = divmod(stake['units'] * total, denominator)
        allocated[stake['user_id']] = units
        remainders.append((-remainder, stake['user_id']))
    left = total - sum(allocated.values())
    for _, uid in sorted(remainders)[:left]:
        allocated[uid] += 1
    return allocated


def _pairwise_match_amounts(stakes_by_option):
    """Maximize disjoint two-option matches without using any stake twice."""
    totals = {option: sum(stake['units'] for stake in stakes)
              for option, stakes in stakes_by_option.items() if stakes}
    options = sorted(totals)
    if len(options) < 2:
        return {}
    if len(options) == 2:
        pair = tuple(options)
        return {pair: min(totals.values())}
    if len(options) != 3:
        raise ValueError('invalid_prediction_option_count')

    largest = max(options, key=lambda option: (totals[option], option))
    others = [option for option in options if option != largest]
    if totals[largest] >= sum(totals[option] for option in others):
        return {tuple(sorted((largest, option))): totals[option] for option in others
                if totals[option]}

    degrees = dict(totals)
    if sum(degrees.values()) % 2:
        # One minimal accounting unit cannot be paired. Removing it from the
        # largest pool keeps all triangle inequalities valid and deterministic.
        degrees[largest] -= 1
    a, b, c = options
    matches = {
        (a, b): (degrees[a] + degrees[b] - degrees[c]) // 2,
        (a, c): (degrees[a] + degrees[c] - degrees[b]) // 2,
        (b, c): (degrees[b] + degrees[c] - degrees[a]) // 2,
    }
    return {pair: amount for pair, amount in matches.items() if amount > 0}


def _pairwise_account_allocations(stakes_by_option, matches):
    """Assign each account's stake to fixed pairwise pools at most once."""
    allocations = {}
    for option, stakes in stakes_by_option.items():
        remaining = {stake['user_id']: stake['units'] for stake in stakes}
        incident = sorted((pair, amount) for pair, amount in matches.items() if option in pair)
        for pair, amount in incident:
            available = [dict(user_id=uid, units=units) for uid, units in remaining.items()]
            share = _proportional(amount, available)
            allocations[(pair, option)] = share
            for uid, units in share.items():
                remaining[uid] -= units
                if remaining[uid] < 0:
                    raise AssertionError('prediction_pairwise_overallocation')
    return allocations


def allocate(stakes, winners, void=False):
    """Settle v3 through fixed pairwise pools; equal hit rates imply zero EV."""
    if void:
        return {s['user_id']: dict(principal=0, profit=0, refund=s['units'], loss=0) for s in stakes}
    stakes_by_option = {}
    for stake in stakes:
        stakes_by_option.setdefault(stake['option_id'], []).append(stake)
    matches = _pairwise_match_amounts(stakes_by_option)
    allocations = _pairwise_account_allocations(stakes_by_option, matches)
    result = {}
    for stake in stakes:
        if stake['option_id'] in winners:
            result[stake['user_id']] = dict(principal=stake['units'], profit=0, refund=0, loss=0)
        else:
            result[stake['user_id']] = dict(principal=0, profit=0, refund=stake['units'], loss=0)

    for pair, amount in matches.items():
        first, second = pair
        first_wins = first in winners
        second_wins = second in winners
        if first_wins == second_wins:
            continue
        winning_option, losing_option = (first, second) if first_wins else (second, first)
        for uid, units in allocations[(pair, winning_option)].items():
            result[uid]['profit'] += units
        for uid, units in allocations[(pair, losing_option)].items():
            result[uid]['refund'] -= units
            result[uid]['loss'] += units
    return result


def _account(db, uid, stamp):
    db.execute('INSERT OR IGNORE INTO token_accounts(user_id,created_at,updated_at) VALUES(?,?,?)', (uid, stamp, stamp))
    return db.execute('SELECT * FROM token_accounts WHERE user_id=?', (uid,)).fetchone()


def _credit(db, uid, delta, event, operation, metadata, stamp):
    account = _account(db, uid, stamp)
    before = account['paid_balance_units'] + account['bonus_balance_units']
    if account['paid_balance_units'] + delta < 0:
        raise HTTPException(402, 'prediction_insufficient_permanent')
    db.execute('UPDATE token_accounts SET paid_balance_units=paid_balance_units+?,updated_at=? WHERE user_id=?',
               (delta, stamp, uid))
    db.execute('''INSERT INTO token_ledger(user_id,event_type,operation_key,paid_delta_units,
        balance_before_units,balance_after_units,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)''',
        (uid, event, operation, delta, before, before + delta, json.dumps(metadata), stamp))


def place(room_id, user_id, body, available, now=None):
    override_now = now
    market_id, option, request_id, amount = (body.get(k) for k in ('market_id','option_id','request_id','amount'))
    kind = body.get('kind', 'winner')
    if kind not in ('winner', 'target65536'):
        raise HTTPException(400, 'prediction_invalid_kind')
    target = kind == 'target65536'
    stakes_table = 'room_prediction_target_stakes' if target else 'room_prediction_stakes'
    requests_table = 'room_prediction_target_requests' if target else 'room_prediction_requests'
    other_requests = 'room_prediction_requests' if target else 'room_prediction_target_requests'
    try:
        request_id = str(uuid.UUID(request_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, 'prediction_invalid_request')
    if type(amount) is not int or amount not in AMOUNTS or not isinstance(option, str) or not isinstance(market_id, str):
        raise HTTPException(400, 'prediction_invalid_stake')
    units = amount * UNIT
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        now = time.time() if override_now is None else override_now
        stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
        market = db.execute('SELECT * FROM room_prediction_markets WHERE id=? AND room_id=?', (market_id,room_id)).fetchone()
        if not market:
            raise HTTPException(404, 'prediction_not_found')
        if db.execute(f'SELECT 1 FROM {other_requests} WHERE user_id=? AND request_id=?', (user_id,request_id)).fetchone():
            raise HTTPException(409, 'prediction_request_conflict')
        previous = db.execute(f'SELECT * FROM {requests_table} WHERE user_id=? AND request_id=?', (user_id,request_id)).fetchone()
        if previous:
            if (previous['market_id'],previous['option_id'],previous['units']) != (market_id,option,units):
                raise HTTPException(409, 'prediction_request_conflict')
            return dict(ok=True, request_id=request_id, market_id=market_id)
        if market['status'] != 'open' or now >= market['deadline'] or not (available() if callable(available) else available):
            raise HTTPException(409, 'prediction_closed')
        if option not in {p['id'] for p in json.loads(market['options'])}:
            raise HTTPException(400, 'prediction_invalid_option')
        stake = db.execute('SELECT * FROM room_prediction_stakes WHERE market_id=? AND user_id=?', (market_id,user_id)).fetchone()
        if target and not stake:
            raise HTTPException(409, 'prediction_main_required')
        if stake and stake['option_id'] != option:
            raise HTTPException(409, 'prediction_cannot_switch')
        if target and db.execute('SELECT 1 FROM room_prediction_target_outcomes WHERE market_id=? AND option_id=?',
                                 (market_id,option)).fetchone():
            raise HTTPException(409, 'prediction_target_closed')
        _credit(db,user_id,-units,'room_prediction_target_stake' if target else 'room_prediction_stake','prediction:'+request_id,
                dict(room_id=room_id,market_id=market_id,option_id=option,kind=kind),stamp)
        db.execute(f'''INSERT INTO {stakes_table} VALUES(?,?,?,?) ON CONFLICT(market_id,user_id)
            DO UPDATE SET units=units+excluded.units''', (market_id,user_id,option,units))
        db.execute(f'INSERT INTO {requests_table} VALUES(?,?,?,?,?)', (user_id,request_id,market_id,option,units))
    return dict(ok=True, request_id=request_id, market_id=market_id)


def settle(room_id, market_id, winners=(), void=False, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        db.execute('BEGIN IMMEDIATE')
        market = db.execute('SELECT * FROM room_prediction_markets WHERE id=? AND room_id=?', (market_id,room_id)).fetchone()
        if not market or market['settled'] is not None:
            return False
        if market['target_rules'] != TARGET_RULE_VERSION:
            raise ValueError('unknown_prediction_target_rules')
        options = {p['id'] for p in json.loads(market['options'])}
        if not void and (not winners or not set(winners) <= options):
            raise ValueError('invalid_prediction_result')
        stakes = [dict(r) for r in db.execute('SELECT * FROM room_prediction_stakes WHERE market_id=?', (market_id,))]
        reason = 'technical_void' if void else 'no_winners' if not any(s['option_id'] in winners for s in stakes) else 'result'
        rules = market['rules']
        if rules != RULE_VERSION:
            raise ValueError('unknown_prediction_rules')
        payouts = allocate(stakes, set(winners), void)
        stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
        for uid, parts in payouts.items():
            payout = parts['principal'] + parts['profit'] + parts['refund']
            _credit(db,uid,payout,'room_prediction_settlement','prediction:'+market_id,
                    dict(room_id=room_id,market_id=market_id,rules=rules,**parts),stamp)
            db.execute('INSERT INTO room_prediction_settlements VALUES(?,?,?,?,?,?)',
                       (market_id,uid,parts['principal'],parts['profit'],parts['refund'],parts['loss']))
        outcomes = _target_outcomes(db, market_id)
        for stake in db.execute('SELECT * FROM room_prediction_target_stakes WHERE market_id=?', (market_id,)).fetchall():
            uid, units = stake['user_id'], stake['units']
            # A milestone already paid its result in real time. Do not pay again
            # at batch end or claw back earned rewards on a later void.
            if db.execute('SELECT 1 FROM room_prediction_target_settlements WHERE market_id=? AND user_id=?',
                          (market_id,uid)).fetchone():
                continue
            outcome = outcomes.get(stake['option_id'])
            reward = BONUS_REWARD if outcome == 'reached_combo' else TARGET_REWARD if outcome == 'reached' else 0
            if void:
                parts = dict(principal=0, profit=0, refund=units, loss=0)
            elif stake['option_id'] not in outcomes:
                raise ValueError('prediction_target_result_missing')
            elif reward:
                parts = dict(principal=0, profit=units*reward, refund=0, loss=0)
            else:
                parts = dict(principal=0, profit=0, refund=0, loss=units)
            _credit(db,uid,parts['principal']+parts['profit']+parts['refund'],
                    'room_prediction_target_settlement','prediction-target:'+market_id,
                    dict(room_id=room_id,market_id=market_id,target=TARGET,bonus_target=BONUS_TARGET,
                         outcome=outcome,reward_multiplier=0 if void else reward,**parts),stamp)
            db.execute('INSERT INTO room_prediction_target_settlements VALUES(?,?,?,?,?,?)',
                       (market_id,uid,parts['principal'],parts['profit'],parts['refund'],parts['loss']))
        db.execute('UPDATE room_prediction_markets SET status=?,winners=?,reason=?,settled=? WHERE id=?',
                   ('void' if void else 'settled',json.dumps(sorted(winners)),reason,now,market_id))
    return True


def listing(room_id, user_id=None, market_id=None, now=None):
    now = time.time() if now is None else now
    with auth_db() as db:
        row = db.execute('''SELECT * FROM room_prediction_markets WHERE room_id=? AND (? IS NULL OR id=?)
            ORDER BY started DESC LIMIT 1''', (room_id,market_id,market_id)).fetchone()
        market = None
        if row:
            market = dict(row)
            market['options'] = json.loads(market['options'])
            market['winners'] = json.loads(market['winners'] or '[]')
            if market['status'] == 'open' and now >= market['deadline']:
                market['status'] = 'closed'
            totals = {r['option_id']: dict(units=r['units'],count=r['count']) for r in db.execute(
                'SELECT option_id,SUM(units) AS units,COUNT(*) AS count FROM room_prediction_stakes WHERE market_id=? GROUP BY option_id', (row['id'],))}
            for option in market['options']:
                option.update(totals.get(option['id'],dict(units=0,count=0)))
            market['pool_units'] = sum(o['units'] for o in market['options'])
            market['target_bet'] = dict(target=TARGET,reward_multiplier=TARGET_REWARD,
                bonus_target=BONUS_TARGET,bonus_reward_multiplier=BONUS_REWARD,
                rules=row['target_rules'],outcomes=_target_outcomes(db, row['id']))
            if user_id is not None:
                stake = db.execute('SELECT option_id,units FROM room_prediction_stakes WHERE market_id=? AND user_id=?', (row['id'],user_id)).fetchone()
                result = db.execute('SELECT principal,profit,refund,loss FROM room_prediction_settlements WHERE market_id=? AND user_id=?', (row['id'],user_id)).fetchone()
                market['mine'] = dict(stake) if stake else None
                market['result'] = dict(result) if result else None
                stake = db.execute('SELECT option_id,units FROM room_prediction_target_stakes WHERE market_id=? AND user_id=?', (row['id'],user_id)).fetchone()
                result = db.execute('SELECT principal,profit,refund,loss FROM room_prediction_target_settlements WHERE market_id=? AND user_id=?', (row['id'],user_id)).fetchone()
                market['target_bet'].update(mine=dict(stake) if stake else None, result=dict(result) if result else None)
        response = dict(market=market, server_time=now, amounts=list(AMOUNTS))
        if user_id is not None:
            account = db.execute('SELECT paid_balance_units FROM token_accounts WHERE user_id=?', (user_id,)).fetchone()
            response['paid_balance_units'] = account[0] if account else 0
            rows = db.execute('''SELECT m.id,m.started AS started_at,m.winners,m.reason,s.* FROM room_prediction_settlements s
                JOIN room_prediction_markets m ON m.id=s.market_id WHERE m.room_id=? AND s.user_id=?
                ORDER BY m.started DESC LIMIT 5''',(room_id,user_id))
            response['recent'] = [dict(r, winners=json.loads(r['winners'] or '[]')) for r in rows]
            for result in response['recent']:
                target = db.execute('''SELECT settlement.principal,settlement.profit,settlement.refund,settlement.loss,
                    stake.units AS stake_units FROM room_prediction_target_settlements settlement
                    LEFT JOIN room_prediction_target_stakes stake
                      ON stake.market_id=settlement.market_id AND stake.user_id=settlement.user_id
                    WHERE settlement.market_id=? AND settlement.user_id=?''', (result['id'],user_id)).fetchone()
                target_stake_units = target['stake_units'] if target else 0
                result['target_bet'] = ({key: target[key] for key in ('principal','profit','refund','loss')}
                                        if target else None)
                result['stake_units'] = (result['principal'] + result['refund'] + result['loss']
                                         + target_stake_units)
                parts = [result] + ([result['target_bet']] if result['target_bet'] else [])
                result['net_profit_units'] = sum(part['profit'] - part['loss'] for part in parts)
        return response
