import copy
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from backend.auth.db import auth_db, init_auth_db
from backend.room_activities import competition_predictions as p

ROOM = 'competition-mtest'


@pytest.fixture()
def market(tmp_path, monkeypatch):
    monkeypatch.setenv('CLOUD_AUTH_DB', str(tmp_path / 'auth.sqlite3'))
    init_auth_db()
    p.init_schema()
    with auth_db() as db:
        for uid in (1, 2):
            db.execute("INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at) VALUES(?,?,?,?,?,'now','now')",
                       (uid, f'{uid}@example.invalid', f'{uid}@example.invalid', 'test', str(uid)))
            db.execute("INSERT INTO token_accounts(user_id,paid_balance_units,bonus_balance_units,created_at,updated_at) VALUES(?,200000000,777000,'now','now')", (uid,))
    facts = dict(match_public_key='mtest', generation=1, content_sequence=1, phase='FIRST_PICK_BAN', suspended=False,
        teams={'yellow': {'roster': [{'display_name':'Alice'}, {'display_name':'Bob'}, {'display_name':'Carol'}]},
               'white': {'roster': [{'display_name':'Dan'}, {'display_name':'Eve'}, {'display_name':'Frank'}]}},
        prediction_window=dict(open=True, opened_at='2026-09-29T00:00:00+00:00', minimum_until='2026-09-29T00:01:00+00:00', closed_at=None),
        public_result=dict(winner_side=None, games=[]))
    p.reconcile(ROOM, facts)
    return facts


def order(kind='winner', option='yellow', amount=1000, uid=1):
    market = next(item for item in p.listing(ROOM, uid)['markets'] if item['kind'] == kind)
    return dict(request_id=str(uuid.uuid4()), market_id=market['id'], option_id=option, amount=amount, revision=market['revision'])


def balance(uid=1):
    with auth_db() as db:
        return tuple(db.execute('SELECT paid_balance_units,bonus_balance_units FROM token_accounts WHERE user_id=?', (uid,)).fetchone())


@pytest.mark.parametrize('count,weights', [(5,[2,3,3,3,3,2]),(7,[2,4,5,5,5,5,4,2])])
def test_extended_score_market_priors(market,count,weights):
    facts={**market,'generation':2,'rules':{'game_count':count,'series_mode':'all'}}
    p.reconcile(ROOM,facts)
    item=next(m for m in p.listing(ROOM)['markets'] if m['kind'].startswith('clinch_'))
    assert [1/o['marginal_odds'] for o in item['options']]==pytest.approx([w/sum(weights) for w in weights])
    assert item['pricing']=='amm'
    assert max(p._initial_reserves(item['kind']).values()) <= p.INITIAL
    stakes=1000*p.UNIT
    for option in item['options']:
        assert option['quotes']['1000'] < stakes*option['marginal_odds']


@pytest.mark.parametrize('wins', [3,4])
def test_all_clinching_scores_and_later_games(market,wins):
    from backend.room_activities.match_prediction_rules import OPTIONS
    for score in OPTIONS[f'clinch_{wins}']:
        y,w=map(int,score.split(':'))
        side='yellow' if y==wins else 'white'
        other='white' if side=='yellow' else 'yellow'
        sequence=[other]*min(y,w)+[side]*wins
        # Later games must not change the clinching score in play-all mode.
        sequence += [other]*(2*wins-1-len(sequence))
        facts={**market,'phase':'FINISHED','public_result':{'winner_side':side,'games':[
            {'game_key':chr(65+i),'winner_side':s} for i,s in enumerate(sequence)]}}
        assert p._outcome(facts,f'clinch_{wins}')==(score,'clinching_score')
        facts['public_result']['games'][0]['winner_side']='draw'
        assert p._outcome(facts,f'clinch_{wins}')[0] is None


def public_facts(market,kind='duel',count=2):
    return {**market,'generation':2,'room_kind':kind,'participant_user_ids':[91,92],
            'prediction_window':{**market['prediction_window'],'minimum_until':(datetime.now(timezone.utc)+timedelta(seconds=60)).isoformat()},
            'rules':{'game_count':count},'phase':'READY_CHECK'}


@pytest.mark.parametrize('kind,count,expected', [('duel',1,1),('duel',2,2),('time_attack',1,1)])
def test_public_market_selection_has_no_virtual_inventory(market,kind,count,expected):
    facts=public_facts(market,kind,count);p.reconcile(ROOM,facts)
    markets=p.listing(ROOM)['markets']
    assert len(markets)==expected
    assert all(m['pricing']=='pool' and m['pool_units']==0 for m in markets)
    assert all(o['marginal_odds'] is None for m in markets for o in m['options'])


@pytest.mark.parametrize('winner,expected', [('yellow',(201000000,199000000)),('white',(199000000,201000000)),('draw',(200000000,200000000)),('cancelled',(200000000,200000000))])
def test_public_pools_conserve_wallet_total_and_settle_once(market,winner,expected):
    facts=public_facts(market);p.reconcile(ROOM,facts)
    first=order();p.place(ROOM,1,first,lambda:facts)
    p.place(ROOM,1,first,lambda:None)  # Retry is idempotent even during source outage.
    p.place(ROOM,2,order(option='white',uid=2),lambda:facts)
    listing=p.listing(ROOM,1)
    item=next(m for m in listing['markets'] if m['kind']=='winner')
    assert item['pool_units']==2000000 and item['mine']['estimated_payout']==2000000
    assert 'mine' not in p.listing(ROOM)['markets'][0]
    final={**facts,'phase':'CANCELLED' if winner=='cancelled' else 'FINISHED','content_sequence':2,'prediction_window':{**facts['prediction_window'],'open':False},
           'public_result':{'winner_side':winner,'games':[]}}
    p.reconcile(ROOM,final);p.reconcile(ROOM,final)
    assert (balance(1)[0],balance(2)[0])==expected
    assert sum(expected)==400000000
    assert p.listing(ROOM,1)['recent'][0]['payout_units']==expected[0]-199000000


def test_public_no_winning_stakes_refund_and_participants_rejected(market):
    facts=public_facts(market);p.reconcile(ROOM,facts)
    with pytest.raises(HTTPException) as exc:
        p.place(ROOM,91,order(),lambda:facts)
    assert exc.value.detail=='competition_prediction_participant'
    absent={**facts};absent.pop('participant_user_ids')
    with pytest.raises(HTTPException) as exc:p.place(ROOM,1,order(),lambda:absent)
    assert exc.value.status_code==503
    p.place(ROOM,1,order(),lambda:facts)
    p.reconcile(ROOM,{**facts,'phase':'FINISHED','content_sequence':3,'public_result':{'winner_side':'white','games':[]}})
    assert balance(1)[0]==200000000
    assert p.listing(ROOM,1)['recent'][0]['status']=='void'


def test_pool_remainder_is_conserved_and_deterministic():
    from backend.room_activities.match_prediction_rules import pool_payouts
    positions=[dict(user_id=i,stake=1,option_id='yellow' if i<4 else 'white') for i in range(1,5)]
    result,refund=pool_payouts(positions,'yellow')
    assert not refund and result=={1:2,2:1,3:1,4:0}
    assert pool_payouts(list(reversed(positions)),'yellow')[0]==result


def test_pool_deadline_rechecked_after_source_read(market):
    facts=public_facts(market)
    facts['prediction_window']['minimum_until']=(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
    p.reconcile(ROOM,facts)
    with pytest.raises(HTTPException,match='competition_prediction_closed'):
        p.place(ROOM,1,order(),lambda:facts)
    assert balance(1)[0]==200000000


def test_opening_first_two_probability_weights_are_one_two_one(market):
    public = p.listing(ROOM)
    winner = next(item for item in public['markets'] if item['kind'] == 'winner')
    scores = next(item for item in public['markets'] if item['kind'] == 'first_two')
    assert [item['marginal_odds'] for item in winner['options']] == [2, 2]
    assert [item['marginal_odds'] for item in scores['options']] == [4, 2, 4]
    assert [1 / item['marginal_odds'] for item in scores['options']] == [.25, .5, .25]
    assert balance() == (200000000, 777000)


@pytest.mark.parametrize('option,odds', [('2:0', 4), ('1:1', 2), ('0:2', 4)])
def test_weighted_first_two_quotes_preserve_product_and_slippage(option, odds):
    reserves = p._initial_reserves('first_two')
    stake = 1000 * p.UNIT
    shares, updated = p.buy(reserves, option, stake)
    assert stake < shares < odds * stake
    assert math.prod(updated.values()) >= math.prod(reserves.values())
    assert p.buy(updated, option, stake)[0] < shares


def _restore_equal_first_two_reserves():
    with auth_db() as db:
        db.execute("UPDATE competition_prediction_markets SET reserves=? WHERE kind='first_two'",
                   (json.dumps(dict.fromkeys(p.KINDS['first_two'], p.INITIAL)),))


def test_untouched_open_market_adopts_weights_and_invalidates_old_quote(market):
    _restore_equal_first_two_reserves()
    old_order = order('first_two', '1:1')
    before = balance()
    with pytest.raises(HTTPException, match='competition_prediction_price_changed'):
        p.place(ROOM, 1, old_order, lambda: market)
    assert balance() == before
    scores = next(item for item in p.listing(ROOM)['markets'] if item['kind'] == 'first_two')
    assert scores['revision'] == 1
    assert [item['marginal_odds'] for item in scores['options']] == [4, 2, 4]
    p.reconcile(ROOM, market)
    assert next(item for item in p.listing(ROOM)['markets'] if item['kind'] == 'first_two')['revision'] == 1


def test_existing_equal_market_with_bets_preserves_inventory_and_payout(market):
    _restore_equal_first_two_reserves()
    # An order accepted under the previous equal-reserve model.
    with patch.object(p, 'reconcile'):
        placed = p.place(ROOM, 1, order('first_two', '1:1'), lambda: market)
    with auth_db() as db:
        before = tuple(db.execute("SELECT reserves,revision FROM competition_prediction_markets WHERE kind='first_two'").fetchone())
    p.reconcile(ROOM, market)
    with auth_db() as db:
        assert tuple(db.execute("SELECT reserves,revision FROM competition_prediction_markets WHERE kind='first_two'").fetchone()) == before
    final = copy.deepcopy(market)
    final.update(phase='FINISHED', content_sequence=2)
    final['prediction_window']['open'] = False
    final['public_result']['games'] = [dict(game_key='A', winner_side='yellow'), dict(game_key='B', winner_side='white')]
    p.reconcile(ROOM, final)
    assert balance() == (199000000 + placed['shares'], 777000)


def test_closed_or_stale_market_is_not_reinitialized(market):
    _restore_equal_first_two_reserves()
    with auth_db() as db:
        db.execute("UPDATE competition_prediction_markets SET source_sequence=5 WHERE kind='first_two'")
    p.reconcile(ROOM, market)
    scores = next(item for item in p.listing(ROOM)['markets'] if item['kind'] == 'first_two')
    assert scores['revision'] == 0
    assert [item['marginal_odds'] for item in scores['options']] == [3, 3, 3]
    market['content_sequence'] = 6
    market['prediction_window']['open'] = False
    p.reconcile(ROOM, market)
    market['content_sequence'] = 7
    market['prediction_window']['open'] = True
    p.reconcile(ROOM, market)
    scores = next(item for item in p.listing(ROOM)['markets'] if item['kind'] == 'first_two')
    assert scores['status'] == 'closed'
    assert scores['revision'] == 0
    assert [item['marginal_odds'] for item in scores['options']] == [3, 3, 3]


@pytest.mark.parametrize('count', [2, 3])
def test_complete_set_constant_product_and_slippage(count):
    reserves = dict.fromkeys(map(str, range(count)), p.INITIAL)
    shares, after = p.buy(reserves, '0', 1000*p.UNIT)
    assert 1000*p.UNIT < shares < count*1000*p.UNIT
    assert math.prod(after.values()) >= math.prod(reserves.values())
    # Integer rounding changes invariant by less than one selected reserve unit.
    assert math.prod(after.values()) - math.prod(reserves.values()) < math.prod(value for key,value in after.items() if key!='0')
    next_shares, _ = p.buy(after, '0', 1000*p.UNIT)
    assert next_shares < shares


def test_two_independent_markets_lock_shares_and_refund_draw(market):
    first = p.place(ROOM, 1, order(), lambda:market)
    second = p.place(ROOM, 1, order('first_two', '1:1'), lambda:market)
    p.place(ROOM, 2, order(option='white',uid=2), lambda:market)
    assert balance() == (198000000, 777000)
    assert first['shares'] > first['stake']
    final=copy.deepcopy(market)
    final.update(phase='FINISHED', content_sequence=9)
    final['prediction_window']['open']=False
    final['public_result']=dict(winner_side='draw',games=[dict(game_key='A',winner_side='yellow'),dict(game_key='B',winner_side='white')])
    p.reconcile(ROOM,final)
    assert balance() == (199000000+second['shares'],777000)
    before=balance()
    p.reconcile(ROOM,final)
    assert balance()==before
    assert not p.pending_rooms()


@pytest.mark.parametrize('phase,winner,games,refund', [
    ('CANCELLED','yellow',['yellow','yellow'],True),
    ('FINISHED',None,[],True),
    ('FINISHED','white',['draw','white'],True),
    ('FINISHED','yellow',['yellow','yellow'],False),
])
def test_first_two_void_and_payout_rules(market,phase,winner,games,refund):
    placed=p.place(ROOM,1,order('first_two','2:0'),lambda:market)
    market.update(phase=phase,content_sequence=3)
    market['prediction_window']['open']=False
    market['public_result']=dict(winner_side=winner,games=[dict(game_key=key,winner_side=value) for key,value in zip('AB',games)])
    p.reconcile(ROOM,market)
    assert balance()[0]==200000000 if refund else balance()[0]==199000000+placed['shares']


def test_single_option_limit_is_per_market(market):
    for _ in range(6):
        p.place(ROOM,1,order(amount=10000),lambda:market)
    with pytest.raises(HTTPException,match='competition_prediction_limit'):
        p.place(ROOM,1,order(amount=500),lambda:market)
    with pytest.raises(HTTPException,match='competition_prediction_cannot_switch'):
        p.place(ROOM,1,order(option='white'),lambda:market)
    p.place(ROOM,1,order('first_two','0:2'),lambda:market)
    assert balance()[0]==139000000


@pytest.mark.parametrize('outcome', ['yellow', 'white', None])
def test_personal_history_survives_new_generation_and_is_private(market, outcome):
    placed = p.place(ROOM, 1, order(), lambda: market)
    assert p.listing(ROOM, 1)['recent'] == []
    final = copy.deepcopy(market)
    final.update(phase='FINISHED', content_sequence=3)
    final['prediction_window']['open'] = False
    final['public_result'] = dict(winner_side=outcome, games=[])
    p.reconcile(ROOM, final)
    result = p.listing(ROOM, 1)['recent'][0]
    payout = placed['shares'] if outcome == 'yellow' else 0 if outcome == 'white' else 1000000
    assert result['payout_units'] == payout
    assert result['stake_units'] == 1000000
    assert result['net_profit_units'] == payout - 1000000
    assert result['status'] == ('void' if outcome is None else 'settled')
    assert result['kind'] == 'winner' and result['option_id'] == 'yellow'
    next_match = copy.deepcopy(market)
    next_match.update(generation=2, content_sequence=1)
    p.reconcile(ROOM, next_match)
    assert p.listing(ROOM, 1)['recent'][0] == result
    assert p.listing(ROOM, 2)['recent'] == []
    assert p.listing('another-room', 1)['recent'] == []
    assert 'recent' not in p.listing(ROOM)


def test_room_activity_broadcast_tracks_pause_without_market_changes(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from backend.room_activities.runtime import RoomActivities
    messages = []
    facts = {'prediction_window': {'open': True}, 'suspended': False}
    hub = SimpleNamespace(room=SimpleNamespace(id=ROOM, content_kind='competition-match'),
                          content=SimpleNamespace(projection=facts), broadcast=messages.append)
    monkeypatch.setattr(p, 'reconcile', lambda *args: None)
    monkeypatch.setattr(p, 'listing', lambda *args: {'markets': [{'id':'m', 'status':'open'}]})
    activity = RoomActivities(hub)
    asyncio.run(activity.refresh())
    assert messages[-1]['available'] is True
    facts['suspended'] = True
    asyncio.run(activity.refresh())
    assert messages[-1]['available'] is False
    facts['suspended'] = False
    asyncio.run(activity.refresh())
    assert messages[-1]['available'] is True
    assert len(messages) == 3


def test_duplicate_concurrent_requests_charge_once_even_after_close(market):
    body=order()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:p.place(ROOM,1,body,lambda:market),range(2)))
    assert results[0]['shares']==results[1]['shares']
    assert balance()[0]==199000000
    market.update(phase='GAME_A_PLAYING',content_sequence=2)
    market['prediction_window']['open']=False
    p.reconcile(ROOM,market)
    assert p.place(ROOM,1,body,lambda:None)['duplicate']
    with pytest.raises(HTTPException,match='competition_prediction_request_conflict'):
        p.place(ROOM,1,{**body,'amount':500},lambda:market)


def test_stale_price_closed_paused_and_outage_never_charge(market):
    stale=order(uid=2)
    p.place(ROOM,1,order(),lambda:market)
    with pytest.raises(HTTPException,match='competition_prediction_price_changed'):
        p.place(ROOM,2,stale,lambda:market)
    before=balance()
    for facts in (None,{**market,'suspended':True},{**market,'prediction_window':{**market['prediction_window'],'open':False}}):
        with pytest.raises(HTTPException):
            p.place(ROOM,1,order(),lambda:facts)
    assert balance()==before
    p.reconcile(ROOM,market)  # Cannot reopen a closed market with cached facts.
    assert all(item['status']=='closed' for item in p.listing(ROOM)['markets'])


def test_losing_option_zero_payout_and_insufficient_balance_rollback(market):
    with auth_db() as db:
        db.execute('UPDATE token_accounts SET paid_balance_units=500000 WHERE user_id=1')
    before=p.listing(ROOM)
    with pytest.raises(HTTPException,match='prediction_insufficient_permanent'):
        p.place(ROOM,1,order(),lambda:market)
    assert p.listing(ROOM)['markets']==before['markets']
    p.place(ROOM,1,order(amount=500),lambda:market)
    market.update(phase='FINISHED',content_sequence=5)
    market['prediction_window']['open']=False
    market['public_result']['winner_side']='white'
    p.reconcile(ROOM,market)
    assert balance()==(0,777000)


def test_results_are_not_paid_before_match_end(market):
    p.place(ROOM,1,order('first_two','1:1'),lambda:market)
    market.update(phase='GAME_B_RESULT',content_sequence=5)
    market['prediction_window']['open']=False
    market['public_result']['games']=[dict(game_key='A',winner_side='white'),dict(game_key='B',winner_side='yellow')]
    p.reconcile(ROOM,market)
    assert balance()[0]==199000000
    assert all(item['status']=='closed' for item in p.listing(ROOM)['markets'])


def test_public_state_has_names_but_no_private_positions(market):
    p.place(ROOM,1,order(),lambda:market)
    public=p.listing(ROOM)
    assert all('mine' not in item for item in public['markets'])
    winner=next(item for item in public['markets'] if item['kind']=='winner')
    assert winner['options'][0]['name']=='Alice / Bob / Carol'
    assert public['amounts']==[500,1000,5000,10000]


def test_live_routes_use_competition_markets_and_survive_broadcast_failure(market, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.live import routes
    from backend.live.rooms import RoomDefinition

    def broken_broadcast(_):
        raise RuntimeError('disconnected broadcaster')

    hub = SimpleNamespace(room=RoomDefinition(id=ROOM,title={},content_kind='competition-match',
        capabilities={'predictions':True},metadata={'public_key':'mtest'}),
        activity_lock=asyncio.Lock(),limit=lambda *args:None,broadcast=broken_broadcast)
    monkeypatch.setattr(routes,'resolve_hub',lambda _:hub)
    monkeypatch.setattr(routes,'require_user',lambda _:{'id':1})
    monkeypatch.setattr(routes,'current_user_from_request',lambda _:{'id':1})
    monkeypatch.setattr(routes.competition_provider,'settlement',lambda _:market)
    app=FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        url=f'/api/live/rooms/{ROOM}/predictions'
        response=client.get(url)
        assert response.status_code==200
        assert response.json()['protocol']==p.RULES
        assert len(response.json()['markets'])==2
        body=order()
        assert client.post(url,json=body,headers={'Origin':'https://foreign.invalid'}).status_code==403
        first=client.post(url,json=body)
        assert first.status_code==200
        retry=client.post(url,json=body)
        assert retry.json()['duplicate']
        assert balance()[0]==199000000
