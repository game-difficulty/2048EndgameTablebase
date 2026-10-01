import copy
import math
import uuid
from concurrent.futures import ThreadPoolExecutor

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
