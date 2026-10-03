from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService

OWNER = Principal(91, 'Host')
GUEST = Principal(92, 'Guest')
ADMIN = Principal(1, 'Admin', 'admin')


@pytest.fixture
def service(tmp_path):
    r = CompetitionService(CompetitionDatabase(tmp_path/'duel.sqlite'), result_rest_seconds=0)
    r.initialize()
    return r


def create(r, count=2, principal=OWNER, command='create-duel-001'):
    projects = [p['project_ref'] for p in r.duels.catalog()][:count]
    return r.duels.create(principal, name='Free duel', projects=projects, command_id=command)


def start(r, count=2):
    room = create(r, count)
    code = room['room_code']
    r.claim_seat(code, GUEST, side='white', position=1, command_id='guest-join-001')
    r.set_ready(code, OWNER, ready=True, command_id='owner-ready-001')
    room = r.set_ready(code, GUEST, ready=True, command_id='guest-ready-001')
    assert room['status'] == 'GAME_A_PLAYING'
    return code


def finish(r, code, user, score):
    room = r.snapshot(code, user)
    runtime = room['match']['my_session']['runtime']
    return r.sync_client_game(code, user, instance_id=runtime['instance_id'], sequence=1,
        phase_token=room['match']['phase_token'], payload={'board':[[2,4],[8,0]],'score':score,'move_count':1},
        checkpoint={'version':1,'state':{}}, result_value=score, elapsed_ms=10, finished=True, outcome='no_moves')


def test_even_sequence_runs_without_draft_or_officials_and_survives_restart(service):
    r = service
    code = start(r)
    room = r.snapshot(code, OWNER)
    assert room['draft'] is None and room['room_kind']=='duel'
    assert room['me']['staff_roles']==[] and not room['me']['is_match_official']
    assert room['match']['my_session']['runtime']['checkpoint'] is None
    assert r.list_live_rooms()==[]
    with r.database.transaction() as db:
        assert not db.execute('SELECT 1 FROM competition_drafts').fetchone()
        assert not db.execute('SELECT 1 FROM competition_prediction_windows').fetchone()
    finish(r, code, OWNER, 200)
    finish(r, code, GUEST, 100)
    r.settle_deadline(code)
    r = CompetitionService(r.database, result_rest_seconds=0)
    r.initialize()
    room=r.snapshot(code, OWNER)
    assert room['status']=='GAME_B_READY'
    assert room['match']['ready_deadline_at'] is None
    assert not room['me']['can_mark_captain_ready']
    r.settle_deadline(code, now=datetime.now(timezone.utc)+timedelta(minutes=2))
    assert r.snapshot(code, OWNER)['status']=='GAME_B_READY'
    for user in (OWNER,GUEST):
        own=r.snapshot(code,user)
        r.set_game_readiness(code,user,readiness_role='player',ready=True,
                            phase_token=own['match']['phase_token'],command_id=f'game-b-ready-{user.user_id}')
    finish(r, code, OWNER, 20)
    finish(r, code, GUEST, 90)
    r.settle_deadline(code)
    room=r.snapshot(code,OWNER)
    assert room['status']=='FINISHED'
    assert room['match']['series_score']=={'yellow':1,'white':1,'draws':0}
    assert room['match']['winner_side']=='draw'
    assert [result['project_key'] for result in room['match']['results']]==list(room['selected_projects'].values())


def test_catalog_limits_idempotency_and_expiry(service):
    r=service
    assert all(not p['test_only'] for p in r.duels.catalog())
    room=create(r)
    assert create(r)['id']==room['id']
    with pytest.raises(CompetitionError, match='') as exc: create(r,command='create-duel-002')
    assert exc.value.code=='DUEL_ACTIVE_ROOM'
    r.set_ready(room['room_code'], OWNER, ready=True, command_id='alone-ready-01')
    r.process_due_rooms(now=datetime.now(timezone.utc)+timedelta(minutes=31))
    assert r.snapshot(room['room_code'],OWNER)['status']=='CANCELLED'
    with pytest.raises(CompetitionError) as exc: create(r,command='create-duel-003')
    assert exc.value.code=='DUEL_CREATE_LIMIT'
    for projects in ([], ['unknown'], ['standard-2048-test'], [r.duels.catalog()[0]['project_ref']]*2):
        with pytest.raises(CompetitionError):
            r.duels.create(GUEST,name='bad room',projects=projects,command_id='invalid-pool-01')


def test_concurrent_creation_and_readiness_are_atomic(service):
    with ThreadPoolExecutor(max_workers=2) as pool:
        created=list(pool.map(lambda _:create(service),range(2)))
    assert created[0]['id']==created[1]['id']
    code=created[0]['room_code']
    service.claim_seat(code,GUEST,side='white',position=1,command_id='claim-other-001')
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda p:service.set_ready(code,p,ready=True,command_id=f'concurrent-{p.user_id}'), (OWNER,GUEST)))
    with service.database.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM competition_game_sessions').fetchone()[0]==2


@pytest.mark.parametrize('user',[OWNER,GUEST,ADMIN])
def test_no_official_privileges_even_for_platform_admin(service,user):
    code=start(service)
    room=service.snapshot(code,user)
    for capability in ('can_manage','can_manage_members','can_suspend','can_override_result','can_force_finish','can_force_advance','can_report_issue','is_match_official'):
        assert not room['me'][capability]
    with service.database.transaction() as db:
        record=service._room_row(db,code)
        for guard in (service._require_match_official,service._require_room_organizer):
            with pytest.raises(CompetitionError) as exc: guard(db,record,user)
            assert exc.value.code=='DUEL_NO_OFFICIALS'
    with pytest.raises(CompetitionError):
        service.manage_member(code,user,user_id=GUEST.user_id,remove=True,command_id='cannot-remove-01')


def test_waiting_next_game_expires_without_forfeiting_or_starting(service):
    code=start(service)
    finish(service,code,OWNER,10)
    finish(service,code,GUEST,10)
    service.settle_deadline(code)
    service.settle_deadline(code,now=datetime.now(timezone.utc)+timedelta(minutes=31))
    room=service.snapshot(code,OWNER)
    assert room['status']=='CANCELLED'
    assert len(room['match']['results'])==1 and room['match']['results'][0]['winner_side']=='draw'


def test_schema_rejects_privilege_fields():
    from competition.backend.schemas import CreateDuelRequest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        CreateDuelRequest(name='Duel',projects=['x'],command_id='abcdefgh',event_slug='cup',rules={'auto_ready':True})


def test_owner_seat_and_opponent_active_limit(service):
    room=create(service)
    code=room['room_code']
    assert not room['me']['can_leave_seat'] and room['me']['can_close']
    with pytest.raises(CompetitionError): service.leave_seat(code,OWNER,command_id='owner-leaves-01')
    with pytest.raises(CompetitionError): service.claim_seat(code,OWNER,side='white',position=1,command_id='self-play-0001')
    other=create(service,principal=GUEST,command='other-room-0001')
    with pytest.raises(CompetitionError) as exc:
        service.claim_seat(code,GUEST,side='white',position=1,command_id='already-active-01')
    assert exc.value.code=='DUEL_ACTIVE_ROOM'
    service.close_competition(other['room_code'],GUEST,command_id='close-other-001')
    service.claim_seat(code,GUEST,side='white',position=1,command_id='join-after-close')
    service.set_ready(code,GUEST,ready=True,command_id='guest-readies-01')
    service.leave_seat(code,GUEST,command_id='guest-leaves-01')
    assert not service.snapshot(code,OWNER)['teams']['white']['ready']


def test_ordinary_user_still_cannot_create_formal_room_or_link_duel(service):
    with pytest.raises(CompetitionError) as exc: service.create_competition(OWNER,name='Formal room')
    assert exc.value.code=='ORGANIZER_REQUIRED'
    room=create(service)
    with service.database.transaction() as db:
        record=service._room_row(db,room['room_code'])
        with pytest.raises(CompetitionError) as exc: service.events.link_in_transaction(db,'anything',record,ADMIN)
        assert exc.value.code=='DUEL_NO_OFFICIALS'
    for user in (GUEST,ADMIN):
        with pytest.raises(CompetitionError): service.close_competition(room['room_code'],user,command_id=f'close-denied-{user.user_id}')


@pytest.mark.parametrize('count',[1,4,15])
def test_fixed_sequence_bounds_and_order(service,count):
    code=start(service,count)
    room=service.snapshot(code,OWNER)
    assert len(room['selected_projects'])==count
    assert list(room['selected_projects'].values())==[p['project_ref'] for p in service.duels.catalog()][:count]
    with service.database.transaction() as db:
        rows=db.execute("SELECT payload_json FROM competition_events WHERE competition_id=? AND event_type='competition.status_changed'",(room['id'],)).fetchall()
        assert all('DRAW' not in row[0] and 'LINEUP' not in row[0] for row in rows)


def test_project_result_contract_preserves_metrics_and_race_ties(service):
    from competition.backend.projects.contracts import ProjectState
    from competition.backend.projects.result_policy import ResultPolicy
    a=ProjectState(((2,),),100,30,True,extra={'result_value':5})
    b=ProjectState(((2,),),200,20,True,extra={'result_value':2})
    assert ResultPolicy('delivered_cargo').compare(a,b)==(5,2,'yellow','delivered_cargo')
    assert ResultPolicy('board_sum').compare(a,b)[2:] == ('yellow','board_sum')
    a=ProjectState(((2,),),0,30,True,outcome='target_reached')
    b=ProjectState(((2,),),0,20,True,outcome='target_reached')
    assert ResultPolicy(race=True).compare(a,b)[2:] == ('white','race_elapsed')
    assert ResultPolicy(race=True).compare(a,a)[2:] == ('draw','race_elapsed')
    for descriptor in service.project_registry.descriptors():
        adapter=service.project_registry.resolve(descriptor.project_ref,descriptor.rules_version)
        assert descriptor.result_policy.race == bool(getattr(getattr(adapter,'rules',None),'race',False))


def test_http_create_and_catalog_use_shared_room_payload(service):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from competition.backend.routes import router, current_principal
    app=FastAPI()
    app.state.competition_service=service
    app.include_router(router)
    app.dependency_overrides[current_principal]=lambda: OWNER
    with TestClient(app) as client:
        projects=client.get('/api/duel-projects').json()['projects']
        payload=dict(name='API duel',projects=[projects[0]['project_ref']],command_id='http-create-001')
        response=client.post('/api/duel-rooms',json=payload)
        assert response.status_code==201
        room=response.json()['competition']
        assert room['room_kind']=='duel' and room['me']['seat']['user_id']==OWNER.user_id
        assert client.post('/api/duel-rooms',json=payload).json()['competition']['id']==room['id']
        assert client.post('/api/duel-rooms',json={**payload,'event_slug':'cup'}).status_code==422
