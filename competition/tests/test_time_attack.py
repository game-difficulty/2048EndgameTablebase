from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import json
import pytest
from backend.human_play import engine
from backend.gamer_ranked.prng import Xoshiro128StarStar
from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService
from competition.backend.projects.target_challenge import configuration, compare_best

HOST, GUEST, OTHER = Principal(91, 'Host'), Principal(92, 'Guest'), Principal(93, 'Other')


@pytest.fixture
def service(tmp_path):
    r = CompetitionService(CompetitionDatabase(tmp_path/'time.sqlite'))
    r.initialize()
    return r


def create(r, **kw):
    return r.time_attacks.create(HOST, **(dict(name='Timed PB', variant='3x4', target_kind='board_sum',
                target_value=10, clock_seconds=60, command_id='create-time-001') | kw))


def start(r, **kw):
    room = create(r, **kw)
    code = room['room_code']
    r.claim_seat(code, GUEST, side='white', position=1, command_id='join-time-001')
    r.set_ready(code, HOST, ready=True, command_id='ready-time-host')
    return r.set_ready(code, GUEST, ready=True, command_id='ready-time-guest')


def move_event(state, variant):
    for direction in range(4):
        board, _ = engine.move(state['board'], *engine.VARIANTS[variant], direction)
        if board != state['board']:
            rng = Xoshiro128StarStar(state['rng'])
            index, value = engine.spawn(board, rng)
            return [direction | (index << 2) | (64 if value == 4 else 0), 0]


def submit(r, room, user=HOST):
    own = r.snapshot(room['room_code'], user)
    side = own['me']['seat']['side']
    attempt = own['time_attack']['players'][side]['attempt']
    return r.time_attacks.command(room['room_code'], user, attempt_id=attempt['id'], action='submit',
            command_id=f"move-{attempt['id']}-{attempt['sequence']}", base_sequence=attempt['sequence'],
            events=[move_event(attempt['state'], own['time_attack']['configuration']['variant'])])


def expire(r, room):
    r.process_due_rooms(now=datetime.fromisoformat(room['time_attack']['deadline_at'])+timedelta(seconds=1))
    return r.snapshot(room['room_code'], HOST)


def test_start_sync_isolation_and_deadline(service):
    r = service
    room = start(r)
    assert room['room_kind'] == 'time_attack'
    assert room['draft'] is None and room['match'] is None
    assert room['status'] == 'GAME_A_PLAYING'
    assert not room['me']['can_manage'] and not room['me']['can_force_finish']
    y, w = room['time_attack']['players'].values()
    assert y['attempt']['started_at'] == w['attempt']['started_at']
    assert 'seed' not in y['attempt'] and 'state' not in y['attempt']
    final = expire(r, room)
    assert final['status'] == 'FINISHED' and final['time_attack']['winner_side'] == 'draw'
    late = r.time_attacks.command(room['room_code'], HOST, attempt_id=y['attempt']['id'], action='restart', command_id='late-restart-1')
    assert late['status'] == 'FINISHED'


def test_verified_attempt_does_not_end_room_and_restart_preserves_pb(service):
    r = service
    room = start(r, target_kind='tile', target_value=8)
    for _ in range(100):
        room = submit(r, room)
        a = room['time_attack']['players']['yellow']['attempt']
        if a['status'] != 'playing':
            break
    assert a['status'] == 'reached'
    assert room['status'] == 'GAME_A_PLAYING'
    assert room['time_attack']['players']['white']['attempt']['status'] == 'playing'
    best = room['time_attack']['players']['yellow']['best']
    with r.database.transaction() as db:
        db.execute('UPDATE competition_time_attempts SET started_at=? WHERE id=?',
                   ((datetime.now(timezone.utc)-timedelta(seconds=2)).isoformat(), a['id']))
    kw = dict(attempt_id=a['id'], action='restart', command_id='restart-time-001')
    after = r.time_attacks.command(room['room_code'], HOST, **kw)
    assert after['time_attack']['players']['yellow']['best'] == best
    fresh = after['time_attack']['players']['yellow']['attempt']
    assert fresh['number'] == 2 and fresh['id'] != a['id'] and fresh['seed'] != a['seed']
    assert r.time_attacks.command(room['room_code'], HOST, **kw)['time_attack']['players']['yellow']['attempt']['id'] == fresh['id']
    reboot = CompetitionService(r.database)
    reboot.initialize()
    final = expire(reboot, after)
    assert final['time_attack']['winner_side'] == 'yellow'
    assert final['time_attack']['players']['yellow']['best'] == best


def test_wait_expiry_and_shared_limits(service):
    r = service
    room = create(r)
    assert create(r)['id'] == room['id']
    with pytest.raises(CompetitionError) as exc:
        r.duels.create(HOST, name='Other room', projects=[r.duels.catalog()[0]['project_ref']], command_id='other-room-001')
    assert exc.value.code == 'DUEL_ACTIVE_ROOM'
    r.process_due_rooms(now=datetime.now(timezone.utc)+timedelta(minutes=31))
    assert r.snapshot(room['room_code'], HOST)['status'] == 'CANCELLED'


def test_forgeries_stale_requests_and_spectators(service):
    r = service
    room = start(r)
    a = r.snapshot(room['room_code'], HOST)['time_attack']['players']['yellow']['attempt']
    for user, aid, events, expected in [
        (OTHER, a['id'], [[0, 0]], 'ACTIVE_PLAYER_REQUIRED'),
        (HOST, 'old-attempt', [[0, 0]], 'STALE_ATTEMPT'),
        (HOST, a['id'], [[128, 0]], 'INVALID_ATTEMPT'),
        (HOST, a['id'], [[0, -1]], 'INVALID_ATTEMPT'),
    ]:
        with pytest.raises(CompetitionError) as exc:
            r.time_attacks.command(room['room_code'], user, attempt_id=aid, action='submit', command_id='bad-action-001', events=events)
        assert exc.value.code == expected


@pytest.mark.parametrize('variant', ['4x4','3x4','2x4','3x3'])
def test_all_variants(service, variant):
    room = start(service, variant=variant)
    room = submit(service, room)
    assert len(room['time_attack']['players']['yellow']['attempt']['board']) == engine.VARIANTS[variant][0]*engine.VARIANTS[variant][1]


@pytest.mark.parametrize('kind,value', [('tile',7),('tile',12),('tile',True),('board_sum',9),('board_sum',11),('board_sum',0)])
def test_invalid_targets(kind, value):
    with pytest.raises(CompetitionError): configuration('4x4', kind, value)


def test_best_comparison():
    assert compare_best(None,None) == 'draw'
    assert compare_best(10,None) == 'yellow'
    assert compare_best(None,10) == 'white'
    assert compare_best(10,10) == 'draw'
    assert compare_best(10,11) == 'yellow'
    assert compare_best(11,10) == 'white'


def test_concurrent_duplicate_batches_are_applied_once(service):
    r = service
    room = start(r)
    a = r.snapshot(room['room_code'], HOST)['time_attack']['players']['yellow']['attempt']
    kw = dict(attempt_id=a['id'], command_id='retry-exact-packet', action='submit', base_sequence=0,
              events=[move_event(a['state'], '3x4')])
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: r.time_attacks.command(room['room_code'], HOST, **kw), range(2)))
    assert all(x['time_attack']['players']['yellow']['attempt']['sequence'] == 1 for x in results)
    with pytest.raises(CompetitionError) as exc:
        r.time_attacks.command(room['room_code'], HOST, **(kw | {'command_id':'outdated-new-command'}))
    assert exc.value.code == 'STALE_ATTEMPT'


def test_exact_deadline_cannot_submit_or_restart_and_settlement_persists(service):
    r = service
    room = start(r)
    a = r.snapshot(room['room_code'], HOST)['time_attack']['players']['yellow']['attempt']
    deadline = datetime.fromisoformat(room['time_attack']['deadline_at'])
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return deadline
    with patch('competition.backend.time_attack_rooms.datetime', Clock):
        final = r.time_attacks.command(room['room_code'], HOST, attempt_id=a['id'], command_id='deadline-submit',
                     action='submit', events=[move_event(a['state'], '3x4')])
    assert final['status'] == 'FINISHED'
    assert final['time_attack']['players']['yellow']['attempt']['sequence'] == 0
    assert r.snapshot(room['room_code'], HOST)['status'] == 'FINISHED'


def test_exact_sum_and_record_replay(service):
    r = service
    room = start(r, target_value=10)
    for _ in range(10):
        room = submit(r, room)
        a = room['time_attack']['players']['yellow']['attempt']
        if a['status'] != 'playing': break
    assert a['status'] in ('reached','overshot')
    total = sum(a['board'])
    assert (a['pb_ms'] is not None) == (total == 10)
    if a['status'] == 'reached':
        record = r.time_attacks.best_record(room['room_code'], OTHER, 'yellow')
        state = engine.initial(record['id'], '3x4', record['header']['seed'])
        replayed = engine.advance(state, '3x4', b''.join(engine.EVENT.pack(*e) for e in record['events']))
        assert replayed['board'] == a['board']
    with pytest.raises(CompetitionError) as exc:
        r.time_attacks.best_record(room['room_code'], HOST, 'white')
    assert exc.value.code == 'NO_VALID_ATTEMPT'


def test_admin_cannot_adjudicate_or_link_event(service):
    r = service
    room = start(r)
    admin = Principal(1,'Admin','admin')
    snapshot = r.snapshot(room['room_code'], admin)
    for key in ('can_manage','can_manage_members','can_suspend','can_override_result','can_force_finish'):
        assert not snapshot['me'][key]
    with pytest.raises(CompetitionError) as exc:
        r.assign_staff(room['room_code'], admin, user_id=93, role='referee')
    assert exc.value.code == 'DUEL_NO_OFFICIALS'
    assert r.list_live_rooms() == []
    with r.database.transaction() as db:
        with pytest.raises(CompetitionError) as exc:
            r.events.link_in_transaction(db, 'any-event', r._room_row(db, room['room_code']), admin)
        assert exc.value.code == 'DUEL_NO_OFFICIALS'


def test_api_schemas_and_auth(service, monkeypatch):
    from fastapi.testclient import TestClient
    from competition.backend.app import create_app
    monkeypatch.setenv('COMPETITION_DB', str(service.database.path))
    monkeypatch.setenv('COMPETITION_ALLOW_DEV_AUTH','1')
    with TestClient(create_app()) as client:
        body=dict(name='Timed duel',variant='2x4',target_kind='tile',target_value=8,clock_seconds=30,command_id='http-create-time')
        assert client.post('/api/time-attack-rooms',json=body).status_code == 401
        headers={'X-Competition-Dev-User':'91:Host:user'}
        assert client.post('/api/time-attack-rooms',json=body | {'target_value':True},headers=headers).status_code == 422
        assert client.post('/api/time-attack-rooms',json=body | {'referee':93},headers=headers).status_code == 422
        response=client.post('/api/time-attack-rooms',json=body,headers=headers)
        assert response.status_code == 201
        code=response.json()['competition']['room_code']
        assert client.get(f'/api/competitions/{code}/time-attack/best/yellow').status_code == 401
