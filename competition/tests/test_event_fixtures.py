from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.event_fixtures import round_robin
from competition.backend.schemas import FixtureActionRequest, CreateFixtureStageRequest
from competition.backend.service import CompetitionService

ADMIN = Principal(900, 'Organizer', 'admin')
SLUG = '819984-cup-3'
TEAM_NAMES = ['PKU', '鲜', 'miles是啥子对', '九月猫', '荷塘月色', 'BBB', 'SSS', 'SCL']


@pytest.fixture
def service(tmp_path):
    r = CompetitionService(CompetitionDatabase(tmp_path / 'fixtures.sqlite'))
    r.initialize()
    e = r.events.enrollment
    e.account_reader = lambda ids: {uid: f'Player {uid}' for uid in ids}
    state = e.action(SLUG, ADMIN, action='settings', revision=0, mode='organizer_team', capacity=0, registration_open=False)
    e.import_roster(SLUG, ADMIN, revision=state['revision'], dry_run=False, entries=[
        dict(user_id=uid, team_name=TEAM_NAMES[(uid-1)//3], captain=(uid-1)%3 == 0,
             position=(uid-1)%3+1, is_external=False) for uid in range(1, 25)])
    state = e.snapshot(SLUG)
    e.action(SLUG, ADMIN, action='lock_roster', revision=state['revision'])
    return r


def create(r, **overrides):
    teams = {t['name']: t['id'] for t in r.fixtures.snapshot(SLUG)['teams']}
    payload = dict(name='小组赛', command_id='stage-create-001', groups=[
        dict(name='A', team_ids=[teams[n] for n in TEAM_NAMES[:4]]),
        dict(name='B', team_ids=[teams[n] for n in TEAM_NAMES[4:]])])
    payload.update(overrides)
    return r.fixtures.create_stage(SLUG, ADMIN, **payload)


def first(r):
    view = r.fixtures.snapshot(SLUG, ADMIN)
    item = view['stages'][0]['fixtures'][0]
    teams = {t['id']: t for g in view['stages'][0]['groups'] for t in g['teams']}
    return item, [teams[item[s+'_team_id']] for s in ('yellow', 'white')]


def actor(team):
    return Principal(team['captain_user_id'], 'Captain')


def book(r, item, principal, *, action='schedule', when=None, command_id='fixture-book-001', **extra):
    if action == 'schedule':
        extra['starts_at'] = (when or datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
    return r.fixtures.action(SLUG, item['id'], principal, action=action, revision=item['revision'], command_id=command_id, **extra)


def refreshed(r, id, principal=ADMIN):
    return next(f for s in r.fixtures.snapshot(SLUG, principal)['stages'] for f in s['fixtures'] if f['id'] == id)


@pytest.mark.parametrize('count', [2, 3, 4, 5, 16])
def test_circle_unique_pairs_and_one_match_per_team_per_round(count):
    rows = list(round_robin(range(count)))
    pairs = {tuple(sorted((a,b))) for _,a,b in rows}
    assert len(rows) == len(pairs) == count*(count-1)//2
    for number in {r[0] for r in rows}:
        players = [t for n,a,b in rows if n == number for t in (a,b)]
        assert len(players) == len(set(players))


def test_eight_teams_generate_twelve_fixtures_without_empty_rooms(service):
    result = create(service)
    stage = result['stages'][0]
    assert len(stage['fixtures']) == 12
    assert {f['round_number'] for f in stage['fixtures']} == {1,2,3}
    assert len(stage['projects']) == 12
    assert all(f['status'] == 'UNSCHEDULED' for f in stage['fixtures'])
    with service.database.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM competitions').fetchone()[0] == 0
    assert create(service)['stages'][0]['id'] == stage['id']
    service.initialize()
    assert service.fixtures.snapshot(SLUG)['stages'][0]['id'] == stage['id']


def test_creation_authorization_locked_roster_and_validation(service):
    with pytest.raises(CompetitionError) as exc:
        service.fixtures.create_stage(SLUG, Principal(1,'Player'), name='越权分组', groups=[], command_id='bad-stage-0001')
    assert exc.value.status_code == 403
    with pytest.raises(CompetitionError):
        create(service, groups=[dict(name='A',team_ids=['bad','bad'])])
    with pytest.raises(CompetitionError):
        create(service, projects=['not-registered'])
    with pytest.raises(CompetitionError):
        create(service, rules=dict(team_size=4))
    state = service.events.enrollment.snapshot(SLUG)
    service.events.enrollment.action(SLUG, ADMIN, action='unlock', revision=state['revision'], reason='解锁测试名单')
    with pytest.raises(CompetitionError) as exc:
        create(service)
    assert exc.value.code == 'ROSTER_NOT_LOCKED'
    with service.database.transaction() as db:
        assert not db.execute('SELECT 1 FROM tournament_fixture_stages').fetchone()


def test_captain_books_private_code_without_official_privileges(service):
    create(service)
    item, teams = first(service)
    for principal in (Principal(2,'Noncaptain'), Principal(13,'Other captain'), Principal(999,'Spectator')):
        if principal.user_id in [t['captain_user_id'] for t in teams]:
            continue
        with pytest.raises(CompetitionError) as exc:
            book(service, item, principal)
        assert exc.value.status_code == 403
    result = book(service, item, actor(teams[0]))
    current = refreshed(service, item['id'], actor(teams[1]))
    code = current['room_code']
    assert code
    assert refreshed(service, item['id'])['room_code'] == code
    assert refreshed(service, item['id'], None)['room_code'] is None
    member = Principal(teams[0]['members'][1]['user_id'], 'Player')
    assert refreshed(service, item['id'], member)['room_code'] == code
    assert not refreshed(service, item['id'], member)['can_schedule']
    assert result['_changed_room_code'] == code
    assert code in [room['room_code'] for room in service.list_competitions(actor(teams[1]))]
    view = service.snapshot(code, actor(teams[0]))
    assert not view['me']['can_manage_members']
    assert not view['me']['can_close']
    with service.database.transaction() as db:
        room = service._room_row(db, code)
        assert room['created_by_user_id'] == ADMIN.user_id
        assert not db.execute('SELECT 1 FROM competition_staff WHERE competition_id=? AND user_id=?', (room['id'], actor(teams[0]).user_id)).fetchone()
    with pytest.raises(CompetitionError):
        service.close_competition(code, actor(teams[0]), command_id='captain-close-001')
    assert service.events.detail(SLUG, actor(teams[1]))['rooms'][0]['fixture_id'] == item['id']
    service.manage_member(code, ADMIN, user_id=member.user_id, remove=True, command_id='fixture-remove-user')
    assert refreshed(service,item['id'],member)['room_code'] is None
    assert code not in [room['room_code'] for room in service.list_competitions(member)]


def test_duplicate_retry_and_racing_captains_only_create_one_room(service):
    create(service)
    item, teams = first(service)
    start = datetime.now(timezone.utc) + timedelta(hours=3)
    def submit(index):
        try:
            return book(service, item, actor(teams[index]), when=start, command_id=f'captain-race-{index}')
        except CompetitionError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (0,1)))
    assert sum(isinstance(x,dict) for x in results) == 1
    assert 'FIXTURE_CHANGED' in results
    winner = next(i for i,r in enumerate(results) if isinstance(r,dict))
    retry = submit(winner)
    assert isinstance(retry,dict)
    with service.database.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM competitions').fetchone()[0] == 1
    with pytest.raises(CompetitionError) as exc:
        book(service, item, actor(teams[winner]), when=start+timedelta(hours=1), command_id=f'captain-race-{winner}')
    assert exc.value.code == 'COMMAND_ID_REUSED'


def test_booking_atomic_failure_and_timezone_validation(service):
    create(service)
    item, teams = first(service)
    for value in ('2026-10-05T12:00:00', 'invalid', '2020-01-01T00:00:00+08:00'):
        with pytest.raises(CompetitionError):
            service.fixtures.action(SLUG, item['id'], actor(teams[0]), action='schedule', revision=0, command_id='invalid-time-0001', starts_at=value)
    with service.database.transaction(immediate=True) as db:
        db.execute('UPDATE tournament_enrollment_config SET roster_locked=0 WHERE event_slug=?', (SLUG,))
    with pytest.raises(CompetitionError):
        book(service, item, actor(teams[0]))
    with service.database.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM competitions').fetchone()[0] == 0
        assert not db.execute('SELECT 1 FROM tournament_fixture_commands').fetchone()


def test_reschedule_keeps_original_until_other_captain_confirms_and_resets_ready(service):
    create(service)
    item, teams = first(service)
    start = datetime.now(timezone.utc) + timedelta(hours=3)
    book(service, item, actor(teams[0]), when=start)
    booked = refreshed(service, item['id'])
    code = booked['room_code']
    for side,team in zip(('yellow','white'), teams):
        for p in team['members']:
            user = Principal(p['user_id'], p['display_name'])
            service.schedule.check_in(code, user)
            service.claim_seat(code, user, side=side, position=p['position'], command_id=f"seat-prepare-{p['user_id']}")
        service.set_ready(code, actor(team), ready=True, command_id=f"ready-prepare-{team['id']}")
    new = start + timedelta(hours=1)
    book(service, booked, actor(teams[0]), when=new, command_id='reschedule-propose-001')
    proposal = refreshed(service, item['id'])
    assert proposal['starts_at'] == start.isoformat()
    assert proposal['proposed_at'] == new.isoformat()
    with pytest.raises(CompetitionError) as exc:
        book(service, proposal, actor(teams[0]), action='confirm', command_id='reschedule-self-001')
    assert exc.value.code == 'FIXTURE_OTHER_CAPTAIN_REQUIRED'
    book(service, proposal, actor(teams[1]), action='confirm', command_id='reschedule-confirm-001')
    final = refreshed(service, item['id'])
    assert final['starts_at'] == new.isoformat() and final['proposed_at'] is None
    assert final['room_code'] == code
    with service.database.transaction() as db:
        cid = service._room_row(db, code)['id']
        assert db.execute('SELECT COUNT(*) FROM competition_team_readiness WHERE competition_id=?', (cid,)).fetchone()[0] == 0
        assert db.execute('SELECT COUNT(*) FROM competition_seats WHERE competition_id=?', (cid,)).fetchone()[0] == 6
    for team in teams:
        service.set_ready(code, actor(team), ready=True, command_id=f"ready-renewed-{team['id']}")
    # A new service instance catches up from persisted readiness and schedule;
    # neither an open browser nor in-memory timers are required.
    recovered = CompetitionService(CompetitionDatabase(service.database.path))
    recovered.initialize()
    assert not recovered.settle_deadline(code, now=new-timedelta(seconds=1))
    assert code in recovered.process_due_rooms(now=new)
    assert recovered.snapshot(code, ADMIN)['status'] == 'DRAW'


@pytest.mark.parametrize('action,side', [('reject',1), ('cancel_proposal',0)])
def test_reject_or_withdraw_keeps_original(service, action, side):
    create(service)
    item, teams = first(service)
    book(service, item, actor(teams[0]))
    booked = refreshed(service, item['id'])
    book(service, booked, actor(teams[0]), command_id='propose-then-reject')
    proposal = refreshed(service, item['id'])
    book(service, proposal, actor(teams[side]), action=action, command_id='reject-or-withdraw')
    final = refreshed(service, item['id'])
    assert final['starts_at'] == booked['starts_at'] and final['proposed_at'] is None


def test_conflicting_team_time_and_close_spacing_warning(service):
    create(service)
    item, teams = first(service)
    start = datetime.now(timezone.utc) + timedelta(hours=3)
    book(service, item, actor(teams[0]), when=start)
    another = next(f for f in service.fixtures.snapshot(SLUG,ADMIN)['stages'][0]['fixtures']
                   if f['id'] != item['id'] and teams[0]['id'] in (f['yellow_team_id'], f['white_team_id']))
    with pytest.raises(CompetitionError) as exc:
        book(service, another, actor(teams[0]), when=start, command_id='same-team-same-time')
    assert exc.value.code == 'FIXTURE_TIME_CONFLICT'
    result = book(service, another, actor(teams[0]), when=start+timedelta(minutes=30), command_id='same-team-near-time')
    assert result['warnings']


def test_closed_reschedule_and_normal_lateness_settlement(service):
    create(service)
    item, teams = first(service)
    start = datetime.now(timezone.utc) + timedelta(hours=3)
    book(service, item, actor(teams[0]), when=start)
    booked = refreshed(service, item['id'])
    code = booked['room_code']
    assert code in service.process_due_rooms(now=start+timedelta(minutes=16))
    assert refreshed(service, item['id'])['exception'] == 'both_late'
    assert refreshed(service, item['id'])['series_score'] == dict(yellow=0,white=0)
    assert not refreshed(service, item['id'], actor(teams[0]))['can_schedule']
    with pytest.raises(CompetitionError) as exc:
        book(service, booked, actor(teams[0]), command_id='too-late-reschedule')
    assert exc.value.code == 'FIXTURE_SCHEDULE_CLOSED'


def test_manager_binds_existing_room_reverse_sides_and_rematch_tracks_replacement(service):
    create(service)
    item, teams = first(service)
    room = service.create_competition(ADMIN, name='已有正式比赛', event_slug=SLUG,
        yellow_team_id=teams[1]['id'], white_team_id=teams[0]['id'], starts_at=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat())
    with pytest.raises(CompetitionError):
        book(service, item, actor(teams[0]), action='bind', room_code=room['room_code'])
    book(service, item, ADMIN, action='bind', room_code=room['room_code'])
    bound = refreshed(service, item['id'])
    assert bound['room_code'] == room['room_code']
    replacement = service.rematch_before_lineup(room['room_code'], ADMIN, command_id='fixture-rematch-001')
    assert refreshed(service, item['id'])['room_code'] == replacement['room_code']
    assert refreshed(service, item['id'])['revision'] == bound['revision']+1
    with service.database.transaction() as db:
        assert db.execute('SELECT COUNT(*) FROM tournament_fixtures WHERE competition_id IS NOT NULL').fetchone()[0] == 1


def test_locked_roster_changes_do_not_silently_retarget_fixtures(service):
    create(service)
    item, teams = first(service)
    with service.database.transaction(immediate=True) as db:
        db.execute('UPDATE tournament_entrants SET user_id=100 WHERE event_slug=? AND user_id=?', (SLUG, teams[0]['members'][1]['user_id']))
    with pytest.raises(CompetitionError) as exc:
        book(service, item, actor(teams[0]))
    assert exc.value.code == 'FIXTURE_ROSTER_CHANGED'


def test_stage_rules_are_frozen_and_support_other_formats(service):
    result = create(service, rules={'preset':'bo7','lineup_policy':'balanced'})
    assert result['stages'][0]['rules']['team_clock_seconds'] == 4800
    item, teams = first(service)
    book(service, item, actor(teams[0]))
    code = refreshed(service,item['id'])['room_code']
    assert service.snapshot(code,ADMIN)['rules']['game_count'] == 7
    assert service.snapshot(code,ADMIN)['rules']['lineup_policy'] == 'balanced'


def test_request_contract_rejects_privilege_and_action_payload_injection():
    with pytest.raises(ValidationError):
        FixtureActionRequest(action='schedule',revision=0,command_id='schema-0001',starts_at='future',owner_user_id=1)
    for payload in (dict(action='confirm',starts_at='future'),dict(action='bind'),dict(action='schedule')):
        with pytest.raises(ValidationError):
            FixtureActionRequest(revision=0,command_id='schema-0002',**payload)
    with pytest.raises(ValidationError):
        CreateFixtureStageRequest(name='小组赛',groups=[],command_id='schema-0003')


def test_http_routes_auth_private_view_and_booking_response(service):
    from dataclasses import replace
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient
    from competition.backend.config import load_settings
    from competition.backend.hub import RoomHub
    from competition.backend.routes import router

    app = FastAPI()
    app.state.competition_service = service
    app.state.competition_settings = replace(load_settings(), allow_dev_auth=True)
    app.state.competition_hub = RoomHub()
    app.include_router(router)
    @app.exception_handler(CompetitionError)
    async def error_handler(request, exc):
        return JSONResponse(status_code=exc.status_code,content={'detail':exc.detail})
    create(service)
    item, teams = first(service)
    captain_headers = {'X-Competition-Dev-User': f"{teams[0]['captain_user_id']}:Captain:user"}
    with TestClient(app) as client:
        public = client.get(f'/api/events/{SLUG}/fixtures')
        assert public.status_code == 200 and 'no-store' in public.headers['cache-control']
        assert not public.json()['schedule']['can_manage']
        payload = dict(action='schedule',revision=0,command_id='http-booking-001',starts_at=(datetime.now(timezone.utc)+timedelta(hours=3)).isoformat())
        url = f"/api/events/{SLUG}/fixtures/{item['id']}/actions"
        assert client.post(url,json=payload).status_code == 401
        response = client.post(url,json=payload,headers=captain_headers)
        assert response.status_code == 200
        assert '_changed_room_code' not in response.json()
        scheduled = next(f for f in response.json()['schedule']['stages'][0]['fixtures'] if f['id']==item['id'])
        assert scheduled['room_code']
        anonymous = client.get(f'/api/events/{SLUG}/fixtures').json()['schedule']
        assert all(f['room_code'] is None for s in anonymous['stages'] for f in s['fixtures'])
        invalid = client.post(url,json={**payload,'rules':{'preset':'bo7'}},headers=captain_headers)
        assert invalid.status_code == 422
