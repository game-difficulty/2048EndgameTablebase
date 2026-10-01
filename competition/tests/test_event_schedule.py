from datetime import datetime, timedelta, timezone

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService

ADMIN = Principal(900, 'Admin', 'admin')
SLUG = '819984-cup-3'


@pytest.fixture
def scheduled(tmp_path):
    service = CompetitionService(CompetitionDatabase(tmp_path / 'schedule.sqlite'))
    service.initialize()
    enrollment = service.events.enrollment
    enrollment.account_reader = lambda ids: {uid: f'Player {uid}' for uid in ids}
    state = enrollment.action(SLUG, ADMIN, action='settings', revision=0, mode='organizer_team', capacity=0, registration_open=False)
    enrollment.import_roster(SLUG, ADMIN, revision=state['revision'], dry_run=False, entries=[
        {'user_id': i, 'team_name': 'AAA' if i <= 3 else 'BBB', 'captain': i in (1,4), 'position':(i-1)%3+1, 'is_external':False} for i in range(1,7)])
    state = enrollment.snapshot(SLUG)
    state = enrollment.action(SLUG, ADMIN, action='lock_roster', revision=state['revision'])
    start = datetime.now(timezone.utc) + timedelta(hours=1)
    kwargs = dict(event_slug=SLUG, starts_at=start.isoformat(), yellow_team_id=state['teams'][0]['id'], white_team_id=state['teams'][1]['id'])
    room = service.create_competition(ADMIN, name='第一轮', room_code='SCHED2', **kwargs)
    return service, room, start, kwargs


def arrive(service, ids):
    for uid in ids:
        service.schedule.check_in('SCHED2', Principal(uid, f'Player {uid}'))
        service.claim_seat('SCHED2', Principal(uid,f'Player {uid}'),side='yellow' if uid<=3 else 'white',position=(uid-1)%3+1,command_id=f'arrive-seat-{uid}')


def test_fixed_roster_directory_and_future_start(scheduled):
    service, room, start, _ = scheduled
    assert room['status'] == 'SEATING'
    assert len(room['seats']) == 0
    assert service.events.detail(SLUG, Principal(1,'P'))['rooms'][0]['room_code'] == 'SCHED2'
    assert service.events.detail(SLUG)['rooms'][0]['room_code'] is None
    for method, kwargs in [(service.claim_seat, dict(side='yellow',position=2)), (service.leave_seat,{})]:
        with pytest.raises(CompetitionError):
            method('SCHED2', Principal(1,'P'), command_id='test-fixed-roster', **kwargs)
    arrive(service, range(1,7))
    for uid in (1,4):
        service.set_ready('SCHED2', Principal(uid,'P'), ready=True, command_id=f'ready-user-{uid}')
    assert service.snapshot('SCHED2', ADMIN)['status'] == 'READY_CHECK'
    assert not service.settle_deadline('SCHED2', now=start-timedelta(seconds=1))
    assert 'SCHED2' in service.process_due_rooms(now=start)
    assert service.snapshot('SCHED2', ADMIN)['status'] == 'DRAW'


def test_one_team_late_is_three_zero_and_idempotent(scheduled):
    service, room, start, _ = scheduled
    arrive(service, (1,2,3,4,5))
    service.set_ready('SCHED2',Principal(1,'P'),ready=True,command_id='yellow-ready-late')
    assert not service.settle_deadline('SCHED2', now=start+timedelta(minutes=5))
    assert not service.settle_deadline('SCHED2', now=start+timedelta(minutes=10))
    assert not service.settle_deadline('SCHED2', now=start+timedelta(minutes=15))
    assert service.settle_deadline('SCHED2', now=start+timedelta(minutes=15,seconds=1))
    result = service.snapshot('SCHED2', ADMIN)
    assert result['status'] == 'FINISHED'
    assert result['match']['series_score'] == {'yellow':3,'white':0,'draws':0}
    assert result['match']['finish_reason'] == 'late_forfeit'
    assert len(result['match']['results']) == 3
    with service.database.transaction() as db:
        key = service._room_row(db, 'SCHED2')['public_key']
    assert service.list_live_rooms() == []
    with pytest.raises(CompetitionError) as captured:
        service.live_projection(key)
    assert captured.value.code == 'LIVE_ROOM_NOT_FOUND'
    assert not service.settle_deadline('SCHED2', now=start+timedelta(minutes=20))
    service.schedule.check_in('SCHED2', Principal(6,'Late'))
    assert service.events.detail(SLUG)['rooms'][0]['series_score'] == {'yellow':3,'white':0}


def test_both_late_finishes_zero_zero(scheduled):
    service, room, start, _ = scheduled
    arrive(service, (1,4))
    service.settle_deadline('SCHED2', now=start+timedelta(minutes=16))
    result = service.snapshot('SCHED2', ADMIN)
    assert result['schedule']['exception'] == 'both_late'
    assert result['status'] == 'FINISHED'
    assert result['match']['series_points']=={'yellow':0,'white':0}
    assert result['match']['results']==[]
    with service.database.transaction(immediate=True) as db:
        room = service._room_row(db, 'SCHED2')
        assert room['live_started_at'] is None
        # Previously written erroneous start timestamps must also stay hidden.
        db.execute('UPDATE competitions SET live_started_at=live_ended_at WHERE id=?', (room['id'],))
        key = room['public_key']
    assert service.list_live_rooms(now=start + timedelta(minutes=16)) == []
    with pytest.raises(CompetitionError) as captured:
        service.live_projection(key)
    assert captured.value.code == 'LIVE_ROOM_NOT_FOUND'


def test_roster_unlock_does_not_change_scheduled_players(scheduled):
    service, room, _, kwargs = scheduled
    enrollment = service.events.enrollment
    state = enrollment.snapshot(SLUG)
    enrollment.action(SLUG, ADMIN, action='unlock', revision=state['revision'], reason='调整下一轮名单')
    with pytest.raises(CompetitionError):
        service.create_competition(ADMIN, name='未锁定', room_code='SCHED3', **kwargs)
    assert len(service.snapshot('SCHED2', ADMIN)['schedule']['players']) == 6
    with service.database.transaction() as db:
        assert not db.execute("SELECT 1 FROM competitions WHERE room_code='SCHED3'").fetchone()


def test_same_team_rejected_and_checkin_not_ready(scheduled):
    service, room, start, kwargs = scheduled
    kwargs['white_team_id'] = kwargs['yellow_team_id']
    with pytest.raises(CompetitionError):
        service.create_competition(ADMIN, name='同队无效', **kwargs)
    arrive(service, (1,4))
    for uid in (1,4):
        with pytest.raises(CompetitionError):
            service.set_ready('SCHED2', Principal(uid,'P'), ready=True, command_id=f'ready-user-{uid}')
    assert not service.settle_deadline('SCHED2', now=start)


def test_expulsion_blocks_checkin_and_unban_restores_assigned_seat(scheduled):
    service, room, start, _ = scheduled
    arrive(service, range(1,7))
    service.manage_member('SCHED2', ADMIN, user_id=2, remove=True, command_id='kick-user-2')
    with pytest.raises(CompetitionError):
        service.schedule.check_in('SCHED2', Principal(2,'P'))
    service.manage_member('SCHED2', ADMIN, user_id=2, remove=False, command_id='unban-user-2')
    result = service.schedule.check_in('SCHED2', Principal(2,'P'))
    assert len(result['seats']) == 5
    result=service.claim_seat('SCHED2',Principal(2,'P'),side='yellow',position=2,command_id='return-user-2')
    assert len(result['seats']) == 6
    assert result['status'] == 'READY_CHECK'
