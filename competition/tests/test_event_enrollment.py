import json

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService
from backend.human_play.rating import top_rating

ADMIN = Principal(900, 'Admin', 'admin')
TEAM = '819984-cup-3'
STATS = '14360-cup-1'


@pytest.fixture
def enrollment(tmp_path):
    service = CompetitionService(CompetitionDatabase(tmp_path / 'events.sqlite'))
    service.initialize()
    service.events.enrollment.account_reader = lambda ids: {i: f'Player {i}' for i in ids if i < 100}
    service.events.statistics.account_reader = service.events.enrollment.account_reader
    service.events.statistics.result_reader = lambda ids, *args: {i: {'completed_games': 0, 'games': []} for i in ids}
    return service.events.enrollment


def act(enrollment, action, uid=900, slug=TEAM, **values):
    principal = ADMIN if uid == 900 else Principal(uid, f'Player {uid}')
    revision = enrollment.snapshot(slug)['revision']
    return enrollment.action(slug, principal, action=action, revision=revision, **values)


def open_registration(enrollment, slug=TEAM, mode='self_team', capacity=0):
    return act(enrollment, 'settings', slug=slug, mode=mode, capacity=capacity, registration_open=True)


def test_consent_privacy_one_team_and_captain_permissions(enrollment):
    open_registration(enrollment)
    team = act(enrollment, 'create_team', uid=1, name='First')['teams'][0]['id']
    other = act(enrollment, 'create_team', uid=4, name='Other')['teams'][1]['id']
    act(enrollment, 'invite', uid=1, team_id=team, user_id=2)
    assert enrollment.snapshot(TEAM)['invitations'] == []
    assert enrollment.snapshot(TEAM, Principal(2, 'Player 2'))['invitations'][0]['team_id'] == team
    assert not any(e['user_id'] == 2 for e in enrollment.snapshot(TEAM)['entries'])
    with pytest.raises(CompetitionError):
        act(enrollment, 'accept_invite', uid=3, team_id=team)
    act(enrollment, 'accept_invite', uid=2, team_id=team)
    with pytest.raises(CompetitionError):
        act(enrollment, 'invite', uid=4, team_id=other, user_id=2)
    with pytest.raises(CompetitionError):
        act(enrollment, 'submit_team', uid=2, team_id=team)
    act(enrollment, 'invite', uid=1, team_id=team, user_id=3)
    act(enrollment, 'accept_invite', uid=3, team_id=team)
    act(enrollment, 'submit_team', uid=1, team_id=team)
    act(enrollment, 'leave_team', uid=2, team_id=team)
    assert not next(t for t in enrollment.snapshot(TEAM)['teams'] if t['id'] == team)['submitted']
    act(enrollment, 'disband_team', uid=1, team_id=team)
    assert len(enrollment.snapshot(TEAM)['entries']) == 4
    assert all(not e['team_id'] for e in enrollment.snapshot(TEAM)['entries'] if e['user_id'] < 4)


def test_solo_capacity_stale_revision_and_no_management_for_players(enrollment):
    open_registration(enrollment, mode='solo', capacity=1)
    revision = enrollment.snapshot(TEAM)['revision']
    act(enrollment, 'signup', uid=1)
    with pytest.raises(CompetitionError):
        enrollment.action(TEAM, Principal(2, 'Player 2'), action='signup', revision=revision)
    with pytest.raises(CompetitionError):
        act(enrollment, 'signup', uid=2)
    with pytest.raises(CompetitionError):
        act(enrollment, 'settings', uid=1, mode='solo', capacity=2, registration_open=True)
    with pytest.raises(CompetitionError):
        act(enrollment, 'create_team', uid=1, name='Invalid')
    act(enrollment, 'withdraw', uid=1)
    act(enrollment, 'signup', uid=2)
    assert len(enrollment.snapshot(TEAM)['entries']) == 1
    act(enrollment, 'lock_roster')
    with pytest.raises(CompetitionError):
        act(enrollment, 'withdraw', uid=2)


def test_pending_groups_two_locks_and_legacy_import_cannot_bypass(enrollment):
    entries = [{'user_id':i,'team_name':'','is_external':i==20} for i in range(1,21)]
    preview = enrollment.import_roster(STATS, ADMIN, entries=entries, revision=0)
    assert not preview['saved'] and enrollment.snapshot(STATS)['entries'] == []
    enrollment.import_roster(STATS, ADMIN, entries=entries, revision=0, dry_run=False)
    assert enrollment.catalog.statistics.standings(STATS)['unassigned_count'] == 20
    assert enrollment.catalog.statistics.standings(STATS)['teams'] == []
    with pytest.raises(CompetitionError):
        act(enrollment, 'lock_roster', slug=STATS)
    act(enrollment, 'lock_registration', slug=STATS)
    revision = enrollment.snapshot(STATS)['revision']
    with pytest.raises(CompetitionError):
        enrollment.import_roster(STATS, ADMIN, entries=entries[:-1], revision=revision, dry_run=False)
    grouped = [e | {'team_name': f'Team {(e["user_id"]-1)//5+1}'} for e in entries]
    enrollment.import_roster(STATS, ADMIN, entries=grouped, revision=revision, dry_run=False)
    locked = act(enrollment, 'lock_roster', slug=STATS)
    with pytest.raises(CompetitionError):
        enrollment.catalog.statistics.import_roster(STATS, ADMIN, entries=grouped, revision=locked['revision'], dry_run=False)
    frozen = json.loads(locked['audit'][0]['payload_json'])
    assert len(frozen['entries']) == 20
    with pytest.raises(CompetitionError):
        act(enrollment, 'unlock', slug=STATS, reason='no')
    unlocked = act(enrollment, 'unlock', slug=STATS, reason='修正举办方分组错误')
    assert not unlocked['registration_open'] and not unlocked['roster_locked']
    assert enrollment.snapshot(STATS)['audit'] == []


def test_captain_can_confirm_after_registration_closes(enrollment):
    open_registration(enrollment)
    tid = act(enrollment, 'create_team', uid=1, name='A')['teams'][0]['id']
    for uid in (2,3):
        act(enrollment, 'invite', uid=1, team_id=tid, user_id=uid)
        act(enrollment, 'accept_invite', uid=uid, team_id=tid)
    act(enrollment, 'lock_registration')
    act(enrollment, 'submit_team', uid=1, team_id=tid)
    with enrollment.database.transaction(immediate=True) as db:
        db.executemany('INSERT INTO tournament_roster_positions VALUES(?,?,?)',[(TEAM,i,i) for i in (1,2,3)])
    act(enrollment, 'lock_roster')
    with pytest.raises(CompetitionError):
        act(enrollment, 'unsubmit_team', uid=1, team_id=tid)


def test_padding_zero_games_and_team_sum(enrollment):
    module = enrollment.catalog.statistics
    module.result_reader = lambda ids, *args: {i: {'completed_games':1,'games':[{'id':'one','score':2000,'board_sum':1022}]} for i in ids if i == 1}
    entries = [{'user_id':i,'team_name':f'Team {(i-1)//5+1}','is_external':i==20} for i in range(1,21)]
    enrollment.import_roster(STATS, ADMIN, entries=entries, revision=0, dry_run=False)
    result = module.standings(STATS)
    one = next(p for p in result['players'] if p['user_id'] == 1)
    assert one['average_board_sum'] == 1022/5
    assert one['rating'] == top_rating('3x3', [1022,0,0,0,0])
    assert next(p for p in result['players'] if p['user_id'] == 2)['rating'] == 0
    assert next(t for t in result['teams'] if t['name']=='Team 1')['rating'] == one['rating']


def test_existing_statistics_roster_migrates_once(enrollment):
    with enrollment.database.transaction(immediate=True) as db:
        db.execute('INSERT INTO tournament_statistics_roster VALUES(?,?,?,?,?)', (STATS,1,'Original','Team A',0))
    enrollment.database.initialize()
    assert enrollment.snapshot(STATS)['entries'][0]['display_name'] == 'Original'
    enrollment.database.initialize()
    assert len(enrollment.snapshot(STATS)['entries']) == 1
