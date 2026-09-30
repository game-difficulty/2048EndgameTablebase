from datetime import datetime, timedelta, timezone

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService, parse_time
from competition.tests.test_match_service import player, prepare_game_a, ready_and_start, force_one_move_completion
from competition.tests.test_draft_service import prepare_draw


@pytest.fixture
def service(tmp_path):
    value = CompetitionService(CompetitionDatabase(tmp_path / 'feedback.sqlite3'),
                               room_creator_ids=frozenset({1}), test_project_target_tile=4)
    value.initialize()
    return value


def test_owner_and_admin_only_and_readmission(service):
    owner, member = player(1), player(10)
    code = 'MGMT23'
    service.create_competition(owner, name='Permission check', room_code=code)
    service.claim_seat(code, member, side='yellow', position=1, command_id='seat-member-10')
    with pytest.raises(CompetitionError, match='Only a tournament'):
        service.manage_member(code, member, user_id=11, remove=True, command_id='deny-captain-11')
    with pytest.raises(CompetitionError):
        service.manage_member(code, player(2), user_id=10, remove=True, command_id='deny-other-owner')
    service.manage_member(code, owner, user_id=10, remove=True, command_id='owner-remove-10')
    with pytest.raises(CompetitionError) as error:
        service.snapshot(code, member)
    assert error.value.code == 'REMOVED_FROM_ROOM'
    service.manage_member(code, player(99, role='admin'), user_id=10, remove=False, command_id='admin-readmit-10')
    assert service.snapshot(code, member)['me']['seat'] is None


def test_removal_pauses_match_without_losing_score(service):
    participants = prepare_game_a(service)
    ready_and_start(service, participants, 'A')
    room = service.manage_member('MATCH5', player(99, role='admin'), user_id=participants[1].user_id,
                                 remove=True, command_id='remove-active-member')
    assert room['match']['suspension']['active']
    assert all(clock['state'] != 'running' for clock in room['match']['clocks'].values())
    with pytest.raises(CompetitionError) as error:
        service.snapshot('MATCH5', participants[1])
    assert error.value.code == 'REMOVED_FROM_ROOM'
    service.manage_member('MATCH5', player(1), user_id=participants[1].user_id,
                          remove=False, command_id='owner-restore-member')
    assert service.snapshot('MATCH5', participants[1])['match']['suspension']['active']


def test_surrender_requires_finished_opponent_preserves_score_and_rest(service):
    participants = prepare_game_a(service)
    ready_and_start(service, participants, 'A')
    snapshot = service.snapshot('MATCH5', participants[0])
    runtime = snapshot['match']['my_session']['runtime']
    packet = dict(instance_id=runtime['instance_id'], sequence=1,
                  phase_token=snapshot['match']['phase_token'],
                  payload={'board': [[2, 4, 0, 0]] + [[0] * 4 for _ in range(3)], 'score': 999, 'move_count': 8},
                  checkpoint={'version': 1, 'state': {}}, result_value=999, elapsed_ms=1234,
                  finished=True, outcome='surrendered')
    with pytest.raises(CompetitionError) as error:
        service.sync_client_game('MATCH5', participants[0], **packet)
    assert error.value.code == 'SURRENDER_NOT_ALLOWED'
    force_one_move_completion(service, participants[3], 'A', 'white')
    result = service.sync_client_game('MATCH5', participants[0], **packet)['competition']
    assert result['match']['current_result']['yellow_score'] == 999
    assert result['match']['current_result']['white_score'] == 4
    assert result['match']['current_result']['winner_side'] == 'white'
    assert result['match']['current_result']['reason'] == 'yellow_surrendered'
    assert service.sync_client_game('MATCH5', participants[0], **packet)['duplicate']
    for captain in (participants[0], participants[3]):
        service.confirm_current_result('MATCH5', captain, result_revision=result['match']['current_result']['result_revision'],
                                       phase_token=result['match']['phase_token'], command_id=f'confirm-rest-{captain.user_id}')
    deadline = parse_time(result['match']['rest_until'])
    assert not service.settle_deadline('MATCH5', now=deadline - timedelta(milliseconds=1))
    assert service.snapshot('MATCH5', participants[0])['status'] == 'GAME_A_RESULT'
    assert service.settle_deadline('MATCH5', now=deadline + timedelta(milliseconds=1))
    assert service.snapshot('MATCH5', participants[0])['status'] == 'GAME_B_READY'


def test_ready_preview_never_spends_team_clock(service):
    participants = prepare_game_a(service)
    with service.database.transaction(immediate=True) as db:
        db.execute('UPDATE competition_prediction_windows SET minimum_until=?',
                   ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),))
    ready = ready_and_start(service, participants, 'A', expire_prediction_window=False)
    assert ready['status'] == 'GAME_A_READY'
    deadline = parse_time(ready['match']['preview_until'])
    assert all(clock['state'] == 'stopped' for clock in ready['match']['clocks'].values())
    assert not service.settle_deadline('MATCH5', now=deadline - timedelta(milliseconds=1))
    assert service.settle_deadline('MATCH5', now=deadline + timedelta(milliseconds=1))
    assert service.snapshot('MATCH5', participants[0])['status'] == 'GAME_A_PLAYING'


def test_draw_candidate_and_result_each_have_ten_seconds(service):
    participants = prepare_draw(service)
    room = service.snapshot('DRAFT2', participants[0])
    deadline = parse_time(room['draft']['deadline_at'])
    assert 9 <= (deadline - parse_time(room['server_time'])).total_seconds() <= 10
    for _ in range(4):
        room = service.snapshot('DRAFT2', participants[0])
        until = max(parse_time(value) for value in room['draft']['deadlines'].values() if value)
        service.settle_deadline('DRAFT2', now=until + timedelta(milliseconds=1))
    candidates = service.snapshot('DRAFT2', participants[0])
    reveal = parse_time(candidates['draft']['c_reveal_at'])
    finish = parse_time(candidates['draft']['deadline_at'])
    assert (finish - reveal).total_seconds() == 10
    assert candidates['draft']['project_c'] is None
    assert not service.settle_deadline('DRAFT2', now=reveal - timedelta(milliseconds=1))
    assert service.settle_deadline('DRAFT2', now=reveal)
    result = service.snapshot('DRAFT2', participants[0])
    assert result['status'] == 'C_DRAW' and result['draft']['project_c']
    assert not service.settle_deadline('DRAFT2', now=finish - timedelta(milliseconds=1))
    assert service.settle_deadline('DRAFT2', now=finish)
    assert service.snapshot('DRAFT2', participants[0])['status'] == 'LINEUP'


def test_removal_during_draft_blocks_manual_and_automatic_progress(service):
    participants = prepare_draw(service)
    room = service.snapshot('DRAFT2', participants[0])
    service.settle_deadline('DRAFT2', now=parse_time(room['draft']['deadline_at']) + timedelta(milliseconds=1))
    room = service.manage_member('DRAFT2', player(1), user_id=participants[1].user_id,
                                 remove=True, command_id='remove-during-draft')
    captain = participants[0] if room['draft']['first_side'] == 'yellow' else participants[3]
    view = service.snapshot('DRAFT2', captain)
    keys = [p['key'] for p in view['projects']]
    with pytest.raises(CompetitionError) as error:
        service.submit_pick_ban('DRAFT2', captain, pick_project_key=keys[0], ban_project_key=keys[1],
                                phase_token=view['draft']['phase_token'], command_id='blocked-draft-submit')
    assert error.value.code == 'MEMBER_REMOVAL_HOLD'
    assert not service.settle_deadline('DRAFT2', now=datetime.now(timezone.utc) + timedelta(days=1))
