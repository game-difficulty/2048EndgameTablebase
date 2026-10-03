from datetime import datetime, timedelta, timezone
import pytest
from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService

HOST,GUEST,VIEWER=Principal(91,'Host'),Principal(92,'Guest'),Principal(93,'Viewer')


@pytest.fixture
def service(tmp_path):
    s=CompetitionService(CompetitionDatabase(tmp_path/'public.sqlite'))
    s.initialize()
    return s


def prepared(s,kind):
    if kind=='duel':
        room=s.duels.create(HOST,name='Pool duel',projects=[p['project_ref'] for p in s.duels.catalog()][:2],
                            command_id='pool-create-duel',predictions_enabled=True)
    else:
        room=s.time_attacks.create(HOST,name='Pool time',variant='3x3',target_kind='tile',target_value=8,
                                   clock_seconds=60,command_id='pool-create-time',predictions_enabled=True)
    code=room['room_code']
    assert s.list_live_rooms()==[]
    s.claim_seat(code,GUEST,side='white',position=1,command_id='pool-guest-join')
    s.set_ready(code,HOST,ready=True,command_id='pool-host-ready')
    assert s.list_live_rooms()==[]
    room=s.set_ready(code,GUEST,ready=True,command_id='pool-guest-ready')
    return room


@pytest.mark.parametrize('kind',['duel','time_attack'])
def test_public_window_locks_roster_and_starts_only_after_sixty_seconds(service,kind):
    s=service;room=prepared(s,kind);code=room['room_code']
    assert room['status']=='READY_CHECK' and room['prediction_window']['open']
    assert not any(room['me'][key] for key in ['can_ready','can_leave_seat','can_claim_seat'])
    until=datetime.fromisoformat(room['prediction_window']['minimum_until'])
    opened=datetime.fromisoformat(room['prediction_window']['opened_at'])
    assert (until-opened).total_seconds()==60
    assert s.list_live_rooms()[0]['public_key']==room['live_public_key']
    facts=s.live_prediction_facts(room['live_public_key'])
    assert facts['room_kind']==kind and set(facts['participant_user_ids'])=={91,92}
    projection=s.live_projection(room['live_public_key'])
    assert 'participant_user_ids' not in projection
    with pytest.raises(CompetitionError):s.set_ready(code,GUEST,ready=False,command_id='pool-unready-attempt')
    with pytest.raises(CompetitionError):s.leave_seat(code,GUEST,command_id='pool-leave-attempt')
    with pytest.raises(CompetitionError):s.claim_seat(code,VIEWER,side='white',position=1,command_id='pool-swap-attempt')
    s.process_due_rooms(now=until-timedelta(milliseconds=1))
    assert s.snapshot(code,HOST)['status']=='READY_CHECK'
    # Reconstructing the service cannot reset/extend the persisted window.
    s=CompetitionService(s.database);s.initialize()
    s.process_due_rooms(now=until+timedelta(milliseconds=1))
    fresh=s.snapshot(code,VIEWER)
    assert fresh['status']=='GAME_A_PLAYING' and not fresh['prediction_window']['open']
    projection=s.live_projection(room['live_public_key'])
    if kind=='duel':
        assert all(g['project_key'] for g in projection['games'])
        assert len(projection['project_public_views'])==2
    else:
        assert datetime.fromisoformat(fresh['time_attack']['started_at'])>=until
        for player in projection['time_attack']['players'].values():
            assert 'seed' not in player['attempt'] and 'state' not in player['attempt']
        end=datetime.fromisoformat(fresh['time_attack']['deadline_at'])
        s.process_due_rooms(now=end)
        facts=s.live_prediction_facts(room['live_public_key'])
        assert facts['phase']=='FINISHED' and facts['public_result']['winner_side']=='draw'


@pytest.mark.parametrize('kind',['duel','time_attack'])
def test_public_window_cancel_keeps_refund_facts(service,kind):
    room=prepared(service,kind)
    service.close_competition(room['room_code'],HOST,command_id='close-public-window')
    facts=service.live_prediction_facts(room['live_public_key'])
    assert facts['phase']=='CANCELLED' and not facts['prediction_window']['open']


def test_late_source_read_closes_public_entries_even_before_scheduler(service):
    room=prepared(service,'time_attack')
    with service.database.transaction(immediate=True) as db:
        db.execute('UPDATE competition_prediction_windows SET minimum_until=?',
                   ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),))
    facts=service.live_prediction_facts(room['live_public_key'])
    assert facts['phase']=='READY_CHECK' and not facts['prediction_window']['open']
