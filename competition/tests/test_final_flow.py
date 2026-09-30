from datetime import datetime, timedelta, timezone
import pytest

from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.tests.test_client_runtime import setup_game, packet
from competition.tests.test_match_service import prepare_game_a
from competition.backend.db import CompetitionDatabase
from competition.backend.service import CompetitionService


def set_yellow_budget(service, elapsed_seconds):
    import json
    with service.database.transaction(immediate=True) as db:
        row=db.execute("SELECT * FROM competition_game_sessions WHERE side='yellow'").fetchone()
        extra=json.loads(row['adapter_state_json']);extra['project_clock_start_ms']=60000
        db.execute('UPDATE competition_game_sessions SET adapter_state_json=? WHERE instance_id=?',(json.dumps(extra),row['instance_id']))
        db.execute("UPDATE competition_team_clocks SET remaining_ms_base=60000,running_since=? WHERE side='yellow'",((datetime.now(timezone.utc)-timedelta(seconds=elapsed_seconds)).isoformat(),))


def test_higher_refund_is_only_credited_after_normal_completion(tmp_path):
    service,players=setup_game(tmp_path,'tournament-cargo-transport-4x4')
    service.sync_client_game('MATCH5',players[3],**packet(service,players[3],finished=True,value=2,elapsed=1000))
    set_yellow_budget(service,0)
    progress=packet(service,players[0],value=3,elapsed=2000)
    progress['checkpoint']['metric_history']=[[0,0],[1500,3]]
    service.sync_client_game('MATCH5',players[0],**progress)
    with service.database.transaction() as db:
        assert db.execute("SELECT remaining_ms_base FROM competition_team_clocks WHERE side='yellow'").fetchone()[0]==60000
        assert db.execute("SELECT COUNT(*) FROM competition_time_refunds").fetchone()[0]==0
    final=packet(service,players[0],sequence=2,finished=True,value=3,elapsed=50000)
    final['checkpoint']['metric_history']=[[0,0],[1500,3]]
    set_yellow_budget(service,50)
    result=service.sync_client_game('MATCH5',players[0],**final)['competition']
    assert result['match']['current_result']['yellow_refund_ms']==48500
    assert result['match']['clocks']['yellow']['remaining_ms']==58500
    assert result['match']['current_result']['winner_side']=='yellow'
    assert result['match']['series_points']=={'yellow':2,'white':0}
    duplicate=service.sync_client_game('MATCH5',players[0],**final)['competition']
    assert duplicate['match']['clocks']['yellow']['remaining_ms']==58500


def test_higher_lead_cannot_rescue_exhausted_team_budget(tmp_path):
    service,players=setup_game(tmp_path,'tournament-cargo-transport-4x4')
    service.sync_client_game('MATCH5',players[3],**packet(service,players[3],finished=True,value=2,elapsed=1000))
    final=packet(service,players[0],finished=True,value=3,elapsed=70000)
    final['checkpoint']['metric_history']=[[0,0],[1500,3]]
    set_yellow_budget(service,70)
    response=service.sync_client_game('MATCH5',players[0],**final)
    result=response['competition']
    assert response['stopped']
    assert result['status']=='FINISHED'
    assert result['match']['finish_reason']=='yellow_clock_expired'
    assert result['match']['series_score']['white']==3
    assert result['match']['clocks']['yellow']['remaining_ms']==0
    with service.database.transaction() as db:
        assert db.execute("SELECT COUNT(*) FROM competition_time_refunds").fetchone()[0]==0


@pytest.mark.parametrize('elapsed', [60000, 60001])
def test_completion_at_or_after_zero_is_not_accepted_for_refund(tmp_path, elapsed):
    service,players=setup_game(tmp_path,'tournament-cargo-transport-4x4')
    service.sync_client_game('MATCH5',players[3],**packet(service,players[3],finished=True,value=2,elapsed=1000))
    set_yellow_budget(service,0)
    final=packet(service,players[0],finished=True,value=3,elapsed=elapsed)
    final['checkpoint']['metric_history']=[[0,0],[1500,3]]
    with pytest.raises(CompetitionError) as error:
        service.sync_client_game('MATCH5',players[0],**final)
    assert error.value.code=='TEAM_CLOCK_EXPIRED'


def test_ready_timeout_and_rest_advance_without_captains(tmp_path):
    service=CompetitionService(CompetitionDatabase(tmp_path/'ready.sqlite'))
    service.initialize();players=prepare_game_a(service)
    snap=service.snapshot('MATCH5',players[0]);deadline=datetime.fromisoformat(snap['match']['ready_deadline_at'])
    service.settle_deadline('MATCH5',now=deadline+timedelta(seconds=60))
    assert service.snapshot('MATCH5',players[0])['status']=='GAME_A_PLAYING'
    for user in (players[0],players[3]):
        result=service.sync_client_game('MATCH5',user,**packet(service,user,finished=True))['competition']
    service.settle_deadline('MATCH5',now=datetime.fromisoformat(result['match']['rest_until'])+timedelta(seconds=1))
    assert service.snapshot('MATCH5',players[0])['status']=='GAME_B_READY'


def test_rematch_cancels_original_and_does_not_copy_seats(tmp_path):
    service=CompetitionService(CompetitionDatabase(tmp_path/'rematch.sqlite'));service.initialize()
    admin=Principal(1,'Admin','admin')
    original=service.create_competition(admin,name='测试赛事',room_code='RMTCH2')
    replacement=service.rematch_before_lineup('RMTCH2',admin,command_id='rematch-once')
    assert replacement['room_code']!='RMTCH2' and replacement['seats']==[]
    assert service.snapshot('RMTCH2',admin)['status']=='CANCELLED'
    assert service.snapshot('RMTCH2',admin)['replacement_room_code']==replacement['room_code']
    assert service.rematch_before_lineup('RMTCH2',admin,command_id='rematch-once')['room_code']==replacement['room_code']


def test_free_scheduled_room_accepts_arbitrary_registered_players(tmp_path):
    service=CompetitionService(CompetitionDatabase(tmp_path/'free.sqlite'));service.initialize()
    room=service.create_competition(Principal(1,'Admin','admin'),name='自由赛',starts_at=(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat())
    room=service.claim_seat(room['room_code'],Principal(44,'Guest'),side='white',position=2,command_id='free-seat-44')
    assert room['seats'][0]['user_id']==44
    assert room['schedule']['players']==[]


def test_rematch_is_closed_once_lineup_is_over(tmp_path):
    service=CompetitionService(CompetitionDatabase(tmp_path/'closed.sqlite'));service.initialize()
    prepare_game_a(service)
    with pytest.raises(CompetitionError) as error:
        service.rematch_before_lineup('MATCH5',Principal(1,'Admin','admin'),command_id='too-late-rematch')
    assert error.value.code=='REMATCH_WINDOW_CLOSED'


def test_refund_cap_and_already_leading_when_opponent_finishes():
    from types import SimpleNamespace
    from competition.backend.final_flow import refund_decision
    winner=SimpleNamespace(outcome='no_moves',elapsed_ms=400000,score=30,
        extra={'checkpoint':{'metric_history':[[0,0],[1000,20],[3000,30]]}})
    loser=SimpleNamespace(outcome='no_moves',elapsed_ms=2000,score=10,extra={})
    assert refund_decision(winner,loser)==(2000,300000)
    winner.outcome='surrendered'
    assert refund_decision(winner,loser) is None


def test_invalid_metric_history_rejected_without_changing_state(tmp_path):
    service,players=setup_game(tmp_path)
    upload=packet(service,players[0])
    upload['checkpoint']['metric_history']=[[100,1],[50,2]]
    with pytest.raises(CompetitionError) as error:
        service.sync_client_game('MATCH5',players[0],**upload)
    assert error.value.code=='INVALID_CLIENT_STATE'
    assert service.snapshot('MATCH5',players[0])['match']['my_session']['runtime']['sequence']==0


@pytest.mark.parametrize('official', [False, True])
def test_full_match_points_and_record_eligibility(tmp_path, official):
    from competition.tests.test_match_service import ready_and_start
    service, players=setup_game(tmp_path)
    service.result_rest_seconds=0
    if official:
        with service.database.transaction(immediate=True) as db:
            cid=db.execute("SELECT id FROM competitions WHERE room_code='MATCH5'").fetchone()[0]
            db.execute("INSERT INTO competition_schedule(competition_id,starts_at,roster_revision,yellow_team_id,white_team_id,yellow_name,white_name) VALUES(?,?,1,'yellow-team','white-team','黄方','白方')",(cid,datetime.now(timezone.utc).isoformat()))
    for index,game in enumerate('ABC'):
        if index:ready_and_start(service,players,game)
        for user,value in [(players[index],100),(players[index+3],50)]:
            service.sync_client_game('MATCH5',user,**packet(service,user,finished=True,value=value))
        result=service.snapshot('MATCH5',players[0])
    assert result['status']=='FINISHED'
    assert result['match']['series_points']=={'yellow':6,'white':0}
    assert all(r['record_eligible_side']==('yellow' if official else None) for r in result['match']['results'])
    public=service.live_projection(service.list_live_rooms()[0]['public_key'])
    assert public['series_points']=={'yellow':6,'white':0}
