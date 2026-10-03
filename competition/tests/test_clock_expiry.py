import json
from datetime import datetime, timedelta, timezone

import pytest

from competition.backend.errors import CompetitionError
from competition.backend.room_rules import normalize_rules
from competition.tests.test_client_runtime import setup_game, upload, packet
from competition.tests.test_match_service import ready_and_start


def expire(service, side='yellow'):
    with service.database.transaction(immediate=True) as db:
        db.execute("UPDATE competition_team_clocks SET remaining_ms_base=1,running_since=? WHERE side=?",
                   ((datetime.now(timezone.utc)-timedelta(seconds=6)).isoformat(),side))


@pytest.mark.parametrize('preset,seconds',[('bo3',1800),('bo5',3600),('bo7',4800)])
def test_clock_defaults_and_explicit_override(preset,seconds):
    assert normalize_rules({'preset':preset})['team_clock_seconds']==seconds
    assert normalize_rules({'preset':preset,'team_clock_seconds':120})['team_clock_seconds']==120


@pytest.mark.parametrize('project,metric',[(None,500),('tournament-cargo-transport-4x4',5)])
def test_higher_timeout_keeps_metric_and_opponent_continues(tmp_path,project,metric):
    service,players=setup_game(tmp_path,project)
    upload(service,players[0],value=metric)
    expire(service)
    assert service.settle_deadline('MATCH5')
    state=service.snapshot('MATCH5',players[0])
    assert state['status']=='GAME_A_PLAYING'
    assert state['match']['my_session']['outcome']=='time_limit'
    result=upload(service,players[3],value=metric-1,finished=True)['competition']
    assert result['match']['current_result']['winner_side']=='yellow'
    assert result['match']['current_result']['yellow_score']==metric
    assert result['match']['clocks']['yellow']['remaining_ms']==0
    with service.database.transaction(immediate=True) as db:
        room=service._room_row(db,'MATCH5')
        service._advance_after_result(db,room,game_key='A',now=datetime.now(timezone.utc),actor_user_id=None)
    later=ready_and_start(service,players,'B')
    assert later['match']['sessions']['yellow']['finished']
    assert not later['match']['sessions']['white']['finished']
    with service.database.transaction() as db:
        row=db.execute("SELECT * FROM competition_game_sessions WHERE game_key='B' AND side='yellow'").fetchone()
        assert json.loads(row['adapter_state_json'])['result_value']==0


def test_race_timeout_is_unfinished_and_never_revives_clock(tmp_path):
    service,players=setup_game(tmp_path,'tournament-grand-full-undo-race-3x3')
    upload(service,players[0],value=9999)
    expire(service)
    # The opponent's upload also settles the expired player, without dropping its own packet.
    data=packet(service,players[3],finished=True,value=10,elapsed=2000,outcome='target_reached')
    result=service.sync_client_game('MATCH5',players[3],**data)['competition']
    assert result['match']['current_result']['winner_side']=='white'
    assert result['match']['clocks']['yellow']['remaining_ms']==0


@pytest.mark.parametrize('race',[False,True])
def test_both_timeout_use_score_only_for_higher(tmp_path,race):
    service,players=setup_game(tmp_path,'tournament-grand-full-undo-race-3x3' if race else None)
    upload(service,players[0],value=500)
    upload(service,players[3],value=100)
    expire(service);expire(service,'white')
    assert service.settle_deadline('MATCH5')
    state=service.snapshot('MATCH5',players[0])
    assert state['match']['current_result']['winner_side']==('draw' if race else 'yellow')
    assert state['match']['clocks']['yellow']['remaining_ms']==0
    assert state['match']['clocks']['white']['remaining_ms']==0


def test_final_boundary_checkpoint_accepted_but_beyond_budget_rejected(tmp_path):
    service,players=setup_game(tmp_path)
    with pytest.raises(CompetitionError) as error:
        upload(service,players[0],finished=True,outcome='time_limit',elapsed=3600001)
    assert error.value.code=='TEAM_CLOCK_EXPIRED'
    result=upload(service,players[0],finished=True,outcome='time_limit',value=888,elapsed=3600000)['competition']
    assert result['match']['my_session']['score']==888
    assert result['match']['clocks']['yellow']['remaining_ms']==0
    assert upload(service,players[3],finished=True,value=800)['competition']['match']['current_result']['winner_side']=='yellow'
