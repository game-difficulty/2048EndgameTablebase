from datetime import datetime, timedelta, timezone
from competition.backend.db import CompetitionDatabase
from competition.backend.service import CompetitionService
from competition.tests.test_match_service import prepare_game_a, ready_and_start


def test_minimum_sixty_seconds_gate_stops_clocks_and_auto_starts(tmp_path):
    service=CompetitionService(CompetitionDatabase(tmp_path/'gate.sqlite3'))
    service.initialize()
    players=prepare_game_a(service)
    waiting=ready_and_start(service,players,'A',expire_prediction_window=False)
    assert waiting['status']=='GAME_A_READY'
    window=waiting['match']['prediction_window']
    opened=datetime.fromisoformat(window['opened_at'])
    deadline=datetime.fromisoformat(window['minimum_until'])
    assert (deadline-opened).total_seconds()==60
    assert all(clock['state']=='stopped' for clock in waiting['match']['clocks'].values())
    assert not service.settle_deadline('MATCH5',now=deadline-timedelta(milliseconds=1))
    assert service.process_due_rooms(now=deadline)==['MATCH5']
    started=service.snapshot('MATCH5',players[0])
    assert started['status']=='GAME_A_PLAYING'
    assert not started['match']['prediction_window']['open']
    assert started['match']['prediction_window']['closed_at']


def test_settlement_facts_survive_lobby_retention(tmp_path):
    service=CompetitionService(CompetitionDatabase(tmp_path/'gate.sqlite3'))
    service.initialize()
    prepare_game_a(service)
    room=service.list_live_rooms()[0]
    with service.database.transaction(immediate=True) as db:
        db.execute("UPDATE competitions SET status='CANCELLED',live_ended_at=?",((datetime.now(timezone.utc)-timedelta(days=2)).isoformat(),))
    assert not service.list_live_rooms()
    facts=service.live_prediction_facts(room['public_key'])
    assert facts['phase']=='CANCELLED'
    assert not facts['prediction_window']['open']
    assert 'project_public_views' not in facts
