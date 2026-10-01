import json
from datetime import datetime, timedelta, timezone

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.service import CompetitionService
from competition.backend.errors import CompetitionError
from competition.backend.projects import tournament_project_catalog
from competition.tests.test_match_service import prepare_game_a, ready_and_start, player


def setup_game(tmp_path, project_ref=None):
    service = CompetitionService(CompetitionDatabase(tmp_path / 'client.sqlite3'), team_clock_seconds=3600)
    service.initialize()
    players = prepare_game_a(service)
    if project_ref:
        item = next(item for item in tournament_project_catalog() if item['project_ref'] == project_ref)
        with service.database.transaction(immediate=True) as db:
            db.execute('UPDATE competition_projects SET project_ref=?, adapter_rules_version=?',
                       (project_ref, item['rules_version']))
    ready_and_start(service, players, 'A')
    return service, players


def packet(service, player, sequence=1, *, finished=False, value=100, elapsed=1000, outcome='no_moves'):
    snapshot = service.snapshot('MATCH5', player)
    runtime = snapshot['match']['my_session']['runtime']
    return dict(instance_id=runtime['instance_id'], sequence=sequence, phase_token=snapshot['match']['phase_token'],
                payload={'board': [[2, 4], [8, 0]], 'score': value, 'move_count': sequence},
                checkpoint={'version': 1, 'state': {'randomState': 1234}}, result_value=value, elapsed_ms=elapsed,
                finished=finished, outcome=outcome if finished else None)


def upload(service, player, **options):
    return service.sync_client_game('MATCH5', player, **packet(service, player, **options))


def test_frame_batches_are_retained_and_recoverable_through_live_cursor(tmp_path):
    service, players = setup_game(tmp_path)
    data = packet(service, players[0], sequence=6)
    frames = [{'sequence': i, 'payload': {**data['payload'], 'move_count': i}} for i in range(1, 7)]
    ack = service.sync_client_game('MATCH5', players[0], **data, frames=frames)
    assert [f['sequence'] for f in ack['update']['public_view']['frames']] == list(range(1, 7))
    key = service.list_live_rooms()[0]['public_key']
    view = service.live_projection(key, after_yellow=3)['project_public_views']['yellow']
    assert [f['sequence'] for f in view['frames']] == [4, 5, 6]
    assert view['frame_start'] == 1
    assert service.sync_client_game('MATCH5', players[0], **data, frames=frames)['duplicate']
    # A legacy checkpoint with a genuine gap advertises a new history floor.
    upload(service, players[0], sequence=10)
    view = service.live_projection(key)['project_public_views']['yellow']
    assert view['frame_start'] == 10
    assert [f['sequence'] for f in view['frames']] == [10]


def test_only_state_upload_gameplay_endpoint_is_registered():
    from competition.backend.routes import router

    paths = {route.path for route in router.routes}
    assert "/api/competitions/{room_code}/games/current/state" in paths
    assert "/api/competitions/{room_code}/games/current/move" not in paths
    assert "/api/competitions/{room_code}/games/current/action" not in paths
    assert not hasattr(CompetitionService, "move_current_game")


@pytest.mark.parametrize('item', tournament_project_catalog(), ids=lambda item: item['project_ref'])
def test_every_formal_project_exposes_seed_and_accepts_states_without_rule_computation(tmp_path, monkeypatch, item):
    # These methods must never be called, even when building observer projections.
    from competition.backend.projects.tournament_variants import Tournament2048Adapter, Tournament2048AdapterV2
    from competition.backend.projects.cargo_transport import CargoTransportAdapter
    from competition.backend.projects.practice_variants import GrowingTilesAdapter, HundredStepSealAdapter
    def forbidden(*args, **kwargs):
        raise AssertionError('server executed game rules')
    for cls in (Tournament2048Adapter, Tournament2048AdapterV2, CargoTransportAdapter, GrowingTilesAdapter, HundredStepSealAdapter):
        for method in ('initial_state', 'apply_move', 'public_payload'):
            monkeypatch.setattr(cls, method, forbidden)
    service, players = setup_game(tmp_path, item['project_ref'])
    yellow = service.snapshot('MATCH5', players[0])['match']['my_session']['runtime']
    white = service.snapshot('MATCH5', players[3])['match']['my_session']['runtime']
    assert yellow['seed'] == white['seed']
    assert yellow['instance_id'] != white['instance_id']
    assert yellow['checkpoint'] is None
    upload(service, players[0], sequence=3)
    upload(service, players[3], sequence=2)
    public = service.live_projection(service.list_live_rooms()[0]['public_key'])
    assert public['project_public_views']['yellow']['sequence'] == 3
    assert public['project_public_views']['yellow']['payload']['score'] == 100
    assert 'checkpoint' not in json.dumps(public)
    assert yellow['seed'] not in json.dumps(public)
    upload(service, players[0], sequence=4, finished=True, value=500)
    result = upload(service, players[3], sequence=3, finished=True, value=200)['competition']
    assert result['status'] == 'GAME_A_RESULT'
    assert result['match']['current_result']['winner_side'] == 'yellow'


def test_old_duplicate_and_unauthorized_uploads_never_replace_latest_state(tmp_path):
    service, players = setup_game(tmp_path)
    old = packet(service, players[0], sequence=2, value=20)
    upload(service, players[0], sequence=5, value=50)
    ack = service.sync_client_game('MATCH5', players[0], **old)
    assert ack['accepted_sequence'] == 5 and ack['duplicate']
    assert service.snapshot('MATCH5', players[0])['match']['my_session']['score'] == 50
    with pytest.raises(CompetitionError) as error:
        service.sync_client_game('MATCH5', players[1], **{**old, 'sequence': 6})
    assert error.value.code == 'ACTIVE_PLAYER_REQUIRED'
    final_packet = packet(service, players[0], sequence=7, finished=True)
    service.sync_client_game('MATCH5', players[0], **final_packet)
    assert service.sync_client_game('MATCH5', players[0], **final_packet)['duplicate']
    assert upload(service, players[0], sequence=8)['stopped']


def test_race_uses_completion_time_not_arrival_order_and_requests_peer_final_state(tmp_path):
    service, players = setup_game(tmp_path, 'tournament-grand-full-undo-race-3x3')
    first = upload(service, players[0], finished=True, elapsed=8000, outcome='target_reached')['competition']
    assert first['status'] == 'GAME_A_PLAYING'
    peer = service.snapshot('MATCH5', players[3])
    assert peer['match']['my_session']['runtime']['race_stop_requested']
    second = upload(service, players[3], finished=True, elapsed=7000, outcome='target_reached')['competition']
    assert second['match']['current_result']['winner_side'] == 'white'
    assert second['match']['sessions']['white']['project_clock']['elapsed_ms'] == 7000
    assert second['match']['clocks']['white']['remaining_ms'] == 3600000 - 7000
    assert second['match']['clocks']['yellow']['remaining_ms'] == 3600000 - 7000


def test_disconnected_race_peer_cannot_block_result_forever(tmp_path):
    service, players = setup_game(tmp_path, 'tournament-grand-full-undo-race-3x3')
    upload(service, players[0], finished=True, outcome='target_reached')
    with service.database.transaction(immediate=True) as db:
        row = db.execute("SELECT * FROM competition_game_sessions WHERE side='white'").fetchone()
        extra = json.loads(row['adapter_state_json'])
        extra['race_stop_at'] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
        db.execute('UPDATE competition_game_sessions SET adapter_state_json=? WHERE instance_id=?',
                   (json.dumps(extra), row['instance_id']))
    assert service.settle_deadline('MATCH5')
    assert service.snapshot('MATCH5', players[0])['status'] == 'GAME_A_RESULT'


def test_final_state_has_transport_grace_and_duplicate_terminal_ack_has_snapshot(tmp_path):
    service, players = setup_game(tmp_path)
    final = packet(service, players[0], finished=True, elapsed=3599500)
    with service.database.transaction(immediate=True) as db:
        db.execute("UPDATE competition_team_clocks SET remaining_ms_base=1000, running_since=? WHERE side='yellow'",
                   ((datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat(),))
    result = service.sync_client_game('MATCH5', players[0], **final)
    assert result['competition']['match']['clocks']['yellow']['remaining_ms'] == 500
    retry = service.sync_client_game('MATCH5', players[0], **final)
    assert retry['duplicate'] and retry['competition']['match']['sessions']['yellow']['finished']


def test_race_final_ack_deadline_does_not_run_during_referee_pause(tmp_path):
    service, players = setup_game(tmp_path, 'tournament-grand-full-undo-race-3x3')
    upload(service, players[0], finished=True, outcome='target_reached')
    official = player(1, role='admin')
    paused = service.suspend_match('MATCH5', official, reason_code='network_device', reason_text='fixture network pause',
        phase_token=service.snapshot('MATCH5', official)['match']['phase_token'], command_id='pause-race-test')
    with service.database.transaction(immediate=True) as db:
        row = db.execute("SELECT * FROM competition_game_sessions WHERE side='white'").fetchone()
        extra = json.loads(row['adapter_state_json'])
        extra['race_stop_at'] = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        db.execute('UPDATE competition_game_sessions SET adapter_state_json=? WHERE instance_id=?',
                   (json.dumps(extra), row['instance_id']))
    assert not service.settle_deadline('MATCH5')
    token = paused['match']['phase_token']
    for captain in (players[0], players[3]):
        service.set_suspension_readiness('MATCH5', captain, ready=True, phase_token=token, command_id=f'resume-{captain.user_id}')
    service.resume_match('MATCH5', official, phase_token=token, command_id='resume-race-test')
    assert not service.settle_deadline('MATCH5')
    assert upload(service, players[3], finished=True, elapsed=500, outcome='target_reached')['competition']['match']['current_result']['winner_side'] == 'white'
