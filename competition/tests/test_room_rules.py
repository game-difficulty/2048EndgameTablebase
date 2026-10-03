from datetime import timedelta
import json
import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.room_rules import normalize_rules, validate_lineup, default_lineup
from competition.backend.service import CompetitionService, parse_time, DEFAULT_PROJECTS

ADMIN = Principal(1, 'Admin', 'admin')


@pytest.fixture
def service(tmp_path):
    value = CompetitionService(CompetitionDatabase(tmp_path/'rules.sqlite'), draw_reveal_seconds=1,
                               draft_turn_seconds=5, c_draw_reveal_seconds=1, lineup_seconds=5, result_rest_seconds=0)
    value.initialize()
    return value


def create(service, preset='bo7', size=4, **extra):
    pool = [dict(DEFAULT_PROJECTS[0], key=f'p{i}', name=f'Project {i}') for i in range(12)]
    room = service.create_competition(ADMIN, name='Reusable series', room_code='RULE22', projects=pool,
                                      rules=dict(preset=preset, team_size=size, **{'lineup_policy':'free', **extra}))
    players = {side: [Principal(10+s*size+i, f'{side} {i}') for i in range(size)] for s, side in enumerate(('yellow', 'white'))}
    for side, team in players.items():
        for i, user in enumerate(team):
            service.claim_seat('RULE22', user, side=side, position=i+1, command_id=f'claim-{user.user_id}')
        service.set_ready('RULE22', team[0], ready=True, command_id=f'ready-{side}')
    room = service.snapshot('RULE22', ADMIN)
    service.settle_deadline('RULE22', now=parse_time(room['draft']['deadline_at'])+timedelta(milliseconds=1))
    return players


def finish_draft(service, players):
    while True:
        room = service.snapshot('RULE22', ADMIN)
        if room['status'] != 'DRAFT_STEP':
            return room
        flow = room['draft']['workflow']; step = flow['current_step']
        captain = players[flow['active_side']][0]
        own = service.snapshot('RULE22', captain)
        pool = own['draft']['available_project_keys']
        service.submit_draft_step('RULE22', captain, picks=pool[:step['picks']], bans=pool[step['picks']:step['picks']+step['bans']],
                                  phase_token=own['draft']['phase_token'], command_id=f'step-{flow["step_index"]}-submit')


@pytest.mark.parametrize('preset,count,minimum', [('bo3',3,5),('bo5',5,7),('bo7',7,11)])
def test_templates(preset,count,minimum):
    rules = normalize_rules({'preset':preset})
    assert rules['game_count'] == count and rules['minimum_pool_size'] == minimum
    with pytest.raises(CompetitionError):normalize_rules({'preset':preset}, pool_size=minimum-1)


@pytest.mark.parametrize('preset,count', [('bo5',5),('bo7',7)])
def test_generic_draft_lineup_restart_and_all_results(service,preset,count):
    players=create(service,preset,size=4,series_mode='all')
    room=finish_draft(service,players)
    assert room['status']=='C_DRAW'
    assert len(set(room['draft']['workflow']['selected']))==count
    assert not set(room['draft']['workflow']['selected']) & set(room['draft']['workflow']['banned'])
    service.initialize()  # Schema init and process reconstruction must preserve the frozen draft.
    service=CompetitionService(service.database,result_rest_seconds=0)
    service.settle_deadline('RULE22',now=parse_time(room['draft']['deadline_at'])+timedelta(milliseconds=1))
    for side,team in players.items():
        own=service.snapshot('RULE22',team[0])
        service.submit_lineup('RULE22',team[0],assignments=default_lineup(own['rules']),phase_token=own['lineup']['phase_token'],command_id=f'lineup-{side}')
    room=service.snapshot('RULE22',ADMIN)
    assert room['status']=='GAME_A_READY'
    # Exercise the actual result transition and DB constraints for every game, including D–G.
    for key in room['rules']['game_keys']:
        with service.database.transaction(immediate=True) as db:
            current=service._room_row(db,'RULE22')
            assert service._match_control_row(db,current['id'])['current_game_key']==key
            from datetime import datetime,timezone
            now=datetime.now(timezone.utc)
            db.execute('INSERT INTO competition_game_results(competition_id,game_key,yellow_score,white_score,winner_side,reason,result_revision,published_at) VALUES(?,?,1,0,\'yellow\',\'score\',1,?)',(current['id'],key,now.isoformat()))
            db.execute('UPDATE competition_match_control SET yellow_wins=yellow_wins+1 WHERE competition_id=?',(current['id'],))
            service._advance_after_result(db,current,game_key=key,now=now,actor_user_id=None)
        snapshot=service.snapshot('RULE22',ADMIN)
        assert len(snapshot['match']['results'])==ord(key)-64
    assert snapshot['status']=='FINISHED'
    with service.database.transaction() as db:
        public=service._public_match_projection(db,service._room_row(db,'RULE22'))
    assert len(public['games'])==count
    assert public['score']['yellow']==count


def test_lineup_policies_and_early_finish():
    rules=normalize_rules({'preset':'bo7','team_size':4,'lineup_policy':'balanced'})
    validate_lineup(rules,default_lineup(rules))
    with pytest.raises(CompetitionError):validate_lineup(rules,{k:1 for k in rules['game_keys']})
    from competition.backend.room_rules import series_complete
    assert series_complete(rules,'D',4,0)
    assert not series_complete(rules,'D',3,1)
    assert series_complete(rules,'G',2,2)  # Draws cannot leave the room stuck after the final game.


@pytest.mark.parametrize('policy,positions,valid', [
    ('everyone', [1,1,1,1,1,2,3], True),
    ('balanced', [1,1,1,1,1,2,3], False),
    ('everyone', [1,1,1,1,1,2,2], False),
    ('free', [1,1,1,1,1,1,1], True),
    ('everyone', [1,2,3,1,2,3,1], True),
    ('balanced', [1,2,3,1,2,3,1], True),
    ('balanced', [1,1,1,1,2,2,3], False),
])
def test_participation_vs_balanced(policy, positions, valid):
    rules=normalize_rules({'preset':'bo7','team_size':3,'lineup_policy':policy})
    assignments=dict(zip(rules['game_keys'],positions))
    if valid:
        validate_lineup(rules,assignments)
    else:
        with pytest.raises(CompetitionError) as exc: validate_lineup(rules,assignments)
        assert exc.value.code=='INVALID_LINEUP'


def test_participation_defaults_and_short_series():
    assert normalize_rules({'preset':'bo3'})['lineup_policy']=='unique'
    for preset in ('bo5','bo7'):
        assert normalize_rules({'preset':preset})['lineup_policy']=='everyone'
        assert normalize_rules({'preset':preset,'lineup_policy':'free'})['lineup_policy']=='free'
    rules=normalize_rules({'preset':'bo3','team_size':5,'lineup_policy':'everyone'})
    validate_lineup(rules,{'A':5,'B':2,'C':4})
    with pytest.raises(CompetitionError): validate_lineup(rules,{'A':5,'B':5,'C':4})
    # Timeout allocation remains valid for all team sizes and both repeated-appearance restrictions.
    for preset in ('bo3','bo5','bo7'):
        for size in range(1,17):
            for policy in ('everyone','balanced'):
                rules=normalize_rules({'preset':preset,'team_size':size,'lineup_policy':policy})
                validate_lineup(rules,default_lineup(rules))


@pytest.mark.parametrize('policy,valid', [('everyone',True),('balanced',False)])
def test_room_submission_enforces_frozen_appearance_policy(service,policy,valid):
    players=create(service,size=3,lineup_policy=policy)
    room=finish_draft(service,players)
    service.settle_deadline('RULE22',now=parse_time(room['draft']['deadline_at'])+timedelta(milliseconds=1))
    service.initialize()
    captain=players['yellow'][0]
    own=service.snapshot('RULE22',captain)
    assert own['rules']['lineup_policy']==policy
    args=dict(assignments=dict(zip('ABCDEFG',[1,1,1,1,1,2,3])),phase_token=own['lineup']['phase_token'],command_id='participation-lineup-command')
    if valid:
        service.submit_lineup('RULE22',captain,**args)
    else:
        with pytest.raises(CompetitionError) as exc: service.submit_lineup('RULE22',captain,**args)
        assert exc.value.code=='INVALID_LINEUP'


def test_timeout_fallback_and_wrong_turn(service):
    players=create(service)
    room=service.snapshot('RULE22',ADMIN);active=room['draft']['workflow']['active_side']
    other='white' if active=='yellow' else 'yellow'
    captain=players[other][0];own=service.snapshot('RULE22',captain)
    with pytest.raises(CompetitionError):
        service.submit_draft_step('RULE22',captain,picks=['p0'],bans=['p1'],phase_token=own['draft']['phase_token'],command_id='bad-turn-command')
    service.settle_deadline('RULE22',now=parse_time(room['draft']['deadline_at'])+timedelta(milliseconds=1))
    room=service.snapshot('RULE22',ADMIN)
    assert room['draft']['workflow']['history'][0]['picks']==['p0']
    assert room['draft']['workflow']['history'][0]['automatic']


def test_generic_blind_does_not_leak_or_reuse_previous_pick(service):
    players=create(service,'bo5',final_selection='blind')
    room=finish_draft(service,players)
    assert room['status']=='BLIND_PICK'
    captain=players['yellow'][0];own=service.snapshot('RULE22',captain)
    pick=own['draft']['available_project_keys'][0]
    service.submit_blind_pick('RULE22',captain,project_key=pick,phase_token=own['draft']['phase_token'],command_id='blind-yellow-command')
    opponent=service.snapshot('RULE22',players['white'][0])
    assert not opponent['draft'].get('my_blind_choice')
    assert not opponent['draft']['workflow']['complete']
    assert opponent['draft']['workflow']['seed_hex'] is None
    assert pick not in opponent['draft']['workflow']['selected']
    assert 'blind_choices' not in opponent['draft']
    service.submit_blind_pick('RULE22',players['white'][0],project_key=pick,phase_token=opponent['draft']['phase_token'],command_id='blind-white-command')
    complete=service.snapshot('RULE22',ADMIN)
    assert complete['draft']['workflow']['selected'][-1]==pick
    assert complete['draft']['workflow']['seed_hex'] is None  # Also seeds future game sessions.


def test_migration_preserves_legacy_rows_indexes_and_constraints(tmp_path):
    import sqlite3
    from competition.backend.db import SCHEMA
    path=tmp_path/'legacy.sqlite'
    legacy=SCHEMA.replace('position BETWEEN 1 AND 16','position BETWEEN 1 AND 3')
    legacy=legacy.replace("length(game_key) = 1 AND game_key BETWEEN 'A' AND 'O'", "game_key IN ('A', 'B', 'C')")
    legacy=legacy.replace("length(current_game_key) = 1 AND current_game_key BETWEEN 'A' AND 'O'", "current_game_key IN ('A', 'B', 'C')")
    # Reinstate the legacy one-game-per-player constraint.
    legacy=legacy.replace('PRIMARY KEY (competition_id, side, game_key),','PRIMARY KEY (competition_id, side, game_key), UNIQUE (competition_id, side, position),',1)
    db=sqlite3.connect(path)
    db.executescript(legacy)
    db.execute("INSERT INTO competitions(id,public_key,room_code,name,status,created_by_user_id,created_at,updated_at) VALUES('old','public-old','OLD222','Legacy','LINEUP',1,'2026-10-01','2026-10-01')")
    db.execute("INSERT INTO competition_seats VALUES('old','yellow',1,1,'Captain','2026-10-01')")
    db.execute("INSERT INTO competition_lineups(competition_id,side,game_key,position,player_user_id,submitted_at) VALUES('old','yellow','A',1,1,'2026-10-01')")
    db.execute('CREATE INDEX legacy_lineup_lookup ON competition_lineups(player_user_id)')
    db.commit();db.close()
    service=CompetitionService(CompetitionDatabase(path));service.initialize();service.initialize()
    with service.database.transaction(immediate=True) as db:
        assert db.execute("SELECT display_name_snapshot FROM competition_seats WHERE competition_id='old'").fetchone()[0]=='Captain'
        assert db.execute("SELECT name FROM sqlite_master WHERE name='legacy_lineup_lookup'").fetchone()
        assert len(db.execute("SELECT * FROM competition_lineups WHERE competition_id='old'").fetchall())==1
        db.execute("INSERT INTO competition_lineups(competition_id,side,game_key,position,player_user_id,submitted_at) VALUES('old','yellow','G',1,1,'2026-10-01')")
        db.execute("INSERT INTO competition_seats VALUES('old','yellow',16,16,'Reserve','2026-10-01')")
        assert not db.execute('PRAGMA foreign_key_check').fetchall()


def test_configurable_commands_are_idempotent_and_reject_stale_phase(service):
    players=create(service)
    before=service.snapshot('RULE22',ADMIN);draft=before['draft'];captain=players[draft['workflow']['active_side']][0]
    draft=service.snapshot('RULE22',captain)['draft']
    args=dict(picks=['p0'],bans=['p1'],phase_token=draft['phase_token'],command_id='draft-repeat-command')
    service.submit_draft_step('RULE22',captain,**args)
    repeated=service.submit_draft_step('RULE22',captain,**args)
    assert len(repeated['draft']['workflow']['history'])==1
    with pytest.raises(CompetitionError) as exc:
        service.submit_draft_step('RULE22',captain,**{**args,'command_id':'new-stale-command'})
    assert exc.value.code=='STALE_PHASE'


def test_custom_room_uses_opaque_projects_and_preserves_rematch_rules(service):
    pool=[dict(DEFAULT_PROJECTS[0],key=f'p{i}',name=f'Opaque {i}') for i in range(2)]
    room=service.create_competition(ADMIN,name='One game',projects=pool,rules={
        'preset':'custom','team_size':1,'lineup_policy':'free','steps':[{'actor':'first','bans':1,'picks':0}]})
    assert room['rules']['game_count']==1
    rematch=service.rematch_before_lineup(room['room_code'],ADMIN,command_id='rematch-rules-command')
    assert rematch['rules']==room['rules']


def test_best_of_ends_at_majority_and_supports_predictions(service):
    from datetime import datetime,timezone
    players=create(service,'bo5')
    room=finish_draft(service,players)
    service.settle_deadline('RULE22',now=parse_time(room['draft']['deadline_at'])+timedelta(milliseconds=1))
    for side,team in players.items():
        own=service.snapshot('RULE22',team[0])
        service.submit_lineup('RULE22',team[0],assignments=default_lineup(own['rules']),phase_token=own['lineup']['phase_token'],command_id=f'lineup-{side}')
    with service.database.transaction(immediate=True) as db:
        row=service._room_row(db,'RULE22')
        assert service._prediction_window(db,row)['open']
        db.execute("UPDATE competition_match_control SET yellow_wins=3,current_game_key='C' WHERE competition_id=?",(row['id'],))
        service._advance_after_result(db,row,game_key='C',now=datetime.now(timezone.utc),actor_user_id=None)
    assert service.snapshot('RULE22',ADMIN)['status']=='FINISHED'


def test_event_rosters_accept_more_than_three_players(service):
    event=service.events.create(ADMIN,slug='larger-teams',name='Larger teams',team_size=5)
    assert event['team_size']==5
    with service.database.transaction() as db:
        assert service.events.enrollment.config(db,event['slug'])['team_size']==5


def test_seven_games_run_through_real_sessions_and_client_results(service):
    from competition.tests.test_match_service import ready_and_start, force_one_move_completion
    service.test_project_target_tile=4
    players=create(service,'bo7',series_mode='all')
    room=finish_draft(service,players)
    service.settle_deadline('RULE22',now=parse_time(room['draft']['deadline_at'])+timedelta(milliseconds=1))
    for side,team in players.items():
        own=service.snapshot('RULE22',team[0])
        service.submit_lineup('RULE22',team[0],assignments=default_lineup(own['rules']),phase_token=own['lineup']['phase_token'],command_id=f'lineup-{side}')
    ordered=players['yellow'][:3]+players['white'][:1]+players['yellow'][3:]+players['white'][1:]
    for i,key in enumerate('ABCDEFG'):
        started=ready_and_start(service,ordered,key,code='RULE22')
        assert started['status']==f'GAME_{key}_PLAYING'
        for side in ('yellow','white'):
            force_one_move_completion(service,players[side][i%4],key,side,code='RULE22')
        room=service.snapshot('RULE22',ADMIN)
        assert len(room['match']['results'])==i+1
        service.settle_deadline('RULE22')
    assert service.snapshot('RULE22',ADMIN)['status']=='FINISHED'


def test_rule_routes_validate_before_creation(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv('COMPETITION_DB',str(tmp_path/'api.sqlite'))
    monkeypatch.setenv('COMPETITION_ALLOW_DEV_AUTH','1')
    monkeypatch.setenv('CLOUD_AUTH_DB',str(tmp_path/'auth.sqlite'))
    from competition.backend.app import create_app
    with TestClient(create_app()) as client:
        presets=client.get('/api/room-rule-presets').json()['presets']
        assert [r['game_count'] for r in presets]==[3,5,7]
        headers={'X-Competition-Dev-User':'1:Admin:admin'}
        invalid=client.post('/api/competitions',headers=headers,json={'name':'Invalid','rules':{'preset':[]}})
        assert invalid.status_code==400
        created=client.post('/api/competitions',headers=headers,json={'name':'API five','rules':{'preset':'bo5'}})
        assert created.status_code==201
        room=created.json()['competition']
        assert room['rules']['game_count']==5
        assert client.post(f'/api/competitions/{room["room_code"]}/draft/step',json={'picks':[],'bans':[],'phase_token':'x','command_id':'anonymous-command'}).status_code==401


def test_five_player_locked_roster_and_seven_game_late_forfeit(service):
    from datetime import datetime,timezone
    slug='five-player-cup'
    service.events.create(ADMIN,slug=slug,name='Five player cup',team_size=5)
    enrollment=service.events.enrollment
    enrollment.account_reader=lambda ids:{uid:f'Player {uid}' for uid in ids}
    state=enrollment.action(slug,ADMIN,action='settings',revision=0,mode='organizer_team',capacity=0,registration_open=False)
    enrollment.import_roster(slug,ADMIN,revision=state['revision'],dry_run=False,entries=[
        {'user_id':uid,'team_name':'Yellow' if uid<15 else 'White','captain':uid in (10,15),'position':(uid-10)%5+1,'is_external':False} for uid in range(10,20)])
    state=enrollment.snapshot(slug)
    state=enrollment.action(slug,ADMIN,action='lock_roster',revision=state['revision'])
    start=datetime.now(timezone.utc)+timedelta(hours=1)
    teams={team['name']:team['id'] for team in state['teams']}
    pool=[dict(DEFAULT_PROJECTS[0],key=f'p{i}',name=f'Project {i}') for i in range(12)]
    room=service.create_competition(ADMIN,name='Scheduled seven',projects=pool,event_slug=slug,starts_at=start.isoformat(),yellow_team_id=teams['Yellow'],white_team_id=teams['White'],rules={'preset':'bo7','team_size':5})
    code=room['room_code']
    assert len(room['schedule']['players'])==10
    for uid in range(10,15):
        player=Principal(uid,f'Player {uid}')
        service.schedule.check_in(code,player)
        service.claim_seat(code,player,side='yellow',position=uid-9,command_id=f'large-seat-{uid}')
    service.set_ready(code,Principal(10,'Player 10'),ready=True,command_id='large-ready-command')
    service.settle_deadline(code,now=start+timedelta(minutes=15,seconds=1))
    final=service.snapshot(code,ADMIN)
    assert final['status']=='FINISHED'
    assert final['match']['series_score']=={'yellow':7,'white':0,'draws':0}
    assert len(final['match']['results'])==7
