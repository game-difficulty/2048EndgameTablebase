import json
import sqlite3

import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.event_statistics import EventStatistics, play_results
from competition.backend.service import CompetitionService
from backend.human_play.rating import top_rating

ADMIN = Principal(1, 'Admin', 'admin')
SLUG = '14360-cup-1'


@pytest.fixture
def service(tmp_path):
    result = CompetitionService(CompetitionDatabase(tmp_path / 'events.sqlite'))
    result.initialize()
    return result


def roster():
    return [{'user_id': i, 'team_name': f'Team {(i-1)//5+1}', 'is_external': i == 20} for i in range(1, 21)]


def stats(service, reader=lambda *args: {}):
    return EventStatistics(service.events, account_reader=lambda ids: {i: f'Player {i}' for i in ids}, result_reader=reader)


def test_statistics_format_rejects_rooms_atomically(service):
    detail = service.events.detail(SLUG)
    assert detail['team_size'] == 5
    assert detail['capabilities']['statistics'] and not detail['capabilities']['rooms']
    with pytest.raises(CompetitionError) as exc:
        service.create_competition(ADMIN, name='Must not create', room_code='STATS2', event_slug=SLUG)
    assert exc.value.code == 'EVENT_FORMAT_MISMATCH'
    assert service.list_competitions(ADMIN) == []


def test_roster_preview_atomic_replace_and_revision(service):
    module = stats(service)
    preview = module.import_roster(SLUG, ADMIN, entries=roster(), revision=0)
    assert len(preview['entries']) == 20 and not preview['saved']
    assert module.standings(SLUG)['players'] == []
    module.import_roster(SLUG, ADMIN, entries=roster(), revision=0, dry_run=False)
    assert len(module.standings(SLUG)['players']) == 20
    with pytest.raises(CompetitionError):
        module.import_roster(SLUG, ADMIN, entries=roster(), revision=0, dry_run=False)
    duplicate = roster(); duplicate[-1]['user_id'] = 1
    with pytest.raises(CompetitionError):
        module.import_roster(SLUG, ADMIN, entries=duplicate, revision=1, dry_run=False)
    with pytest.raises(CompetitionError):
        module.import_roster(SLUG, Principal(99, 'Other'), entries=roster(), revision=1)
    assert module.standings(SLUG)['roster_revision'] == 1


def test_rating_uses_mean_board_sum_not_mean_ratings(service):
    games = [{'id': str(i), 'score': 100-i, 'board_sum': value} for i, value in enumerate([256, 512, 768, 1024, 1280])]
    module = stats(service, lambda ids, *args: {i: {'completed_games': 7, 'games': games} for i in ids})
    module.import_roster(SLUG, ADMIN, entries=roster(), revision=0, dry_run=False)
    result = module.standings(SLUG)
    assert result['players'][0]['rating'] == top_rating('3x3', [256, 512, 768, 1024, 1280])
    assert result['teams'][0]['rating'] == pytest.approx(result['players'][0]['rating'] * 5)
    assert result['teams'][0]['board_sum'] == sum([256, 512, 768, 1024, 1280]) * 5
    assert result['teams'][0]['selected_games'] == 25


def test_source_filters_boundaries_imports_and_top_score(tmp_path, monkeypatch):
    path = tmp_path / 'play.sqlite'
    monkeypatch.setenv('HUMAN_PLAY_DB', str(path))
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE human_runs(id TEXT,user_id INTEGER,variant TEXT,status TEXT,source TEXT,
            visible INTEGER,eligibility TEXT,reason TEXT,created REAL,ended REAL,state TEXT);
            CREATE TABLE human_rank_approvals(run_id TEXT);''')
        def add(key, score=100, board=100, created=1000, ended=1500, source='native', status='sealed', variant='3x3', eligible='eligible'):
            db.execute('INSERT INTO human_runs VALUES(?,1,?,?,?,1,?,\'game_over\',?,?,?)',
                       (key, variant, status, source, eligible, created, ended, json.dumps({'score': score, 'board': [board]})))
        for i in range(7): add(f'good{i}', score=100+i, board=1000-i)
        add('before-start', 9999, created=999)
        add('at-end', 9999, ended=2000)
        add('verse', 9999, source='verse')
        add('manual', 9999, source='manual')
        add('active', 9999, status='active')
        add('other-variant', 9999, variant='4x4')
        add('invalid', 9999, eligible='invalid')
    result = play_results([1], 1000, 2000)[1]
    assert result['completed_games'] == 7
    assert [game['id'] for game in result['games']] == ['good6','good5','good4','good3','good2']
    assert result['games'][0]['board_sum'] == 994


def test_missing_source_is_not_zero(tmp_path, monkeypatch):
    monkeypatch.setenv('HUMAN_PLAY_DB', str(tmp_path / 'missing.sqlite'))
    with pytest.raises(CompetitionError) as exc:
        play_results([1], 1000, 2000)
    assert exc.value.code == 'EVENT_SOURCE_UNAVAILABLE'
    assert not (tmp_path / 'missing.sqlite').exists()
