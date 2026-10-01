import sqlite3

import pytest
from pydantic import ValidationError

from backend.profile.validation import canonical_display_name_key
from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.event_statistics import accounts_by_username
from competition.backend.schemas import EnrollmentRosterRequest, StatisticsRosterEntry
from competition.backend.service import CompetitionService

ADMIN = Principal(900, 'Admin', 'admin')


@pytest.fixture
def enrollment(tmp_path):
    service = CompetitionService(CompetitionDatabase(tmp_path / 'events.sqlite'))
    service.initialize()
    result = service.events.enrollment
    result.account_reader = lambda ids: {uid: f'Player {uid}' for uid in ids if uid < 100}
    result.username_reader = lambda names: {canonical_display_name_key(f'Player {uid}'): [uid] for uid in range(1, 21)}
    service.events.statistics.account_reader = result.account_reader
    return result


@pytest.mark.parametrize('slug', ['819984-cup-3', '14360-cup-1'])
def test_current_username_import_preview_and_save(enrollment, slug):
    entries = [{'username': f'  PLAYER {uid}  '} for uid in range(1, 21)]
    revision = enrollment.snapshot(slug)['revision']
    importer = enrollment if slug == '819984-cup-3' else enrollment.catalog.statistics
    preview = importer.import_roster(slug, ADMIN, entries=entries, revision=revision)
    assert [e['user_id'] for e in preview['entries']] == list(range(1, 21))
    assert preview['entries'][0]['display_name'] == 'Player 1'
    assert enrollment.snapshot(slug)['entries'] == []
    request = EnrollmentRosterRequest(entries=preview['entries'], revision=revision, dry_run=False)
    importer.import_roster(slug, ADMIN, **request.model_dump())
    assert len(enrollment.snapshot(slug)['entries']) == 20


@pytest.mark.parametrize('dry_run', [True, False])
@pytest.mark.parametrize('slug', ['819984-cup-3', '14360-cup-1'])
def test_unmatched_name_blocks_entire_import(enrollment, slug, dry_run):
    revision = enrollment.snapshot(slug)['revision']
    enrollment.import_roster(slug, ADMIN, entries=[{'user_id': 1}], revision=revision, dry_run=False)
    before = enrollment.snapshot(slug)
    with pytest.raises(CompetitionError) as exc:
        enrollment.import_roster(slug, ADMIN, entries=[{'username': 'Player 2'}, {'username': 'Old Name'}, {'username': 'Missing'}], revision=before['revision'], dry_run=dry_run)
    assert exc.value.code == 'ROSTER_USERNAME_INVALID'
    assert '第 2 行「Old Name」' in exc.value.message and '第 3 行「Missing」' in exc.value.message
    assert enrollment.snapshot(slug) == before


def test_ambiguous_and_duplicate_identity_rejected(enrollment):
    slug = '819984-cup-3'
    revision = enrollment.snapshot(slug)['revision']
    with pytest.raises(CompetitionError):
        enrollment.import_roster(slug, ADMIN, entries=[{'username': 'Player 1'}, {'user_id': 1}], revision=revision, dry_run=False)
    enrollment.username_reader = lambda names: {'same': [1, 2]}
    with pytest.raises(CompetitionError, match='匹配到多个账号'):
        enrollment.import_roster(slug, ADMIN, entries=[{'username': 'Same'}], revision=revision, dry_run=False)
    assert enrollment.snapshot(slug)['entries'] == []


@pytest.mark.parametrize('entry', [{}, {'username': ' '}, {'username': 'Player', 'user_id': 1}])
def test_request_requires_exactly_one_identity(entry):
    with pytest.raises(ValidationError):
        StatisticsRosterEntry(**entry)


def test_database_resolver_only_current_active_names(tmp_path, monkeypatch):
    path = tmp_path / 'auth.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE users(id INTEGER,display_name TEXT,display_name_key TEXT,status TEXT)')
        db.executemany('INSERT INTO users VALUES(?,?,?,?)', [(1, 'New Name', 'new name', 'active'), (2, 'Disabled', 'disabled', 'disabled'), (3, '123', '123', 'active'), (4, 'Different Name', 'old name', 'active')])
    monkeypatch.setattr('backend.auth.db.get_auth_db_path', lambda: path)
    assert accounts_by_username(['ＮＥＷ  NAME', 'Old Name', 'Disabled', '123']) == {'new name': [1], '123': [3]}
    monkeypatch.setattr('backend.auth.db.get_auth_db_path', lambda: tmp_path / 'missing.sqlite')
    with pytest.raises(CompetitionError) as exc:
        accounts_by_username(['New Name'])
    assert exc.value.code == 'EVENT_SOURCE_UNAVAILABLE'
