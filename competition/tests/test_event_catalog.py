import pytest

from competition.backend.db import CompetitionDatabase
from competition.backend.domain import Principal
from competition.backend.errors import CompetitionError
from competition.backend.service import CompetitionService


@pytest.fixture
def service(tmp_path):
    service = CompetitionService(CompetitionDatabase(tmp_path / 'events.sqlite'), room_creator_ids=frozenset({2}))
    service.initialize()
    return service


ADMIN = Principal(1, 'Admin', 'admin')
PLAYER = Principal(2, 'Player')


def test_assign_organizer_scoped_permissions_and_validation(service):
    catalog = service.events
    catalog.enrollment.account_reader = lambda ids: {i: f'User {i}' for i in ids if i in (2, 3)}
    slug = '14360-cup-1'
    assert catalog.detail(slug, ADMIN)['can_assign_organizer']
    for principal in (PLAYER, Principal(9, 'Organizer', 'organizer')):
        with pytest.raises(CompetitionError) as exc:
            catalog.assign_organizer(slug, principal, user_id=2, dry_run=False)
        assert exc.value.status_code == 403
    with pytest.raises(CompetitionError):
        catalog.assign_organizer(slug, ADMIN, user_id=999, dry_run=False)
    assert catalog.assign_organizer(slug, ADMIN, user_id=2)['organizer']['display_name'] == 'User 2'
    assert not catalog.detail(slug, PLAYER)['can_manage']
    result = catalog.assign_organizer(slug, ADMIN, user_id=2, dry_run=False)
    assert result['event']['organizer']['user_id'] == 2
    assert catalog.detail(slug, PLAYER)['can_manage']
    assert not catalog.detail('819984-cup-3', PLAYER)['can_manage']
    catalog.update(slug, PLAYER, name='第一届14360杯', description='举办方公告', rules='规则', status='preparing')
    snapshot = catalog.enrollment.snapshot(slug, PLAYER)
    assert snapshot['me']['can_manage']
    imported = catalog.enrollment.import_roster(slug, PLAYER, entries=[{'user_id':2,'team_name':'','is_external':False,'captain':False}], revision=snapshot['revision'], dry_run=False)
    assert imported
    catalog.assign_organizer(slug, ADMIN, user_id=3, dry_run=False)
    assert not catalog.detail(slug, PLAYER)['can_manage']
    with pytest.raises(CompetitionError):
        catalog.update(slug, PLAYER, name='越权修改', description='', rules='', status='preparing')


def test_catalog_seed_is_idempotent_and_does_not_adopt_old_rooms(service):
    service.create_competition(ADMIN, name='第三届819984杯测试', room_code='EVNT23')
    service.initialize()
    assert {event['slug'] for event in service.events.list()} == {'819984-cup-3', '14360-cup-1'}
    assert service.events.detail('819984-cup-3')['room_count'] == 0


def test_link_public_privacy_idempotency_and_conflict(service):
    service.create_competition(ADMIN, name='正式房间', room_code='EVNT23', event_slug='819984-cup-3')
    service.events.link('819984-cup-3', 'EVNT23', ADMIN)
    public = service.events.detail('819984-cup-3')
    assert public['room_count'] == 1
    assert public['rooms'][0]['room_code'] is None
    assert public['rooms'][0]['public_key'] is None
    assert service.events.detail('819984-cup-3', ADMIN)['rooms'][0]['room_code'] == 'EVNT23'
    service.events.create(ADMIN, slug='another-cup', name='另一赛事')
    with pytest.raises(CompetitionError) as exc:
        service.events.link('another-cup', 'EVNT23', ADMIN)
    assert exc.value.code == 'ROOM_EVENT_CONFLICT'


def test_permissions_and_atomic_create(service):
    with pytest.raises(CompetitionError):
        service.events.create(PLAYER, slug='bad-cup', name='无权创建')
    with pytest.raises(CompetitionError):
        service.create_competition(PLAYER, name='无权关联', room_code='EVENT2', event_slug='819984-cup-3')
    with service.database.transaction() as db:
        assert not db.execute("SELECT 1 FROM competitions WHERE room_code='EVENT2'").fetchone()
    with pytest.raises(CompetitionError):
        service.events.detail('missing')


def test_expulsions_hide_directory_entry_code(service):
    service.create_competition(ADMIN, name='正式房间', room_code='EVNT23', event_slug='819984-cup-3')
    service.claim_seat('EVNT23', PLAYER, side='yellow', position=1, command_id='claim-event-seat')
    assert service.events.detail('819984-cup-3', PLAYER)['rooms'][0]['room_code'] == 'EVNT23'
    service.manage_member('EVNT23', ADMIN, user_id=2, remove=True, command_id='remove-event-player')
    assert service.events.detail('819984-cup-3', PLAYER)['rooms'][0]['room_code'] is None


def test_event_settings_and_room_context(service):
    values = dict(name='第三届819984杯', description='公告', rules='规则', status='active')
    with pytest.raises(CompetitionError):
        service.events.update('819984-cup-3', PLAYER, **values)
    service.events.update('819984-cup-3', ADMIN, **values)
    assert service.events.detail('819984-cup-3')['status'] == 'active'
    created = service.create_competition(ADMIN, name='房间测试', room_code='EVNT23')
    service.events.link('819984-cup-3', 'EVNT23', ADMIN)
    linked = service.snapshot('EVNT23', ADMIN)
    assert linked['version'] > created['version']
    assert linked['event']['slug'] == '819984-cup-3'
    assert service.list_competitions(ADMIN)[0]['event']['name'] == values['name']
    service.events.update('819984-cup-3', ADMIN, **(values | {'status': 'finished'}))
    with pytest.raises(CompetitionError):
        service.create_competition(ADMIN, name='结束后不能新增', room_code='EVNT24', event_slug='819984-cup-3')
