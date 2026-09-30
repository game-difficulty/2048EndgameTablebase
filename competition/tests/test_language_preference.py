import asyncio
from types import SimpleNamespace
from starlette.responses import Response
from competition.backend import routes
from backend.profile import preferences


def test_language_endpoint_is_read_only_and_returns_only_language(monkeypatch):
    monkeypatch.setattr(routes, 'optional_principal', lambda request: SimpleNamespace(user_id=17))
    monkeypatch.setattr(preferences, 'get_preferences', lambda user_id: {
        'preferences': {'language': 'en', 'theme': 'private'}, 'revision': 8,
    })
    def forbidden_write(*args, **kwargs):
        raise AssertionError('Language sync must not write account preferences')
    monkeypatch.setattr(preferences, 'patch_preferences', forbidden_write)
    response = Response()
    assert asyncio.run(routes.preferred_language(None, response)) == {'language': 'en'}
    assert response.headers['cache-control'] == 'private, no-store'


def test_language_endpoint_guest_does_not_read_account(monkeypatch):
    monkeypatch.setattr(routes, 'optional_principal', lambda request: None)
    def forbidden_read(*args):
        raise AssertionError('Guest must not read account preferences')
    monkeypatch.setattr(preferences, 'get_preferences', forbidden_read)
    assert asyncio.run(routes.preferred_language(None, Response())) == {'language': None}
