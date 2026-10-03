from http.cookies import SimpleCookie
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import Response
from starlette.requests import Request
from starlette.websockets import WebSocket

from backend.auth.dependencies import (
    auth_tokens_from_request,
    current_user_from_request,
    current_user_from_websocket,
    shared_cookie_domain,
)
from backend.auth.routes import _clear_session_cookie, _set_session_cookie
from backend.auth.service import SESSION_COOKIE_NAME, SHARED_SESSION_COOKIE_NAME


def scope(host, kind='http', cookie=''):
    return {
        'type': kind, 'scheme': 'wss' if kind == 'websocket' else 'https',
        'server': (host, 443), 'path': '/api/auth/me', 'query_string': b'',
        'headers': [(b'host', host.encode()), (b'cookie', cookie.encode())],
        'method': 'GET',
    }


@pytest.fixture(autouse=True)
def cookie_config(monkeypatch):
    monkeypatch.setenv('AUTH_SHARED_COOKIE_DOMAIN', '2048tables.online')
    monkeypatch.delenv('AUTH_SHARED_COOKIE_HOSTS', raising=False)
    monkeypatch.setenv('AUTH_COOKIE_SECURE', '1')


def test_tables_inherits_main_session_for_http_and_websocket():
    response = Response()
    _set_session_cookie(response, 'main-session', '2099-01-01T00:00:00+00:00',
                        Request(scope('2048tables.online')))
    cookies = SimpleCookie()
    for header in response.headers.getlist('set-cookie'):
        cookies.load(header)
    shared = cookies[SHARED_SESSION_COOKIE_NAME]
    assert shared['domain'] == '2048tables.online'
    assert shared['httponly'] and shared['secure']
    cookie = f'{SHARED_SESSION_COOKIE_NAME}={shared.value}; {SESSION_COOKIE_NAME}=stale'
    user = {'id': 23}
    with patch('backend.auth.dependencies.authenticate_session_token',
               side_effect=lambda token: user if token == 'main-session' else None), \
            patch('backend.auth.dependencies.bind_activity_account'):
        request = Request(scope('tables.2048tables.online', cookie=cookie))
        assert auth_tokens_from_request(request) == ['main-session', 'stale']
        assert current_user_from_request(request) == user
        assert current_user_from_websocket(WebSocket(scope(
            'tables.2048tables.online', 'websocket', cookie), AsyncMock(), AsyncMock())) == user


@pytest.mark.parametrize('host', ['tables.2048tables.online.evil.test',
                                  'unknown.2048tables.online', 'localhost'])
def test_untrusted_hosts_do_not_accept_shared_cookie(host):
    request = Request(scope(host, cookie=f'{SHARED_SESSION_COOKIE_NAME}=main-session'))
    assert shared_cookie_domain(request) is None
    assert auth_tokens_from_request(request) == []


def test_tables_login_and_logout_use_the_shared_domain():
    request = Request(scope('tables.2048tables.online'))
    response = Response()
    _set_session_cookie(response, 'tables-session', '2099-01-01T00:00:00+00:00', request)
    assert any(f'{SHARED_SESSION_COOKIE_NAME}=tables-session' in value
               and 'Domain=2048tables.online' in value
               for value in response.headers.getlist('set-cookie'))
    cleared = Response()
    _clear_session_cookie(cleared, request)
    assert any(f'{SHARED_SESSION_COOKIE_NAME}=' in value
               and 'Domain=2048tables.online' in value and 'Max-Age=0' in value
               for value in cleared.headers.getlist('set-cookie'))
