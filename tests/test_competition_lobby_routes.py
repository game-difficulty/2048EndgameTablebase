import pytest

from tools.enable_competition_lobby_routes import add_lobby_routes


ANCHOR = '    location = /practice { try_files /index.html =404; add_header Cache-Control no-cache; }'


def test_lobbies_have_exact_routes_without_relaxing_fallback():
    original = f'server {{\n{ANCHOR}\n    location / {{ return 404; }}\n}}'
    result = add_lobby_routes(original)
    for route in ('duels', 'time-attacks'):
        assert f'location = /{route} {{ try_files /index.html =404;' in result
        assert f'location = /{route}/ {{ return 308 /{route}; }}' in result
    assert 'location / { return 404; }' in result
    assert add_lobby_routes(result) == result


def test_partial_configuration_only_adds_missing_routes():
    old = '    location = /duels { try_files /index.html =404; }\n' + ANCHOR
    result = add_lobby_routes(old)
    assert result.count('location = /duels {') == 1
    assert 'location = /duels/ ' in result


@pytest.mark.parametrize('config', ['', ANCHOR+'\n'+ANCHOR])
def test_unknown_layout_fails_before_writing(config):
    with pytest.raises(ValueError):
        add_lobby_routes(config)
