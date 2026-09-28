from __future__ import annotations

import asyncio

from fastapi import Response

from backend.profile import preferences, themes
from competition.backend.domain import Principal
from competition.backend.routes import practice_appearance


def test_practice_appearance_reads_only_the_signed_in_players_theme(monkeypatch):
    calls = []

    def get_preferences(user_id):
        calls.append(("preferences", user_id))
        return {"preferences": {
            "theme": "Classic", "font_size_factor": 125,
            "saved_theme_id": 9, "language": "zh",
        }}

    def get_theme(user_id, theme_id):
        calls.append(("theme", user_id, theme_id))
        return {"theme": {"light": {"2": {"--tile-background": "#123456"}}}}

    monkeypatch.setattr(preferences, "get_preferences", get_preferences)
    monkeypatch.setattr(themes, "get_theme", get_theme)
    response = Response()
    result = asyncio.run(practice_appearance(Principal(42, "选手"), response))

    assert calls == [("preferences", 42), ("theme", 42, 9)]
    assert result["theme"] == "Classic"
    assert result["saved_theme"]["light"]["2"]["--tile-background"] == "#123456"
    assert "language" not in result
    assert response.headers["cache-control"] == "private, no-store"
