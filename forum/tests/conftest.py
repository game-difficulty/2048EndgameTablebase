"""Tests create and drop ONLY a fresh, randomly named PostgreSQL database."""
from dataclasses import replace
import os
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy.engine import make_url

from forum.backend.app import create_app
from forum.backend.config import Settings
from forum.backend.db import execute, make_engine

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def settings():
    admin_url = os.environ.get("FORUM_TEST_ADMIN_URL")
    if not admin_url:
        pytest.skip("Set FORUM_TEST_ADMIN_URL to an isolated PostgreSQL admin URL; no SQLite fallback.")
    parsed = make_url(admin_url)
    name = "forum_test_" + uuid4().hex
    connection = psycopg.connect(parsed.set(drivername="postgresql").render_as_string(hide_password=False), autocommit=True)
    connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = parsed.set(drivername="postgresql+psycopg", database=name).render_as_string(hide_password=False)
    old = os.environ.get("FORUM_DATABASE_URL")
    os.environ["FORUM_DATABASE_URL"] = url
    try:
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        # Repeating upgrade is safe; no seed duplication or implicit schema creation.
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        yield Settings(url, public_origin="http://testserver", environment="test",
                       allow_dev_auth=True, admin_ids=frozenset({1}))
    finally:
        if old is None:
            os.environ.pop("FORUM_DATABASE_URL", None)
        else:
            os.environ["FORUM_DATABASE_URL"] = old
        assert name.startswith("forum_test_") and len(name) == 43
        connection.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        connection.close()


@pytest.fixture
def app(settings):
    engine = make_engine(settings.database_url)
    with engine.begin() as conn:
        # This fixture's engine points exclusively at its generated test database.
        execute(conn, """TRUNCATE forum_profiles,forum_roles,forum_sanctions,forum_topics,
            forum_posts,forum_post_revisions,forum_drafts,forum_reactions,forum_bookmarks,
            forum_notifications,forum_reports,forum_moderation_actions,forum_idempotency,
            forum_outbox,forum_rate_windows,forum_media,forum_post_media,forum_subscriptions,
            forum_notification_preferences,forum_follows,forum_mentions,forum_reading,
            forum_reply_drafts,forum_appeals CASCADE""")
    engine.dispose()
    yield create_app(settings)


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


def headers(user=2, key=None):
    result = {"Origin": "http://testserver", "X-Forum-Dev-User": f"{user}:Player{user}"}
    if key:
        result["Idempotency-Key"] = str(key)
    return result


def document(value="这里讨论一个残局。"):
    return {"version": 1, "blocks": [{"type": "paragraph", "text": value}]}


def topic(client, user=2, **changes):
    payload = {"board_slug": "general", "title": "一个残局讨论", "body": document(), "tags": ["L3"]}
    payload.update(changes)
    response = client.post("/api/forum/v1/topics", json=payload, headers=headers(user, uuid4()))
    assert response.status_code == 201, response.text
    return response.json()
