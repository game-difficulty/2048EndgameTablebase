from __future__ import annotations

import sqlite3

from competition.backend.db import CompetitionDatabase, SCHEMA_VERSION


def test_v4_match_tables_are_upgraded_and_suspension_is_backfilled(tmp_path) -> None:
    path = tmp_path / "v4.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE competition_team_clocks (
          competition_id TEXT NOT NULL,
          side TEXT NOT NULL,
          remaining_ms_base INTEGER NOT NULL,
          running_since TEXT,
          state TEXT NOT NULL,
          revision INTEGER NOT NULL DEFAULT 1,
          updated_at TEXT NOT NULL,
          PRIMARY KEY (competition_id, side)
        );
        CREATE TABLE competition_game_results (
          competition_id TEXT NOT NULL,
          game_key TEXT NOT NULL,
          yellow_score INTEGER NOT NULL,
          white_score INTEGER NOT NULL,
          winner_side TEXT NOT NULL,
          reason TEXT NOT NULL,
          result_revision INTEGER NOT NULL DEFAULT 1,
          published_at TEXT NOT NULL,
          PRIMARY KEY (competition_id, game_key)
        );
        """
    )
    connection.commit()
    connection.close()

    database = CompetitionDatabase(path)
    database.initialize()
    with database.transaction(immediate=True) as db:
        db.execute(
            """
            INSERT INTO competitions
              (id, room_code, name, status, created_by_user_id, version, created_at, updated_at)
            VALUES ('legacy-room', 'LEGACY', 'Legacy', 'GAME_A_READY', 1, 1, 'now', 'now')
            """
        )
        db.execute(
            """
            INSERT INTO competition_match_control
              (competition_id, current_game_key, phase_token, created_at, updated_at)
            VALUES ('legacy-room', 'A', 'legacy-token', 'now', 'now')
            """
        )
    database.initialize()

    with database.transaction() as db:
        clock_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(competition_team_clocks)")
        }
        result_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(competition_game_results)")
        }
        suspension_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(competition_suspensions)")
        }
        competition_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(competitions)")
        }
        project_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(competition_projects)")
        }
        session_columns = {
            str(row[1]) for row in db.execute("PRAGMA table_info(competition_game_sessions)")
        }
        schema_version = db.execute(
            "SELECT value FROM competition_schema_meta WHERE key = 'schema_version'"
        ).fetchone()["value"]
        suspension = db.execute(
            "SELECT active FROM competition_suspensions WHERE competition_id = 'legacy-room'"
        ).fetchone()
    assert "resume_after_suspension" in clock_columns
    assert {"corrected_by_user_id", "correction_reason", "corrected_at"} <= result_columns
    assert "started_by_display_name" in suspension_columns
    assert {"public_key", "live_generation", "live_started_at", "live_ended_at"} <= competition_columns
    assert {"project_ref", "adapter_rules_version", "adapter_snapshot_json"} <= project_columns
    assert {"project_ref", "rules_version", "instance_id", "public_generation"} <= session_columns
    assert schema_version == str(SCHEMA_VERSION)
    assert suspension is not None and int(suspension["active"]) == 0
