from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_VERSION = 19


SCHEMA = """
CREATE TABLE IF NOT EXISTS tournament_fixture_stages (
  id TEXT PRIMARY KEY,
  event_slug TEXT NOT NULL REFERENCES tournament_events(slug),
  name TEXT NOT NULL,
  groups_json TEXT NOT NULL,
  projects_json TEXT NOT NULL,
  rules_json TEXT NOT NULL,
  organizer_user_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  command_id TEXT NOT NULL,
  request_json TEXT NOT NULL,
  UNIQUE(event_slug, name), UNIQUE(event_slug, command_id)
);
CREATE TABLE IF NOT EXISTS tournament_fixtures (
  id TEXT PRIMARY KEY,
  stage_id TEXT NOT NULL REFERENCES tournament_fixture_stages(id),
  group_name TEXT NOT NULL,
  round_number INTEGER NOT NULL,
  yellow_team_id TEXT NOT NULL,
  white_team_id TEXT NOT NULL,
  competition_id TEXT UNIQUE REFERENCES competitions(id),
  revision INTEGER NOT NULL DEFAULT 0,
  proposed_at TEXT,
  proposed_by_user_id INTEGER,
  UNIQUE(stage_id, yellow_team_id, white_team_id)
);
CREATE TABLE IF NOT EXISTS tournament_fixture_commands (
  fixture_id TEXT NOT NULL REFERENCES tournament_fixtures(id),
  command_id TEXT NOT NULL,
  actor_user_id INTEGER NOT NULL,
  request_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY(fixture_id, command_id)
);
CREATE INDEX IF NOT EXISTS idx_fixture_stage_event ON tournament_fixture_stages(event_slug);
CREATE INDEX IF NOT EXISTS idx_fixture_stage ON tournament_fixtures(stage_id);
CREATE TABLE IF NOT EXISTS competition_time_attack (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  configuration_json TEXT NOT NULL,
  started_at TEXT,
  deadline_at TEXT,
  winner_side TEXT
);
CREATE TABLE IF NOT EXISTS competition_time_attempts (
  id TEXT PRIMARY KEY,
  competition_id TEXT NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
  side TEXT NOT NULL,
  number INTEGER NOT NULL,
  started_at TEXT NOT NULL,
  seed TEXT NOT NULL,
  state_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'playing',
  pb_ms INTEGER,
  events BLOB NOT NULL DEFAULT X'',
  UNIQUE(competition_id, side, number)
);
CREATE INDEX IF NOT EXISTS idx_time_attempt_best ON competition_time_attempts(competition_id,side,pb_ms);
CREATE TABLE IF NOT EXISTS competition_fixed_series (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  seed_hex TEXT NOT NULL,
  projects_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS competition_duel_rooms (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  owner_user_id INTEGER NOT NULL,
  command_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  UNIQUE(owner_user_id, command_id)
);
CREATE INDEX IF NOT EXISTS idx_duel_owner_created ON competition_duel_rooms(owner_user_id, created_at);
CREATE TABLE IF NOT EXISTS competition_room_rules (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  rules_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS competition_draft_steps (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  step_index INTEGER NOT NULL DEFAULT 0,
  selected_json TEXT NOT NULL DEFAULT '[]',
  banned_json TEXT NOT NULL DEFAULT '[]',
  history_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS competition_schedule (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  starts_at TEXT NOT NULL, roster_revision INTEGER NOT NULL,
  yellow_team_id TEXT NOT NULL, white_team_id TEXT NOT NULL,
  yellow_name TEXT NOT NULL, white_name TEXT NOT NULL,
  attendance_resolved INTEGER NOT NULL DEFAULT 0,
  exception TEXT
);
CREATE TABLE IF NOT EXISTS competition_scheduled_players (
  competition_id TEXT NOT NULL REFERENCES competitions(id) ON DELETE CASCADE,
  user_id INTEGER NOT NULL, side TEXT NOT NULL, position INTEGER NOT NULL,
  display_name TEXT NOT NULL, arrived_at TEXT,
  PRIMARY KEY(competition_id,user_id), UNIQUE(competition_id,side,position)
);
CREATE TABLE IF NOT EXISTS competition_flow_rules (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id) ON DELETE CASCADE,
  version TEXT NOT NULL, team_clock_ms INTEGER NOT NULL,
  late_minutes INTEGER NOT NULL DEFAULT 15, ready_seconds INTEGER NOT NULL DEFAULT 60
);
CREATE TABLE IF NOT EXISTS tournament_roster_positions (
  event_slug TEXT NOT NULL, user_id INTEGER NOT NULL, position INTEGER NOT NULL CHECK(position BETWEEN 1 AND 16),
  PRIMARY KEY(event_slug,user_id)
);
CREATE TABLE IF NOT EXISTS competition_time_refunds (
  competition_id TEXT NOT NULL, game_key TEXT NOT NULL, side TEXT NOT NULL,
  decisive_ms INTEGER NOT NULL, amount_ms INTEGER NOT NULL,
  PRIMARY KEY(competition_id,game_key,side)
);
CREATE TABLE IF NOT EXISTS tournament_events (
  slug TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  rules TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'preparing' CHECK(status IN ('preparing','active','finished')),
  format_key TEXT NOT NULL DEFAULT 'team-draft-v1',
  owner_user_id INTEGER,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tournament_room_links (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id),
  event_slug TEXT NOT NULL REFERENCES tournament_events(slug),
  linked_by INTEGER NOT NULL,
  linked_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tournament_room_event ON tournament_room_links(event_slug);
CREATE TABLE IF NOT EXISTS tournament_statistics_config (
  event_slug TEXT PRIMARY KEY REFERENCES tournament_events(slug),
  starts_at TEXT NOT NULL, ends_at TEXT NOT NULL,
  roster_revision INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tournament_enrollment_config (
  event_slug TEXT PRIMARY KEY REFERENCES tournament_events(slug),
  mode TEXT NOT NULL CHECK(mode IN ('solo','self_team','organizer_team')),
  team_size INTEGER NOT NULL, capacity INTEGER NOT NULL DEFAULT 0,
  registration_open INTEGER NOT NULL DEFAULT 0,
  registration_locked INTEGER NOT NULL DEFAULT 0,
  roster_locked INTEGER NOT NULL DEFAULT 0,
  revision INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS tournament_teams (
  id TEXT PRIMARY KEY, event_slug TEXT NOT NULL REFERENCES tournament_events(slug),
  name TEXT NOT NULL, captain_user_id INTEGER, submitted INTEGER NOT NULL DEFAULT 0,
  UNIQUE(event_slug,name)
);
CREATE TABLE IF NOT EXISTS tournament_entrants (
  event_slug TEXT NOT NULL REFERENCES tournament_events(slug), user_id INTEGER NOT NULL,
  display_name TEXT NOT NULL, team_id TEXT REFERENCES tournament_teams(id),
  is_external INTEGER NOT NULL DEFAULT 0, source TEXT NOT NULL,
  PRIMARY KEY(event_slug,user_id)
);
CREATE TABLE IF NOT EXISTS tournament_team_invites (
  team_id TEXT NOT NULL REFERENCES tournament_teams(id), user_id INTEGER NOT NULL,
  display_name TEXT NOT NULL, PRIMARY KEY(team_id,user_id)
);
CREATE TABLE IF NOT EXISTS tournament_enrollment_audit (
  event_slug TEXT NOT NULL, revision INTEGER NOT NULL, actor_user_id INTEGER NOT NULL,
  action TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(event_slug,revision)
);
CREATE TABLE IF NOT EXISTS tournament_statistics_roster (
  event_slug TEXT NOT NULL REFERENCES tournament_events(slug),
  user_id INTEGER NOT NULL, display_name TEXT NOT NULL,
  team_name TEXT NOT NULL, is_external INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY(event_slug,user_id)
);
CREATE TABLE IF NOT EXISTS tournament_roster_audit (
  event_slug TEXT NOT NULL, revision INTEGER NOT NULL, actor_user_id INTEGER NOT NULL,
  payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(event_slug,revision)
);
INSERT OR IGNORE INTO tournament_events(slug,name,description,rules,format_key,created_at)
VALUES ('14360-cup-1','第一届14360杯','20 位选手，四队各五人，其中一位外援。仅统计 Table 对局站原生 3×3 成绩。',
  '比赛时间：北京时间 2026 年 10 月 1 日 00:00 至 10 月 8 日 00:00（结束时刻不含）。对局必须在比赛期间开始并完成。每人按得分取最佳五局，以其盘面和计算个人与团队成绩。外站对局无效。分队结果由举办方导入，本赛事不创建对战房间。',
  'team-top5-3x3-v1',strftime('%Y-%m-%dT%H:%M:%fZ','now'));
INSERT OR IGNORE INTO tournament_statistics_config(event_slug,starts_at,ends_at)
VALUES('14360-cup-1','2026-10-01T00:00:00+08:00','2026-10-08T00:00:00+08:00');
INSERT OR IGNORE INTO tournament_events(slug, name, description, rules, created_at)
VALUES ('819984-cup-3', '第三届819984杯', '2048 团队赛事。赛事公告、比赛房间与后续参赛服务将在此汇集。',
        '双方各三名选手，通过抽签、选 Ban 和秘密布阵，依次进行 A、B、C 三个项目的对决。具体项目规则以房间内公布的规则为准。整届赛事的分组和晋级安排以举办方公告为准。',
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now'));
CREATE TABLE IF NOT EXISTS competition_expulsions (
  competition_id TEXT NOT NULL, user_id INTEGER NOT NULL,
  expelled_by INTEGER NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, user_id)
);
CREATE TABLE IF NOT EXISTS competition_stage_holds (
  competition_id TEXT NOT NULL, stage TEXT NOT NULL, until_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, stage)
);
CREATE TABLE IF NOT EXISTS competition_schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS competitions (
  id TEXT PRIMARY KEY,
  public_key TEXT UNIQUE,
  room_code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  status TEXT NOT NULL,
  created_by_user_id INTEGER NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  live_generation INTEGER NOT NULL DEFAULT 1,
  live_started_at TEXT,
  live_ended_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS competition_staff (
  competition_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  role TEXT NOT NULL,
  assigned_by_user_id INTEGER NOT NULL,
  assigned_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, user_id, role),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (role IN ('organizer', 'referee'))
);

CREATE TABLE IF NOT EXISTS competition_prediction_windows (
  competition_id TEXT PRIMARY KEY REFERENCES competitions(id),
  opened_at TEXT NOT NULL,
  minimum_until TEXT NOT NULL,
  closed_at TEXT
);

CREATE TABLE IF NOT EXISTS competition_seats (
  competition_id TEXT NOT NULL,
  side TEXT NOT NULL,
  position INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  display_name_snapshot TEXT NOT NULL,
  seated_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, side, position),
  UNIQUE (competition_id, user_id),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (side IN ('yellow', 'white')),
  CHECK (position BETWEEN 1 AND 16)
);

CREATE TABLE IF NOT EXISTS competition_team_readiness (
  competition_id TEXT NOT NULL,
  side TEXT NOT NULL,
  ready_by_user_id INTEGER NOT NULL,
  ready_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, side),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (side IN ('yellow', 'white'))
);

CREATE TABLE IF NOT EXISTS competition_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  competition_id TEXT NOT NULL,
  sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  actor_user_id INTEGER,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  UNIQUE (competition_id, sequence),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS competition_commands (
  competition_id TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  command_id TEXT NOT NULL,
  action TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, user_id, command_id),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS competition_projects (
  competition_id TEXT NOT NULL,
  project_key TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  project_ref TEXT NOT NULL DEFAULT 'standard-2048-test',
  adapter_rules_version TEXT NOT NULL DEFAULT 'standard-v1',
  rules_version TEXT NOT NULL DEFAULT 'pending',
  adapter_snapshot_json TEXT NOT NULL DEFAULT '{}',
  sort_order INTEGER NOT NULL,
  enabled INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (competition_id, project_key),
  UNIQUE (competition_id, sort_order),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS competition_drafts (
  competition_id TEXT PRIMARY KEY,
  random_seed_hex TEXT NOT NULL,
  commitment TEXT NOT NULL,
  algorithm_version TEXT NOT NULL,
  first_side TEXT NOT NULL,
  phase_token TEXT NOT NULL,
  phase_started_at TEXT NOT NULL,
  yellow_deadline_at TEXT,
  white_deadline_at TEXT,
  project_a TEXT,
  ban_m TEXT,
  project_b TEXT,
  ban_n TEXT,
  blind_yellow TEXT,
  blind_white TEXT,
  project_c TEXT,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (first_side IN ('yellow', 'white'))
);

CREATE TABLE IF NOT EXISTS competition_lineup_state (
  competition_id TEXT PRIMARY KEY,
  phase_token TEXT NOT NULL,
  phase_started_at TEXT NOT NULL,
  yellow_deadline_at TEXT NOT NULL,
  white_deadline_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS competition_lineups (
  competition_id TEXT NOT NULL,
  side TEXT NOT NULL,
  game_key TEXT NOT NULL,
  position INTEGER NOT NULL,
  player_user_id INTEGER NOT NULL,
  automatic INTEGER NOT NULL DEFAULT 0,
  submitted_by_user_id INTEGER,
  submitted_at TEXT NOT NULL,
  revealed_at TEXT,
  PRIMARY KEY (competition_id, side, game_key),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (side IN ('yellow', 'white')),
  CHECK (length(game_key) = 1 AND game_key BETWEEN 'A' AND 'O'),
  CHECK (position BETWEEN 1 AND 16),
  CHECK (automatic IN (0, 1))
);

CREATE TABLE IF NOT EXISTS competition_match_control (
  competition_id TEXT PRIMARY KEY,
  current_game_key TEXT NOT NULL,
  phase_token TEXT NOT NULL,
  yellow_wins INTEGER NOT NULL DEFAULT 0,
  white_wins INTEGER NOT NULL DEFAULT 0,
  draws INTEGER NOT NULL DEFAULT 0,
  winner_side TEXT,
  finish_reason TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (length(current_game_key) = 1 AND current_game_key BETWEEN 'A' AND 'O')
);

CREATE TABLE IF NOT EXISTS competition_game_readiness (
  competition_id TEXT NOT NULL,
  game_key TEXT NOT NULL,
  side TEXT NOT NULL,
  player_ready_by_user_id INTEGER,
  player_ready_at TEXT,
  captain_ready_by_user_id INTEGER,
  captain_ready_at TEXT,
  PRIMARY KEY (competition_id, game_key, side),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (length(game_key) = 1 AND game_key BETWEEN 'A' AND 'O'),
  CHECK (side IN ('yellow', 'white'))
);

CREATE TABLE IF NOT EXISTS competition_team_clocks (
  competition_id TEXT NOT NULL,
  side TEXT NOT NULL,
  remaining_ms_base INTEGER NOT NULL,
  running_since TEXT,
  state TEXT NOT NULL,
  resume_after_suspension INTEGER NOT NULL DEFAULT 0,
  revision INTEGER NOT NULL DEFAULT 1,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, side),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (side IN ('yellow', 'white')),
  CHECK (state IN ('stopped', 'running', 'expired')),
  CHECK (resume_after_suspension IN (0, 1)),
  CHECK (remaining_ms_base >= 0)
);

CREATE TABLE IF NOT EXISTS competition_game_sessions (
  competition_id TEXT NOT NULL,
  game_key TEXT NOT NULL,
  side TEXT NOT NULL,
  project_key TEXT NOT NULL,
  project_ref TEXT NOT NULL DEFAULT 'standard-2048-test',
  rules_version TEXT NOT NULL DEFAULT 'standard-v1',
  instance_id TEXT NOT NULL DEFAULT '',
  public_generation INTEGER NOT NULL DEFAULT 1,
  player_user_id INTEGER NOT NULL,
  seed_hex TEXT NOT NULL,
  board_json TEXT NOT NULL,
  score INTEGER NOT NULL DEFAULT 0,
  move_count INTEGER NOT NULL DEFAULT 0,
  rng_counter INTEGER NOT NULL DEFAULT 0,
  adapter_state_json TEXT NOT NULL DEFAULT '{}',
  state TEXT NOT NULL,
  outcome_reason TEXT,
  started_at TEXT NOT NULL,
  completed_at TEXT,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, game_key, side),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (length(game_key) = 1 AND game_key BETWEEN 'A' AND 'O'),
  CHECK (side IN ('yellow', 'white')),
  CHECK (state IN ('playing', 'completed'))
);

CREATE TABLE IF NOT EXISTS competition_live_outbox (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  competition_id TEXT NOT NULL,
  generation INTEGER NOT NULL,
  sequence INTEGER NOT NULL,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  delivered_at TEXT,
  UNIQUE (competition_id, generation, sequence),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS competition_game_results (
  competition_id TEXT NOT NULL,
  game_key TEXT NOT NULL,
  yellow_score INTEGER NOT NULL,
  white_score INTEGER NOT NULL,
  winner_side TEXT NOT NULL,
  reason TEXT NOT NULL,
  result_revision INTEGER NOT NULL DEFAULT 1,
  corrected_by_user_id INTEGER,
  correction_reason TEXT,
  corrected_at TEXT,
  published_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, game_key),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (length(game_key) = 1 AND game_key BETWEEN 'A' AND 'O'),
  CHECK (winner_side IN ('yellow', 'white', 'draw'))
);

CREATE TABLE IF NOT EXISTS competition_suspensions (
  competition_id TEXT PRIMARY KEY,
  active INTEGER NOT NULL DEFAULT 0,
  reason_code TEXT,
  reason_text TEXT,
  started_by_user_id INTEGER,
  started_by_display_name TEXT,
  started_at TEXT,
  ended_by_user_id INTEGER,
  ended_at TEXT,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS competition_suspension_readiness (
  competition_id TEXT NOT NULL,
  side TEXT NOT NULL,
  ready_by_user_id INTEGER NOT NULL,
  ready_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, side),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (side IN ('yellow', 'white'))
);

CREATE TABLE IF NOT EXISTS competition_issue_reports (
  id TEXT PRIMARY KEY,
  competition_id TEXT NOT NULL,
  reported_by_user_id INTEGER NOT NULL,
  category TEXT NOT NULL,
  details TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',
  created_at TEXT NOT NULL,
  resolved_by_user_id INTEGER,
  resolution_note TEXT,
  resolved_at TEXT,
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (category IN ('network_device', 'project', 'rules', 'other')),
  CHECK (status IN ('open', 'resolved', 'dismissed'))
);

CREATE TABLE IF NOT EXISTS competition_result_confirmations (
  competition_id TEXT NOT NULL,
  game_key TEXT NOT NULL,
  side TEXT NOT NULL,
  result_revision INTEGER NOT NULL,
  confirmed_by_user_id INTEGER NOT NULL,
  confirmed_at TEXT NOT NULL,
  PRIMARY KEY (competition_id, game_key, side),
  FOREIGN KEY (competition_id) REFERENCES competitions(id) ON DELETE CASCADE,
  CHECK (length(game_key) = 1 AND game_key BETWEEN 'A' AND 'O'),
  CHECK (side IN ('yellow', 'white'))
);

CREATE TABLE IF NOT EXISTS practice_bests (
  project_id TEXT NOT NULL,
  rules_version INTEGER NOT NULL,
  user_id INTEGER NOT NULL,
  display_name TEXT NOT NULL,
  result_value INTEGER NOT NULL,
  elapsed_ms INTEGER NOT NULL,
  outcome TEXT NOT NULL,
  achieved_at TEXT NOT NULL,
  PRIMARY KEY (project_id, rules_version, user_id)
);

CREATE INDEX IF NOT EXISTS competition_staff_user_idx
  ON competition_staff(user_id, competition_id);
CREATE INDEX IF NOT EXISTS competition_seats_user_idx
  ON competition_seats(user_id, competition_id);
CREATE INDEX IF NOT EXISTS competition_events_room_idx
  ON competition_events(competition_id, sequence);
CREATE INDEX IF NOT EXISTS competition_projects_order_idx
  ON competition_projects(competition_id, sort_order);
CREATE INDEX IF NOT EXISTS competition_lineups_room_idx
  ON competition_lineups(competition_id, side, game_key);
CREATE INDEX IF NOT EXISTS competition_sessions_room_idx
  ON competition_game_sessions(competition_id, game_key, side);
CREATE INDEX IF NOT EXISTS competition_results_room_idx
  ON competition_game_results(competition_id, game_key);
CREATE INDEX IF NOT EXISTS competition_issues_room_idx
  ON competition_issue_reports(competition_id, status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS competition_open_issue_dedupe_idx
  ON competition_issue_reports(competition_id, reported_by_user_id, category)
  WHERE status = 'open';
CREATE INDEX IF NOT EXISTS competition_live_outbox_pending_idx
  ON competition_live_outbox(delivered_at, id);
CREATE INDEX IF NOT EXISTS practice_bests_project_idx
  ON practice_bests(project_id, rules_version);
"""


class CompetitionDatabase:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=8.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 8000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            if immediate:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        setup = sqlite3.connect(self.path, timeout=8.0)
        try:
            setup.execute("PRAGMA journal_mode = WAL")
            setup.execute("PRAGMA synchronous = NORMAL")
            setup.executescript(SCHEMA)
            # Existing grouped statistics rosters are preserved once; future edits use enrollment.
            setup.execute("""INSERT OR IGNORE INTO tournament_enrollment_config(event_slug,mode,team_size,capacity)
                SELECT slug,CASE WHEN format_key='team-top5-3x3-v1' THEN 'organizer_team' ELSE 'self_team' END,
                  CASE WHEN format_key='team-top5-3x3-v1' THEN 5 ELSE 3 END,
                  CASE WHEN format_key='team-top5-3x3-v1' THEN 20 ELSE 0 END FROM tournament_events""")
            setup.execute("""INSERT OR IGNORE INTO tournament_teams(id,event_slug,name)
                SELECT 'legacy:'||r.event_slug||':'||r.team_name,r.event_slug,r.team_name
                FROM tournament_statistics_roster r JOIN tournament_enrollment_config c ON c.event_slug=r.event_slug
                WHERE c.revision=0 GROUP BY r.event_slug,r.team_name""")
            setup.execute("""INSERT OR IGNORE INTO tournament_entrants
                SELECT r.event_slug,r.user_id,r.display_name,'legacy:'||r.event_slug||':'||r.team_name,r.is_external,'imported'
                FROM tournament_statistics_roster r JOIN tournament_enrollment_config c ON c.event_slug=r.event_slug
                WHERE c.revision=0""")
            setup.execute("""UPDATE tournament_enrollment_config SET revision=1 WHERE revision=0
                AND EXISTS(SELECT 1 FROM tournament_entrants e WHERE e.event_slug=tournament_enrollment_config.event_slug)""")
            competition_columns = {
                str(row[1])
                for row in setup.execute("PRAGMA table_info(competitions)").fetchall()
            }
            for column, definition in (
                ("public_key", "TEXT"),
                ("live_generation", "INTEGER NOT NULL DEFAULT 1"),
                ("live_started_at", "TEXT"),
                ("live_ended_at", "TEXT"),
            ):
                if column not in competition_columns:
                    setup.execute(f"ALTER TABLE competitions ADD COLUMN {column} {definition}")
            setup.execute(
                """
                UPDATE competitions
                SET public_key = 'm' || substr(replace(id, '-', ''), 1, 20)
                WHERE public_key IS NULL OR public_key = ''
                """
            )
            setup.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS competitions_public_key_idx ON competitions(public_key)"
            )
            setup.execute(
                """
                UPDATE competitions
                SET live_started_at = updated_at
                WHERE live_started_at IS NULL
                  AND status NOT IN ('CREATED', 'SEATING', 'READY_CHECK')
                """
            )
            setup.execute(
                """
                UPDATE competitions
                SET live_ended_at = updated_at
                WHERE live_ended_at IS NULL AND status IN ('FINISHED', 'CANCELLED')
                """
            )
            project_columns = {
                str(row[1])
                for row in setup.execute(
                    "PRAGMA table_info(competition_projects)"
                ).fetchall()
            }
            for column, definition in (
                ("project_ref", "TEXT NOT NULL DEFAULT 'standard-2048-test'"),
                ("adapter_rules_version", "TEXT NOT NULL DEFAULT 'standard-v1'"),
                ("adapter_snapshot_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                if column not in project_columns:
                    setup.execute(
                        f"ALTER TABLE competition_projects ADD COLUMN {column} {definition}"
                    )
            setup.execute(
                """
                UPDATE competition_projects
                SET rules_version = 'standard-v1'
                WHERE rules_version = 'pending'
                """
            )
            session_columns = {
                str(row[1])
                for row in setup.execute(
                    "PRAGMA table_info(competition_game_sessions)"
                ).fetchall()
            }
            for column, definition in (
                ("project_ref", "TEXT NOT NULL DEFAULT 'standard-2048-test'"),
                ("rules_version", "TEXT NOT NULL DEFAULT 'standard-v1'"),
                ("instance_id", "TEXT NOT NULL DEFAULT ''"),
                ("public_generation", "INTEGER NOT NULL DEFAULT 1"),
                ("adapter_state_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                if column not in session_columns:
                    setup.execute(
                        f"ALTER TABLE competition_game_sessions ADD COLUMN {column} {definition}"
                    )
            setup.execute(
                """
                UPDATE competition_game_sessions
                SET instance_id = competition_id || ':' || game_key || ':' || side
                WHERE instance_id = ''
                """
            )
            setup.execute(
                """
                INSERT OR IGNORE INTO competition_live_outbox
                  (competition_id, generation, sequence, event_type,
                   payload_json, created_at)
                SELECT id, live_generation, 1, 'projection.snapshot_required',
                       json_object('phase', status, 'revision', version), updated_at
                FROM competitions WHERE live_started_at IS NOT NULL
                """
            )
            clock_columns = {
                str(row[1])
                for row in setup.execute(
                    "PRAGMA table_info(competition_team_clocks)"
                ).fetchall()
            }
            if "resume_after_suspension" not in clock_columns:
                setup.execute(
                    """
                    ALTER TABLE competition_team_clocks
                    ADD COLUMN resume_after_suspension INTEGER NOT NULL DEFAULT 0
                    """
                )
            result_columns = {
                str(row[1])
                for row in setup.execute(
                    "PRAGMA table_info(competition_game_results)"
                ).fetchall()
            }
            for column, definition in (
                ("corrected_by_user_id", "INTEGER"),
                ("correction_reason", "TEXT"),
                ("corrected_at", "TEXT"),
            ):
                if column not in result_columns:
                    setup.execute(
                        f"ALTER TABLE competition_game_results ADD COLUMN {column} {definition}"
                    )
            suspension_columns = {
                str(row[1])
                for row in setup.execute(
                    "PRAGMA table_info(competition_suspensions)"
                ).fetchall()
            }
            if "started_by_display_name" not in suspension_columns:
                setup.execute(
                    "ALTER TABLE competition_suspensions ADD COLUMN started_by_display_name TEXT"
                )
            setup.execute(
                """
                INSERT OR IGNORE INTO competition_suspensions
                  (competition_id, active, updated_at)
                SELECT competition_id, 0, updated_at
                FROM competition_match_control
                """
            )
            from .room_rules_migration import migrate_room_constraints
            migrate_room_constraints(setup)
            setup.execute(
                "INSERT OR REPLACE INTO competition_schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            setup.commit()
        finally:
            setup.close()
