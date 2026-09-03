from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.profile.validation import canonical_display_name_key


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTH_DB = PROJECT_ROOT / "docs_and_configs" / "cloud_runtime" / "auth.sqlite3"
PLUS_ALIAS_DOMAINS = {
    "gmail.com",
    "googlemail.com",
    "outlook.com",
    "hotmail.com",
    "icloud.com",
}


def _sqlite_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("CLOUD_SQLITE_BUSY_TIMEOUT_SECONDS", "8")))
    except ValueError:
        return 8.0


def _canonical_email_identity(email: str) -> str:
    value = str(email or "").strip().lower()
    local, separator, domain = value.rpartition("@")
    if not separator or not local or not domain or "@" in local:
        return value
    if domain == "googlemail.com":
        domain = "gmail.com"
    if domain == "gmail.com":
        local = local.split("+", 1)[0].replace(".", "")
    elif domain in PLUS_ALIAS_DOMAINS:
        local = local.split("+", 1)[0]
    return f"{local}@{domain}"


def get_auth_db_path() -> Path:
    return Path(os.getenv("CLOUD_AUTH_DB") or DEFAULT_AUTH_DB)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def auth_db() -> Iterator[sqlite3.Connection]:
    path = get_auth_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout_seconds = _sqlite_timeout_seconds()
    connection = sqlite3.connect(path, timeout=timeout_seconds)
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = NORMAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_auth_db() -> None:
    path = get_auth_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout_seconds = _sqlite_timeout_seconds()
    setup = sqlite3.connect(path, timeout=timeout_seconds)
    try:
        setup.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        setup.execute("PRAGMA journal_mode = WAL")
        setup.execute("PRAGMA synchronous = NORMAL")
        setup.commit()
    finally:
        setup.close()
    with auth_db() as db:
        profile_table_existed = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'user_profiles'"
        ).fetchone() is not None
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT NOT NULL UNIQUE,
              email_identity TEXT,
              email_verified_at TEXT,
              password_hash TEXT NOT NULL,
              display_name TEXT,
              display_name_key TEXT,
              registered_with_invite INTEGER NOT NULL DEFAULT 1,
              invite_code_id INTEGER,
              role TEXT NOT NULL DEFAULT 'user',
              status TEXT NOT NULL DEFAULT 'active',
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              last_login_at TEXT,
              FOREIGN KEY(invite_code_id) REFERENCES invite_codes(id)
            );

            CREATE TABLE IF NOT EXISTS sessions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_token_hash TEXT NOT NULL UNIQUE,
              user_agent TEXT,
              ip_address TEXT,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              revoked_at TEXT,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS guest_sessions (
              guest_id TEXT PRIMARY KEY,
              token_hash TEXT NOT NULL UNIQUE,
              display_name TEXT NOT NULL,
              created_ip_hash TEXT NOT NULL,
              last_ip_hash TEXT NOT NULL,
              created_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              revoked_at TEXT
            );

            CREATE TABLE IF NOT EXISTS guest_query_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              guest_id TEXT NOT NULL,
              request_id TEXT NOT NULL,
              full_pattern TEXT NOT NULL,
              ip_bucket_hash TEXT NOT NULL,
              status TEXT NOT NULL DEFAULT 'reserved',
              created_at TEXT NOT NULL,
              finalized_at TEXT,
              UNIQUE(guest_id, request_id),
              FOREIGN KEY(guest_id) REFERENCES guest_sessions(guest_id) ON DELETE CASCADE,
              CHECK(status IN ('reserved', 'consumed', 'cancelled'))
            );

            CREATE TABLE IF NOT EXISTS refresh_tokens (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              token_hash TEXT NOT NULL UNIQUE,
              session_id INTEGER,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              revoked_at TEXT,
              replaced_by_id INTEGER,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS invite_codes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              code_hash TEXT NOT NULL UNIQUE,
              label TEXT,
              created_by_user_id INTEGER,
              max_uses INTEGER NOT NULL DEFAULT 1,
              used_count INTEGER NOT NULL DEFAULT 0,
              allowed_email TEXT,
              allowed_domain TEXT,
              expires_at TEXT,
              disabled_at TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(created_by_user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS email_verification_codes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT NOT NULL,
              purpose TEXT NOT NULL,
              code_hash TEXT NOT NULL,
              invite_code_id INTEGER,
              attempts INTEGER NOT NULL DEFAULT 0,
              max_attempts INTEGER NOT NULL DEFAULT 5,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              consumed_at TEXT,
              ip_address TEXT,
              FOREIGN KEY(invite_code_id) REFERENCES invite_codes(id)
            );

            CREATE TABLE IF NOT EXISTS auth_browser_cooldowns (
              browser_token_hash TEXT NOT NULL,
              purpose TEXT NOT NULL,
              last_sent_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(browser_token_hash, purpose)
            );

            CREATE TABLE IF NOT EXISTS user_quotas (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              quota_key TEXT NOT NULL,
              period TEXT NOT NULL DEFAULT 'lifetime',
              limit_value INTEGER NOT NULL DEFAULT 0,
              used_value INTEGER NOT NULL DEFAULT 0,
              resets_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              UNIQUE(user_id, quota_key, period),
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS usage_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_id INTEGER,
              event_type TEXT NOT NULL,
              quota_key TEXT,
              cost INTEGER NOT NULL DEFAULT 0,
              metadata_json TEXT,
              ip_address TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS token_accounts (
              user_id INTEGER PRIMARY KEY,
              bonus_balance_units INTEGER NOT NULL DEFAULT 0,
              paid_balance_units INTEGER NOT NULL DEFAULT 0,
              last_weekly_grant_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS user_entitlements (
              user_id INTEGER PRIMARY KEY,
              tier TEXT NOT NULL DEFAULT 'free',
              supporter_since TEXT,
              supporter_until TEXT,
              show_supporter_badge INTEGER NOT NULL DEFAULT 1,
              can_upload_avatar INTEGER NOT NULL DEFAULT 1,
              notes TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS user_profiles (
              user_id INTEGER PRIMARY KEY,
              avatar_key TEXT,
              avatar_sha256 TEXT,
              avatar_changed_at TEXT,
              display_name_changed_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS user_profile_change_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              change_type TEXT NOT NULL,
              old_value TEXT,
              new_value TEXT,
              ip_address TEXT,
              user_agent TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS token_ledger (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              session_id INTEGER,
              event_type TEXT NOT NULL,
              operation_key TEXT,
              table_pattern TEXT,
              table_multiplier_units INTEGER NOT NULL DEFAULT 1000,
              base_cost_units INTEGER NOT NULL DEFAULT 0,
              final_cost_units INTEGER NOT NULL DEFAULT 0,
              bonus_delta_units INTEGER NOT NULL DEFAULT 0,
              paid_delta_units INTEGER NOT NULL DEFAULT 0,
              balance_before_units INTEGER NOT NULL DEFAULT 0,
              balance_after_units INTEGER NOT NULL DEFAULT 0,
              metadata_json TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS token_operation_requests (
              request_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              session_id INTEGER,
              operation_key TEXT NOT NULL,
              ledger_id INTEGER,
              created_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(session_id) REFERENCES sessions(id),
              FOREIGN KEY(ledger_id) REFERENCES token_ledger(id)
            );

            CREATE TABLE IF NOT EXISTS token_reservation_settlements (
              reservation_ledger_id INTEGER PRIMARY KEY,
              settlement_type TEXT NOT NULL,
              settlement_ledger_id INTEGER,
              created_at TEXT NOT NULL,
              FOREIGN KEY(reservation_ledger_id) REFERENCES token_ledger(id),
              FOREIGN KEY(settlement_ledger_id) REFERENCES token_ledger(id),
              CHECK(settlement_type IN ('finalize', 'cancel'))
            );

            CREATE TABLE IF NOT EXISTS uploads (
              upload_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              session_id INTEGER,
              kind TEXT NOT NULL,
              filename TEXT NOT NULL,
              path TEXT NOT NULL,
              size INTEGER NOT NULL,
              content_type TEXT,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS analysis_jobs (
              job_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              session_id INTEGER,
              pattern TEXT NOT NULL,
              target TEXT NOT NULL,
              status TEXT NOT NULL,
              total INTEGER NOT NULL DEFAULT 0,
              done INTEGER NOT NULL DEFAULT 0,
              failed INTEGER NOT NULL DEFAULT 0,
              output_dir TEXT,
              zip_path TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS leaderboard_snapshots (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              board_key TEXT NOT NULL,
              period_start TEXT,
              period_end TEXT NOT NULL,
              generated_at TEXT NOT NULL,
              entry_count INTEGER NOT NULL DEFAULT 0,
              UNIQUE(board_key, period_end)
            );

            CREATE TABLE IF NOT EXISTS leaderboard_entries (
              snapshot_id INTEGER NOT NULL,
              user_id INTEGER NOT NULL,
              rank INTEGER NOT NULL,
              score_units INTEGER NOT NULL DEFAULT 0,
              display_name TEXT NOT NULL,
              is_supporter INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(snapshot_id, user_id),
              FOREIGN KEY(snapshot_id) REFERENCES leaderboard_snapshots(id) ON DELETE CASCADE,
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS gamer_ranked_runs (
              run_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              request_id TEXT NOT NULL,
              seed_hex TEXT NOT NULL,
              rules_version INTEGER NOT NULL,
              spawn_rate4_millis INTEGER NOT NULL DEFAULT 100,
              lease_token_hash TEXT,
              lease_expires_at TEXT,
              lease_last_seen_at TEXT,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              validation_started_at TEXT,
              submitted_at TEXT,
              completed_at TEXT,
              pending_record TEXT,
              claimed_score INTEGER,
              claimed_final_board TEXT,
              start_ip TEXT,
              submit_ip TEXT,
              error_code TEXT,
              board_key TEXT,
              new_personal_best INTEGER NOT NULL DEFAULT 0,
              UNIQUE(user_id, request_id),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS gamer_high_scores (
              user_id INTEGER NOT NULL,
              board_key TEXT NOT NULL,
              score INTEGER NOT NULL,
              max_tile INTEGER NOT NULL,
              move_count INTEGER NOT NULL,
              used_ai INTEGER NOT NULL DEFAULT 0,
              final_board TEXT NOT NULL,
              record_blob TEXT NOT NULL,
              replay_id TEXT NOT NULL UNIQUE,
              run_id TEXT NOT NULL UNIQUE,
              achieved_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(user_id, board_key),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
              FOREIGN KEY(run_id) REFERENCES gamer_ranked_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS gamer_weekly_high_scores (
              user_id INTEGER NOT NULL,
              board_key TEXT NOT NULL,
              week_start TEXT NOT NULL,
              score INTEGER NOT NULL,
              max_tile INTEGER NOT NULL,
              move_count INTEGER NOT NULL,
              used_ai INTEGER NOT NULL DEFAULT 0,
              final_board TEXT NOT NULL,
              record_blob TEXT NOT NULL,
              replay_id TEXT NOT NULL UNIQUE,
              run_id TEXT NOT NULL UNIQUE,
              achieved_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(user_id, board_key, week_start),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
              FOREIGN KEY(run_id) REFERENCES gamer_ranked_runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS minigame_high_scores (
              user_id INTEGER NOT NULL,
              game_id TEXT NOT NULL,
              difficulty INTEGER NOT NULL,
              best_score INTEGER NOT NULL DEFAULT 0,
              trophy_tier INTEGER NOT NULL DEFAULT 0,
              highest_tile_exp INTEGER NOT NULL DEFAULT 0,
              final_board_json TEXT NOT NULL,
              board_rows INTEGER NOT NULL DEFAULT 4,
              board_cols INTEGER NOT NULL DEFAULT 4,
              score_achieved_at TEXT NOT NULL,
              trophy_achieved_at TEXT,
              score_run_id TEXT,
              trophy_run_id TEXT,
              verification_level TEXT NOT NULL DEFAULT 'legacy',
              score_verified_at TEXT,
              trophy_verified_at TEXT,
              record_hash TEXT,
              record_blob TEXT,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(user_id, game_id, difficulty),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS minigame_ranked_runs (
              run_id TEXT PRIMARY KEY,
              user_id INTEGER NOT NULL,
              request_id TEXT NOT NULL,
              game_id TEXT NOT NULL,
              difficulty INTEGER NOT NULL,
              rules_version INTEGER NOT NULL,
              seed_salt_hex TEXT NOT NULL,
              seed_hex TEXT NOT NULL,
              lease_token_hash TEXT,
              lease_expires_at TEXT,
              lease_last_seen_at TEXT,
              lease_generation INTEGER NOT NULL DEFAULT 1,
              status TEXT NOT NULL,
              started_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              start_ip TEXT,
              qualified_at TEXT,
              qualification_expires_at TEXT,
              submission_token_hash TEXT,
              submission_token_consumed_at TEXT,
              claimed_summary_json TEXT,
              pending_record TEXT,
              record_hash TEXT,
              action_count INTEGER,
              submitted_at TEXT,
              submit_ip TEXT,
              validation_started_at TEXT,
              completed_at TEXT,
              verified_summary_json TEXT,
              error_code TEXT,
              UNIQUE(user_id, request_id),
              FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS minigame_ranked_checkpoints (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              run_id TEXT NOT NULL,
              revision INTEGER NOT NULL,
              status TEXT NOT NULL,
              claimed_summary_json TEXT NOT NULL,
              pending_record TEXT,
              record_hash TEXT NOT NULL,
              action_count INTEGER NOT NULL,
              submitted_at TEXT NOT NULL,
              submit_ip TEXT,
              validation_started_at TEXT,
              completed_at TEXT,
              verified_summary_json TEXT,
              error_code TEXT,
              UNIQUE(run_id, revision),
              UNIQUE(run_id, record_hash),
              FOREIGN KEY(run_id) REFERENCES minigame_ranked_runs(run_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
            CREATE INDEX IF NOT EXISTS idx_guest_sessions_expires
              ON guest_sessions(expires_at, revoked_at);
            CREATE INDEX IF NOT EXISTS idx_guest_sessions_created_ip
              ON guest_sessions(created_ip_hash, created_at);
            CREATE INDEX IF NOT EXISTS idx_guest_query_events_guest
              ON guest_query_events(guest_id, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_guest_query_events_ip
              ON guest_query_events(ip_bucket_hash, status, created_at);
            CREATE INDEX IF NOT EXISTS idx_usage_user_created ON usage_events(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_token_ledger_user_created ON token_ledger(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_token_operation_requests_user_created ON token_operation_requests(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_user_entitlements_tier ON user_entitlements(tier);
            CREATE INDEX IF NOT EXISTS idx_profile_changes_user_created
              ON user_profile_change_events(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user ON analysis_jobs(user_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_email_codes_email ON email_verification_codes(email, purpose);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_snapshots_key_generated
              ON leaderboard_snapshots(board_key, generated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_leaderboard_entries_rank
              ON leaderboard_entries(snapshot_id, rank);
            CREATE INDEX IF NOT EXISTS idx_gamer_runs_user_status
              ON gamer_ranked_runs(user_id, status, submitted_at);
            CREATE INDEX IF NOT EXISTS idx_gamer_runs_pending
              ON gamer_ranked_runs(status, submitted_at);
            CREATE INDEX IF NOT EXISTS idx_gamer_runs_start_ip
              ON gamer_ranked_runs(start_ip, started_at);
            CREATE INDEX IF NOT EXISTS idx_gamer_runs_submit_ip
              ON gamer_ranked_runs(submit_ip, submitted_at);
            CREATE INDEX IF NOT EXISTS idx_gamer_scores_board_score
              ON gamer_high_scores(board_key, score DESC, achieved_at ASC);
            CREATE INDEX IF NOT EXISTS idx_gamer_weekly_scores_period
              ON gamer_weekly_high_scores(
                board_key, week_start, score DESC, achieved_at ASC
              );
            CREATE INDEX IF NOT EXISTS idx_minigame_scores_game
              ON minigame_high_scores(
                game_id, difficulty, best_score DESC, score_achieved_at ASC
              );
            CREATE INDEX IF NOT EXISTS idx_minigame_scores_trophies
              ON minigame_high_scores(
                difficulty, trophy_tier DESC, trophy_achieved_at ASC
              );
            CREATE INDEX IF NOT EXISTS idx_minigame_runs_user_status
              ON minigame_ranked_runs(user_id, status, started_at);
            CREATE INDEX IF NOT EXISTS idx_minigame_runs_active_game
              ON minigame_ranked_runs(user_id, game_id, difficulty, status);
            CREATE INDEX IF NOT EXISTS idx_minigame_runs_pending
              ON minigame_ranked_runs(status, submitted_at);
            CREATE INDEX IF NOT EXISTS idx_minigame_runs_start_ip
              ON minigame_ranked_runs(start_ip, started_at);
            CREATE INDEX IF NOT EXISTS idx_minigame_checkpoints_pending
              ON minigame_ranked_checkpoints(status, submitted_at);
            CREATE INDEX IF NOT EXISTS idx_minigame_checkpoints_run_revision
              ON minigame_ranked_checkpoints(run_id, revision DESC);
            """
        )
        existing_user_columns = {
            row["name"] for row in db.execute("PRAGMA table_info(users)").fetchall()
        }
        if "password_changed_at" not in existing_user_columns:
            db.execute("ALTER TABLE users ADD COLUMN password_changed_at TEXT")
        if "deactivated_at" not in existing_user_columns:
            db.execute("ALTER TABLE users ADD COLUMN deactivated_at TEXT")
        if "email_identity" not in existing_user_columns:
            db.execute("ALTER TABLE users ADD COLUMN email_identity TEXT")
        if "registered_with_invite" not in existing_user_columns:
            db.execute("ALTER TABLE users ADD COLUMN registered_with_invite INTEGER NOT NULL DEFAULT 1")
        if "invite_code_id" not in existing_user_columns:
            db.execute("ALTER TABLE users ADD COLUMN invite_code_id INTEGER REFERENCES invite_codes(id)")
        if "display_name_key" not in existing_user_columns:
            db.execute("ALTER TABLE users ADD COLUMN display_name_key TEXT")

        existing_gamer_run_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(gamer_ranked_runs)").fetchall()
        }
        if "spawn_rate4_millis" not in existing_gamer_run_columns:
            db.execute(
                "ALTER TABLE gamer_ranked_runs "
                "ADD COLUMN spawn_rate4_millis INTEGER NOT NULL DEFAULT 100"
            )
        gamer_run_migrations = (
            ("lease_token_hash", "TEXT"),
            ("lease_expires_at", "TEXT"),
            ("lease_last_seen_at", "TEXT"),
        )
        for column_name, column_type in gamer_run_migrations:
            if column_name not in existing_gamer_run_columns:
                db.execute(
                    f"ALTER TABLE gamer_ranked_runs ADD COLUMN {column_name} {column_type}"
                )
        db.execute(
            """
            UPDATE gamer_ranked_runs
            SET status = 'expired',
                completed_at = COALESCE(completed_at, ?),
                error_code = COALESCE(error_code, 'lease_required')
            WHERE status = 'active'
              AND (lease_token_hash IS NULL OR lease_token_hash = '')
            """,
            (_iso_now(),),
        )

        existing_minigame_score_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(minigame_high_scores)").fetchall()
        }
        minigame_score_migrations = (
            ("score_run_id", "TEXT"),
            ("trophy_run_id", "TEXT"),
            ("verification_level", "TEXT NOT NULL DEFAULT 'legacy'"),
            ("score_verified_at", "TEXT"),
            ("trophy_verified_at", "TEXT"),
            ("record_hash", "TEXT"),
            ("record_blob", "TEXT"),
        )
        for column_name, column_type in minigame_score_migrations:
            if column_name not in existing_minigame_score_columns:
                db.execute(
                    f"ALTER TABLE minigame_high_scores ADD COLUMN {column_name} {column_type}"
                )

        existing_minigame_run_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(minigame_ranked_runs)").fetchall()
        }
        minigame_run_migrations = (
            ("lease_token_hash", "TEXT"),
            ("lease_expires_at", "TEXT"),
            ("lease_last_seen_at", "TEXT"),
            ("lease_generation", "INTEGER NOT NULL DEFAULT 1"),
        )
        for column_name, column_type in minigame_run_migrations:
            if column_name not in existing_minigame_run_columns:
                db.execute(
                    f"ALTER TABLE minigame_ranked_runs ADD COLUMN {column_name} {column_type}"
                )
        db.execute(
            """
            UPDATE minigame_ranked_runs
            SET status = 'expired',
                completed_at = COALESCE(completed_at, ?),
                error_code = COALESCE(error_code, 'lease_required')
            WHERE status IN ('active', 'qualified')
              AND (lease_token_hash IS NULL OR lease_token_hash = '')
            """,
            (_iso_now(),),
        )

        used_identities = {
            row["email_identity"]
            for row in db.execute(
                "SELECT email_identity FROM users WHERE email_identity IS NOT NULL AND email_identity != ''"
            ).fetchall()
        }
        for row in db.execute(
            "SELECT id, email FROM users WHERE email_identity IS NULL OR email_identity = '' ORDER BY id"
        ).fetchall():
            identity = _canonical_email_identity(row["email"])
            stored_identity = identity
            if stored_identity in used_identities:
                stored_identity = f"legacy-conflict:{row['id']}:{identity}"
            used_identities.add(stored_identity)
            db.execute(
                "UPDATE users SET email_identity = ? WHERE id = ?",
                (stored_identity, row["id"]),
            )

        db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_identity
            ON users(email_identity)
            WHERE email_identity IS NOT NULL
            """
        )

        used_display_names = {
            row["display_name_key"]
            for row in db.execute(
                "SELECT display_name_key FROM users WHERE display_name_key IS NOT NULL AND display_name_key != ''"
            ).fetchall()
        }
        for row in db.execute(
            "SELECT id, display_name FROM users WHERE display_name_key IS NULL OR display_name_key = '' ORDER BY id"
        ).fetchall():
            name_key = canonical_display_name_key(row["display_name"] or "") or f"user-{row['id']}"
            stored_key = name_key
            if stored_key in used_display_names:
                stored_key = f"legacy-conflict:{row['id']}:{name_key}"
            used_display_names.add(stored_key)
            db.execute(
                "UPDATE users SET display_name_key = ? WHERE id = ?",
                (stored_key, int(row["id"])),
            )

        db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_users_display_name_key
            ON users(display_name_key)
            WHERE display_name_key IS NOT NULL
            """
        )

        now = _iso_now()
        db.execute(
            """
            INSERT OR IGNORE INTO user_entitlements
            (user_id, tier, supporter_since, show_supporter_badge, can_upload_avatar, created_at, updated_at)
            SELECT
              users.id,
              CASE WHEN COALESCE(token_accounts.paid_balance_units, 0) > 0 THEN 'supporter' ELSE 'free' END,
              CASE WHEN COALESCE(token_accounts.paid_balance_units, 0) > 0 THEN ? ELSE NULL END,
              1,
              1,
              ?,
              ?
            FROM users
            LEFT JOIN token_accounts ON token_accounts.user_id = users.id
            """,
            (now, now, now),
        )
        if not profile_table_existed:
            db.execute(
                "UPDATE user_entitlements SET can_upload_avatar = 1, updated_at = ?",
                (now,),
            )
        db.execute(
            """
            INSERT OR IGNORE INTO user_profiles
            (user_id, display_name_changed_at, created_at, updated_at)
            SELECT id, created_at, ?, ? FROM users
            """,
            (now, now),
        )
        db.execute(
            """
            UPDATE user_entitlements
            SET tier = 'supporter',
                supporter_since = COALESCE(supporter_since, ?),
                show_supporter_badge = 1,
                updated_at = ?
            WHERE tier != 'supporter'
              AND user_id IN (
                SELECT user_id FROM token_accounts WHERE paid_balance_units > 0
              )
            """,
            (now, now),
        )
