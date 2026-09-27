from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
import secrets

from backend.auth.db import get_auth_db_path


def db_path():
    return Path(os.environ.get("HUMAN_PLAY_DB") or get_auth_db_path().with_name("human-play.sqlite3"))


@contextmanager
def database():
    db = sqlite3.connect(db_path(), timeout=10)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA synchronous=FULL")
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    db_path().parent.mkdir(parents=True, exist_ok=True)
    with database() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS human_runs (
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, browser TEXT NOT NULL,
            variant TEXT NOT NULL, request_id TEXT NOT NULL, seed TEXT NOT NULL,
            threshold INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'active',
            eligibility TEXT NOT NULL DEFAULT 'eligible', reason TEXT,
            created REAL NOT NULL, ended REAL, writer TEXT NOT NULL, epoch INTEGER NOT NULL DEFAULT 1,
            permit_until REAL NOT NULL DEFAULT 0, monitored INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL, archive BLOB,
            display_threshold INTEGER NOT NULL DEFAULT 0,
            visible INTEGER NOT NULL DEFAULT 1,
            deleted_by_user INTEGER NOT NULL DEFAULT 0,
            deleted_by_user_at REAL,
            has_replay INTEGER NOT NULL DEFAULT 1,
            single_rating REAL, single_rating_version INTEGER,
            UNIQUE(user_id, browser, request_id)
        );
        CREATE UNIQUE INDEX IF NOT EXISTS human_active_slot
            ON human_runs(user_id, browser, variant) WHERE status='active';
        CREATE INDEX IF NOT EXISTS human_player_history ON human_runs(user_id, ended DESC);
        CREATE INDEX IF NOT EXISTS human_user_created ON human_runs(user_id, created);
        CREATE INDEX IF NOT EXISTS human_board_scores
            ON human_runs(variant, json_extract(state,'$.score') DESC, ended, id) WHERE status='sealed';
        CREATE TABLE IF NOT EXISTS human_keys (id INTEGER PRIMARY KEY, secret BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS human_chunks (
            run_id TEXT NOT NULL REFERENCES human_runs(id), start INTEGER NOT NULL,
            count INTEGER NOT NULL, digest BLOB NOT NULL, previous_hash BLOB NOT NULL,
            data BLOB NOT NULL, received REAL NOT NULL,
            PRIMARY KEY(run_id,start)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS human_audit (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, kind TEXT NOT NULL,
            local_seq INTEGER, server_seq INTEGER, created REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS human_rank_approvals (
            run_id TEXT PRIMARY KEY REFERENCES human_runs(id),
            operator TEXT NOT NULL, note TEXT NOT NULL, created REAL NOT NULL,
            seq INTEGER NOT NULL, prefix_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS human_rank_reviews (
            id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES human_runs(id),
            approved INTEGER NOT NULL, operator TEXT NOT NULL, note TEXT NOT NULL,
            created REAL NOT NULL, seq INTEGER NOT NULL, prefix_hash TEXT NOT NULL
        );
        """)
        schema = db.execute("SELECT sql FROM sqlite_master WHERE name='human_chunks'").fetchone()[0]
        if 'WITHOUT ROWID' not in schema.upper():
            # One atomic, lossless schema migration, not a background compaction job.
            # Preserve every batch, boundary, timestamp and full 256-bit proof.
            db.execute('BEGIN IMMEDIATE')
            db.create_function('human_hash_bytes', 1, hash_bytes)
            db.execute('''CREATE TABLE human_chunks_compact (
                run_id TEXT NOT NULL REFERENCES human_runs(id), start INTEGER NOT NULL,
                count INTEGER NOT NULL, digest BLOB NOT NULL, previous_hash BLOB NOT NULL,
                data BLOB NOT NULL, received REAL NOT NULL, PRIMARY KEY(run_id,start)) WITHOUT ROWID''')
            db.execute('''INSERT INTO human_chunks_compact
                SELECT run_id,start,count,human_hash_bytes(digest),human_hash_bytes(previous_hash),data,received
                FROM human_chunks''')
            db.execute('DROP TABLE human_chunks')
            db.execute('ALTER TABLE human_chunks_compact RENAME TO human_chunks')
        db.execute("INSERT OR IGNORE INTO human_keys VALUES(1,?)", (secrets.token_bytes(32),))
        # HPR/gzip already carries CRC32 and length; reviews replay the entire game.
        # Drop only the redundant archive digest, keeping anti-rollback prefix hashes.
        columns = {row["name"] for row in db.execute("PRAGMA table_info(human_runs)")}
        if "display_threshold" not in columns:
            db.execute("ALTER TABLE human_runs ADD COLUMN display_threshold INTEGER NOT NULL DEFAULT 0")
        if "visible" not in columns:
            db.execute("ALTER TABLE human_runs ADD COLUMN visible INTEGER NOT NULL DEFAULT 1")
        if "deleted_by_user" not in columns:
            db.execute("ALTER TABLE human_runs ADD COLUMN deleted_by_user INTEGER NOT NULL DEFAULT 0")
        if "deleted_by_user_at" not in columns:
            db.execute("ALTER TABLE human_runs ADD COLUMN deleted_by_user_at REAL")
        if "has_replay" not in columns:
            db.execute("ALTER TABLE human_runs ADD COLUMN has_replay INTEGER NOT NULL DEFAULT 1")
        if "source" not in columns:
            db.execute("ALTER TABLE human_runs ADD COLUMN source TEXT NOT NULL DEFAULT 'native'")
        db.execute("""CREATE TABLE IF NOT EXISTS human_external_claims (
            id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, provider TEXT NOT NULL,
            username TEXT NOT NULL, username_key TEXT NOT NULL,
            status TEXT NOT NULL, requested REAL NOT NULL, updated REAL NOT NULL,
            approved_by INTEGER, approved_at REAL, proof_note TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '', counts TEXT NOT NULL DEFAULT '{}'
        )""")
        db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS human_external_claim_user
            ON human_external_claims(provider,user_id)
            WHERE status NOT IN ('rejected','cancelled')""")
        db.execute("""CREATE UNIQUE INDEX IF NOT EXISTS human_external_claim_name
            ON human_external_claims(provider,username_key)
            WHERE status IN ('approved','importing','complete')""")
        db.execute("""CREATE INDEX IF NOT EXISTS human_external_claim_status_updated
            ON human_external_claims(status,updated DESC)""")
        db.execute("""CREATE TABLE IF NOT EXISTS human_external_audit (
            id INTEGER PRIMARY KEY, claim_id INTEGER NOT NULL,
            operator_id INTEGER, action TEXT NOT NULL, note TEXT NOT NULL,
            created REAL NOT NULL
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS human_external_run_source
            ON human_runs(source,user_id,variant,ended) WHERE source!='native'""")
        if "archive_hash" in columns:
            db.execute("ALTER TABLE human_runs DROP COLUMN archive_hash")
        db.execute("""CREATE TABLE IF NOT EXISTS human_player_settings (
            user_id INTEGER PRIMARY KEY,
            display_thresholds TEXT NOT NULL DEFAULT '{}',
            timer_splits TEXT NOT NULL DEFAULT '{}'
        )""")
        settings_columns = {row[1] for row in db.execute("PRAGMA table_info(human_player_settings)")}
        if "timer_splits" not in settings_columns:
            db.execute("ALTER TABLE human_player_settings ADD COLUMN timer_splits TEXT NOT NULL DEFAULT '{}'")
        from backend.rolling_leaderboards import init_schema as init_rolling_schema
        init_rolling_schema(db)
        from .manual_archive import init_schema as init_manual_archive_schema
        init_manual_archive_schema(db)
        from .rating import init_schema as init_rating_schema
        init_rating_schema(db)
        from .analysis_summary import init_schema as init_analysis_summary_schema
        init_analysis_summary_schema(db)
        from .statistics import init_schema as init_statistics_schema
        init_statistics_schema(db)
        from .leaderboards import init_schema as init_full_leaderboards_schema
        init_full_leaderboards_schema(db)
    from .traffic import initialize
    initialize()


def hash_bytes(value):
    result = bytes.fromhex(value) if isinstance(value, str) else bytes(value)
    if len(result) != 32:
        raise ValueError('invalid_prefix_proof')
    return result
