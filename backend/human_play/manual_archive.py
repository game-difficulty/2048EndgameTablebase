"""Validated player-submitted replay applications and audited archival."""
from __future__ import annotations

import gzip
import json
import struct
import time
import uuid
import zlib

from . import engine, service, statistics, verse_replay
from .codec import gunzip_limited
from .store import database


MAX_INPUT = 2 * 1024 * 1024
MAX_PENDING = 3
MAX_DAILY = 10
MAX_GLOBAL_PENDING = 200
MIN_ENDED_AT = 1394323200.0  # 2048's original public release date.


def init_schema(db) -> None:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS human_archive_applications (
        id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL,
        variant TEXT NOT NULL, claimed_ended_at REAL NOT NULL,
        claimed_score INTEGER NOT NULL, status TEXT NOT NULL,
        replay_crc INTEGER NOT NULL, replay_size INTEGER NOT NULL,
        moves INTEGER NOT NULL, final_board_json TEXT NOT NULL,
        is_game_over INTEGER NOT NULL, timing_summary_json TEXT NOT NULL,
        warning_flags_json TEXT NOT NULL DEFAULT '[]',
        original_filename TEXT NOT NULL DEFAULT '',
        requested_at REAL NOT NULL, updated_at REAL NOT NULL,
        approved_by INTEGER, approved_at REAL, review_note TEXT NOT NULL DEFAULT '',
        run_id TEXT UNIQUE
    );
    CREATE INDEX IF NOT EXISTS human_archive_application_user
        ON human_archive_applications(user_id,requested_at DESC);
    CREATE INDEX IF NOT EXISTS human_archive_application_pending
        ON human_archive_applications(status,updated_at);
    CREATE INDEX IF NOT EXISTS human_archive_application_duplicate
        ON human_archive_applications(variant,claimed_score,moves,replay_crc,replay_size,status);
    CREATE TABLE IF NOT EXISTS human_archive_application_payloads (
        application_id INTEGER PRIMARY KEY REFERENCES human_archive_applications(id) ON DELETE CASCADE,
        archive BLOB NOT NULL
    );
    CREATE TABLE IF NOT EXISTS human_archive_application_audit (
        id INTEGER PRIMARY KEY,
        application_id INTEGER NOT NULL REFERENCES human_archive_applications(id),
        operator_id INTEGER, action TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS human_archive_application_audit_claim
        ON human_archive_application_audit(application_id,id);
    """)


def _payload(row) -> dict:
    return {"id": row["id"], "user_id": row["user_id"], "variant": row["variant"],
            "ended_at": row["claimed_ended_at"], "score": row["claimed_score"],
            "status": row["status"], "moves": row["moves"],
            "board": json.loads(row["final_board_json"]),
            "game_over": bool(row["is_game_over"]),
            "timing": json.loads(row["timing_summary_json"]),
            "warnings": json.loads(row["warning_flags_json"]),
            "filename": row["original_filename"], "requested_at": row["requested_at"],
            "updated_at": row["updated_at"], "review_note": row["review_note"],
            "run_id": row["run_id"]}


def _inspect_hpr(raw: bytes, expected_variant: str) -> dict:
    if raw.startswith(b"\x1f\x8b"):
        raw = gunzip_limited(raw, MAX_INPUT)
    header, events = engine.parse_replay(raw)
    if header["variant"] != expected_variant:
        raise ValueError("replay_variant_mismatch")
    initial_state = engine.initial(header["run_id"], expected_variant, header["seed"])
    rate_tracker = statistics.Rate32kTracker(expected_variant)
    rate_tracker.observe(initial_state)
    state = engine.advance(initial_state, expected_variant, events, observer=rate_tracker.observe)
    moves = []
    for offset in range(0, len(events), engine.EVENT.size):
        code, delta = engine.EVENT.unpack_from(events, offset)
        moves.append((code & 3, (code >> 2) & 15, 4 if code & 64 else 2, delta))
    normalized = verse_replay.encode_rpl1(expected_variant, initial_state["board"], moves)
    return {"normalized": normalized, "variant": expected_variant,
            "initial_board": initial_state["board"], "board": state["board"],
            "score": state["score"], "moves": state["seq"], "elapsed": state["elapsed"],
            "timed_moves": state["seq"], "nodes": state["nodes"],
            "spawn_count": state["spawnCount"], "four_count": state["fourCount"],
            "game_over": engine.game_over(state["board"], *engine.VARIANTS[expected_variant]),
            "rate": rate_tracker.result()}


def inspect_upload(raw: bytes, expected_variant: str, expected_score: int) -> dict:
    if expected_variant not in engine.VARIANTS:
        raise service.RunError("invalid_variant", 400)
    if type(expected_score) is not int or expected_score < 0 or expected_score > 2_000_000_000:
        raise service.RunError("invalid_score", 400)
    if not raw or len(raw) > MAX_INPUT:
        raise service.RunError("replay_size_invalid", 413)
    try:
        probe = gunzip_limited(raw, MAX_INPUT) if raw.startswith(b"\x1f\x8b") else raw
        result = (_inspect_hpr(raw, expected_variant)
                  if probe.startswith((b"HPR1", b"HPR2"))
                  else verse_replay.inspect_replay(raw, expected_variant))
    except (ValueError, KeyError, TypeError, OSError, EOFError, struct.error) as exc:
        raise service.RunError(str(exc) or "replay_invalid", 400) from exc
    if result["score"] != expected_score:
        raise service.RunError("replay_score_mismatch", 400,
                               calculated_score=result["score"])
    return result


def submit(user_id: int, variant: str, ended_at: float, score: int,
           filename: str, raw: bytes) -> dict:
    now = time.time()
    try:
        ended_at = float(ended_at)
    except (TypeError, ValueError) as exc:
        raise service.RunError("invalid_ended_at", 400) from exc
    if not MIN_ENDED_AT <= ended_at <= now + 300:
        raise service.RunError("invalid_ended_at", 400)
    result = inspect_upload(raw, variant, score)
    normalized = result["normalized"]
    archive = gzip.compress(normalized, compresslevel=6, mtime=0)
    replay_crc = zlib.crc32(normalized)
    warnings = [] if result["game_over"] else ["replay_not_game_over"]
    timing = {"elapsed_ms": result["elapsed"], "timed_moves": result["timed_moves"]}
    safe_filename = str(filename or "")[:180].replace("\x00", "")
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        pending = db.execute("""SELECT count(*) FROM human_archive_applications
            WHERE user_id=? AND status='pending'""", (user_id,)).fetchone()[0]
        if pending >= MAX_PENDING:
            raise service.RunError("archive_application_pending_limit", 429)
        if db.execute("""SELECT count(*) FROM human_archive_applications
                WHERE status='pending'""").fetchone()[0] >= MAX_GLOBAL_PENDING:
            raise service.RunError("archive_application_queue_full", 429)
        recent = db.execute("""SELECT count(*) FROM human_archive_applications
            WHERE user_id=? AND requested_at>?""", (user_id, now - 86400)).fetchone()[0]
        if recent >= MAX_DAILY:
            raise service.RunError("archive_application_rate_limit", 429)
        candidates = db.execute("""SELECT a.id,a.run_id,p.archive AS pending_archive,
                r.archive AS run_archive FROM human_archive_applications a
            LEFT JOIN human_archive_application_payloads p ON p.application_id=a.id
            LEFT JOIN human_runs r ON r.id=a.run_id
            WHERE a.variant=? AND a.claimed_score=? AND a.moves=? AND a.replay_crc=?
              AND a.replay_size=? AND a.status IN ('pending','approved')""",
            (variant, score, result["moves"], replay_crc, len(normalized))).fetchall()
        if any((row["pending_archive"] or row["run_archive"]) == archive for row in candidates):
            raise service.RunError("replay_already_submitted", 409)
        cursor = db.execute("""INSERT INTO human_archive_applications
            (user_id,variant,claimed_ended_at,claimed_score,status,replay_crc,replay_size,
             moves,final_board_json,is_game_over,timing_summary_json,warning_flags_json,
             original_filename,requested_at,updated_at)
            VALUES(?,?,?,?,'pending',?,?,?,?,?,?,?,?,?,?)""",
            (user_id, variant, ended_at, score, replay_crc, len(normalized), result["moves"],
             json.dumps(result["board"], separators=(",", ":")), int(result["game_over"]),
             json.dumps(timing, separators=(",", ":")), json.dumps(warnings),
             safe_filename, now, now))
        application_id = cursor.lastrowid
        db.execute("INSERT INTO human_archive_application_payloads VALUES(?,?)",
                   (application_id, archive))
        db.execute("""INSERT INTO human_archive_application_audit
            (application_id,operator_id,action,note,created_at)
            VALUES(?,NULL,'submitted','',?)""", (application_id, now))
        row = db.execute("SELECT * FROM human_archive_applications WHERE id=?",
                         (application_id,)).fetchone()
        return _payload(row)


def mine(user_id: int) -> list[dict]:
    with database() as db:
        rows = db.execute("""SELECT * FROM human_archive_applications
            WHERE user_id=? ORDER BY requested_at DESC LIMIT 50""", (user_id,)).fetchall()
    return [_payload(row) for row in rows]


def pending(user_id: int | None = None) -> list[dict]:
    with database() as db:
        rows = db.execute("""SELECT * FROM human_archive_applications
            WHERE (? IS NULL OR user_id=?)
            ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END,updated_at DESC LIMIT 100""",
            (user_id, user_id)).fetchall()
    return [_payload(row) for row in rows]


def pending_approval_user_ids() -> set[int]:
    with database() as db:
        return {int(row[0]) for row in db.execute("""SELECT DISTINCT user_id
            FROM human_archive_applications WHERE status='pending'""")}


def _validated_saved(db, application_id: int):
    row = db.execute("""SELECT a.*,p.archive FROM human_archive_applications a
        JOIN human_archive_application_payloads p ON p.application_id=a.id WHERE a.id=?""",
        (application_id,)).fetchone()
    if not row:
        raise service.RunError("archive_application_not_found", 404)
    normalized = gunzip_limited(row["archive"], MAX_INPUT)
    try:
        result = verse_replay.inspect_replay(normalized, row["variant"])
    except ValueError as exc:
        raise service.RunError("stored_replay_invalid", 409) from exc
    if (result["score"] != row["claimed_score"] or result["moves"] != row["moves"]
            or result["board"] != json.loads(row["final_board_json"])):
        raise service.RunError("stored_replay_changed", 409)
    return dict(row), result


def decide(application_id: int, operator_id: int, approved: bool, note: str) -> dict:
    note = note.strip()
    if not note or len(note) > 1000:
        raise service.RunError("review_reason_required", 400)
    with database() as db:
        row = db.execute("SELECT * FROM human_archive_applications WHERE id=?",
                         (application_id,)).fetchone()
        if not row:
            raise service.RunError("archive_application_not_found", 404)
        if row["status"] != "pending":
            raise service.RunError("archive_application_not_pending", 409)
        if not approved:
            now = time.time()
            db.execute("""UPDATE human_archive_applications SET status='rejected',
                updated_at=?,review_note=? WHERE id=?""", (now, note, application_id))
            db.execute("""INSERT INTO human_archive_application_audit
                (application_id,operator_id,action,note,created_at)
                VALUES(?,?,'rejected',?,?)""", (application_id, operator_id, note, now))
            return _payload(db.execute("SELECT * FROM human_archive_applications WHERE id=?",
                                       (application_id,)).fetchone())
        saved, result = _validated_saved(db, application_id)

    now = time.time()
    run_id = str(uuid.uuid4())
    state = {"score": result["score"], "board": result["board"], "seq": result["moves"],
             "elapsed": result["elapsed"], "nodes": result["nodes"], "hash": None,
             "spawnCount": result["spawn_count"], "fourCount": result["four_count"],
             "imported": {"application_id": application_id}}
    created = max(MIN_ENDED_AT, saved["claimed_ended_at"] - result["elapsed"] / 1000)
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("""SELECT a.*,p.archive FROM human_archive_applications a
            JOIN human_archive_application_payloads p ON p.application_id=a.id WHERE a.id=?""",
            (application_id,)).fetchone()
        if not current or current["status"] != "pending" or current["archive"] != saved["archive"]:
            raise service.RunError("archive_application_changed", 409)
        duplicate = db.execute("""SELECT a.id FROM human_archive_applications a
            JOIN human_runs r ON r.id=a.run_id
            WHERE a.id<>? AND a.status='approved' AND a.variant=? AND a.claimed_score=?
              AND a.moves=? AND a.replay_crc=? AND a.replay_size=? AND r.archive=? LIMIT 1""",
            (application_id, saved["variant"], saved["claimed_score"], saved["moves"],
             saved["replay_crc"], saved["replay_size"], saved["archive"])).fetchone()
        if duplicate:
            raise service.RunError("replay_already_archived", 409)
        from . import rating
        db.execute("""INSERT INTO human_runs
            (id,user_id,browser,variant,request_id,seed,threshold,status,eligibility,reason,
             created,ended,writer,epoch,permit_until,monitored,state,archive,display_threshold,
             visible,has_replay,source,single_rating,single_rating_version)
            VALUES(?,?,?,?,?,'',0,'sealed','eligible','imported',?,?,'',0,0,0,?,?,0,1,1,
                   'manual',?,?)""",
            (run_id, saved["user_id"], f"manual:{application_id}", saved["variant"],
             str(application_id), created, saved["claimed_ended_at"],
             json.dumps(state, separators=(",", ":")), saved["archive"],
             rating.single_rating(saved["variant"], result["board"]), rating.RATING_VERSION))
        db.execute("""UPDATE human_archive_applications SET status='approved',approved_by=?,
            approved_at=?,updated_at=?,review_note=?,run_id=? WHERE id=?""",
            (operator_id, now, now, note, run_id, application_id))
        db.execute("""INSERT INTO human_archive_application_audit
            (application_id,operator_id,action,note,created_at)
            VALUES(?,?,'approved',?,?)""", (application_id, operator_id, note, now))
        run = dict(db.execute("SELECT * FROM human_runs WHERE id=?", (run_id,)).fetchone())
        statistics.upsert_fact(db, run, result["board"], result["score"], result["rate"])
        rating.refresh_player(db, saved["user_id"], saved["variant"])
        statistics.rebuild_player(db, saved["user_id"], saved["variant"])
        # Deliberately do not call rolling.add_run: manual archives never enter
        # the rolling 168-hour board or its weekly token settlement.
        db.execute("DELETE FROM human_archive_application_payloads WHERE application_id=?",
                   (application_id,))
        return _payload(db.execute("SELECT * FROM human_archive_applications WHERE id=?",
                                   (application_id,)).fetchone())


def revoke(application_id: int, operator_id: int, note: str) -> dict:
    note = note.strip()
    if not note or len(note) > 1000:
        raise service.RunError("review_reason_required", 400)
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM human_archive_applications WHERE id=?",
                         (application_id,)).fetchone()
        if not row or row["status"] != "approved" or not row["run_id"]:
            raise service.RunError("archive_application_not_approved", 409)
        db.execute("UPDATE human_runs SET visible=0,eligibility='disqualified' WHERE id=?",
                   (row["run_id"],))
        db.execute("""UPDATE human_archive_applications SET status='revoked',updated_at=?,
            review_note=? WHERE id=?""", (now, note, application_id))
        db.execute("""INSERT INTO human_archive_application_audit
            (application_id,operator_id,action,note,created_at)
            VALUES(?,?,'revoked',?,?)""", (application_id, operator_id, note, now))
        from . import leaderboards, rating
        rating.refresh_player(db, row["user_id"], row["variant"])
        statistics.rebuild_player(db, row["user_id"], row["variant"])
        leaderboards.refresh_run(db, row["run_id"])
        return _payload(db.execute("SELECT * FROM human_archive_applications WHERE id=?",
                                   (application_id,)).fetchone())
