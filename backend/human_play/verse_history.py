"""Verified Verse history snapshots and one-time account claims."""
from __future__ import annotations

import json
import os
from pathlib import Path
import psutil
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import uuid

from . import engine, rating, service, statistics
from .store import database

VARIANTS = ("4x4", "3x4", "3x3", "2x4")
USERNAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ROOT = Path(__file__).resolve().parents[2]
_worker_lock = threading.Lock()


class BulkCollectorBusy(RuntimeError):
    pass


def _bulk_collector_running() -> bool:
    cohort_root = archive_root() / "cohorts"
    if not cohort_root.is_dir():
        return False
    for lock in cohort_root.glob("*/bulk-import.lock"):
        try:
            pid = int(json.loads(lock.read_text(encoding="utf-8"))["pid"])
            process = psutil.Process(pid)
            if any("bulk-import-verse-history.mjs" in part for part in process.cmdline()):
                return True
        except psutil.NoSuchProcess:
            continue
        except (OSError, ValueError, KeyError, psutil.AccessDenied):
            return True
    return False


def _retry_after_collector() -> None:
    timer = threading.Timer(60, start_worker)
    timer.daemon = True
    timer.start()


def archive_root() -> Path:
    return Path(os.environ.get("HUMAN_VERSE_ARCHIVE_ROOT") or ROOT / "data" / "verse-history")


class Reader:
    def __init__(self, raw: bytes):
        self.raw, self.at = raw, 0

    def take(self, size: int) -> bytes:
        if self.at + size > len(self.raw):
            raise ValueError("verse_segment_truncated")
        data = self.raw[self.at:self.at + size]
        self.at += size
        return data

    def uint(self) -> int:
        value = 0
        for shift in range(0, 56, 7):
            byte = self.take(1)[0]
            value += (byte & 127) << shift
            if byte < 128:
                if value > 2**53 - 1:
                    raise ValueError("verse_integer_too_large")
                return value
        raise ValueError("verse_integer_invalid")


def decode_segment(path: Path, variant: str) -> dict:
    raw = path.read_bytes()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError("verse_segment_too_large")
    rd = Reader(raw)
    if rd.take(4) != b"VHS2" or rd.take(1)[0] != VARIANTS.index(variant):
        raise ValueError("verse_segment_format")
    pass_mask = rd.take(1)[0]
    if not 1 <= pass_mask <= 15:
        raise ValueError("verse_pass_mask")
    collected_ms, declared, raw_read, unique, maximum = (rd.uint() for _ in range(5))
    if unique != declared or raw_read < unique or unique > 5_000_000:
        raise ValueError("verse_segment_incomplete")
    rows, cols = engine.VARIANTS[variant]
    cell_count = rows * cols
    board_bytes = (cell_count * 5 + 7) // 8
    records, seen = [], set()
    previous_ms = 0
    for index in range(unique):
        stamp = rd.uint()
        played_ms = stamp if index == 0 else previous_ms + stamp
        game_id, score = rd.uint(), rd.uint()
        packed = int.from_bytes(rd.take(board_bytes), "little")
        if packed >> (cell_count * 5):
            raise ValueError("verse_board_padding_invalid")
        board = [0 if (exponent := (packed >> (cell * 5)) & 31) == 0 else 1 << exponent
                 for cell in range(cell_count)]
        if game_id in seen or played_ms < previous_ms or played_ms > collected_ms + 86_400_000:
            raise ValueError("verse_segment_conflict")
        seen.add(game_id)
        records.append((game_id, played_ms, score, board))
        previous_ms = played_ms
    if rd.at != len(raw) or max((row[2] for row in records), default=0) != maximum:
        raise ValueError("verse_segment_mismatch")
    return {"variant": variant, "collected_ms": collected_ms, "count": unique,
            "maximum": maximum, "records": records}


def snapshot(username: str) -> dict:
    if not USERNAME.fullmatch(username):
        raise service.RunError("invalid_verse_username", 400)
    root = archive_root()
    directory = root / username
    if not directory.is_dir():
        matches = [candidate for candidate in root.iterdir()
                   if candidate.is_dir() and candidate.name.casefold() == username.casefold()] if root.is_dir() else []
        if len(matches) != 1:
            raise FileNotFoundError("account_snapshot_missing" if not matches else "account_snapshot_ambiguous")
        directory = matches[0]
    segments = {}
    for variant in VARIANTS:
        path = directory / f"{variant}.vhs"
        if not path.is_file():
            raise FileNotFoundError(variant)
        segments[variant] = decode_segment(path, variant)
    return segments


def claim_payload(row) -> dict:
    return {"id": row["id"], "provider": row["provider"], "username": row["username"],
            "status": row["status"], "counts": json.loads(row["counts"]),
            "error": row["error"], "requested_at": row["requested"],
            "updated_at": row["updated"]}


def own_claim(user_id: int):
    with database() as db:
        row = db.execute("""SELECT * FROM human_external_claims
            WHERE provider='verse' AND user_id=? AND status NOT IN ('rejected','cancelled')
            ORDER BY id DESC LIMIT 1""", (user_id,)).fetchone()
    return claim_payload(row) if row else None


def request_claim(user_id: int, username: str) -> dict:
    username = username.strip()
    if not USERNAME.fullmatch(username):
        raise service.RunError("invalid_verse_username", 400)
    key = username.casefold()
    now = time.time()
    try:
        counts = {variant: segment["count"] for variant, segment in snapshot(username).items()}
    except (FileNotFoundError, ValueError):
        counts = {}
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("""SELECT * FROM human_external_claims
            WHERE provider='verse' AND user_id=? AND status NOT IN ('rejected','cancelled')
            ORDER BY id DESC LIMIT 1""", (user_id,)).fetchone()
        if existing:
            if existing["username_key"] == key:
                return claim_payload(existing)
            raise service.RunError("verse_claim_already_exists", 409)
        owner = db.execute("""SELECT 1 FROM human_external_claims WHERE provider='verse'
            AND username_key=? AND status IN ('approved','importing','complete')""", (key,)).fetchone()
        if owner:
            raise service.RunError("verse_account_claimed", 409)
        recent = db.execute("""SELECT count(*) FROM human_external_claims WHERE user_id=?
            AND requested>?""", (user_id, now - 86400)).fetchone()[0]
        if recent >= 3:
            raise service.RunError("verse_claim_rate_limit", 429)
        cursor = db.execute("""INSERT INTO human_external_claims
            (user_id,provider,username,username_key,status,requested,updated,counts)
            VALUES (?,'verse',?,?,'pending',?,?,?)""",
            (user_id, username, key, now, now, json.dumps(counts)))
        row = db.execute("SELECT * FROM human_external_claims WHERE id=?", (cursor.lastrowid,)).fetchone()
        return claim_payload(row)


def pending_claims(user_id: int | None = None) -> list[dict]:
    with database() as db:
        rows = db.execute("""SELECT * FROM human_external_claims
            WHERE (? IS NULL OR user_id=?)
            ORDER BY CASE WHEN status IN ('pending','approved','importing','failed') THEN 0 ELSE 1 END,
                updated DESC LIMIT 100""", (user_id, user_id)).fetchall()
    return [{**claim_payload(row), "user_id": row["user_id"],
             "approved_by": row["approved_by"], "proof_note": row["proof_note"]} for row in rows]


def pending_approval_user_ids() -> set[int]:
    """Return users that currently need an administrator's decision."""
    with database() as db:
        rows = db.execute("""SELECT DISTINCT user_id FROM human_external_claims
            WHERE status='pending'""").fetchall()
    return {int(row["user_id"]) for row in rows}


def decide_claim(claim_id: int, operator_id: int, approved: bool, note: str) -> dict:
    note = note.strip()
    if len(note) > 1000:
        raise service.RunError("verse_proof_note_too_long", 400)
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone()
        if not row:
            raise service.RunError("verse_claim_not_found", 404)
        if row["status"] == "complete":
            raise service.RunError("verse_claim_complete")
        if row["status"] not in ("pending", "approved", "failed"):
            raise service.RunError("verse_claim_not_pending")
        status = "approved" if approved else "rejected"
        try:
            db.execute("""UPDATE human_external_claims SET status=?,approved_by=?,approved_at=?,
                proof_note=?,error='',updated=? WHERE id=?""",
                (status, operator_id if approved else None, now if approved else None, note, now, claim_id))
        except sqlite3.IntegrityError as exc:
            raise service.RunError("verse_account_claimed", 409) from exc
        db.execute("""INSERT INTO human_external_audit
            (claim_id,operator_id,action,note,created) VALUES (?,?,?,?,?)""",
            (claim_id, operator_id, status, note, now))
        return claim_payload(db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone())


def retry_claim(claim_id: int, operator_id: int, note: str) -> dict:
    note = note.strip()
    if len(note) > 1000:
        raise service.RunError("verse_proof_note_too_long", 400)
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone()
        if not row:
            raise service.RunError("verse_claim_not_found", 404)
        if row["status"] not in ("failed", "importing") or (
                row["status"] == "importing" and now - row["updated"] < 7200):
            raise service.RunError("verse_claim_not_retryable")
        db.execute("""UPDATE human_external_claims SET status='approved',error='',updated=?
            WHERE id=?""", (now, claim_id))
        db.execute("""INSERT INTO human_external_audit
            (claim_id,operator_id,action,note,created) VALUES (?,?,'retry',?,?)""",
            (claim_id, operator_id, note, now))
        return claim_payload(db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone())


def revoke_claim(claim_id: int, operator_id: int, note: str) -> dict:
    note = note.strip()
    if len(note) > 1000:
        raise service.RunError("verse_proof_note_too_long", 400)
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone()
        if not row or row["status"] != "complete":
            raise service.RunError("verse_claim_not_complete", 409)
        db.execute("""UPDATE human_runs SET visible=0,eligibility='disqualified'
            WHERE source='verse' AND user_id=? AND browser=?""",
            (row["user_id"], f"verse:{claim_id}"))
        affected_runs = [item["id"] for item in db.execute("""SELECT id FROM human_runs
            WHERE source='verse' AND user_id=? AND browser=?""",
            (row["user_id"], f"verse:{claim_id}")).fetchall()]
        from . import rolling
        for candidate in db.execute("""SELECT c.run_id FROM rolling_candidates c
            JOIN human_runs r ON r.id=c.run_id WHERE r.source='verse'
            AND r.user_id=? AND r.browser=? AND c.active=1""",
            (row['user_id'], f'verse:{claim_id}')).fetchall():
            rolling.revoke_run(db, candidate['run_id'], now, refresh=False)
        from backend import rolling_leaderboards as core
        for variant in VARIANTS:
            core.refresh_user(db, variant, row['user_id'], now)
            rating.refresh_player(db, row['user_id'], variant)
            statistics.rebuild_player(db, row['user_id'], variant)
        from . import leaderboards
        for run_id in affected_runs:
            leaderboards.refresh_run(db, run_id, now)
        db.execute("""UPDATE human_external_claims SET status='revoked',updated=?
            WHERE id=?""", (now, claim_id))
        db.execute("""INSERT INTO human_external_audit
            (claim_id,operator_id,action,note,created) VALUES (?,?,'revoke',?,?)""",
            (claim_id, operator_id, note, now))
        return claim_payload(db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone())


def _fetch(username: str, *, incremental: bool = False) -> dict:
    if _bulk_collector_running():
        raise BulkCollectorBusy("verse_bulk_collector_running")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("node_unavailable")
    script = ROOT / "tools" / "verse-history" / ("refresh-verse-history-api.mjs" if incremental
                                                 else "import-verse-history-api.mjs")
    if not script.is_file():
        raise RuntimeError("verse_importer_unavailable")
    root = archive_root()
    if root.is_dir():
        matches = [item.name for item in root.iterdir()
                   if item.is_dir() and item.name.casefold() == username.casefold()]
        if len(matches) > 1:
            raise RuntimeError("verse_cache_directory_ambiguous")
        if matches:
            username = matches[0]
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as errors:
        result = subprocess.run([node, str(script), "--username", username,
            "--output-root", str(root), "--delay-ms", "500", "--retries", "4"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors,
            timeout=7200, check=False, text=True)
        if result.returncode:
            errors.seek(0)
            detail = errors.read().strip()
            reason = next((line.removeprefix("Error: ").strip() for line in reversed(detail.splitlines())
                           if line.startswith("Error: ")), detail[-300:])
            raise RuntimeError(f"verse_fetch_failed:{reason[:500]}")
        try:
            return json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("verse_fetch_output_invalid") from exc


def _remote_removal_note(reports: list[dict]) -> str | None:
    warnings = []
    for report in reports:
        for variant in report.get("variants", []):
            removed = int(variant.get("remoteRemoved") or 0)
            if not removed:
                continue
            ids = [int(value) for value in (variant.get("remoteRemovedIds") or [])]
            warnings.append({"variant": Path(variant.get("file") or "").stem,
                "remote_removed": removed, "remote_total": int(variant.get("remoteTotal") or 0),
                "retained_total": int(variant.get("count") or 0),
                "remote_removed_ids": ids[:200], "ids_truncated": len(ids) > 200})
    return json.dumps({"message": "Verse 远端已移除记录，保留此前完整采集证据",
                       "variants": warnings}, ensure_ascii=False) if warnings else None


def _import(claim_id: int) -> None:
    with database() as db:
        row = db.execute("SELECT * FROM human_external_claims WHERE id=?", (claim_id,)).fetchone()
    if not row or row["status"] != "importing":
        return
    refresh_reports = []
    try:
        snapshot(row["username"])
    except (FileNotFoundError, ValueError):
        root = archive_root()
        had_cache = root.is_dir() and any(
            item.is_dir() and item.name.casefold() == row["username"].casefold()
            and any(item.glob("*.vhs")) for item in root.iterdir())
        _fetch(row["username"])
        if had_cache:
            refresh_reports.append(_fetch(row["username"], incremental=True))
    else:
        refresh_reports.append(_fetch(row["username"], incremental=True))
    segments = snapshot(row["username"])
    browser = f"verse:{claim_id}"
    counts = {variant: segment["count"] for variant, segment in segments.items()}
    for variant, segment in segments.items():
        for start in range(0, len(segment["records"]), 500):
            batch = segment["records"][start:start + 500]
            with database() as db:
                db.execute("BEGIN IMMEDIATE")
                for game_id, played_ms, score, board in batch:
                    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
                        f"2048verse:{claim_id}:{variant}:{game_id}"))
                    state = {"score": score, "board": board, "seq": None, "elapsed": None,
                             "nodes": {}, "hash": None, "imported": {
                                 "source_game_id": game_id,
                                 "collected_at_ms": segment["collected_ms"]}}
                    prior = db.execute("""SELECT user_id,variant,state,source,ended,single_rating,single_rating_version FROM human_runs
                        WHERE id=?""", (run_id,)).fetchone()
                    if prior:
                        old = json.loads(prior["state"])
                        if (prior["user_id"] != row["user_id"] or prior["variant"] != variant
                                or prior["source"] != "verse" or old["score"] != score
                                or round(prior["ended"] * 1000) != played_ms
                                or old["board"] != board
                                or old.get("imported", {}).get("source_game_id") != game_id):
                            raise ValueError("verse_record_conflict")
                        if prior["single_rating"] is None or prior["single_rating_version"] != rating.RATING_VERSION:
                            db.execute("UPDATE human_runs SET single_rating=?,single_rating_version=? WHERE id=?",
                                       (rating.single_rating(variant, board), rating.RATING_VERSION, run_id))
                        # Incremental imports commonly revisit the local archive.  An
                        # older run may predate the statistics tables, so ensure it
                        # has a fact row without replacing replay-derived coverage.
                        fact = db.execute("SELECT 1 FROM human_run_statistics WHERE run_id=?",
                                          (run_id,)).fetchone()
                        if not fact:
                            statistics.upsert_fact(db, {"id": run_id, "user_id": row["user_id"],
                                "variant": variant, "ended": played_ms / 1000}, board, score,
                                statistics.unavailable_rate())
                        continue
                    db.execute("""INSERT INTO human_runs
                        (id,user_id,browser,variant,request_id,seed,threshold,status,
                         eligibility,reason,created,ended,writer,epoch,permit_until,
                         monitored,state,archive,display_threshold,visible,has_replay,source,single_rating,single_rating_version)
                         VALUES (?,?,?,?,?,'',0,'sealed','eligible','imported',?,?,'',0,0,0,?,NULL,0,0,0,'verse',?,?)""",
                        (run_id, row["user_id"], browser, variant, str(game_id),
                         played_ms / 1000, played_ms / 1000,
                         json.dumps(state, separators=(",", ":")), rating.single_rating(variant, board), rating.RATING_VERSION))
                    statistics.upsert_fact(db, {"id": run_id, "user_id": row["user_id"],
                        "variant": variant, "ended": played_ms / 1000}, board, score,
                        statistics.unavailable_rate())
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute("SELECT status FROM human_external_claims WHERE id=?", (claim_id,)).fetchone()
        if not current or current["status"] != "importing":
            return
        db.execute("""UPDATE human_runs SET visible=1 WHERE source='verse' AND deleted_by_user=0
            AND user_id=? AND browser=?""", (row["user_id"], browser))
        db.execute("""UPDATE human_external_claims SET status='complete',error='',
            counts=?,updated=? WHERE id=?""", (json.dumps(counts), time.time(), claim_id))
        for variant in VARIANTS:
            rating.refresh_player(db, row['user_id'], variant)
            statistics.rebuild_player(db, row['user_id'], variant)
        from . import rolling
        from backend import rolling_leaderboards as core
        completion = time.time()
        was_initialized = bool(db.execute("SELECT 1 FROM rolling_meta WHERE key='human_rolling_v1'").fetchone())
        rolling.ensure_backfill(db, completion)
        if was_initialized:
            recent = db.execute("""SELECT id,user_id,variant,state,ended,source,has_replay
                FROM human_runs WHERE source='verse' AND user_id=? AND browser=?
                AND ended>=? AND ended<?""",
                (row['user_id'], browser, completion-core.WINDOW_SECONDS, completion)).fetchall()
            for game in recent:
                rolling._add_row(db, game, completion, now=completion)
            for variant in VARIANTS:
                core.refresh_user(db, variant, row['user_id'], completion)
        db.execute("""INSERT INTO human_external_audit
            (claim_id,operator_id,action,note,created) VALUES (?,NULL,'complete',?,?)""",
            (claim_id, json.dumps(counts), time.time()))
        if removal_note := _remote_removal_note([report for report in refresh_reports if report]):
            db.execute("""INSERT INTO human_external_audit
                (claim_id,operator_id,action,note,created)
                VALUES (?,NULL,'remote_removed',?,?)""", (claim_id, removal_note, time.time()))


def process_approved() -> None:
    if not _worker_lock.acquire(blocking=False):
        return
    try:
        while True:
            if _bulk_collector_running():
                with database() as db:
                    waiting = db.execute("""SELECT 1 FROM human_external_claims
                        WHERE status='approved' LIMIT 1""").fetchone()
                if waiting:
                    _retry_after_collector()
                return
            with database() as db:
                db.execute("BEGIN IMMEDIATE")
                active = db.execute("""SELECT 1 FROM human_external_claims
                    WHERE status='importing' LIMIT 1""").fetchone()
                if active:
                    return
                row = db.execute("""SELECT id FROM human_external_claims
                    WHERE status='approved' ORDER BY approved_at LIMIT 1""").fetchone()
                if not row:
                    return
                claim_id = row["id"]
                db.execute("""UPDATE human_external_claims SET status='importing',updated=?
                    WHERE id=?""", (time.time(), claim_id))
            try:
                _import(claim_id)
            except BulkCollectorBusy:
                with database() as db:
                    db.execute("""UPDATE human_external_claims SET status='approved',updated=?
                        WHERE id=? AND status='importing'""", (time.time(), claim_id))
                _retry_after_collector()
                return
            except Exception as exc:
                with database() as db:
                    db.execute("""UPDATE human_external_claims SET status='failed',
                        error=?,updated=? WHERE id=? AND status='importing'""",
                        (str(exc)[:500], time.time(), claim_id))
    finally:
        _worker_lock.release()


def start_worker() -> None:
    threading.Thread(target=process_approved, daemon=True, name="verse-history-import").start()
