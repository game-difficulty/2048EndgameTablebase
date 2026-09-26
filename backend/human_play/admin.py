"""Site-owner ranking review, available only through local database access.

No player-facing HTTP endpoint exposes these operations.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import time

from . import engine
from .service import RunError, summary
from .store import database, init_db, hash_bytes
from .codec import gunzip_limited, MAX_REPLAY_BYTES


def get_run(db, run_id):
    row = db.execute("SELECT * FROM human_runs WHERE id=?", (run_id,)).fetchone()
    if not row:
        raise RunError("run_not_found", 404)
    if row["source"] != "native":
        raise RunError("native_run_required", 400)
    return dict(row)


def inspect(run_id):
    with database() as db:
        run = get_run(db, run_id)
        state = json.loads(run["state"])
        return {**summary(run), "status": run["status"], "prefix_hash": state["hash"],
                "manually_approved": bool(db.execute("SELECT 1 FROM human_rank_approvals WHERE run_id=?", (run_id,)).fetchone()),
                "verification_events": [dict(row) for row in db.execute("SELECT * FROM human_audit WHERE run_id=? ORDER BY id", (run_id,))],
                "ranking_reviews": [dict(row) for row in db.execute("SELECT * FROM human_rank_reviews WHERE run_id=? ORDER BY id", (run_id,))]}


def verified_events(db, run):
    """Recompute the retained prefix; never accept a score supplied by an operator."""
    state = json.loads(run["state"])
    if run["status"] == "sealed":
        binary = gunzip_limited(run["archive"], MAX_REPLAY_BYTES)
        header, raw = engine.parse_replay(binary)
        if binary != engine.replay_bytes(run, raw, version=header['version']) or header["run_id"] != run["id"]:
            raise ValueError("archive_header_mismatch")
        received = run["ended"]
    else:
        current = engine.initial(run["id"], run["variant"], run["seed"])
        raw_buffer = bytearray()
        received = run["created"]
        for chunk in db.execute("SELECT * FROM human_chunks WHERE run_id=? ORDER BY start", (run["id"],)):
            data = chunk["data"]
            if (chunk["start"] != current["seq"] or hash_bytes(chunk["previous_hash"]) != hash_bytes(current["hash"])
                    or chunk["count"] * 5 != len(data) or hashlib.sha256(data).digest() != hash_bytes(chunk["digest"])):
                raise ValueError("checkpoint_integrity_mismatch")
            current = engine.advance(current, run["variant"], data, run["threshold"])
            raw_buffer.extend(data)
            received = chunk["received"]
        raw = bytes(raw_buffer)
    full = engine.advance(engine.initial(run["id"], run["variant"], run["seed"]), run["variant"], raw, run["threshold"])
    if 'spawnCount' not in state:
        full.pop('spawnCount', None)
        full.pop('fourCount', None)
    if full != state:
        raise ValueError("archive_validation_mismatch")
    if not full["seq"]:
        raise ValueError("no_retained_progress")
    return raw, received


def review(run_id, *, approved, operator, note, expected_seq=None, expected_hash=None):
    if not operator.strip() or not note.strip():
        raise RunError("review_reason_required", 400)
    archive = None
    with database() as db:
        db.execute("BEGIN")
        run = get_run(db, run_id)
        state = json.loads(run["state"])
        if approved:
            if run["eligibility"] != "eligible":
                raise RunError("run_disqualified")
            if run["reason"] == "game_over":
                raise RunError("already_automatically_ranked")
            if state["seq"] != expected_seq or state["hash"] != expected_hash:
                raise RunError("review_progress_changed")
            try:
                raw, received = verified_events(db, run)
            except (ValueError, KeyError, TypeError, OSError, EOFError, struct.error) as exc:
                raise RunError("retained_replay_invalid", 409) from exc
    if approved and run["status"] != "sealed":
        end_reason = run["reason"] or "interrupted"
        archive = gzip.compress(engine.replay_bytes({**run, 'reason': end_reason}, raw, version=2), compresslevel=6, mtime=0)
    rate = None
    if approved:
        from . import statistics
        tracker = statistics.Rate32kTracker(run["variant"])
        initial = engine.initial(run["id"], run["variant"], run["seed"])
        tracker.observe(initial)
        engine.advance(initial, run["variant"], raw, run["threshold"],
                       observer=tracker.observe)
        rate = tracker.result()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        if get_run(db, run_id) != run:
            raise RunError("review_progress_changed")
        from . import rating, statistics
        if approved:
            if archive is not None:
                db.execute("""UPDATE human_runs SET status='sealed',reason=?,ended=?,archive=?,permit_until=0,
                    visible=?,has_replay=1,single_rating=?,single_rating_version=? WHERE id=?""", (end_reason, received, archive,
                    int(state['score'] >= run['display_threshold']),
                    rating.single_rating(run['variant'], state['board']), rating.RATING_VERSION, run_id))
                db.execute("DELETE FROM human_chunks WHERE run_id=?", (run_id,))
            db.execute("""INSERT OR REPLACE INTO human_rank_approvals
                (run_id,operator,note,created,seq,prefix_hash) VALUES(?,?,?,?,?,?)""",
                (run_id, operator.strip(), note.strip(), time.time(), state["seq"], state["hash"]))
            final = db.execute("SELECT visible,ended FROM human_runs WHERE id=?", (run_id,)).fetchone()
            statistics.upsert_fact(db, {**run, "ended": final["ended"]},
                                   state["board"], state["score"], rate)
            if final['visible']:
                from . import rolling
                rolling.add_run(db, run_id, time.time())
        else:
            if not db.execute("SELECT 1 FROM human_rank_approvals WHERE run_id=?", (run_id,)).fetchone():
                raise RunError("no_manual_approval")
            db.execute("DELETE FROM human_rank_approvals WHERE run_id=?", (run_id,))
            from . import rolling
            rolling.revoke_run(db, run_id)
        rating.refresh_player(db, run['user_id'], run['variant'])
        statistics.rebuild_player(db, run['user_id'], run['variant'])
        from . import leaderboards
        leaderboards.refresh_run(db, run_id)
        db.execute("""INSERT INTO human_rank_reviews
            (run_id,approved,operator,note,created,seq,prefix_hash) VALUES(?,?,?,?,?,?,?)""",
            (run_id, int(approved), operator.strip(), note.strip(), time.time(), state["seq"], state["hash"]))
    return inspect(run_id)


def main():
    parser = argparse.ArgumentParser(description="Review a retained human game for exceptional ranking.")
    parser.add_argument("--db", required=True, help="Existing human-play SQLite database path")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("inspect", "approve", "revoke"):
        command = sub.add_parser(action)
        command.add_argument("run_id")
        if action != "inspect":
            command.add_argument("--operator", required=True)
            command.add_argument("--note", required=True)
        if action == "approve":
            command.add_argument("--expected-seq", required=True, type=int)
            command.add_argument("--expected-hash", required=True)
    args = parser.parse_args()
    if not Path(args.db).is_file():
        parser.error("--db must name an existing database")
    os.environ["HUMAN_PLAY_DB"] = str(Path(args.db).resolve())
    init_db()
    try:
        result = inspect(args.run_id) if args.action == "inspect" else review(args.run_id,
            approved=args.action == "approve", operator=args.operator, note=args.note,
            expected_seq=getattr(args, "expected_seq", None), expected_hash=getattr(args, "expected_hash", None))
    except RunError as exc:
        parser.exit(1, f"{exc.code}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
