from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone

from backend.auth.db import auth_db
from . import engine, permits
from .store import database, hash_bytes

PERMIT_SECONDS = 12
RUN_COLUMNS = "id,user_id,browser,variant,request_id,seed,threshold,status,eligibility,reason,created,ended,writer,epoch,permit_until,monitored,state"
SUMMARY_COLUMNS = "id,user_id,variant,state,ended,reason,eligibility"
# Used by rankings, public history, PBs and replay access alike.
RANKABLE_SQL = "eligibility='eligible' AND (reason='game_over' OR id IN (SELECT run_id FROM human_rank_approvals))"


class RunError(Exception):
    def __init__(self, code, status=409, **details):
        self.code, self.status, self.details = code, status, details
        super().__init__(code)


def config():
    thresholds = dict(engine.THRESHOLDS)
    override = json.loads(os.getenv("HUMAN_PLAY_THRESHOLDS", "{}"))
    for key, value in override.items():
        if key not in thresholds or type(value) is not int or value < 0:
            raise ValueError("Invalid HUMAN_PLAY_THRESHOLDS")
        thresholds[key] = value
    return {"variants": [{"id": key, "rows": dims[0], "cols": dims[1],
                          "threshold": thresholds[key], "restart_threshold": engine.RESTART_THRESHOLDS[key],
                          "nodes": engine.NODES[key]} for key, dims in engine.VARIANTS.items()],
            "upload_seconds": 10, "upload_moves": 32, "permit_seconds": PERMIT_SECONDS,
            "rules_version": 1, "max_moves": engine.MAX_MOVES}


def owned(db, run_id, user_id, browser):
    run = db.execute(f"SELECT {RUN_COLUMNS} FROM human_runs WHERE id=?", (run_id,)).fetchone()
    if not run or run["user_id"] != user_id or run["browser"] != browser:
        raise RunError("run_not_found", 404)
    return dict(run)


def receipt(run, until=None):
    state = json.loads(run["state"])
    expiry = run["permit_until"] if until is None else until
    return {"permit": permits.issue(run, expiry) if expiry else "", "run_id": run["id"], "seq": state["seq"], "prefix_hash": state["hash"],
            "monitored": bool(run["monitored"]), "threshold": run["threshold"],
            "epoch": run["epoch"], "status": run["status"], "eligibility": run["eligibility"],
            "server_time": time.time(), "permit_until": expiry}


def create(user_id, browser, variant, request_id, writer, replace_id=None):
    policies = {v["id"]: v for v in config()["variants"]}
    if variant not in policies:
        raise RunError("invalid_variant", 400)
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute(f"SELECT {RUN_COLUMNS} FROM human_runs WHERE user_id=? AND browser=? AND request_id=?",
                              (user_id, browser, request_id)).fetchone()
        if existing:
            if existing["variant"] != variant or existing["writer"] != writer:
                raise RunError("request_conflict")
            run = dict(existing)
        else:
            recent = db.execute("SELECT count(*) FROM human_runs WHERE user_id=? AND created>?",
                                (user_id, now - 60)).fetchone()[0]
            if recent >= 60:
                raise RunError("creation_rate_limit", 429)
            active = db.execute(f"SELECT {RUN_COLUMNS} FROM human_runs WHERE user_id=? AND browser=? AND variant=? AND status='active'",
                                (user_id, browser, variant)).fetchone()
            if active:
                if active["id"] != replace_id:
                    raise RunError("slot_exists", active_id=active["id"])
                # Explicit abandonment frees only this browser/variant slot. Evidence remains.
                db.execute("UPDATE human_runs SET status='pending_archive',reason='restarted',ended=? WHERE id=?",
                           (now, active["id"]))
            run_id = str(uuid.uuid4())
            seed = "".join(f"{secrets.randbelow(0xffffffff) + 1:08x}" for _ in range(4))
            state = engine.initial(run_id, variant, seed)
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,created,writer,state)
                VALUES (?,?,?,?,?,?,?,?,?,?)""", (run_id, user_id, browser, variant, request_id, seed,
                policies[variant]["threshold"], now, writer, json.dumps(state)))
            run = owned(db, run_id, user_id, browser)
        return {**receipt(run), "seed": run["seed"], "variant": variant, "created_at": run["created"],
                "rules_version": 1}


def status(user_id, browser, run_id):
    with database() as db:
        return receipt(owned(db, run_id, user_id, browser))


def reject(db, run, code, local_seq):
    state = json.loads(run["state"])
    db.execute("UPDATE human_runs SET eligibility='disqualified',permit_until=0 WHERE id=?", (run["id"],))
    db.execute("INSERT INTO human_audit(run_id,kind,local_seq,server_seq,created) VALUES(?,?,?,?,?)",
               (run["id"], code, local_seq, state["seq"], time.time()))
    db.commit()  # Preserve the rejection even though the request raises afterwards.
    raise RunError(code)


def assert_snapshot(db, original):
    current = owned(db, original['id'], original['user_id'], original['browser'])
    # Covers sequence/hash (state), writer/epoch, eligibility, threshold and retired status.
    if any(current[key] != original[key] for key in original):
        raise RunError("progress_changed")
    return current


def reject_snapshot(run, code, local_seq):
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        assert_snapshot(db, run)
        reject(db, run, code, local_seq)


def submit(user_id, browser, run_id, *, action, writer, epoch, start, prefix_hash, local_seq, data, reason="", permit=""):
    if action not in {"monitor", "append", "reentry", "seal"}:
        raise RunError("invalid_action", 400)
    if len(data) % 5 or len(data) > engine.MAX_BYTES or not 0 <= local_seq <= engine.MAX_MOVES:
        raise RunError("invalid_record_size", 400)
    if action == "seal" and reason not in {"game_over", "restarted", "abandoned"}:
        raise RunError("invalid_end_reason", 400)
    now = time.time()
    with database() as db:
        db.execute("BEGIN")
        run = owned(db, run_id, user_id, browser)
        prior = db.execute("SELECT digest,previous_hash FROM human_chunks WHERE run_id=? AND start=?",
                           (run_id, start)).fetchone()
        retained = bytearray()
        if action == 'seal' and run['status'] != 'sealed':
            for row in db.execute('SELECT data FROM human_chunks WHERE run_id=? ORDER BY start', (run_id,)):
                if len(retained) + len(row[0]) > engine.MAX_BYTES:
                    raise RuntimeError('retained_record_too_large')
                retained.extend(row[0])
    if run["status"] == "sealed":
        if action == "seal" and run["reason"] == reason and json.loads(run["state"])["seq"] == local_seq:
            return receipt(run)
        raise RunError("run_sealed")
    if run["eligibility"] != "eligible" and action != "seal":
        raise RunError("run_disqualified")
    # A retired run may retry its missing first checkpoint solely on the way
    # to archival. It never regains append/reentry or its active browser slot.
    archive_checkpoint = (run["status"] == "pending_archive" and action == "monitor" and not run["monitored"])
    if run["status"] != "active" and action != "seal" and not archive_checkpoint:
        raise RunError("run_inactive")
    state = json.loads(run["state"])
    if action == "reentry":
        if epoch != run["epoch"]:
            raise RunError("writer_changed")
        if local_seq < state["seq"]:
            reject_snapshot(run, "rollback_detected", local_seq)
    elif run["writer"] != writer or run["epoch"] != epoch:
        raise RunError("writer_changed")
    digest = hashlib.sha256(data).digest()
    if start < state["seq"] and action != "reentry":
        try:
            if prior and hash_bytes(prior['digest']) == digest and hash_bytes(prior['previous_hash']) == hash_bytes(prefix_hash):
                return receipt(run)
        except ValueError:
            pass
        raise RunError("stale_upload")
    if start != state["seq"]:
        raise RunError("progress_changed")
    if prefix_hash != state["hash"]:
        reject_snapshot(run, "prefix_conflict", local_seq)
    if local_seq != start + len(data) // 5:
        raise RunError("incomplete_tail", 400)
    if action == "append" and (not run["monitored"] or (run["permit_until"] < now and not permits.valid(run, permit, now))):
        raise RunError("reentry_required")
    try:
        next_state = engine.advance(state, run["variant"], data, run["threshold"])
    except ValueError as exc:
        reject_snapshot(run, str(exc), local_seq)
    if action == "monitor" and next_state["score"] <= run["threshold"]:
        raise RunError("below_threshold", 400)
    if not run["monitored"] and next_state["first_over"] is not None and next_state["seq"] > next_state["first_over"]:
        reject_snapshot(run, "monitoring_required", local_seq)
    # Reentry cannot be used as a below-threshold periodic upload API.
    if action == "reentry" and not run["monitored"] and next_state["score"] <= run["threshold"] and data:
        raise RunError("below_threshold", 400)
    if action == "seal" and reason == "game_over" and not engine.game_over(next_state["board"], *engine.VARIANTS[run["variant"]]):
        raise RunError("game_not_over", 400)
    archive = None
    if action == "seal":
        raw = bytes(retained) + data
        full = engine.advance(engine.initial(run_id, run['variant'], run['seed']), run['variant'], raw, run['threshold'])
        if full != next_state:
            raise RuntimeError("archive_validation_mismatch")
        archive = gzip.compress(engine.replay_bytes({**run, 'reason': reason}, raw, version=2), compresslevel=6, mtime=0)
    # Replay and compression have finished; only the compare-and-commit holds a write lock.
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        assert_snapshot(db, run)
        if data:
            db.execute("INSERT INTO human_chunks VALUES(?,?,?,?,?,?,?)",
                       (run_id, start, len(data) // 5, digest, hash_bytes(prefix_hash), data, now))
        monitored = bool(run["monitored"] or next_state["score"] > run["threshold"])
        next_epoch = run["epoch"] + (1 if action == "reentry" and writer != run["writer"] else 0)
        expiry = time.time() + PERMIT_SECONDS if monitored else 0
        db.execute("UPDATE human_runs SET state=?,monitored=?,writer=?,epoch=?,permit_until=? WHERE id=?",
                   (json.dumps(next_state), int(monitored), writer, next_epoch, expiry, run_id))
        if archive is not None:
            db.execute("UPDATE human_runs SET status='sealed',reason=?,ended=?,archive=?,permit_until=0 WHERE id=?",
                       (reason, now, archive, run_id))
            db.execute("DELETE FROM human_chunks WHERE run_id=?", (run_id,))
        result = owned(db, run_id, user_id, browser)
    return receipt(result)


def claim_low(user_id, browser, run_id, writer, expected_epoch):
    """Acquire a low-score writer without uploading low-score local progress."""
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        run = owned(db, run_id, user_id, browser)
        if run["monitored"] or run["status"] != "active":
            raise RunError("reentry_required")
        if run["epoch"] != expected_epoch:
            raise RunError("writer_changed")
        if run["eligibility"] != "eligible":
            raise RunError("run_disqualified")
        db.execute("UPDATE human_runs SET writer=?,epoch=epoch+1 WHERE id=?", (writer, run_id))
        return receipt(owned(db, run_id, user_id, browser))


def online_check(user_id, browser, run_id, writer, epoch, permit=""):
    with database() as db:
        run = owned(db, run_id, user_id, browser)
    if run["status"] != "active" or run["eligibility"] != "eligible":
        raise RunError("run_disqualified")
    if run["writer"] != writer or run["epoch"] != epoch:
        raise RunError("writer_changed")
    if not run["monitored"] or (run["permit_until"] < time.time() and not permits.valid(run, permit)):
        raise RunError("reentry_required")
    result = receipt(run, time.time() + PERMIT_SECONDS)
    return {key: result[key] for key in ('seq', 'epoch', 'monitored', 'eligibility',
                                        'server_time', 'permit_until', 'permit')}


def identity_map(ids):
    if not ids:
        return {}
    with auth_db() as db:
        rows = db.execute(f"SELECT id,display_name,status FROM users WHERE id IN ({','.join('?' for _ in ids)})", list(ids))
        return {r["id"]: (r["display_name"] or "玩家") for r in rows if r["status"] == "active"}


def summary(run):
    state = json.loads(run["state"])
    return {"id": run["id"], "user_id": run["user_id"], "variant": run["variant"],
            "score": state["score"], "max_tile": max(state["board"]), "moves": state["seq"],
            "elapsed": state["elapsed"], "nodes": state["nodes"], "ended_at": run["ended"],
            "reason": run["reason"], "eligibility": run["eligibility"]}


def history(user_id, viewer_id, before=None, limit=30):
    with database() as db:
        public = "" if user_id == viewer_id else " AND " + RANKABLE_SQL
        where = "user_id=? AND status='sealed'" + public
        args = [user_id]
        if before:
            where += " AND ended<?"
            args.append(before)
        rows = db.execute(f"SELECT {SUMMARY_COLUMNS} FROM human_runs WHERE {where} ORDER BY ended DESC,id DESC LIMIT ?", (*args, limit + 1)).fetchall()
        aggregate = db.execute("""SELECT count(*) AS games, sum(reason='game_over') AS completed,
            max(json_extract(state,'$.score')) AS best FROM human_runs
            WHERE user_id=? AND status='sealed'""" + public, (user_id,)).fetchone()
        bests = db.execute("""SELECT variant,max(json_extract(state,'$.score')) AS score FROM human_runs
            WHERE user_id=? AND status='sealed' AND """ + RANKABLE_SQL + " GROUP BY variant", (user_id,)).fetchall()
    names = identity_map({user_id})
    if user_id not in names:
        raise RunError("player_not_found", 404)
    return {"player": {"id": user_id, "display_name": names[user_id]}, "stats": dict(aggregate),
            "bests": {r["variant"]: r["score"] for r in bests}, "entries": [summary(r) for r in rows[:limit]],
            "next_cursor": rows[limit - 1]["ended"] if len(rows) > limit else None}


def personal_bests(user_id):
    with database() as db:
        rows = db.execute("SELECT variant,max(json_extract(state,'$.score')) AS score FROM human_runs "
            "WHERE user_id=? AND status='sealed' AND " + RANKABLE_SQL + " GROUP BY variant", (user_id,)).fetchall()
    return {"bests": {row['variant']: row['score'] for row in rows}}


def leaderboard(variant, period="all", limit=10):
    if variant not in engine.VARIANTS or period not in {"all", "week"}:
        raise RunError("invalid_board", 400)
    cutoff = 0
    if period == "week":
        local = datetime.now(timezone(timedelta(hours=8)))
        cutoff = (local - timedelta(days=local.weekday())).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with database() as db:
        rows = db.execute(f"""SELECT * FROM (SELECT {SUMMARY_COLUMNS},row_number() OVER
            (PARTITION BY user_id ORDER BY json_extract(state,'$.score') DESC,ended,id) AS personal_rank
            FROM human_runs WHERE variant=? AND status='sealed' AND {RANKABLE_SQL}
            AND ended>=?) WHERE personal_rank=1
            ORDER BY json_extract(state,'$.score') DESC,ended,id LIMIT ?""", (variant, cutoff, max(1, min(100, limit)))).fetchall()
    names = identity_map({r["user_id"] for r in rows})
    entries = []
    previous_score, rank = None, 0
    for row in rows:
        if row["user_id"] not in names:
            continue
        item = {key: value for key, value in summary(row).items() if key in {"id", "user_id", "score", "max_tile"}}
        if previous_score != item["score"]:
            rank = len(entries) + 1
        previous_score = item["score"]
        entries.append({**item, "rank": rank, "display_name": names[row["user_id"]]})
    return {"entries": entries, "variant": variant, "period": period}


def replay(run_id, viewer_id):
    with database() as db:
        run = db.execute(f"SELECT user_id,archive,({RANKABLE_SQL}) AS publicly_ranked FROM human_runs WHERE id=? AND status='sealed'", (run_id,)).fetchone()
        if not run or (run["user_id"] != viewer_id and not run["publicly_ranked"]):
            raise RunError("replay_not_found", 404)
        if run["user_id"] not in identity_map({run["user_id"]}):
            raise RunError("replay_not_found", 404)
        return run["archive"]
