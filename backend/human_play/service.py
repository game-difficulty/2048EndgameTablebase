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
RUN_COLUMNS = "id,user_id,browser,variant,request_id,seed,threshold,status,eligibility,reason,created,ended,writer,epoch,permit_until,monitored,state,display_threshold,visible,source"
SUMMARY_COLUMNS = "id,user_id,variant,state,ended,reason,eligibility,has_replay,source"
HISTORY_COLUMNS = "id,variant,ended,reason,has_replay,source,json_extract(state,'$.score') AS score,json_extract(state,'$.board') AS board"
DEFAULT_TIMER_SPLITS = {variant: [str(value) for value in engine.NODES[variant]] for variant in engine.VARIANTS}
def rankable_sql(alias: str = "") -> str:
    """Return the shared public/rankable predicate, optionally table-qualified."""
    # Keep the default qualified as well: the predicate contains correlated
    # EXISTS clauses whose inner tables also have id/user_id columns.
    prefix = f"{alias}." if alias else "human_runs."
    return f"""{prefix}visible=1 AND {prefix}eligibility='eligible' AND (
    ({prefix}source='native' AND ({prefix}reason='game_over' OR {prefix}id IN (SELECT run_id FROM human_rank_approvals)))
    OR ({prefix}source='verse' AND {prefix}reason='imported' AND EXISTS (
        SELECT 1 FROM human_external_claims c WHERE c.provider='verse'
        AND c.status='complete' AND c.user_id={prefix}user_id
        AND {prefix}browser='verse:' || c.id))
    OR ({prefix}source='manual' AND {prefix}reason='imported' AND EXISTS (
        SELECT 1 FROM human_archive_applications a WHERE a.run_id={prefix}id
        AND a.status='approved' AND a.user_id={prefix}user_id)))"""


# Used by rankings, public history, PBs and replay access alike.
PUBLIC_SQL = "visible=1 AND eligibility='eligible'"
RANKABLE_SQL = rankable_sql()


def player_settings(user_id):
    with database() as db:
        row = db.execute("SELECT display_thresholds,timer_splits FROM human_player_settings WHERE user_id=?", (user_id,)).fetchone()
    saved_splits = json.loads(row[1]) if row and row[1] else {}
    return {"display_thresholds": json.loads(row[0]) if row else {},
            "timer_splits": {variant: saved_splits.get(variant, values) for variant, values in DEFAULT_TIMER_SPLITS.items()}}


def _timer_split(value):
    if not isinstance(value, str) or len(value) > 100:
        raise RunError("invalid_timer_splits", 400)
    parts = value.split("+")
    if not 1 <= len(parts) <= 8:
        raise RunError("invalid_timer_splits", 400)
    numbers = []
    for part in parts:
        if not part.isdigit():
            raise RunError("invalid_timer_splits", 400)
        number = int(part)
        if number < 2 or number > 2 ** 31 or number & (number - 1):
            raise RunError("invalid_timer_splits", 400)
        numbers.append(number)
    if any(left < right for left, right in zip(numbers, numbers[1:])):
        raise RunError("invalid_timer_splits", 400)
    return "+".join(str(number) for number in numbers)


def _timer_splits(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(engine.VARIANTS):
        raise RunError("invalid_timer_splits", 400)
    result = {}
    for variant, splits in value.items():
        if not isinstance(splits, list) or len(splits) > 32:
            raise RunError("invalid_timer_splits", 400)
        normalized = [_timer_split(split) for split in splits]
        if len(set(normalized)) != len(normalized):
            raise RunError("invalid_timer_splits", 400)
        result[variant] = normalized
    return result


def save_player_settings(user_id, thresholds, timer_splits=None):
    if set(thresholds) != set(engine.VARIANTS) or any(type(v) is not int or not 0 <= v <= 100000000 for v in thresholds.values()):
        raise RunError("invalid_display_threshold", 400)
    normalized_splits = _timer_splits(timer_splits)
    with database() as db:
        row = db.execute("SELECT timer_splits FROM human_player_settings WHERE user_id=?", (user_id,)).fetchone()
        if normalized_splits is None:
            normalized_splits = json.loads(row[0]) if row and row[0] else DEFAULT_TIMER_SPLITS
        db.execute("INSERT INTO human_player_settings(user_id,display_thresholds,timer_splits) VALUES(?,?,?) "
                   "ON CONFLICT(user_id) DO UPDATE SET display_thresholds=excluded.display_thresholds,timer_splits=excluded.timer_splits",
                   (user_id, json.dumps(thresholds, separators=(',', ':')),
                    json.dumps(normalized_splits, separators=(',', ':'))))
    return {"display_thresholds": thresholds, "timer_splits": normalized_splits}


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
    if not run or run["source"] != "native" or run["user_id"] != user_id or run["browser"] != browser:
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
            recent = db.execute("SELECT count(*) FROM human_runs WHERE user_id=? AND source='native' AND created>?",
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
            settings = db.execute("SELECT display_thresholds FROM human_player_settings WHERE user_id=?", (user_id,)).fetchone()
            display_threshold = json.loads(settings[0]).get(variant, 0) if settings else 0
            db.execute("""INSERT INTO human_runs
                (id,user_id,browser,variant,request_id,seed,threshold,created,writer,state,display_threshold)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""", (run_id, user_id, browser, variant, request_id, seed,
                policies[variant]["threshold"], now, writer, json.dumps(state), display_threshold))
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
    if action not in {"monitor", "append", "reentry", "seal", "live"}:
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
    if action == "live":
        before_nodes = {int(value) for value in state.get("nodes", {})}
        after_nodes = {int(value) for value in next_state.get("nodes", {})}
        if not ({32768, 65536} & (after_nodes - before_nodes)):
            raise RunError("live_milestone_required", 400)
        # Once ordinary high-score monitoring applies, it remains the only
        # allowed append path and carries its stronger online permit checks.
        if next_state["score"] > run["threshold"] or run["monitored"]:
            raise RunError("monitoring_required", 409)
    if not run["monitored"] and next_state["first_over"] is not None and next_state["seq"] > next_state["first_over"]:
        reject_snapshot(run, "monitoring_required", local_seq)
    # Reentry cannot be used as a below-threshold periodic upload API.
    if action == "reentry" and not run["monitored"] and next_state["score"] <= run["threshold"] and data:
        raise RunError("below_threshold", 400)
    if action == "seal" and reason == "game_over" and not engine.game_over(next_state["board"], *engine.VARIANTS[run["variant"]]):
        raise RunError("game_not_over", 400)
    archive = None
    statistics_rate = None
    if action == "seal":
        from . import statistics
        raw = bytes(retained) + data
        rate_tracker = statistics.Rate32kTracker(run['variant'])
        initial = engine.initial(run_id, run['variant'], run['seed'])
        rate_tracker.observe(initial)
        full = engine.advance(initial, run['variant'], raw,
                              run['threshold'], observer=rate_tracker.observe)
        statistics_rate = rate_tracker.result()
        if 'spawnCount' not in next_state:
            # Runs created before spawn counters remain verifiable without a migration.
            full.pop('spawnCount', None)
            full.pop('fourCount', None)
        if full != next_state:
            raise RuntimeError("archive_validation_mismatch")
        archive = gzip.compress(engine.replay_bytes({**run, 'reason': reason}, raw, version=2), compresslevel=6, mtime=0)
    # Replay and compression have finished; only the compare-and-commit holds a write lock.
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        assert_snapshot(db, run)
        if archive is not None and reason == 'game_over':
            from . import rolling
            rolling.ensure_backfill(db)
        if data:
            db.execute("INSERT INTO human_chunks VALUES(?,?,?,?,?,?,?)",
                       (run_id, start, len(data) // 5, digest, hash_bytes(prefix_hash), data, now))
        monitored = bool(run["monitored"] or next_state["score"] > run["threshold"])
        next_epoch = run["epoch"] + (1 if action == "reentry" and writer != run["writer"] else 0)
        expiry = time.time() + PERMIT_SECONDS if monitored else 0
        db.execute("UPDATE human_runs SET state=?,monitored=?,writer=?,epoch=?,permit_until=? WHERE id=?",
                   (json.dumps(next_state), int(monitored), writer, next_epoch, expiry, run_id))
        if archive is not None:
            from . import rating, statistics
            db.execute("UPDATE human_runs SET status='sealed',reason=?,ended=?,archive=?,permit_until=0,visible=?,has_replay=1,single_rating=?,single_rating_version=? WHERE id=?",
                       (reason, now, archive, int(next_state['score'] >= run['display_threshold']),
                        rating.single_rating(run['variant'], next_state['board']), rating.RATING_VERSION, run_id))
            db.execute("DELETE FROM human_chunks WHERE run_id=?", (run_id,))
            statistics.upsert_fact(db, {**run, "ended": now}, next_state["board"],
                                   next_state["score"], statistics_rate)
            if reason == 'game_over' and next_state['score'] >= run['display_threshold'] and run['eligibility'] == 'eligible':
                from . import rolling
                # Eligibility begins only after replay verification, not when
                # the seal request first entered the process.
                qualified_at = time.time()
                rolling.add_run(db, run_id, qualified_at, qualified_at)
                rating.refresh_player(db, user_id, run['variant'])
                statistics.refresh_after_run(db, run_id)
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


def player_id_for_name(username):
    from backend.profile.validation import canonical_display_name_key
    key = canonical_display_name_key(username)
    with auth_db() as db:
        row = db.execute("""SELECT id FROM users WHERE status='active'
            AND (display_name=? OR display_name_key=? OR (display_name_key IS NULL AND display_name=?))
            ORDER BY CASE WHEN display_name=? THEN 0 WHEN display_name_key=? THEN 1 ELSE 2 END, id
            LIMIT 1""", (username, key, username, username, key)).fetchone()
    if not row:
        raise RunError("player_not_found", 404)
    return row["id"]


def summary(run):
    state = json.loads(run["state"])
    return {"id": run["id"], "user_id": run["user_id"], "variant": run["variant"],
            "score": state["score"], "max_tile": max(state["board"]), "moves": state["seq"],
            "elapsed": state["elapsed"], "nodes": state["nodes"], "ended_at": run["ended"],
            "reason": run["reason"], "eligibility": run["eligibility"], "source": run["source"],
            "has_replay": bool(run["has_replay"] if "has_replay" in run.keys() else run["archive"] is not None)}


def history(user_id, viewer_id, before=None, limit=30, variant="all", sort="newest", offset=0,
            page=None):
    if variant != "all" and variant not in engine.VARIANTS:
        raise RunError("invalid_variant", 400)
    if (sort not in {"newest", "oldest", "score_desc", "score_asc"}
            or not 0 <= offset <= 1000000 or not 1 <= limit <= 100
            or (page is not None and page < 1)):
        raise RunError("invalid_history_query", 400)
    with database() as db:
        public = "" if user_id == viewer_id else " AND eligibility='eligible'"
        where = "user_id=? AND status='sealed' AND visible=1" + public
        args = [user_id]
        if variant != "all":
            where += " AND variant=?"
            args.append(variant)
        if before:
            where += " AND ended<?"
            args.append(before)
        total = db.execute(f"SELECT count(*) FROM human_runs WHERE {where}", args).fetchone()[0]
        page_count = max(1, (total + limit - 1) // limit)
        if page is not None:
            page = min(page, page_count)
            offset = (page - 1) * limit
        else:
            page = offset // limit + 1
        order = {"newest": "ended DESC,id DESC", "oldest": "ended ASC,id ASC",
                 "score_desc": "json_extract(state,'$.score') DESC,ended DESC,id DESC",
                 "score_asc": "json_extract(state,'$.score') ASC,ended DESC,id DESC"}[sort]
        rows = db.execute(f"SELECT {HISTORY_COLUMNS} FROM human_runs WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                          (*args, limit + 1, offset)).fetchall()
        stats_where = "user_id=? AND status='sealed' AND visible=1" + public
        aggregate = db.execute("""SELECT count(*) AS games, sum(reason='game_over') AS completed,
            max(json_extract(state,'$.score')) AS best FROM human_runs
            WHERE """ + stats_where, (user_id,)).fetchone()
        bests = db.execute("""SELECT variant,max(json_extract(state,'$.score')) AS score FROM human_runs
            WHERE user_id=? AND status='sealed' AND """ + RANKABLE_SQL + " GROUP BY variant", (user_id,)).fetchall()
    names = identity_map({user_id})
    if user_id not in names:
        raise RunError("player_not_found", 404)
    return {"player": {"id": user_id, "display_name": names[user_id]},
            "is_owner": viewer_id is not None and user_id == viewer_id, "stats": dict(aggregate),
            "bests": {r["variant"]: r["score"] for r in bests},
            "entries": [{"id": r["id"], "variant": r["variant"], "score": r["score"],
                         "ended_at": r["ended"], "reason": r["reason"],
                         "source": r["source"], "has_replay": bool(r["has_replay"]),
                         "board": json.loads(r["board"])} for r in rows[:limit]],
            "next_offset": offset + limit if len(rows) > limit else None,
            "total": total, "page": page, "page_size": limit, "page_count": page_count}


def delete_history_run(run_id: str, user_id: int) -> dict:
    """Soft-delete one owned archived game and refresh every derived public view."""
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("""SELECT id,user_id,variant,status,visible,deleted_by_user
            FROM human_runs WHERE id=? AND user_id=? AND status='sealed'""",
            (run_id, int(user_id))).fetchone()
        if not row:
            raise RunError("run_not_found", 404)
        if row["deleted_by_user"]:
            return {"run_id": run_id, "deleted": True, "already_deleted": True}
        db.execute("""UPDATE human_runs SET visible=0,deleted_by_user=1,deleted_by_user_at=?
            WHERE id=? AND user_id=? AND status='sealed'""", (time.time(), run_id, int(user_id)))

        # Keep the archive and audit evidence, but immediately remove the score
        # from every maintained projection which otherwise outlives the row flag.
        from . import leaderboards, rating, rolling, statistics
        rolling.revoke_run(db, run_id)
        rating.refresh_player(db, int(user_id), row["variant"])
        statistics.rebuild_player(db, int(user_id), row["variant"])
        leaderboards.refresh_run(db, run_id)
    return {"run_id": run_id, "deleted": True, "already_deleted": False}


def best_ten(user_id, viewer_id, variant):
    if variant not in engine.VARIANTS:
        raise RunError("invalid_variant", 400)
    names = identity_map({user_id})
    if user_id not in names:
        raise RunError("player_not_found", 404)
    with database() as db:
        rows = db.execute("""SELECT id,variant,ended,reason,has_replay,source,single_rating,
            json_extract(state,'$.score') AS score,json_extract(state,'$.board') AS board
            FROM human_runs WHERE user_id=? AND variant=? AND status='sealed' AND """ + RANKABLE_SQL +
            " ORDER BY score DESC,ended DESC,id DESC LIMIT 10", (user_id, variant)).fetchall()
        from . import rating
        standings = rating.player_snapshot(db, user_id, variant)
    entries = []
    for row in rows:
        entry = dict(row)
        entry["board"] = json.loads(row["board"])
        entry["four_spawn_rate"] = four_spawn_rate(entry["board"], entry["score"])
        entries.append(entry)
    return {"player": {"id": user_id, "display_name": names[user_id]}, "variant": variant,
            "entries": entries, **standings}


def player_statistics(user_id, variant):
    names = identity_map({user_id})
    if user_id not in names:
        raise RunError("player_not_found", 404)
    from . import statistics
    try:
        with database() as db:
            result = statistics.payload(db, user_id, variant)
    except ValueError as exc:
        raise RunError(str(exc), 400) from exc
    return {"player": {"id": user_id, "display_name": names[user_id]}, **result}


def personal_bests(user_id):
    with database() as db:
        rows = db.execute("SELECT variant,max(json_extract(state,'$.score')) AS score FROM human_runs "
            "WHERE user_id=? AND status='sealed' AND " + RANKABLE_SQL + " GROUP BY variant", (user_id,)).fetchall()
    return {"bests": {row['variant']: row['score'] for row in rows}}


def four_spawn_rate(board, score):
    """Derive the observed 4-spawn rate from a final board and score."""
    if not isinstance(board, list) or not board or isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    board_sum = weighted = 0
    for raw in board:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0 or (raw and raw & (raw - 1)):
            return None
        board_sum += raw
        if raw:
            weighted += raw * (raw.bit_length() - 2)
    # Each spawned 4 contributes four points to this difference. This
    # includes either of the two initial tiles, so no replay is required.
    four_points = weighted - score
    if four_points < 0 or four_points % 4 or (board_sum - four_points) < 0 or (board_sum - four_points) % 2:
        return None
    fours = four_points // 4
    twos = (board_sum - four_points) // 2
    total = fours + twos
    return fours / total if total > 0 else None


def leaderboard(variant, period="all", limit=10, viewer_id=None):
    if variant not in engine.VARIANTS or period not in {"all", "week"}:
        raise RunError("invalid_board", 400)
    if period == "week":
        from . import rolling
        from backend import rolling_leaderboards as core
        with database() as db:
            if (not db.execute("SELECT 1 FROM rolling_meta WHERE key='human_rolling_v1'").fetchone()
                    or core.due(db, variant)):
                db.execute('BEGIN IMMEDIATE')
            rows = rolling.entries(db, variant, limit)
            own = rolling.player_entry(db, variant, viewer_id) if viewer_id is not None else None
        names = identity_map({r['user_id'] for r in rows} | ({own[0]['user_id']} if own else set()))
        entries = []
        previous_score, rank = None, 0
        for row in rows:
            if row['user_id'] not in names:
                continue
            if previous_score != row['score']:
                rank = len(entries) + 1
            previous_score = row['score']
            entries.append({'id': row['run_id'], 'user_id': row['user_id'],
                            'score': row['score'], 'max_tile': row['max_tile'],
                            'source': row['source'], 'has_replay': bool(row['has_replay']),
                            'rank': rank, 'display_name': names[row['user_id']]})
        me = None
        if own and own[0]['user_id'] in names:
            row, own_rank = own
            me = {'id': row['run_id'], 'user_id': row['user_id'], 'score': row['score'],
                  'max_tile': row['max_tile'], 'source': row['source'],
                  'has_replay': bool(row['has_replay']), 'rank': own_rank,
                  'display_name': names[row['user_id']]}
        return {'entries': entries, 'me': me, 'variant': variant, 'period': period}
    with database() as db:
        rows = db.execute(f"""WITH personal AS (
            SELECT {SUMMARY_COLUMNS},json_extract(state,'$.score') AS board_score,
              row_number() OVER (PARTITION BY user_id
                ORDER BY json_extract(state,'$.score') DESC,ended,id) AS personal_rank
            FROM human_runs WHERE variant=? AND status='sealed' AND {RANKABLE_SQL}
          ), best AS (SELECT * FROM personal WHERE personal_rank=1), ranked AS (
            SELECT best.*,rank() OVER (ORDER BY board_score DESC) AS board_rank,
              row_number() OVER (ORDER BY board_score DESC,ended,id) AS board_position
            FROM best
          ) SELECT * FROM ranked WHERE board_position<=? OR user_id=?
          ORDER BY board_position""", (variant, max(1, min(100, limit)), viewer_id or -1)).fetchall()
    names = identity_map({r["user_id"] for r in rows})
    entries, me = [], None
    for row in rows:
        if row["user_id"] not in names:
            continue
        item = {key: value for key, value in summary(row).items() if key in {"id", "user_id", "score", "max_tile", "source", "has_replay"}}
        item = {**item, "rank": row['board_rank'], "display_name": names[row["user_id"]]}
        if row['board_position'] <= limit:
            entries.append(item)
        if viewer_id is not None and row['user_id'] == viewer_id:
            me = item
    return {"entries": entries, "me": me, "variant": variant, "period": period}


def full_leaderboard_catalog():
    from . import leaderboards
    with database() as db:
        return leaderboards.catalog(db)


def full_leaderboard(kind, variant, period="all", page=1, pattern="", target=""):
    from . import leaderboards
    try:
        with database() as db:
            return leaderboards.full_page(db, kind=kind, variant=variant, period=period,
                                          page=page, pattern=pattern, target=target)
    except ValueError as exc:
        raise RunError(str(exc), 400) from exc


def maintain_full_leaderboards():
    from . import leaderboards
    with database() as db:
        return leaderboards.maintain_due_analysis_weeks(db)


def replay_record(run_id, viewer_id):
    with database() as db:
        run = db.execute(f"SELECT user_id,archive,source,({PUBLIC_SQL}) AS publicly_visible FROM human_runs WHERE id=? AND status='sealed'", (run_id,)).fetchone()
        if not run or run["archive"] is None or (run["user_id"] != viewer_id and not run["publicly_visible"]):
            raise RunError("replay_not_found", 404)
        if run["user_id"] not in identity_map({run["user_id"]}):
            raise RunError("replay_not_found", 404)
        return run["archive"], run["source"]


def replay(run_id, viewer_id):
    return replay_record(run_id, viewer_id)[0]
