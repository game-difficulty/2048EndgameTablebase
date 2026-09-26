"""Indexed, replay-free full leaderboards for the Play site."""
from __future__ import annotations

import json
import math
import time

from backend.rolling_leaderboards import WINDOW_SECONDS

from . import engine


PAGE_SIZE = 50
RATE_MIN_32K_GAMES = 10
KINDS = frozenset(("score", "rating", "rate32k", "count", "strength"))


def _board_key(variant: str, pattern: str, target: str) -> str:
    return json.dumps([variant, pattern, target], separators=(",", ":"))


def init_schema(db) -> None:
    had_analysis_catalog = db.execute("""SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='human_analysis_catalog'""").fetchone() is not None
    db.executescript("""
    CREATE TABLE IF NOT EXISTS human_analysis_results (
        summary_id INTEGER PRIMARY KEY REFERENCES human_analysis_summaries(id) ON DELETE CASCADE,
        run_id TEXT NOT NULL, user_id INTEGER NOT NULL, variant TEXT NOT NULL,
        pattern TEXT NOT NULL, target TEXT NOT NULL,
        run_ended_at REAL NOT NULL, result_created_at REAL NOT NULL,
        metric_version INTEGER NOT NULL, grade_version INTEGER NOT NULL,
        weighted_score REAL NOT NULL, grade TEXT NOT NULL,
        mean_goodness_of_fit REAL NOT NULL, mean_ms_per_timed_move REAL NOT NULL,
        max_combo INTEGER NOT NULL, stage_count INTEGER NOT NULL,
        evaluated_moves INTEGER NOT NULL, final_score INTEGER NOT NULL,
        active INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX IF NOT EXISTS human_analysis_results_board
        ON human_analysis_results(variant,pattern,target,active,weighted_score DESC,
            mean_goodness_of_fit DESC,final_score DESC,mean_ms_per_timed_move,run_ended_at DESC,summary_id);
    CREATE INDEX IF NOT EXISTS human_analysis_results_week
        ON human_analysis_results(variant,pattern,target,active,run_ended_at,user_id);
    CREATE INDEX IF NOT EXISTS human_analysis_results_run
        ON human_analysis_results(run_id);
    CREATE TABLE IF NOT EXISTS human_analysis_best_all (
        variant TEXT NOT NULL, pattern TEXT NOT NULL, target TEXT NOT NULL,
        user_id INTEGER NOT NULL, result_id INTEGER NOT NULL,
        weighted_score REAL NOT NULL,
        PRIMARY KEY(variant,pattern,target,user_id)
    );
    CREATE INDEX IF NOT EXISTS human_analysis_best_all_order
        ON human_analysis_best_all(variant,pattern,target,weighted_score DESC,user_id);
    CREATE TABLE IF NOT EXISTS human_analysis_best_week (
        variant TEXT NOT NULL, pattern TEXT NOT NULL, target TEXT NOT NULL,
        user_id INTEGER NOT NULL, result_id INTEGER NOT NULL,
        weighted_score REAL NOT NULL,
        PRIMARY KEY(variant,pattern,target,user_id)
    );
    CREATE INDEX IF NOT EXISTS human_analysis_best_week_order
        ON human_analysis_best_week(variant,pattern,target,weighted_score DESC,user_id);
    CREATE TABLE IF NOT EXISTS human_analysis_week_state (
        board_key TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL, next_expiry_at REAL
    );
    CREATE TABLE IF NOT EXISTS human_full_leaderboard_meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS human_analysis_catalog (
        variant TEXT NOT NULL, pattern TEXT NOT NULL, target TEXT NOT NULL,
        result_count INTEGER NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY(variant,pattern,target)
    );
    """)
    if not had_analysis_catalog:
        _rebuild_analysis_catalog(db)
    from .analysis_grade import GRADE_VERSION
    marker = db.execute("""SELECT value FROM human_full_leaderboard_meta
        WHERE key='analysis_grade_version'""").fetchone()
    if not marker or marker[0] != str(GRADE_VERSION):
        rebuild_analysis_results(db)


def _analysis_values(db, summary_id: int, summary: dict) -> dict | None:
    from .analysis_grade import GRADE_VERSION, grade_result
    from .analysis_summary import poster_goal_tile
    from .service import RANKABLE_SQL

    row = db.execute(f"""SELECT s.run_id,s.user_id,s.pattern,s.target,s.metric_version,s.created,
        r.variant,r.ended,r.state,EXISTS(SELECT 1 FROM human_runs
            WHERE id=s.run_id AND status='sealed' AND {RANKABLE_SQL}) AS rankable
        FROM human_analysis_summaries s JOIN human_runs r ON r.id=s.run_id
        WHERE s.id=?""", (summary_id,)).fetchone()
    if not row:
        return None
    run = summary.get("run") or {}
    state = json.loads(row["state"])
    aggregate = summary.get("aggregate") or {}
    goal = summary.get("goal_tile")
    if goal is None:
        goal = poster_goal_tile(row["pattern"], row["target"], row["variant"])
    result = grade_result(variant=row["variant"], goal_tile=goal,
                          score=int(run.get("score", state.get("score", 0))),
                          aggregate=aggregate)
    if not result or not row["rankable"]:
        return None
    weighted, grade = result
    fit = float(aggregate["mean_goodness_of_fit"])
    mean_ms = float(aggregate["mean_ms_per_timed_move"])
    if not all(math.isfinite(value) for value in (weighted, fit, mean_ms)):
        return None
    return {
        "summary_id": summary_id, "run_id": row["run_id"], "user_id": row["user_id"],
        "variant": row["variant"], "pattern": row["pattern"], "target": row["target"],
        "run_ended_at": row["ended"], "result_created_at": row["created"],
        "metric_version": row["metric_version"], "grade_version": GRADE_VERSION,
        "weighted_score": weighted, "grade": grade,
        "mean_goodness_of_fit": fit, "mean_ms_per_timed_move": mean_ms,
        "max_combo": int(aggregate.get("max_combo") or 0),
        "stage_count": int(aggregate.get("stage_count") or 0),
        "evaluated_moves": int(aggregate.get("evaluated_moves") or 0),
        "final_score": int(run.get("score", state.get("score", 0))), "active": 1,
    }


def _insert_result(db, values: dict) -> None:
    columns = tuple(values)
    db.execute(f"""INSERT INTO human_analysis_results ({','.join(columns)})
        VALUES ({','.join('?' for _ in columns)})
        ON CONFLICT(summary_id) DO UPDATE SET
        {','.join(f'{column}=excluded.{column}' for column in columns if column != 'summary_id')}""",
        tuple(values[column] for column in columns))


def _result_order() -> str:
    return ("weighted_score DESC,mean_goodness_of_fit DESC,final_score DESC,"
            "mean_ms_per_timed_move ASC,run_ended_at DESC,summary_id ASC")


def _refresh_analysis_catalog_board(db, variant: str, pattern: str, target: str,
                                    now: float | None = None) -> None:
    """Update one tiny catalog row on the low-frequency result write path."""
    result_count = int(db.execute("""SELECT count(*) FROM human_analysis_results
        WHERE variant=? AND pattern=? AND target=? AND active=1""",
        (variant, pattern, target)).fetchone()[0])
    if result_count:
        db.execute("""INSERT INTO human_analysis_catalog
            (variant,pattern,target,result_count,updated_at) VALUES(?,?,?,?,?)
            ON CONFLICT(variant,pattern,target) DO UPDATE SET
              result_count=excluded.result_count,updated_at=excluded.updated_at""",
            (variant, pattern, target, result_count,
             time.time() if now is None else float(now)))
    else:
        db.execute("""DELETE FROM human_analysis_catalog
            WHERE variant=? AND pattern=? AND target=?""", (variant, pattern, target))


def _rebuild_analysis_catalog(db, now: float | None = None) -> None:
    """Rebuild once for migrations/full repairs; page reads never aggregate results."""
    now = time.time() if now is None else float(now)
    db.execute("DELETE FROM human_analysis_catalog")
    db.execute("""INSERT INTO human_analysis_catalog
        (variant,pattern,target,result_count,updated_at)
        SELECT variant,pattern,target,count(*),? FROM human_analysis_results
        WHERE active=1 GROUP BY variant,pattern,target""", (now,))


def _refresh_best_user(db, variant: str, pattern: str, target: str,
                       user_id: int, *, week: bool, now: float | None = None) -> None:
    table = "human_analysis_best_week" if week else "human_analysis_best_all"
    now = time.time() if now is None else float(now)
    time_sql = " AND run_ended_at>=? AND run_ended_at<?" if week else ""
    args = [variant, pattern, target, user_id]
    if week:
        args.extend((now - WINDOW_SECONDS, now))
    result = db.execute(f"""SELECT * FROM human_analysis_results
        WHERE variant=? AND pattern=? AND target=? AND user_id=? AND active=1{time_sql}
        ORDER BY {_result_order()} LIMIT 1""", args).fetchone()
    if result:
        db.execute(f"""INSERT INTO {table}
            (variant,pattern,target,user_id,result_id,weighted_score) VALUES(?,?,?,?,?,?)
            ON CONFLICT(variant,pattern,target,user_id) DO UPDATE SET
              result_id=excluded.result_id,weighted_score=excluded.weighted_score""",
            (variant, pattern, target, user_id, result["summary_id"], result["weighted_score"]))
    else:
        db.execute(f"DELETE FROM {table} WHERE variant=? AND pattern=? AND target=? AND user_id=?",
                   (variant, pattern, target, user_id))


def _refresh_week_state(db, variant: str, pattern: str, target: str,
                        now: float | None = None) -> None:
    now = time.time() if now is None else float(now)
    earliest = db.execute("""SELECT MIN(r.run_ended_at) FROM human_analysis_best_week b
        JOIN human_analysis_results r ON r.summary_id=b.result_id
        WHERE b.variant=? AND b.pattern=? AND b.target=?""",
        (variant, pattern, target)).fetchone()[0]
    key = _board_key(variant, pattern, target)
    db.execute("""INSERT INTO human_analysis_week_state(board_key,version,updated_at,next_expiry_at)
        VALUES(?,1,?,?) ON CONFLICT(board_key) DO UPDATE SET
          version=human_analysis_week_state.version+1,updated_at=excluded.updated_at,
          next_expiry_at=excluded.next_expiry_at""",
        (key, now, earliest + WINDOW_SECONDS if earliest is not None else None))


def maintain_analysis_week(db, variant: str, pattern: str, target: str,
                           now: float | None = None) -> None:
    now = time.time() if now is None else float(now)
    key = _board_key(variant, pattern, target)
    state = db.execute("SELECT next_expiry_at FROM human_analysis_week_state WHERE board_key=?",
                       (key,)).fetchone()
    if state and (state["next_expiry_at"] is None or state["next_expiry_at"] >= now):
        return
    if state is None:
        users = db.execute("""SELECT DISTINCT user_id FROM human_analysis_results
            WHERE variant=? AND pattern=? AND target=? AND active=1
              AND run_ended_at>=? AND run_ended_at<?""",
            (variant, pattern, target, now - WINDOW_SECONDS, now)).fetchall()
    else:
        users = db.execute("""SELECT b.user_id FROM human_analysis_best_week b
            JOIN human_analysis_results r ON r.summary_id=b.result_id
            WHERE b.variant=? AND b.pattern=? AND b.target=? AND r.run_ended_at<?""",
            (variant, pattern, target, now - WINDOW_SECONDS)).fetchall()
    for row in users:
        _refresh_best_user(db, variant, pattern, target, row["user_id"], week=True, now=now)
    _refresh_week_state(db, variant, pattern, target, now)


def maintain_due_analysis_weeks(db, now: float | None = None, limit: int = 100) -> int:
    """Refresh a bounded batch so page reads normally find a ready weekly index."""
    now = time.time() if now is None else float(now)
    rows = db.execute("""SELECT board_key FROM human_analysis_week_state
        WHERE next_expiry_at IS NOT NULL AND next_expiry_at<?
        ORDER BY next_expiry_at LIMIT ?""", (now, max(1, min(int(limit), 1000)))).fetchall()
    for row in rows:
        variant, pattern, target = json.loads(row["board_key"])
        maintain_analysis_week(db, variant, pattern, str(target), now)
    return len(rows)


def upsert_analysis_summary(db, summary_id: int, summary: dict,
                            *, refresh: bool = True, now: float | None = None) -> None:
    old = db.execute("""SELECT variant,pattern,target,user_id FROM human_analysis_results
        WHERE summary_id=?""", (summary_id,)).fetchone()
    values = _analysis_values(db, summary_id, summary)
    affected = set()
    if old:
        affected.add((old["variant"], old["pattern"], old["target"], old["user_id"]))
    if values:
        _insert_result(db, values)
        affected.add((values["variant"], values["pattern"], values["target"], values["user_id"]))
    else:
        db.execute("DELETE FROM human_analysis_results WHERE summary_id=?", (summary_id,))
    if refresh:
        for variant, pattern, target, user_id in affected:
            _refresh_best_user(db, variant, pattern, target, user_id, week=False, now=now)
            _refresh_best_user(db, variant, pattern, target, user_id, week=True, now=now)
            _refresh_week_state(db, variant, pattern, target, now)
        for variant, pattern, target in {
                (item[0], item[1], item[2]) for item in affected}:
            _refresh_analysis_catalog_board(db, variant, pattern, target, now)


def refresh_run(db, run_id: str, now: float | None = None) -> None:
    rows = db.execute("""SELECT s.id,s.summary_json FROM human_analysis_summaries s
        WHERE s.run_id=?""", (run_id,)).fetchall()
    for row in rows:
        upsert_analysis_summary(db, row["id"], json.loads(row["summary_json"]), now=now)


def rebuild_analysis_bests(db, now: float | None = None) -> None:
    now = time.time() if now is None else float(now)
    db.execute("DELETE FROM human_analysis_best_all")
    db.execute("DELETE FROM human_analysis_best_week")
    db.execute("DELETE FROM human_analysis_week_state")
    keys = db.execute("""SELECT DISTINCT variant,pattern,target,user_id
        FROM human_analysis_results WHERE active=1""").fetchall()
    boards = set()
    for row in keys:
        key = (row["variant"], row["pattern"], row["target"])
        boards.add(key)
        _refresh_best_user(db, *key, row["user_id"], week=False, now=now)
        _refresh_best_user(db, *key, row["user_id"], week=True, now=now)
    for board in boards:
        _refresh_week_state(db, *board, now)
    _rebuild_analysis_catalog(db, now)


def rebuild_analysis_results(db) -> None:
    from .analysis_grade import GRADE_VERSION
    db.execute("DELETE FROM human_analysis_results")
    rows = db.execute("SELECT id,summary_json FROM human_analysis_summaries ORDER BY id").fetchall()
    for row in rows:
        upsert_analysis_summary(db, row["id"], json.loads(row["summary_json"]), refresh=False)
    rebuild_analysis_bests(db)
    db.execute("""INSERT INTO human_full_leaderboard_meta(key,value)
        VALUES('analysis_grade_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value""",
        (str(GRADE_VERSION),))


def catalog(db) -> dict:
    rows = db.execute("""SELECT variant,pattern,target,result_count
        FROM human_analysis_catalog WHERE result_count>0
        ORDER BY variant,pattern,target""").fetchall()
    strength = [{"variant": row["variant"], "pattern": row["pattern"],
                 "target": str(row["target"]), "results": int(row["result_count"])}
                for row in rows]
    strength.sort(key=lambda item: (item["variant"], item["pattern"],
                                    (0, int(item["target"]))
                                    if item["target"].isdigit() else (1, item["target"])))
    return {"strength": strength, "page_size": PAGE_SIZE,
            "rate_min_32k_games": RATE_MIN_32K_GAMES}


def _page(value: int) -> int:
    return max(1, min(100000, int(value)))


def _names(rows) -> dict:
    from .service import identity_map
    return identity_map({row["user_id"] for row in rows})


def _finish(kind: str, variant: str, period: str, page: int, total: int,
            rows, entries: list[dict], **extra) -> dict:
    names = _names(rows)
    visible = [{**entry, "display_name": names[entry["user_id"]]}
               for entry in entries if entry["user_id"] in names]
    return {"type": kind, "variant": variant, "period": period,
            "page": page, "page_size": PAGE_SIZE, "total": total,
            "page_count": max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
            "entries": visible, **extra}


def full_page(db, *, kind: str, variant: str, period: str = "all",
              page: int = 1, pattern: str = "", target: str = "") -> dict:
    if kind not in KINDS or variant not in engine.VARIANTS:
        raise ValueError("invalid_leaderboard")
    page = _page(page)
    offset = (page - 1) * PAGE_SIZE
    if kind == "score":
        if period not in {"all", "week"}:
            raise ValueError("invalid_leaderboard_period")
        if period == "week":
            from . import rolling
            from backend import rolling_leaderboards as core
            rolling.ensure_backfill(db)
            core.maintain(db, variant)
            total = db.execute("SELECT count(*) FROM rolling_player_best WHERE board_key=?",
                               (variant,)).fetchone()[0]
            rows = db.execute("""SELECT p.user_id,p.run_id,c.score,c.achieved_at,
                c.has_replay,c.source FROM rolling_player_best p
                JOIN rolling_candidates c ON c.run_id=p.run_id
                WHERE p.board_key=? ORDER BY p.score DESC,p.achieved_at DESC,p.user_id,p.run_id
                LIMIT ? OFFSET ?""", (variant, PAGE_SIZE, offset)).fetchall()
            entries = [{"rank": offset + index + 1, "user_id": row["user_id"],
                        "run_id": row["run_id"], "score": row["score"],
                        "ended_at": row["achieved_at"], "has_replay": bool(row["has_replay"]),
                        "source": row["source"]} for index, row in enumerate(rows)]
        else:
            total = db.execute("SELECT count(*) FROM human_player_ratings WHERE variant=?",
                               (variant,)).fetchone()[0]
            rows = db.execute("""SELECT p.user_id,p.pb_run_id AS run_id,p.pb_score AS score,
                p.pb_ended_at AS ended_at,r.has_replay,r.source
                FROM human_player_ratings p JOIN human_runs r ON r.id=p.pb_run_id
                WHERE p.variant=? ORDER BY p.pb_score DESC,p.pb_ended_at DESC,p.user_id
                LIMIT ? OFFSET ?""", (variant, PAGE_SIZE, offset)).fetchall()
            entries = [{"rank": offset + index + 1, "user_id": row["user_id"],
                        "run_id": row["run_id"], "score": row["score"],
                        "ended_at": row["ended_at"], "has_replay": bool(row["has_replay"]),
                        "source": row["source"]} for index, row in enumerate(rows)]
        return _finish(kind, variant, period, page, total, rows, entries)

    if period != "all" and kind not in {"strength"}:
        raise ValueError("invalid_leaderboard_period")
    if kind == "rating":
        total = db.execute("SELECT count(*) FROM human_player_ratings WHERE variant=?",
                           (variant,)).fetchone()[0]
        rows = db.execute("""SELECT r.user_id,r.rating,r.game_count,s.b10_score
            FROM human_player_ratings r LEFT JOIN human_player_statistics s
              ON s.user_id=r.user_id AND s.variant=r.variant
            WHERE r.variant=? ORDER BY r.rating DESC,r.pb_score DESC,r.user_id LIMIT ? OFFSET ?""",
            (variant, PAGE_SIZE, offset)).fetchall()
        entries = [{"rank": offset + index + 1, **dict(row)} for index, row in enumerate(rows)]
        return _finish(kind, variant, "all", page, total, rows, entries)

    if kind == "rate32k":
        if variant != "4x4":
            raise ValueError("invalid_leaderboard_variant")
        where = "variant='4x4' AND rate_32k_candidate_games>=? AND rate_32k_total>0"
        total = db.execute(f"SELECT count(*) FROM human_player_statistics WHERE {where}",
                           (RATE_MIN_32K_GAMES,)).fetchone()[0]
        rows = db.execute(f"""SELECT user_id,rate_32k_value,rate_32k_passed,rate_32k_total,
            rate_32k_covered_games,rate_32k_candidate_games FROM human_player_statistics
            WHERE {where} ORDER BY rate_32k_value DESC,rate_32k_total DESC,
              rate_32k_candidate_games DESC,user_id LIMIT ? OFFSET ?""",
            (RATE_MIN_32K_GAMES, PAGE_SIZE, offset)).fetchall()
        entries = [{"rank": offset + index + 1, **dict(row)} for index, row in enumerate(rows)]
        return _finish(kind, variant, "all", page, total, rows, entries,
                       minimum_32k_games=RATE_MIN_32K_GAMES)

    if kind == "count":
        total = db.execute("SELECT count(*) FROM human_player_statistics WHERE variant=?",
                           (variant,)).fetchone()[0]
        rows = db.execute("""SELECT user_id,primary_achievement_count,game_count
            FROM human_player_statistics WHERE variant=?
            ORDER BY primary_achievement_count DESC,game_count DESC,user_id LIMIT ? OFFSET ?""",
            (variant, PAGE_SIZE, offset)).fetchall()
        entries = [{"rank": offset + index + 1, **dict(row)} for index, row in enumerate(rows)]
        labels = {"4x4": "32K", "3x4": "4096", "3x3": "1024", "2x4": "512"}
        return _finish(kind, variant, "all", page, total, rows, entries,
                       achievement=labels[variant])

    if variant != "4x4" or period not in {"all", "week"} or not pattern or not target:
        raise ValueError("invalid_strength_leaderboard")
    if period == "week":
        maintain_analysis_week(db, variant, pattern, target)
    table = "human_analysis_best_week" if period == "week" else "human_analysis_best_all"
    total = db.execute(f"SELECT count(*) FROM {table} WHERE variant=? AND pattern=? AND target=?",
                       (variant, pattern, target)).fetchone()[0]
    rows = db.execute(f"""SELECT b.user_id,r.summary_id,r.run_id,r.grade,
        r.mean_goodness_of_fit,r.mean_ms_per_timed_move,r.max_combo,r.stage_count,
        r.evaluated_moves,r.final_score,r.run_ended_at,hr.source,hr.has_replay
        FROM {table} b JOIN human_analysis_results r ON r.summary_id=b.result_id
        JOIN human_runs hr ON hr.id=r.run_id
        WHERE b.variant=? AND b.pattern=? AND b.target=?
        ORDER BY b.weighted_score DESC,r.mean_goodness_of_fit DESC,r.final_score DESC,
          r.mean_ms_per_timed_move ASC,r.run_ended_at DESC,r.summary_id ASC
        LIMIT ? OFFSET ?""", (variant, pattern, target, PAGE_SIZE, offset)).fetchall()
    entries = [{"rank": offset + index + 1, **dict(row)} for index, row in enumerate(rows)]
    return _finish(kind, variant, period, page, total, rows, entries,
                   pattern=pattern, target=target)
