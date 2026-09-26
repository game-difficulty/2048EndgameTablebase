"""Persist single-game and per-variant Top 10 ratings without reading replay BLOBs."""
from __future__ import annotations

import json
import math
import time

RATING_VERSION = 2


def rating_from_board_sum(variant: str, board_sum: float) -> float:
    board_sum = float(board_sum)
    if board_sum <= 0:
        raise ValueError("invalid_rating_board")
    log_sum = math.log2(board_sum)
    if variant == "4x4":
        return 562.5 * log_sum - 6000
    if variant == "3x4":
        return 562.5 * log_sum - 4280
    if variant == "3x3":
        return 800 * log_sum - 5468
    if variant == "2x4":
        return 0.996313 ** (board_sum - 2365) + 2000 * log_sum - 16531
    raise ValueError("invalid_rating_variant")


def single_rating(variant: str, board: list[int]) -> float:
    return rating_from_board_sum(variant, sum(int(value) for value in board))


def top_rating(variant: str, board_sums) -> float:
    """Rate a B10 set by applying the formula to its mean terminal board sum."""
    values = [float(value) for value in board_sums]
    if not values:
        raise ValueError("invalid_rating_games")
    return rating_from_board_sum(variant, sum(values) / len(values))


def refresh_player(db, user_id: int, variant: str) -> None:
    # The eligibility expression is the same one used by all-time rankings and PBs.
    from .service import RANKABLE_SQL

    rows = db.execute(f"""SELECT id,ended, json_extract(state,'$.score') AS score,
        json_extract(state,'$.board') AS board, single_rating, single_rating_version
        FROM human_runs WHERE user_id=? AND variant=? AND status='sealed'
        AND {RANKABLE_SQL}
        ORDER BY score DESC, ended DESC, id DESC LIMIT 10""", (user_id, variant)).fetchall()
    if not rows:
        db.execute("DELETE FROM human_player_ratings WHERE user_id=? AND variant=?", (user_id, variant))
        return
    board_sums = []
    for row in rows:
        value = row["single_rating"]
        if value is None or row["single_rating_version"] != RATING_VERSION:
            value = single_rating(variant, json.loads(row["board"]))
            db.execute("UPDATE human_runs SET single_rating=?,single_rating_version=? WHERE id=?",
                       (value, RATING_VERSION, row["id"]))
        board_sums.append(sum(int(cell) for cell in json.loads(row["board"])))
    db.execute("""INSERT INTO human_player_ratings
        (user_id,variant,pb_score,pb_run_id,pb_ended_at,rating,game_count,updated_at)
        VALUES(?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id,variant) DO UPDATE SET
          pb_score=excluded.pb_score,pb_run_id=excluded.pb_run_id,
          pb_ended_at=excluded.pb_ended_at,rating=excluded.rating,
          game_count=excluded.game_count,updated_at=excluded.updated_at""",
        (user_id, variant, rows[0]["score"], rows[0]["id"], rows[0]["ended"],
         top_rating(variant, board_sums), len(board_sums), time.time()))


def player_snapshot(db, user_id: int, variant: str) -> dict:
    row = db.execute("""SELECT pb_score,rating,game_count FROM human_player_ratings
        WHERE user_id=? AND variant=?""", (user_id, variant)).fetchone()
    if not row:
        return {"pb_score": None, "pb_rank": None, "rating": None, "ra_rank": None, "rating_games": 0}
    pb_rank = db.execute("""SELECT count(*)+1 FROM human_player_ratings
        WHERE variant=? AND pb_score>?""", (variant, row["pb_score"])).fetchone()[0]
    ra_rank = db.execute("""SELECT count(*)+1 FROM human_player_ratings
        WHERE variant=? AND rating>?""", (variant, row["rating"])).fetchone()[0]
    return {"pb_score": row["pb_score"], "pb_rank": pb_rank, "rating": row["rating"],
            "ra_rank": ra_rank, "rating_games": row["game_count"]}


def init_schema(db) -> None:
    run_columns = {row["name"] for row in db.execute("PRAGMA table_info(human_runs)")}
    column_added = "single_rating" not in run_columns
    version_column_added = "single_rating_version" not in run_columns
    table_missing = not db.execute("""SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='human_player_ratings'""").fetchone()
    if column_added:
        db.execute("ALTER TABLE human_runs ADD COLUMN single_rating REAL")
    if version_column_added:
        db.execute("ALTER TABLE human_runs ADD COLUMN single_rating_version INTEGER")
    db.execute("""CREATE TABLE IF NOT EXISTS human_player_ratings (
        user_id INTEGER NOT NULL, variant TEXT NOT NULL,
        pb_score INTEGER NOT NULL, pb_run_id TEXT, pb_ended_at REAL,
        rating REAL NOT NULL,
        game_count INTEGER NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY(user_id,variant))""")
    rating_columns = {row["name"] for row in db.execute("PRAGMA table_info(human_player_ratings)")}
    leaderboard_columns_added = "pb_run_id" not in rating_columns or "pb_ended_at" not in rating_columns
    if "pb_run_id" not in rating_columns:
        db.execute("ALTER TABLE human_player_ratings ADD COLUMN pb_run_id TEXT")
    if "pb_ended_at" not in rating_columns:
        db.execute("ALTER TABLE human_player_ratings ADD COLUMN pb_ended_at REAL")
    db.execute("""CREATE TABLE IF NOT EXISTS human_rating_meta (
        key TEXT PRIMARY KEY, value TEXT NOT NULL)""")
    db.execute("""CREATE INDEX IF NOT EXISTS human_rating_order
        ON human_player_ratings(variant,rating DESC)""")
    db.execute("""CREATE INDEX IF NOT EXISTS human_rating_pb_order
        ON human_player_ratings(variant,pb_score DESC,pb_ended_at DESC,user_id)""")
    db.execute("""CREATE INDEX IF NOT EXISTS human_rating_candidates
        ON human_runs(user_id,variant,json_extract(state,'$.score') DESC,ended DESC,id DESC)
        WHERE status='sealed' AND visible=1 AND eligibility='eligible'""")
    version = db.execute("SELECT value FROM human_rating_meta WHERE key='formula_version'").fetchone()
    rebuild = (column_added or version_column_added or table_missing or leaderboard_columns_added
               or not version or version[0] != str(RATING_VERSION))
    if not rebuild:
        return
    # Rebuild only each player's top ten. Older runs are calculated if they
    # subsequently enter the top ten; startup never rewrites the full archive.
    db.execute("DELETE FROM human_player_ratings")
    for row in db.execute("""SELECT DISTINCT user_id,variant FROM human_runs
        WHERE status='sealed' AND visible=1 AND eligibility='eligible'"""):
        refresh_player(db, row["user_id"], row["variant"])
    db.execute("""INSERT INTO human_rating_meta(key,value) VALUES('formula_version',?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value""", (str(RATING_VERSION),))
