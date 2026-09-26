"""Small, replay-free player statistics and progression series."""
from __future__ import annotations

import json
import math
import time


STATS_VERSION = 5
RATE_SERIES_MIN_GAMES = 10
VARIANTS = ("4x4", "3x4", "2x4", "3x3")

FEATURES = {
    "4x4": (
        ("16k", "16K"), ("32k", "32K"),
        ("32k_16k", "16/32"),
        ("32k_16k_8k", "24/32"),
        ("32k_16k_8k_4k", "28/32"),
        ("32k_16k_8k_4k_2k", "30/32"),
        ("32k_16k_8k_4k_2k_1k", "31/32"),
    ),
    "3x4": (
        ("4k", "4K"), ("4k_2k", "4K + 2K"),
        ("4k_2k_1k", "4K + 2K + 1K"),
        ("4k_2k_1k_512", "4K + 2K + 1K + 512"),
        ("4k_2k_1k_512_256", "4K + 2K + 1K + 512 + 256"),
        ("4k_2k_1k_512_256_128", "4K + 2K + 1K + 512 + 256 + 128"),
    ),
    "3x3": (
        ("full", "满盘"), ("1k", "1K"), ("second_full", "二阶满盘"),
        ("1k_512", "1K + 512"), ("third_full", "三阶满盘"),
    ),
    "2x4": (
        ("full", "满盘"), ("tile_512", "512"), ("second_full", "二阶满盘"),
        ("512_256", "512 + 256"), ("third_full", "三阶满盘"),
    ),
}


def feature_mask(variant: str, board: list[int]) -> int:
    values = [int(value) for value in board]
    present = set(values)
    maximum = max(values, default=0)
    flags: list[bool]
    if variant == "4x4":
        flags = [maximum >= 16384, maximum >= 32768,
                 maximum >= 32768 and 16384 in present,
                 maximum >= 32768 and {16384, 8192}.issubset(present),
                 maximum >= 32768 and {16384, 8192, 4096}.issubset(present),
                 maximum >= 32768 and {16384, 8192, 4096, 2048}.issubset(present),
                 maximum >= 32768 and {16384, 8192, 4096, 2048, 1024}.issubset(present)]
    elif variant == "3x4":
        flags = [maximum >= 4096,
                 maximum >= 4096 and 2048 in present,
                 maximum >= 4096 and {2048, 1024}.issubset(present),
                 maximum >= 4096 and {2048, 1024, 512}.issubset(present),
                 maximum >= 4096 and {2048, 1024, 512, 256}.issubset(present),
                 maximum >= 4096 and {2048, 1024, 512, 256, 128}.issubset(present)]
    elif variant == "3x3":
        flags = [sorted(values) == [2, 4, 8, 16, 32, 64, 128, 256, 512],
                 maximum >= 1024,
                 sorted(values) == [2, 4, 8, 16, 32, 64, 128, 256, 1024],
                 maximum >= 1024 and 512 in present,
                 sorted(values) == [2, 4, 8, 16, 32, 64, 128, 512, 1024]]
    elif variant == "2x4":
        flags = [sorted(values) == [2, 4, 8, 16, 32, 64, 128, 256],
                 maximum >= 512,
                 sorted(values) == [2, 4, 8, 16, 32, 64, 128, 512],
                 maximum >= 512 and 256 in present,
                 sorted(values) == [2, 4, 8, 16, 32, 64, 256, 512]]
    else:
        raise ValueError("invalid_statistics_variant")
    return sum((1 << index) for index, enabled in enumerate(flags) if enabled)


class Rate32kTracker:
    """Retain the terminal board for the endpoint-based 32K aggregate rate."""

    def __init__(self, variant: str):
        self.variant = variant
        self.previous: list[int] | None = None
        self.counter = None

    def observe(self, state: dict) -> None:
        if self.variant == "4x4":
            self.previous = list(state["board"])

    def result(self) -> tuple[int | None, int | None, int]:
        if self.variant != "4x4" or self.previous is None:
            return None, None, 1
        passed, total = rate32k_fact(feature_mask("4x4", self.previous))
        return passed, total, 1


def rate32k_fact(mask: int) -> tuple[int | None, int | None]:
    """Return this terminal board's passed and attempted 32K stages.

    The five transitions are 32K -> 16/32 -> 24/32 -> 28/32 ->
    30/32 -> 31/32. A run attempts the next stage after every level it
    reaches; reaching 31/32 completes all five attempts.
    """
    if not (mask & (1 << 1)):
        return None, None
    passed = 0
    for index in range(2, 7):
        if not (mask & (1 << index)):
            break
        passed += 1
    return passed, min(passed + 1, 5)


def unavailable_rate() -> tuple[None, None, int]:
    return None, None, 0


def init_schema(db) -> None:
    rate_series_missing = not db.execute("""SELECT 1 FROM sqlite_master
        WHERE type='table' AND name='human_player_rate32k_series'""").fetchone()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS human_run_statistics (
        run_id TEXT PRIMARY KEY REFERENCES human_runs(id),
        user_id INTEGER NOT NULL, variant TEXT NOT NULL, ended_at REAL NOT NULL,
        score INTEGER NOT NULL, board_sum INTEGER NOT NULL, max_tile INTEGER NOT NULL,
        single_rating REAL NOT NULL, feature_mask INTEGER NOT NULL,
        rate_32k_passed INTEGER, rate_32k_total INTEGER,
        rate_32k_coverage INTEGER NOT NULL DEFAULT 0,
        stats_version INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS human_run_statistics_player
        ON human_run_statistics(user_id,variant,ended_at,run_id);
    CREATE INDEX IF NOT EXISTS human_run_statistics_score
        ON human_run_statistics(user_id,variant,score DESC,ended_at DESC,run_id DESC);
    CREATE TABLE IF NOT EXISTS human_player_statistics (
        user_id INTEGER NOT NULL, variant TEXT NOT NULL,
        game_count INTEGER NOT NULL, pb_score INTEGER,
        b10_score INTEGER, b10_rating REAL,
        primary_achievement_count INTEGER NOT NULL DEFAULT 0,
        rate_32k_value REAL,
        features_json TEXT NOT NULL, rate_32k_passed INTEGER NOT NULL,
        rate_32k_total INTEGER NOT NULL, rate_32k_covered_games INTEGER NOT NULL,
        rate_32k_candidate_games INTEGER NOT NULL,
        stats_version INTEGER NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY(user_id,variant)
    );
    CREATE TABLE IF NOT EXISTS human_player_stat_series (
        user_id INTEGER NOT NULL, variant TEXT NOT NULL, run_id TEXT NOT NULL,
        game_index INTEGER NOT NULL, ended_at REAL NOT NULL,
        pb_score INTEGER NOT NULL, b10_score INTEGER, b10_rating REAL NOT NULL,
        stats_version INTEGER NOT NULL,
        PRIMARY KEY(user_id,variant,run_id)
    );
    CREATE INDEX IF NOT EXISTS human_player_stat_series_order
        ON human_player_stat_series(user_id,variant,game_index);
    CREATE TABLE IF NOT EXISTS human_player_rate32k_series (
        user_id INTEGER NOT NULL, run_id TEXT NOT NULL,
        game_index INTEGER NOT NULL, ended_at REAL NOT NULL,
        passed INTEGER NOT NULL, total INTEGER NOT NULL,
        value REAL NOT NULL, stats_version INTEGER NOT NULL,
        PRIMARY KEY(user_id,run_id)
    );
    CREATE INDEX IF NOT EXISTS human_player_rate32k_series_order
        ON human_player_rate32k_series(user_id,game_index);
    """)
    player_columns = {row["name"] for row in db.execute("PRAGMA table_info(human_player_statistics)")}
    leaderboard_columns_added = ("primary_achievement_count" not in player_columns
                                 or "rate_32k_value" not in player_columns)
    if "primary_achievement_count" not in player_columns:
        db.execute("ALTER TABLE human_player_statistics ADD COLUMN primary_achievement_count INTEGER NOT NULL DEFAULT 0")
    if "rate_32k_value" not in player_columns:
        db.execute("ALTER TABLE human_player_statistics ADD COLUMN rate_32k_value REAL")
    db.execute("""CREATE INDEX IF NOT EXISTS human_statistics_achievement_order
        ON human_player_statistics(variant,primary_achievement_count DESC,game_count DESC,user_id)""")
    db.execute("""CREATE INDEX IF NOT EXISTS human_statistics_rate_order
        ON human_player_statistics(variant,rate_32k_value DESC,rate_32k_total DESC,user_id)""")
    _backfill_missing_facts(db)
    if rate_series_missing or leaderboard_columns_added:
        for row in db.execute("""SELECT DISTINCT user_id,variant
            FROM human_run_statistics""").fetchall():
            rebuild_player(db, row["user_id"], row["variant"])


def _backfill_missing_facts(db) -> None:
    rows = db.execute("""SELECT r.id,r.user_id,r.variant,r.ended,r.state
        FROM human_runs r LEFT JOIN human_run_statistics s ON s.run_id=r.id
        WHERE r.status='sealed' AND r.ended IS NOT NULL
        AND (s.run_id IS NULL OR s.stats_version<>?)""", (STATS_VERSION,)).fetchall()
    affected: set[tuple[int, str]] = set()
    for row in rows:
        state = json.loads(row["state"])
        upsert_fact(db, dict(row), state["board"], state["score"])
        affected.add((row["user_id"], row["variant"]))
    for user_id, variant in affected:
        rebuild_player(db, user_id, variant)


def upsert_fact(db, run: dict, board: list[int], score: int,
                _rate: tuple[int | None, int | None, int] | None = None) -> None:
    from .rating import single_rating

    mask = feature_mask(run["variant"], board)
    passed, total = rate32k_fact(mask) if run["variant"] == "4x4" else (None, None)
    coverage = int(run["variant"] == "4x4")
    db.execute("""INSERT INTO human_run_statistics
        (run_id,user_id,variant,ended_at,score,board_sum,max_tile,single_rating,
         feature_mask,rate_32k_passed,rate_32k_total,rate_32k_coverage,stats_version)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
          user_id=excluded.user_id,variant=excluded.variant,ended_at=excluded.ended_at,
          score=excluded.score,board_sum=excluded.board_sum,max_tile=excluded.max_tile,
          single_rating=excluded.single_rating,feature_mask=excluded.feature_mask,
          rate_32k_passed=excluded.rate_32k_passed,
          rate_32k_total=excluded.rate_32k_total,
          rate_32k_coverage=excluded.rate_32k_coverage,
          stats_version=excluded.stats_version""",
        (run["id"], run["user_id"], run["variant"], run["ended"], int(score),
         sum(board), max(board, default=0), single_rating(run["variant"], board),
         mask, passed, total, coverage, STATS_VERSION))


def _insert_top(top: list[dict], row: dict) -> list[dict]:
    top.append(row)
    top.sort(key=lambda item: (item["score"], item["ended_at"], item["run_id"]), reverse=True)
    return top[:10]


def rebuild_player(db, user_id: int, variant: str) -> None:
    from .rating import top_rating
    from .service import RANKABLE_SQL

    rows = db.execute(f"""SELECT s.* FROM human_run_statistics s
        JOIN human_runs ON human_runs.id=s.run_id
        WHERE s.user_id=? AND s.variant=? AND human_runs.status='sealed' AND {RANKABLE_SQL}
        ORDER BY s.ended_at,s.run_id""", (user_id, variant)).fetchall()
    db.execute("DELETE FROM human_player_stat_series WHERE user_id=? AND variant=?",
               (user_id, variant))
    if variant == "4x4":
        db.execute("DELETE FROM human_player_rate32k_series WHERE user_id=?", (user_id,))
    if not rows:
        db.execute("DELETE FROM human_player_statistics WHERE user_id=? AND variant=?",
                   (user_id, variant))
        return
    counts = [0] * len(FEATURES[variant])
    rate_passed = rate_total = covered = candidates = 0
    top: list[dict] = []
    previous = None
    for game_index, raw in enumerate(rows, 1):
        row = dict(raw)
        for index in range(len(counts)):
            counts[index] += int(bool(row["feature_mask"] & (1 << index)))
        if variant == "4x4" and row["max_tile"] >= 32768:
            candidates += 1
            if row["rate_32k_coverage"] and row["rate_32k_total"] is not None:
                covered += 1
                rate_passed += row["rate_32k_passed"]
                rate_total += row["rate_32k_total"]
                db.execute("""INSERT INTO human_player_rate32k_series
                    (user_id,run_id,game_index,ended_at,passed,total,value,stats_version)
                    VALUES(?,?,?,?,?,?,?,?)""",
                    (user_id, row["run_id"], candidates, row["ended_at"], rate_passed,
                     rate_total, rate_passed / rate_total, STATS_VERSION))
        top = _insert_top(top, row)
        current = (top[0]["score"], top[9]["score"] if len(top) == 10 else None,
                   top_rating(variant, (item["board_sum"] for item in top)))
        if current != previous:
            db.execute("""INSERT INTO human_player_stat_series
                (user_id,variant,run_id,game_index,ended_at,pb_score,b10_score,
                 b10_rating,stats_version) VALUES(?,?,?,?,?,?,?,?,?)""",
                (user_id, variant, row["run_id"], game_index, row["ended_at"],
                 current[0], current[1], current[2], STATS_VERSION))
            previous = current
    features = {key: counts[index] for index, (key, _) in enumerate(FEATURES[variant])}
    primary_count = counts[1 if variant == "4x4" else 0 if variant == "3x4" else 1]
    rate_value = rate_passed / rate_total if rate_total else None
    db.execute("""INSERT INTO human_player_statistics
        (user_id,variant,game_count,pb_score,b10_score,b10_rating,
         primary_achievement_count,rate_32k_value,features_json,
         rate_32k_passed,rate_32k_total,rate_32k_covered_games,
         rate_32k_candidate_games,stats_version,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id,variant) DO UPDATE SET
          game_count=excluded.game_count,pb_score=excluded.pb_score,
          b10_score=excluded.b10_score,b10_rating=excluded.b10_rating,
          primary_achievement_count=excluded.primary_achievement_count,
          rate_32k_value=excluded.rate_32k_value,
          features_json=excluded.features_json,
          rate_32k_passed=excluded.rate_32k_passed,
          rate_32k_total=excluded.rate_32k_total,
          rate_32k_covered_games=excluded.rate_32k_covered_games,
          rate_32k_candidate_games=excluded.rate_32k_candidate_games,
          stats_version=excluded.stats_version,updated_at=excluded.updated_at""",
        (user_id, variant, len(rows), top[0]["score"],
         top[9]["score"] if len(top) == 10 else None,
          top_rating(variant, (item["board_sum"] for item in top)),
         primary_count, rate_value, json.dumps(features, separators=(",", ":")), rate_passed, rate_total,
         covered, candidates, STATS_VERSION, time.time()))


def refresh_after_run(db, run_id: str) -> None:
    """Append the usual chronological native result; rebuild only for backfills."""
    from .rating import top_rating
    from .service import RANKABLE_SQL

    row = db.execute(f"""SELECT s.* FROM human_run_statistics s
        JOIN human_runs ON human_runs.id=s.run_id WHERE s.run_id=? AND human_runs.status='sealed'
        AND {RANKABLE_SQL}""", (run_id,)).fetchone()
    if not row:
        return
    row = dict(row)
    current = db.execute("""SELECT * FROM human_player_statistics
        WHERE user_id=? AND variant=?""", (row["user_id"], row["variant"])).fetchone()
    later = db.execute(f"""SELECT 1 FROM human_run_statistics s
        JOIN human_runs ON human_runs.id=s.run_id
        WHERE s.user_id=? AND s.variant=? AND s.run_id<>? AND human_runs.status='sealed'
        AND {RANKABLE_SQL} AND (s.ended_at>? OR (s.ended_at=? AND s.run_id>?)) LIMIT 1""",
        (row["user_id"], row["variant"], run_id, row["ended_at"],
         row["ended_at"], run_id)).fetchone()
    if not current or later:
        rebuild_player(db, row["user_id"], row["variant"])
        return
    top = db.execute(f"""SELECT s.score,s.board_sum FROM human_run_statistics s
        JOIN human_runs ON human_runs.id=s.run_id
        WHERE s.user_id=? AND s.variant=? AND human_runs.status='sealed' AND {RANKABLE_SQL}
        ORDER BY s.score DESC,s.ended_at DESC,s.run_id DESC LIMIT 10""",
        (row["user_id"], row["variant"])).fetchall()
    game_count = current["game_count"] + 1
    pb_score = top[0]["score"]
    b10_score = top[9]["score"] if len(top) == 10 else None
    b10_rating = top_rating(row["variant"], (item["board_sum"] for item in top))
    features = json.loads(current["features_json"])
    for index, (key, _) in enumerate(FEATURES[row["variant"]]):
        features[key] = features.get(key, 0) + int(bool(row["feature_mask"] & (1 << index)))
    rate_passed, rate_total = current["rate_32k_passed"], current["rate_32k_total"]
    covered, candidates = current["rate_32k_covered_games"], current["rate_32k_candidate_games"]
    if row["variant"] == "4x4" and row["max_tile"] >= 32768:
        candidates += 1
        if row["rate_32k_coverage"] and row["rate_32k_total"] is not None:
            covered += 1
            rate_passed += row["rate_32k_passed"]
            rate_total += row["rate_32k_total"]
            db.execute("""INSERT OR REPLACE INTO human_player_rate32k_series
                (user_id,run_id,game_index,ended_at,passed,total,value,stats_version)
                VALUES(?,?,?,?,?,?,?,?)""",
                (row["user_id"], run_id, candidates, row["ended_at"], rate_passed,
                 rate_total, rate_passed / rate_total, STATS_VERSION))
    prior = db.execute("""SELECT pb_score,b10_score,b10_rating
        FROM human_player_stat_series WHERE user_id=? AND variant=?
        ORDER BY game_index DESC LIMIT 1""", (row["user_id"], row["variant"])).fetchone()
    values = (pb_score, b10_score, b10_rating)
    if not prior or values != tuple(prior):
        db.execute("""INSERT OR REPLACE INTO human_player_stat_series
            (user_id,variant,run_id,game_index,ended_at,pb_score,b10_score,
             b10_rating,stats_version) VALUES(?,?,?,?,?,?,?,?,?)""",
            (row["user_id"], row["variant"], run_id, game_count, row["ended_at"],
             pb_score, b10_score, b10_rating, STATS_VERSION))
    primary_key = "32k" if row["variant"] == "4x4" else "4k" if row["variant"] == "3x4" else "1k" if row["variant"] == "3x3" else "tile_512"
    rate_value = rate_passed / rate_total if rate_total else None
    db.execute("""UPDATE human_player_statistics SET game_count=?,pb_score=?,
        b10_score=?,b10_rating=?,primary_achievement_count=?,rate_32k_value=?,
        features_json=?,rate_32k_passed=?,rate_32k_total=?,
        rate_32k_covered_games=?,rate_32k_candidate_games=?,stats_version=?,updated_at=?
        WHERE user_id=? AND variant=?""",
        (game_count, pb_score, b10_score, b10_rating, features.get(primary_key, 0), rate_value,
         json.dumps(features, separators=(",", ":")), rate_passed, rate_total,
         covered, candidates, STATS_VERSION, time.time(), row["user_id"], row["variant"]))


def _sample(rows: list[dict], limit: int = 600) -> list[dict]:
    if len(rows) <= limit:
        return rows
    indexes = {0, len(rows) - 1}
    for position in range(1, limit - 1):
        indexes.add(round(position * (len(rows) - 1) / (limit - 1)))
    return [rows[index] for index in sorted(indexes)]


def payload(db, user_id: int, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError("invalid_statistics_variant")
    summaries = {}
    for row in db.execute("""SELECT * FROM human_player_statistics
        WHERE user_id=? ORDER BY variant""", (user_id,)):
        summaries[row["variant"]] = {
            "game_count": row["game_count"], "pb_score": row["pb_score"],
            "b10_score": row["b10_score"], "b10_rating": row["b10_rating"],
            "features": json.loads(row["features_json"]),
            "rate_32k": {
                "passed": row["rate_32k_passed"], "total": row["rate_32k_total"],
                "covered_games": row["rate_32k_covered_games"],
                "candidate_games": row["rate_32k_candidate_games"],
                "value": (row["rate_32k_passed"] / row["rate_32k_total"]
                          if row["rate_32k_total"] else None),
            },
        }
    points = [dict(row) for row in db.execute("""SELECT game_index,ended_at,pb_score,
        b10_score,b10_rating FROM human_player_stat_series
        WHERE user_id=? AND variant=? ORDER BY game_index""", (user_id, variant))]
    for item in points:
        if item["b10_rating"] is not None and not math.isfinite(item["b10_rating"]):
            item["b10_rating"] = None
    rate_points = [dict(row) for row in db.execute("""SELECT game_index,ended_at,
        passed,total,value FROM human_player_rate32k_series
        WHERE user_id=? AND game_index>=? ORDER BY game_index""",
        (user_id, RATE_SERIES_MIN_GAMES))] if variant == "4x4" else []
    return {"version": STATS_VERSION, "variant": variant, "summaries": summaries,
            "feature_labels": {key: label for key, label in FEATURES[variant]},
            "series": _sample(points), "series_total": len(points),
            "rate_32k_series": _sample(rate_points),
            "rate_32k_series_total": len(rate_points)}
