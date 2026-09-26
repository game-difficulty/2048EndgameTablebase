"""Durable, per-formation summaries produced by the existing analysis pass."""
from __future__ import annotations

import base64
import json
import math
import time
from pathlib import Path

from engine_core.performance_evaluation import PERFORMANCE_PERFECT_LABEL

from .store import database
from .verse_replay import PREFIX, UNKNOWN_TIMING_MS, _rpl1


METRIC_VERSION = 1
POSTER_GOALS = frozenset((16384, 32768, 65536))
MAX_COUNTED_MOVE_MS = 20 * 60 * 1000


def _personal_rank(db, user_id: int, variant: str, run_id: str) -> int | None:
    """Return this archived game position in the owner's current score order."""
    from .service import RANKABLE_SQL
    row = db.execute(f"""SELECT personal_rank FROM (
        SELECT id,ROW_NUMBER() OVER (ORDER BY json_extract(state,'$.score') DESC,
            ended DESC,id DESC) AS personal_rank
        FROM human_runs WHERE user_id=? AND variant=? AND status='sealed' AND {RANKABLE_SQL}
    ) WHERE id=?""", (user_id, variant, run_id)).fetchone()
    return int(row["personal_rank"]) if row else None


def poster_goal_tile(pattern: str, target: str, variant: str) -> int | None:
    """Return a verified 4x4 terminal tile; variant patterns need their own mapping."""
    if variant != "4x4":
        return None
    from Config import category_info, pattern_32k_tiles_map
    if pattern in category_info.get("variant", []) or pattern not in pattern_32k_tiles_map:
        return None
    try:
        return int(target) * (1 << int(pattern_32k_tiles_map[pattern][0]))
    except (TypeError, ValueError, OverflowError):
        return None


def intervals_from_analysis_input(path: Path) -> list[int | None]:
    text = path.read_text(encoding="ascii").strip()
    if not text.startswith(PREFIX):
        raise ValueError("analysis_replay_format_invalid")
    raw = base64.b64decode(text[len(PREFIX):], validate=True)
    _, _, moves = _rpl1(raw)
    return [None if move[3] == UNKNOWN_TIMING_MS else move[3] for move in moves]


def build_summary(segments: list[dict], intervals: list[int | None], *,
                  source: str = "native", timing_lossy: bool = False) -> dict:
    if not source:
        raise ValueError("analysis_source_invalid")
    if len(intervals) > 200000:
        raise ValueError("analysis_replay_too_long")
    result_segments: list[dict] = []
    included: list[dict] = []
    previous_end = 0
    for segment in segments:
        start = int(segment["start_index"])
        end = int(segment["end_index"])
        if start < previous_end or end <= start or end > len(intervals):
            raise ValueError("analysis_segment_boundary_invalid")
        previous_end = end
        elapsed = intervals[start:end]
        timed = [value for value in elapsed
                 if value is not None and 0 <= value <= MAX_COUNTED_MOVE_MS]
        fit = float(segment["goodness_of_fit"])
        eligible = end - start >= 20 and math.isfinite(fit) and fit > 0
        item = {
            "start_index": start, "end_index": end, "total_moves": end - start,
            "evaluated_moves": int(segment["evaluated_moves"]),
            "goodness_of_fit": fit if math.isfinite(fit) else None,
            "max_combo": int(segment["max_combo"]),
            "performance_counts": {str(key): int(value)
                                   for key, value in segment["performance_counts"].items()},
            "timed_moves": len(timed),
            "valid_timing_ms": sum(timed),
            "excluded_timing_moves": len(elapsed) - len(timed),
            "mean_ms_per_timed_move": sum(timed) / len(timed) if timed else None,
            "report_file": segment.get("report_file"),
            "replay_file": segment.get("replay_file"),
            "artifact_id": segment.get("artifact_id"),
            "replay_move_count": segment.get("replay_move_count"),
            "included": eligible,
            "exclusion_reason": None if eligible else (
                "too_few_moves" if end - start < 20 else "zero_or_invalid_fit"),
        }
        result_segments.append(item)
        if eligible:
            included.append(item)
    counts: dict[str, int] = {}
    for item in included:
        for key, value in item["performance_counts"].items():
            counts[key] = counts.get(key, 0) + value
    timed_moves = sum(item["timed_moves"] for item in included)
    valid_timing_ms = sum(item["valid_timing_ms"] for item in included)
    evaluated_moves = sum(counts.values())
    aggregate = {
        "stage_count": len(included),
        "total_moves": sum(item["total_moves"] for item in included),
        "evaluated_moves": evaluated_moves,
        "mean_goodness_of_fit": (
            sum(item["goodness_of_fit"] for item in included) / len(included)
            if included else None),
        "max_combo": max((item["max_combo"] for item in included), default=0),
        "performance_counts": counts,
        "perfect_rate": counts.get(PERFORMANCE_PERFECT_LABEL, 0) / evaluated_moves
        if evaluated_moves else None,
        "timed_moves": timed_moves,
        "valid_timing_ms": valid_timing_ms,
        "excluded_timing_moves": sum(item["excluded_timing_moves"] for item in included),
        "mean_ms_per_timed_move": valid_timing_ms / timed_moves if timed_moves else None,
        # During the current closed test, normalized Verse intervals (including
        # the historical 100 ms values) are accepted as recorded timing.
        "speed_grade_eligible": bool(timed_moves),
    }
    return {"metric_version": METRIC_VERSION, "source": source,
            "timing_lossy": timing_lossy, "segments": result_segments,
            "aggregate": aggregate}


def init_schema(db) -> None:
    db.execute("""CREATE TABLE IF NOT EXISTS human_analysis_summaries (
        id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES human_runs(id),
        user_id INTEGER NOT NULL, pattern TEXT NOT NULL, target TEXT NOT NULL,
        metric_version INTEGER NOT NULL, job_id TEXT NOT NULL,
        aggregate_json TEXT NOT NULL, summary_json TEXT NOT NULL, created REAL NOT NULL,
        UNIQUE(run_id,pattern,target,metric_version)
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(human_analysis_summaries)")}
    if "aggregate_json" not in columns:
        db.execute("ALTER TABLE human_analysis_summaries ADD COLUMN aggregate_json TEXT NOT NULL DEFAULT '{}'")
    db.execute("""CREATE INDEX IF NOT EXISTS human_analysis_summaries_user
        ON human_analysis_summaries(user_id,run_id)""")


def save_summary(*, run_id: str, user_id: int, pattern: str, target: str,
                 job_id: str, summary: dict) -> int:
    from .analysis_grade import GRADE_VERSION, grade_for_summary
    from .service import RANKABLE_SQL
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        init_schema(db)
        row = db.execute("""SELECT id,variant,ended,state,source FROM human_runs WHERE id=? AND user_id=?
            AND status='sealed' AND visible=1 AND archive IS NOT NULL AND has_replay=1""",
            (run_id, user_id)).fetchone()
        if not row:
            raise ValueError("analysis_run_not_found")
        run_state = json.loads(row["state"])
        prior = db.execute(f"""SELECT MAX(json_extract(state,'$.score')) AS score
            FROM human_runs WHERE user_id=? AND variant=? AND status='sealed'
                AND (ended<? OR (ended=? AND id<?)) AND {RANKABLE_SQL}""",
            (user_id, row["variant"], row["ended"], row["ended"], run_id)).fetchone()
        previous_best = int(prior["score"] or 0)
        rankable = bool(db.execute(f"""SELECT 1 FROM human_runs WHERE id=?
            AND status='sealed' AND {RANKABLE_SQL}""", (run_id,)).fetchone())
        score = int(run_state["score"])
        summary = {**summary, "run": {
            "id": run_id, "variant": row["variant"], "score": score,
            "board": run_state["board"], "ended_at": row["ended"],
            "source": row["source"],
            "personal_rank": _personal_rank(db, user_id, row["variant"], run_id),
            "pb": {"new_best": rankable and score > previous_best,
                   "previous_best": previous_best,
                   "delta": score - previous_best if rankable and score > previous_best else 0},
        }}
        summary["grade_version"] = GRADE_VERSION
        summary["grade"] = grade_for_summary(
            variant=row["variant"], goal_tile=summary.get("goal_tile"),
            score=score, aggregate=summary["aggregate"])
        db.execute("""INSERT INTO human_analysis_summaries
            (run_id,user_id,pattern,target,metric_version,job_id,aggregate_json,summary_json,created)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,pattern,target,metric_version) DO UPDATE SET
                job_id=excluded.job_id,aggregate_json=excluded.aggregate_json,
                summary_json=excluded.summary_json,created=excluded.created""",
            (run_id, user_id, pattern, target, METRIC_VERSION, job_id,
             json.dumps(summary["aggregate"], separators=(",", ":"), allow_nan=False),
             json.dumps(summary, separators=(",", ":"), allow_nan=False), time.time()))
        summary_id = db.execute("""SELECT id FROM human_analysis_summaries
            WHERE run_id=? AND pattern=? AND target=? AND metric_version=?""",
            (run_id, pattern, target, METRIC_VERSION)).fetchone()["id"]
        from .leaderboards import upsert_analysis_summary
        upsert_analysis_summary(db, summary_id, summary)
        return summary_id


def list_summaries(run_id: str, user_id: int) -> list[dict]:
    with database() as db:
        init_schema(db)
        rows = db.execute("""SELECT s.id,s.pattern,s.target,s.metric_version,s.aggregate_json,s.created
            FROM human_analysis_summaries s JOIN human_runs r ON r.id=s.run_id
            WHERE s.run_id=? AND s.user_id=? AND r.user_id=? AND r.status='sealed'
                AND r.visible=1 AND r.archive IS NOT NULL AND r.has_replay=1
            ORDER BY s.created DESC,s.id DESC""", (run_id, user_id, user_id)).fetchall()
    return [{"id": row["id"], "pattern": row["pattern"], "target": row["target"],
             "metric_version": row["metric_version"], "created": row["created"],
             "aggregate": json.loads(row["aggregate_json"])} for row in rows]


def get_summary(summary_id: int, user_id: int) -> dict | None:
    from .analysis_grade import GRADE_VERSION, grade_for_summary
    with database() as db:
        init_schema(db)
        row = db.execute("""SELECT s.id,s.run_id,s.pattern,s.target,s.created,s.summary_json
            FROM human_analysis_summaries s JOIN human_runs r ON r.id=s.run_id
            WHERE s.id=? AND s.user_id=? AND r.user_id=? AND r.status='sealed'
                AND r.visible=1 AND r.archive IS NOT NULL AND r.has_replay=1""",
            (summary_id, user_id, user_id)).fetchone()
    if not row:
        return None
    summary = json.loads(row["summary_json"])
    run = summary.get("run") or {}
    with database() as db:
        run["personal_rank"] = _personal_rank(
            db, user_id, run.get("variant", ""), row["run_id"])
    if summary.get("grade_version") != GRADE_VERSION:
        # Older paid analyses can be graded from their durable small summary;
        # the full replay and tablebase do not need to be read again.
        goal = summary.get("goal_tile")
        if goal is None:
            goal = poster_goal_tile(row["pattern"], row["target"], run.get("variant", ""))
        aggregate = summary.get("aggregate") or {}
        aggregate["speed_grade_eligible"] = bool(aggregate.get("timed_moves"))
        summary["grade_version"] = GRADE_VERSION
        summary["grade"] = grade_for_summary(
            variant=run.get("variant", ""), goal_tile=goal,
            score=run.get("score", 0), aggregate=aggregate)
        with database() as db:
            db.execute("""UPDATE human_analysis_summaries
                SET aggregate_json=?,summary_json=? WHERE id=? AND user_id=?""",
                (json.dumps(aggregate, separators=(",", ":"), allow_nan=False),
                 json.dumps(summary, separators=(",", ":"), allow_nan=False),
                 summary_id, user_id))
            from .leaderboards import upsert_analysis_summary
            upsert_analysis_summary(db, summary_id, summary)
    return {"id": row["id"], "run_id": row["run_id"], "pattern": row["pattern"],
            "target": row["target"], "created": row["created"], **summary}
