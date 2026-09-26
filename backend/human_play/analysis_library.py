"""Public, replay-free index of admitted play-site analysis results."""
from __future__ import annotations

import base64
from typing import Any

from backend.auth.db import auth_db
from backend.profile.service import public_profile

from .analysis_summary import get_summary, init_schema
from .service import RunError, player_id_for_name, rankable_sql
from .store import database


GRADES = frozenset(("SSS", "SS", "S", "A", "B", "C", "D", "E", "F"))


def _encode_cursor(ended: float, summary_id: int) -> str:
    raw = f"{float(ended)}|{int(summary_id)}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str) -> tuple[float, int]:
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()
        ended, summary_id = raw.split("|", 1)
        return float(ended), int(summary_id)
    except (ValueError, UnicodeDecodeError) as exc:
        raise RunError("invalid_analysis_library_cursor", 400) from exc


def _subjects(user_ids: set[int]) -> dict[int, dict]:
    if not user_ids:
        return {}
    placeholders = ",".join("?" for _ in user_ids)
    result = {}
    with auth_db() as db:
        rows = db.execute(f"""SELECT id,display_name,status FROM users
            WHERE id IN ({placeholders})""", sorted(user_ids)).fetchall()
        for row in rows:
            if row["status"] != "active":
                continue
            profile = public_profile(int(row["id"]), db=db)
            result[int(row["id"])] = {
                "id": int(row["id"]), "display_name": row["display_name"] or "玩家",
                "avatar_url": profile.get("avatar_url"),
            }
    return result


def list_entries(*, limit: int = 20, cursor: str = "", username: str = "",
                 variant: str = "", pattern: str = "", target: str = "",
                 grade: str = "") -> dict:
    size = min(50, max(1, int(limit)))
    params: list[Any] = []
    where = ["s.listed=1", "s.admitted=1", "r.status='sealed'", rankable_sql("r")]
    if username:
        where.append("s.user_id=?")
        params.append(player_id_for_name(username))
    if variant:
        where.append("r.variant=?")
        params.append(variant)
    if pattern:
        where.append("s.pattern=?")
        params.append(pattern)
    if target:
        where.append("s.target=?")
        params.append(target)
    if grade:
        if grade == "unrated":
            where.append("a.summary_id IS NULL")
        elif grade in GRADES:
            where.append("a.grade=?")
            params.append(grade)
        else:
            raise RunError("invalid_analysis_grade", 400)
    scan_cursor = _decode_cursor(cursor) if cursor else None
    batch_size = max(50, min(200, size * 3))
    items = []
    last_visible = None
    has_more = False
    while not has_more:
        scan_where = list(where)
        scan_params = list(params)
        if scan_cursor:
            ended, summary_id = scan_cursor
            scan_where.append("(r.ended<? OR (r.ended=? AND s.id<?))")
            scan_params.extend((ended, ended, summary_id))
        with database() as db:
            init_schema(db)
            rows = db.execute(f"""SELECT s.id,s.run_id,s.user_id,s.pattern,s.target,
                s.metric_version,s.analyzer_version,s.created,s.first_listed_at,s.final_score,
                r.variant,r.ended AS run_ended_at,r.source,
                a.weighted_score,a.grade,a.mean_goodness_of_fit,a.max_combo,
                a.stage_count,a.evaluated_moves
                FROM human_analysis_summaries s
                JOIN human_runs r ON r.id=s.run_id
                LEFT JOIN human_analysis_results a ON a.summary_id=s.id AND a.active=1
                WHERE {' AND '.join(scan_where)}
                ORDER BY r.ended DESC,s.id DESC LIMIT ?""",
                (*scan_params, batch_size)).fetchall()
        if not rows:
            break
        subjects = _subjects({int(row["user_id"]) for row in rows})
        from backend.analysis_history import library_artifacts
        artifacts = library_artifacts({int(row["id"]) for row in rows})
        for row in rows:
            subject = subjects.get(int(row["user_id"]))
            stages = artifacts.get(int(row["id"]), [])
            if not subject or not any(stage["available"] for stage in stages):
                continue
            if len(items) >= size:
                has_more = True
                break
            items.append({
                "id": int(row["id"]), "run_id": row["run_id"], "subject": subject,
                "variant": row["variant"], "score": int(row["final_score"] or 0),
                "run_ended_at": row["run_ended_at"], "source": row["source"],
                "pattern": row["pattern"], "target": row["target"],
                "metric_version": row["metric_version"], "analyzer_version": row["analyzer_version"],
                "analyzed_at": row["created"], "grade": row["grade"],
                "mean_goodness_of_fit": row["mean_goodness_of_fit"],
                "max_combo": int(row["max_combo"] or 0),
                "stage_count": int(row["stage_count"] if row["stage_count"] is not None else len(stages)),
                "evaluated_moves": int(row["evaluated_moves"] or 0),
                "replay_available": True,
            })
            last_visible = row
        if has_more or len(rows) < batch_size:
            break
        tail = rows[-1]
        scan_cursor = (float(tail["run_ended_at"]), int(tail["id"]))
    next_cursor = (_encode_cursor(last_visible["run_ended_at"], last_visible["id"])
                   if has_more and last_visible is not None else "")
    return {"items": items, "next_cursor": next_cursor}


def detail(summary_id: int) -> dict:
    summary = get_summary(int(summary_id))
    if summary is None:
        raise RunError("analysis_summary_not_found", 404)
    from backend.analysis_history import library_artifacts
    summary["artifacts"] = library_artifacts([int(summary_id)]).get(int(summary_id), [])
    return summary
