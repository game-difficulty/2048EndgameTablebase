"""Single-archive, multi-formation analysis entry for the play site."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from backend.analysis import normalize_target_value
from backend.auth.db import auth_db
from backend.cloud_analysis_jobs import AnalysisWorkItem, check_analysis_capacity, create_analysis_job, get_analysis_job, get_analysis_root
from backend.quota.config import TOKEN_UNIT, apply_pricing_multipliers, operation_cost_units, resolve_pricing_snapshot, table_multiplier_units
from backend.quota.service import cancel_reservation, get_token_balance, reserve_operation_tokens_many
from backend.tablebase_catalog import get_available_tablebases, get_catalog_version, resolve_configured_tablebase
from Config import category_info
from . import engine, service
from .analysis_bridge import archived_to_analysis_text
from .analysis_summary import get_summary, list_summaries
from .store import database


def _subject(user_id: int, *, allow_missing: bool = False) -> dict:
    from backend.profile.service import public_profile
    with auth_db() as db:
        row = db.execute("SELECT id,display_name,status FROM users WHERE id=?", (int(user_id),)).fetchone()
        if not row and allow_missing:
            return {"id": int(user_id), "display_name": "玩家", "avatar_url": None}
        if not row or row["status"] != "active":
            raise service.RunError("analysis_run_not_found", 404)
        profile = public_profile(int(user_id), db=db)
    return {"id": int(user_id), "display_name": row["display_name"] or "玩家",
            "avatar_url": profile.get("avatar_url")}


def _listing_enabled(user_id: int) -> bool:
    from backend.profile.preferences import get_preferences
    return bool(get_preferences(int(user_id))["preferences"].get("share_play_analysis", True))


def _run_metadata(run_id: str, _viewer_id: int | None = None) -> dict:
    from .service import RANKABLE_SQL
    with database() as db:
        row = db.execute(f"""SELECT id,user_id,variant,state,reason,ended,has_replay,source,
            visible,({RANKABLE_SQL}) AS rankable FROM human_runs
            WHERE id=? AND status='sealed' AND archive IS NOT NULL""",
            (run_id,)).fetchone()
    if (not row or not row["visible"] or not row["has_replay"]
            or (int(row["user_id"]) != int(_viewer_id or -1) and not row["rankable"])):
        raise service.RunError("analysis_run_not_found", 404)
    state = json.loads(row["state"])
    return {"id": row["id"], "variant": row["variant"], "score": state["score"],
            "moves": state["seq"], "reason": row["reason"], "ended_at": row["ended"],
            "source": row["source"],
            "subject": _subject(row["user_id"], allow_missing=int(row["user_id"]) == int(_viewer_id or -1))}


def _compatible(pattern: str, variant: str) -> bool:
    variants = set(category_info.get("variant", []))
    return (pattern not in variants) if variant == "4x4" else (pattern in variants and pattern.startswith(variant))


def options(run_id: str, user_id: int) -> dict:
    run = _run_metadata(run_id, user_id)
    tables = [{"pattern": table["pattern"], "target": table["target"],
               "full_pattern": table["full_pattern"]}
              for table in get_available_tablebases() if _compatible(table["pattern"], run["variant"])]
    return {"run": run, "catalog_version": get_catalog_version(), "tables": tables,
            "max_items": 6, "max_waiting_items": 60}


def summaries(run_id: str, user_id: int) -> dict:
    run = _run_metadata(run_id, user_id)
    return {"run": run, "items": list_summaries(run_id, user_id)}


def summary(summary_id: int, user_id: int) -> dict:
    result = get_summary(summary_id, user_id)
    if result is None:
        raise service.RunError("analysis_summary_not_found", 404)
    return result


def _validated_items(run_id: str, user_id: int, items: list[dict]) -> tuple[dict, list[dict]]:
    run = _run_metadata(run_id, user_id)
    if not 1 <= len(items) <= 6:
        raise service.RunError("invalid_analysis_items", 400)
    seen = set()
    valid = []
    for item in items:
        pattern = str(item.get("pattern") or "").strip()
        target = str(item.get("target") or "").strip()
        try:
            target_tile, _, numeric_target = normalize_target_value(target)
        except (ValueError, TypeError) as exc:
            raise service.RunError("invalid_analysis_target", 400) from exc
        full_pattern = f"{pattern}_{numeric_target}"
        if not _compatible(pattern, run["variant"]) or full_pattern in seen:
            raise service.RunError("invalid_analysis_items", 400)
        descriptor = resolve_configured_tablebase(full_pattern)
        if descriptor is None or (descriptor.get("_provider") == "remote" and not descriptor.get("_available", False)):
            raise service.RunError("analysis_tablebase_unavailable", 409)
        seen.add(full_pattern)
        valid.append({"pattern": pattern, "target": target_tile, "full_pattern": full_pattern})
    return run, valid


def _price(items: list[dict]) -> tuple[list[dict], int]:
    pricing = resolve_pricing_snapshot()
    base = operation_cost_units("analysis_per_replay")
    result = []
    for item in items:
        units = apply_pricing_multipliers(base, table_multiplier_units(item["full_pattern"]),
                                          pricing.global_multiplier_units)
        result.append({**item, "cost_units": units, "cost": units / TOKEN_UNIT})
    return result, sum(item["cost_units"] for item in result)


def quote(run_id: str, user_id: int, items: list[dict]) -> dict:
    run, valid = _validated_items(run_id, user_id, items)
    priced, total = _price(valid)
    return {"run": run, "items": priced, "total_cost_units": total,
            "total_cost": total / TOKEN_UNIT, "token_balance": get_token_balance(user_id),
            "catalog_version": get_catalog_version()}


def _claim_request(user_id: int, request_id: str) -> str | None:
    with auth_db() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE IF NOT EXISTS human_analysis_requests (
            user_id INTEGER NOT NULL, request_id TEXT NOT NULL, job_id TEXT,
            created_at REAL NOT NULL, PRIMARY KEY(user_id,request_id))""")
        row = db.execute("SELECT job_id,created_at FROM human_analysis_requests WHERE user_id=? AND request_id=?",
                         (user_id, request_id)).fetchone()
        if row:
            if row["job_id"]:
                return row["job_id"]
            if time.time() - row["created_at"] < 300:
                raise service.RunError("analysis_request_in_progress", 409)
            db.execute("DELETE FROM human_analysis_requests WHERE user_id=? AND request_id=?", (user_id, request_id))
        db.execute("INSERT INTO human_analysis_requests VALUES(?,?,NULL,?)", (user_id, request_id, time.time()))
    return None


def _finish_request(user_id: int, request_id: str, job_id: str | None) -> None:
    with auth_db() as db:
        if job_id:
            db.execute("UPDATE human_analysis_requests SET job_id=? WHERE user_id=? AND request_id=?",
                       (job_id, user_id, request_id))
        else:
            db.execute("DELETE FROM human_analysis_requests WHERE user_id=? AND request_id=? AND job_id IS NULL",
                       (user_id, request_id))


def create(run_id: str, user_id: int, session_id: int | None, items: list[dict],
           expected_cost_units: int, expected_catalog_version: str, request_id: str):
    if not 16 <= len(request_id) <= 128:
        raise service.RunError("invalid_analysis_request_id", 400)
    existing = _claim_request(user_id, request_id)
    if existing:
        try:
            return get_analysis_job(existing, user_id=user_id)
        except FileNotFoundError as exc:
            raise service.RunError("analysis_job_expired", 404) from exc
    reservations = []
    path: Path | None = None
    job = None
    try:
        run, valid = _validated_items(run_id, user_id, items)
        if expected_catalog_version != get_catalog_version():
            raise service.RunError("analysis_catalog_changed", 409)
        _, total = _price(valid)
        if expected_cost_units != total:
            raise service.RunError("analysis_price_changed", 409)
        check_analysis_capacity(user_id, len(valid))
        # Only the worker input is materialized; clients never download and upload the archive.
        with database() as db:
            row = db.execute("""SELECT archive FROM human_runs WHERE id=? AND user_id=?
                AND status='sealed' AND visible=1 AND archive IS NOT NULL""",
                (run_id, run["subject"]["id"])).fetchone()
        if not row:
            raise service.RunError("analysis_run_not_found", 404)
        if run["source"] in {"verse", "manual"}:
            from .verse_replay import archived_to_analysis_text as verse_analysis_text
            text = verse_analysis_text(row["archive"], run["variant"], run["moves"])
        else:
            text = archived_to_analysis_text(row["archive"], expected_run_id=run_id,
                                             expected_variant=run["variant"], expected_seq=run["moves"])
        path = get_analysis_root() / f"human-{uuid.uuid4().hex}.vrs"
        path.write_text(text, encoding="ascii")
        try:
            reservations = reserve_operation_tokens_many(
                user_id=user_id, session_id=session_id, operation_key="analysis_per_replay",
                full_patterns=[item["full_pattern"] for item in valid], expected_total_units=expected_cost_units)
        except ValueError as exc:
            if str(exc) == "analysis_price_changed":
                raise service.RunError("analysis_price_changed", 409) from exc
            raise
        listing_snapshot = _listing_enabled(run["subject"]["id"])
        work_items = [AnalysisWorkItem(
            path=path, filename=f"{run_id}.vrs", pattern=item["pattern"], target=item["target"],
            reservation=reservations[index], source_run_id=run_id,
            subject_user_id=run["subject"]["id"], listing_snapshot=listing_snapshot,
            source_ended_at=run["ended_at"],
        ) for index, item in enumerate(valid)]
        job = create_analysis_job(work_items=work_items, user_id=user_id, session_id=session_id)
        path.unlink(missing_ok=True)
        _finish_request(user_id, request_id, job.job_id)
        return job
    except Exception:
        if job is not None:
            # Admission succeeded; the worker now owns both input and reservations.
            # A request-index write failure must never refund or delete its input.
            return job
        for reservation in reservations:
            cancel_reservation(reservation, reason="analysis_job_not_created")
        if path:
            path.unlink(missing_ok=True)
        _finish_request(user_id, request_id, None)
        raise
