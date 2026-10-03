"""Read-only, paged view over every administrator approval workflow."""
from __future__ import annotations

import json
import time
from typing import Any, Iterable

from .store import database


KINDS = {"all", "verse", "archive", "assistance"}
STAGES = {"all", "pending", "processing", "approved", "rejected", "revoked"}


def _actions(kind: str, status: str, updated_at: float) -> list[str]:
    if kind == 'assistance':
        return ['confirm', 'dismiss'] if status == 'pending' else []
    if kind == "archive":
        if status == "pending":
            return ["approve", "reject"]
        return ["revoke"] if status == "approved" else []
    if status == "pending":
        return ["approve", "reject"]
    if status == "failed" or (status == "importing" and updated_at < time.time() - 7200):
        return ["retry"]
    if status == "complete":
        return ["revoke"]
    return []


def _where(kind: str, stage: str, query: str, identity_user_ids: Iterable[int]) -> tuple[str, list[Any]]:
    parts: list[str] = []
    params: list[Any] = []
    if kind != "all":
        parts.append("kind = ?")
        params.append(kind)
    if stage != "all":
        parts.append("stage = ?")
        params.append(stage)
    normalized = str(query or "").strip().lower()
    matched_ids = sorted({int(item) for item in identity_user_ids})
    if normalized:
        search_parts = ["lower(subject) LIKE ?", "CAST(transaction_id AS TEXT) = ?", "CAST(user_id AS TEXT) = ?"]
        search_params: list[Any] = [f"%{normalized}%", normalized, normalized]
        if matched_ids:
            placeholders = ",".join("?" for _ in matched_ids)
            search_parts.append(f"user_id IN ({placeholders})")
            search_params.extend(matched_ids)
        parts.append("(" + " OR ".join(search_parts) + ")")
        params.extend(search_params)
    return ("WHERE " + " AND ".join(parts)) if parts else "", params


def list_transactions(
    *,
    kind: str = "all",
    stage: str = "all",
    query: str = "",
    identity_user_ids: Iterable[int] = (),
    page: int = 1,
    page_size: int = 30,
) -> dict[str, Any]:
    resolved_kind = kind if kind in KINDS else "all"
    resolved_stage = stage if stage in STAGES else "all"
    resolved_page_size = max(1, min(100, int(page_size)))
    source = """
      SELECT 'verse' AS kind, id AS transaction_id, user_id, status AS raw_status,
             CASE
               WHEN status='pending' THEN 'pending'
               WHEN status IN ('approved','importing','failed') THEN 'processing'
               WHEN status='complete' THEN 'approved'
               WHEN status='rejected' THEN 'rejected'
               WHEN status IN ('revoked','cancelled') THEN 'revoked'
               ELSE 'processing' END AS stage,
             requested AS requested_at, updated AS updated_at, username AS subject,
             NULL AS variant, NULL AS score, NULL AS moves, NULL AS ended_at, NULL AS started_at,
             NULL AS game_over, counts AS detail_json, error, proof_note AS review_note,
             approved_by AS operator_id
      FROM human_external_claims
      UNION ALL
      SELECT 'archive' AS kind, id AS transaction_id, user_id, status AS raw_status,
             CASE
               WHEN status='pending' THEN 'pending'
               WHEN status='approved' THEN 'approved'
               WHEN status='rejected' THEN 'rejected'
               WHEN status IN ('revoked','cancelled') THEN 'revoked'
               ELSE 'processing' END AS stage,
             requested_at, updated_at, original_filename AS subject,
             variant, claimed_score AS score, moves, claimed_ended_at AS ended_at, claimed_started_at AS started_at,
             is_game_over AS game_over, warning_flags_json AS detail_json,
             '' AS error, review_note, approved_by AS operator_id
      FROM human_archive_applications
      UNION ALL
      SELECT 'assistance' AS kind, a.id AS transaction_id, a.user_id, a.status AS raw_status,
             CASE a.status WHEN 'pending' THEN 'pending' WHEN 'confirmed' THEN 'approved' ELSE 'rejected' END AS stage,
             a.requested_at, a.updated_at, a.run_id AS subject,
             r.variant, json_extract(r.state,'$.score') AS score, json_extract(r.state,'$.seq') AS moves,
             r.ended AS ended_at, r.first_move_at AS started_at, NULL AS game_over,
             a.details AS detail_json, '' AS error, a.review_note, a.operator_id
      FROM human_assistance_reviews a JOIN human_runs r ON r.id=a.run_id
    """
    where, params = _where(resolved_kind, resolved_stage, query, identity_user_ids)
    with database() as db:
        total = int(db.execute(
            f"SELECT COUNT(*) FROM ({source}) transactions {where}", tuple(params)
        ).fetchone()[0])
        page_count = max(1, (total + resolved_page_size - 1) // resolved_page_size)
        resolved_page = max(1, min(int(page), page_count))
        rows = db.execute(
            f"""SELECT * FROM ({source}) transactions {where}
                ORDER BY updated_at DESC, kind, transaction_id DESC LIMIT ? OFFSET ?""",
            (*params, resolved_page_size, (resolved_page - 1) * resolved_page_size),
        ).fetchall()

    transactions = []
    for row in rows:
        item = dict(row)
        raw_detail = item.pop("detail_json", "")
        try:
            item["details"] = json.loads(raw_detail or "{}")
        except (TypeError, json.JSONDecodeError):
            item["details"] = {}
        item["game_over"] = None if item["game_over"] is None else bool(item["game_over"])
        item["actions"] = _actions(item["kind"], item["raw_status"], float(item["updated_at"] or 0))
        item["key"] = f'{item["kind"]}:{item["transaction_id"]}'
        transactions.append(item)
    return {
        "transactions": transactions,
        "page": {
            "page": resolved_page,
            "page_size": resolved_page_size,
            "total": total,
            "page_count": page_count,
        },
        "filters": {"kind": resolved_kind, "stage": resolved_stage, "q": str(query or "").strip()},
    }
