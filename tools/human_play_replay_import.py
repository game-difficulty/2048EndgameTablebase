"""Plan and apply bulk replay attachments for already-imported Play records.

The default mode is read-only and writes a JSON report.  ``--apply`` attaches
only uniquely matched replays.  Public archive-upload parsing is deliberately
not changed by this tool; 2048next support lives here only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.human_play import engine, leaderboards, rating, statistics, verse_replay  # noqa: E402
from backend.human_play.store import database  # noqa: E402
from backend.replay_2048next import (  # noqa: E402
    CheckpointRecord,
    EndRecord,
    EXT_AI_USED,
    EXT_DIFFICULTY_CHANGE,
    ExtensionRecord,
    MoveRecord,
    Replay2048NextError,
    UndoRecord,
    decode_2048next_replay,
)


TOOL_VERSION = 1
NAME_RE = re.compile(
    r"^(?P<player>.+?)_(?P<variant>[234]x[34])_(?P<label>.+?)_#(?P<ordinal>\d+)_"
    r"(?P<year>20\d{2})[-_](?P<month>\d{2})[-_](?P<day>\d{2})_"
    r"(?P<score>\d+)_(?P<source>verse|2048next|pku)\.(?:txt|vrs)$",
    re.IGNORECASE,
)
SUPPORTED_SUFFIXES = {".txt", ".vrs"}
OLD_VERSE_ALPHABET = (bytes(range(32, 127)).decode("ascii")
                      + bytes(range(128, 161)).decode("cp850"))
OLD_VERSE_VALUES = {character: index for index, character in enumerate(OLD_VERSE_ALPHABET)}


@dataclass(frozen=True)
class FileMetadata:
    relative_path: str
    variant: str
    played_date: str
    score: int
    source: str
    ordinal: int


@dataclass
class PlannedReplay:
    path: Path
    metadata: FileMetadata | None
    status: str
    detail: str = ""
    run_id: str | None = None
    normalized: bytes | None = None
    result: dict | None = None

    def payload(self) -> dict:
        item = {
            "file": self.metadata.relative_path if self.metadata else self.path.name,
            "status": self.status,
        }
        if self.metadata:
            item.update({
                "source": self.metadata.source,
                "variant": self.metadata.variant,
                "date": self.metadata.played_date,
                "score": self.metadata.score,
                "ordinal": self.metadata.ordinal,
            })
        if self.run_id:
            item["run_id"] = self.run_id
        if self.result:
            item.update({
                "moves": self.result["moves"],
                "game_over": bool(self.result["game_over"]),
                "calculated_score": self.result["score"],
            })
        if self.detail:
            item["detail"] = self.detail
        return item


def parse_filename(path: Path, root: Path) -> FileMetadata:
    match = NAME_RE.match(path.name)
    if not match:
        raise ValueError("filename_metadata_unrecognized")
    fields = match.groupdict()
    return FileMetadata(
        relative_path=path.relative_to(root).as_posix(),
        variant=fields["variant"].lower(),
        played_date=f'{fields["year"]}-{fields["month"]}-{fields["day"]}',
        score=int(fields["score"]),
        source=fields["source"].lower(),
        ordinal=int(fields["ordinal"]),
    )


def _checkpoint_codes(board: list[int]) -> tuple[int, ...]:
    return tuple(0 if value == 0 else int(value).bit_length() - 1 for value in board)


def inspect_2048next(raw: bytes) -> dict:
    """Convert a standard 2048next replay into the site's canonical RPL1."""
    if not raw or len(raw) > verse_replay.MAX_INPUT:
        raise ValueError("replay_size_invalid")
    try:
        text = raw.decode("utf-8-sig")
        replay = decode_2048next_replay(text)
    except (UnicodeDecodeError, Replay2048NextError) as exc:
        raise ValueError(f"2048next_invalid:{exc}") from exc
    ruleset = replay.text_extension(2)
    if ruleset not in (None, "pow2"):
        raise ValueError(f"2048next_ruleset_unsupported:{ruleset}")
    if any(isinstance(record, ExtensionRecord)
           and record.extension_type in {EXT_AI_USED, EXT_DIFFICULTY_CHANGE}
           for record in replay.records):
        raise ValueError("2048next_assistance_or_difficulty_change")
    variant = f"{replay.height}x{replay.width}"
    if variant not in engine.VARIANTS:
        raise ValueError("replay_variant_invalid")
    board = [0] * (replay.width * replay.height)
    for index, value_bit in replay.initial_tiles:
        if board[index]:
            raise ValueError("replay_initial_invalid")
        board[index] = 4 if value_bit else 2
    if sum(bool(value) for value in board) != 2:
        raise ValueError("replay_initial_invalid")

    initial = board.copy()
    score = 0
    effective: list[tuple[int, int, int, int]] = []
    history: list[tuple[list[int], int]] = []
    ended = False
    for record in replay.records:
        if isinstance(record, ExtensionRecord):
            continue
        if isinstance(record, CheckpointRecord):
            if record.board_codes != _checkpoint_codes(board):
                raise ValueError("2048next_checkpoint_mismatch")
            continue
        if isinstance(record, EndRecord):
            ended = True
            continue
        if ended:
            raise ValueError("2048next_event_after_end")
        if isinstance(record, UndoRecord):
            if record.count > len(history):
                raise ValueError("2048next_undo_exceeds_history")
            for _ in range(record.count):
                board, score = history.pop()
                effective.pop()
            continue
        if not isinstance(record, MoveRecord) or record.direction not in range(4):
            raise ValueError("2048next_event_unsupported")
        if record.delta_ms > 0xFFFFFFFF:
            raise ValueError("replay_timing_invalid")
        history.append((board.copy(), score))
        moved, gained = engine.move(board, replay.height, replay.width, record.direction)
        value = 4 if record.spawn_value_bit else 2
        if moved == board or moved[record.spawn_index]:
            raise ValueError("replay_move_invalid")
        moved[record.spawn_index] = value
        board = moved
        score += gained
        effective.append((record.direction, record.spawn_index, value, record.delta_ms))

    normalized = verse_replay.encode_rpl1(variant, initial, effective)
    # Reuse the canonical validator for derived fields and a second full replay.
    result = verse_replay.inspect_replay(normalized, variant)
    if replay.start_unix_ms is not None:
        # The 2048next field name is historical; current exports store Unix
        # seconds.  Also accept millisecond values from future/older exporters.
        stamp = float(replay.start_unix_ms)
        result["source_started_at"] = stamp / 1000 if stamp > 10_000_000_000 else stamp
    return result


def inspect_old_verse(raw: bytes, expected_variant: str) -> dict:
    """Normalize Verse's older one-character-per-spawn ``replay_`` export."""
    try:
        text = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError:
        text = raw.decode("latin-1").strip()
    if expected_variant not in engine.VARIANTS:
        raise ValueError("replay_variant_invalid")
    if expected_variant == "4x4":
        if not text.startswith("replay_"):
            raise ValueError("replay_header_invalid")
        payload, row_offset, virtual_cols = text[7:], 0, 4
    else:
        header = next((text[:10] for _ in (0,) if expected_variant in text[:12]), None)
        if header is None:
            raise ValueError("old_verse_variant_header_missing")
        payload = text[10:]
        row_offset = 1 if expected_variant == "2x4" else 0
        virtual_cols = 4
    rows, cols = engine.VARIANTS[expected_variant]
    if len(payload) < 2 or len(payload) - 2 > engine.MAX_MOVES:
        raise ValueError("replay_length_invalid")
    board = [0] * (rows * cols)
    initial = board.copy()
    moves: list[tuple[int, int, int, int]] = []
    for sequence, character in enumerate(payload):
        try:
            encoded = OLD_VERSE_VALUES[character]
        except KeyError as exc:
            raise ValueError("replay_character_invalid") from exc
        direction = encoded >> 5
        visual = ((encoded & 3) << 2) + ((encoded & 15) >> 2)
        virtual_row, column = divmod(visual, virtual_cols)
        row = virtual_row - row_offset
        if direction not in range(4) or not (0 <= row < rows and 0 <= column < cols):
            raise ValueError("replay_spawn_invalid")
        index = row * cols + column
        value = 4 if encoded & 16 else 2
        if sequence < 2:
            if board[index]:
                raise ValueError("replay_initial_invalid")
            board[index] = value
            initial[index] = value
            continue
        moved, _ = engine.move(board, rows, cols, direction)
        if moved == board or moved[index]:
            raise ValueError("replay_move_invalid")
        moved[index] = value
        board = moved
        moves.append((direction, index, value, verse_replay.UNKNOWN_TIMING_MS))
    normalized = verse_replay.encode_rpl1(expected_variant, initial, moves)
    return verse_replay.inspect_replay(normalized, expected_variant)


def inspect_file(path: Path, metadata: FileMetadata) -> dict:
    raw = path.read_bytes()
    if metadata.source == "verse":
        probe = raw.lstrip(b"\xef\xbb\xbf")
        result = (inspect_old_verse(raw, metadata.variant)
                  if probe.startswith(b"replay_")
                  else verse_replay.inspect_replay(raw, metadata.variant))
    elif metadata.source == "2048next":
        result = inspect_2048next(raw)
    else:
        raise ValueError("source_unsupported")
    if result["variant"] != metadata.variant:
        raise ValueError("filename_variant_mismatch")
    if result["score"] != metadata.score:
        raise ValueError(
            f'filename_score_mismatch:expected={metadata.score},calculated={result["score"]}'
        )
    return result


def load_overrides(path: Path | None) -> dict[str, str | None]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
            isinstance(key, str) and (value is None or isinstance(value, str))
            for key, value in data.items()):
        raise ValueError("overrides_must_map_relative_paths_to_run_ids_or_null")
    return {key.replace("\\", "/"): value for key, value in data.items()}


def _date_bounds(date: str, timezone_name: str) -> tuple[float, float]:
    from datetime import datetime, timedelta
    zone = ZoneInfo(timezone_name)
    start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=zone)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def _target(db, *, user_id: int, metadata: FileMetadata, result: dict,
            timezone_name: str, override_run_id: str | None | object) -> tuple[str, str]:
    if override_run_id is None:
        return "skipped_override", "override_requested_skip"
    if isinstance(override_run_id, str):
        rows = db.execute("""SELECT id,user_id,variant,state,source,archive,has_replay,ended,status
            FROM human_runs WHERE id=?""", (override_run_id,)).fetchall()
    else:
        start, end = _date_bounds(metadata.played_date, timezone_name)
        rows = db.execute("""SELECT id,user_id,variant,state,source,archive,has_replay,ended,status
            FROM human_runs WHERE user_id=? AND variant=? AND status='sealed'
              AND json_extract(state,'$.score')=? AND ended>=? AND ended<?
            ORDER BY ended,id""", (user_id, metadata.variant, metadata.score, start, end)).fetchall()
    exact = []
    rejected = []
    for row in rows:
        state = json.loads(row["state"])
        if row["user_id"] != user_id:
            rejected.append(f'{row["id"]}:wrong_user')
        elif row["status"] != "sealed":
            rejected.append(f'{row["id"]}:not_sealed')
        elif row["variant"] != metadata.variant or state.get("score") != metadata.score:
            rejected.append(f'{row["id"]}:metadata_mismatch')
        elif state.get("board") != result["board"]:
            rejected.append(f'{row["id"]}:board_mismatch')
        elif row["source"] not in {"verse", "manual"}:
            rejected.append(f'{row["id"]}:target_source_{row["source"]}_unsupported')
        else:
            exact.append(row)
    if not exact:
        return "no_match", ",".join(rejected[:8]) or "no_score_date_candidate"
    if len(exact) != 1:
        return "ambiguous", ",".join(row["id"] for row in exact)
    row = exact[0]
    normalized = result["normalized"]
    if row["archive"] is not None:
        try:
            same = gzip.decompress(row["archive"]) == normalized
        except (OSError, EOFError):
            same = False
        return ("already_attached", row["id"]) if same else ("archive_conflict", row["id"])
    return "matched", row["id"]


_NO_OVERRIDE = object()


def build_plan(input_root: Path, *, user_id: int, timezone_name: str,
               overrides: dict[str, str | None] | None = None,
               create_missing: bool = False) -> list[PlannedReplay]:
    input_root = input_root.resolve()
    overrides = overrides or {}
    plan: list[PlannedReplay] = []
    paths = sorted(path for path in input_root.rglob("*")
                   if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)
    seen_targets: dict[str, PlannedReplay] = {}
    with database() as db:
        for path in paths:
            try:
                metadata = parse_filename(path, input_root)
            except ValueError as exc:
                plan.append(PlannedReplay(path, None, "invalid_filename", str(exc)))
                continue
            if metadata.source == "pku":
                plan.append(PlannedReplay(path, metadata, "skipped_pku",
                                          "PKU replay was already entered manually"))
                continue
            try:
                result = inspect_file(path, metadata)
            except (ValueError, OSError) as exc:
                plan.append(PlannedReplay(path, metadata, "invalid_replay", str(exc)))
                continue
            override = overrides.get(metadata.relative_path, _NO_OVERRIDE)
            status, detail = _target(db, user_id=user_id, metadata=metadata,
                                     result=result, timezone_name=timezone_name,
                                     override_run_id=override)
            if status == "no_match" and create_missing and override is _NO_OVERRIDE:
                status, detail = "create_missing", "create audited manual archive"
            run_id = detail if status in {"matched", "already_attached", "archive_conflict"} else None
            item = PlannedReplay(path, metadata, status,
                                 "" if status in {"matched", "already_attached"} else detail,
                                 run_id=run_id, normalized=result["normalized"], result=result)
            if status == "matched" and run_id in seen_targets:
                first = seen_targets[run_id]
                if first.normalized == item.normalized:
                    item.status, item.detail = "duplicate_input", first.metadata.relative_path
                else:
                    item.status, item.detail = "target_collision", first.metadata.relative_path
            elif status == "matched":
                seen_targets[run_id] = item
            plan.append(item)
    return plan


def _init_audit(db) -> None:
    db.executescript("""
    CREATE TABLE IF NOT EXISTS human_bulk_replay_import_batches (
        id TEXT PRIMARY KEY,user_id INTEGER NOT NULL,input_root TEXT NOT NULL,
        tool_version INTEGER NOT NULL,created_at REAL NOT NULL,summary_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS human_bulk_replay_import_items (
        batch_id TEXT NOT NULL,relative_path TEXT NOT NULL,run_id TEXT,
        replay_source TEXT NOT NULL,status TEXT NOT NULL,archive_crc INTEGER,
        archive_size INTEGER,note TEXT NOT NULL DEFAULT '',
        PRIMARY KEY(batch_id,relative_path)
    );
    CREATE INDEX IF NOT EXISTS human_bulk_replay_import_run
        ON human_bulk_replay_import_items(run_id,status);
    """)


def _ended_at(item: PlannedReplay, timezone_name: str) -> float:
    assert item.metadata and item.result
    if item.result.get("source_started_at") is not None:
        return float(item.result["source_started_at"]) + item.result["elapsed"] / 1000
    start, end = _date_bounds(item.metadata.played_date, timezone_name)
    return (start + end) / 2


def _create_approved_archive(db, item: PlannedReplay, *, user_id: int,
                             operator_id: int, batch_id: str,
                             timezone_name: str, now: float) -> None:
    assert item.metadata and item.result and item.normalized
    result = item.result
    ended = _ended_at(item, timezone_name)
    archive = gzip.compress(item.normalized, compresslevel=6, mtime=0)
    replay_crc = zlib.crc32(item.normalized)
    warnings = [] if result["game_over"] else ["replay_not_game_over"]
    timing = {"elapsed_ms": result["elapsed"], "timed_moves": result["timed_moves"]}
    run_id = str(uuid.uuid5(uuid.NAMESPACE_URL,
        f"2048tables:bulk-replay:{user_id}:{item.metadata.relative_path}:{replay_crc}"))
    existing = db.execute("SELECT id FROM human_runs WHERE id=?", (run_id,)).fetchone()
    if existing:
        raise RuntimeError(f"generated_run_exists:{run_id}")
    cursor = db.execute("""INSERT INTO human_archive_applications
        (user_id,variant,claimed_ended_at,claimed_score,status,replay_crc,replay_size,
         moves,final_board_json,is_game_over,timing_summary_json,warning_flags_json,
         original_filename,requested_at,updated_at,approved_by,approved_at,review_note,run_id)
        VALUES(?,?,?,?,'approved',?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
        user_id, item.metadata.variant, ended, result["score"], replay_crc,
        len(item.normalized), result["moves"], json.dumps(result["board"], separators=(",", ":")),
        int(result["game_over"]), json.dumps(timing, separators=(",", ":")),
        json.dumps(warnings), item.path.name[:180], now, now, operator_id, now,
        f"bulk replay import {batch_id}; source={item.metadata.source}", run_id,
    ))
    application_id = cursor.lastrowid
    created = max(1394323200.0, ended - result["elapsed"] / 1000)
    state = {
        "score": result["score"], "board": result["board"], "seq": result["moves"],
        "elapsed": result["elapsed"], "nodes": result["nodes"], "hash": None,
        "spawnCount": result["spawn_count"], "fourCount": result["four_count"],
        "replay_timing_version": 2,
        "imported": {"application_id": application_id, "bulk_batch_id": batch_id,
                     "replay_source": item.metadata.source,
                     "ordinal": item.metadata.ordinal},
    }
    db.execute("""INSERT INTO human_runs
        (id,user_id,browser,variant,request_id,seed,threshold,status,eligibility,reason,
         created,ended,writer,epoch,permit_until,monitored,state,archive,display_threshold,
         visible,has_replay,source,single_rating,single_rating_version)
        VALUES(?,?,?,?,?,'',0,'sealed','eligible','imported',?,?,'',0,0,0,?,?,0,1,1,
               'manual',?,?)""", (
        run_id, user_id, f"manual:{application_id}", item.metadata.variant,
        str(application_id), created, ended, json.dumps(state, separators=(",", ":")),
        archive, rating.single_rating(item.metadata.variant, result["board"]),
        rating.RATING_VERSION,
    ))
    for action in ("submitted_bulk", "approved_bulk"):
        db.execute("""INSERT INTO human_archive_application_audit
            (application_id,operator_id,action,note,created_at) VALUES(?,?,?,?,?)""",
            (application_id, operator_id, action,
             f"batch={batch_id}; source={item.metadata.source}", now))
    run = dict(db.execute("SELECT * FROM human_runs WHERE id=?", (run_id,)).fetchone())
    statistics.upsert_fact(db, run, result["board"], result["score"], result["rate"])
    item.run_id = run_id
    item.status = "created"


def apply_plan(plan: list[PlannedReplay], *, user_id: int, input_root: Path,
               operator_id: int | None = None,
               timezone_name: str = "Asia/Shanghai") -> dict:
    selected = [item for item in plan if item.status in {"matched", "create_missing"}]
    if any(item.status == "create_missing" for item in selected) and operator_id is None:
        raise ValueError("operator_id_required_to_create_missing")
    batch_id = uuid.uuid4().hex
    affected: set[tuple[int, str]] = set()
    now = time.time()
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        _init_audit(db)
        for item in selected:
            if item.status == "create_missing":
                _create_approved_archive(db, item, user_id=user_id,
                                         operator_id=int(operator_id), batch_id=batch_id,
                                         timezone_name=timezone_name, now=now)
                affected.add((user_id, item.metadata.variant))
                continue
            assert item.run_id and item.normalized and item.result and item.metadata
            row = db.execute("""SELECT id,user_id,variant,state,source,archive,status,ended
                FROM human_runs WHERE id=?""", (item.run_id,)).fetchone()
            state = json.loads(row["state"]) if row else {}
            if (not row or row["user_id"] != user_id or row["status"] != "sealed"
                    or row["variant"] != item.metadata.variant
                    or row["source"] not in {"verse", "manual"}
                    or state.get("score") != item.result["score"]
                    or state.get("board") != item.result["board"]):
                raise RuntimeError(f"target_changed:{item.run_id}")
            archive = gzip.compress(item.normalized, compresslevel=6, mtime=0)
            if row["archive"] is not None:
                if row["archive"] != archive:
                    raise RuntimeError(f"archive_changed:{item.run_id}")
                item.status = "already_attached"
                continue
            state.update({
                "seq": item.result["moves"], "elapsed": item.result["elapsed"],
                "nodes": item.result["nodes"], "spawnCount": item.result["spawn_count"],
                "fourCount": item.result["four_count"], "replay_timing_version": 2,
            })
            db.execute("""UPDATE human_runs SET archive=?,has_replay=1,state=? WHERE id=?""",
                       (archive, json.dumps(state, separators=(",", ":")), item.run_id))
            db.execute("""UPDATE rolling_candidates SET has_replay=1,replay_id=?
                WHERE run_id=?""", (item.run_id, item.run_id))
            statistics.upsert_fact(db, dict(row), item.result["board"],
                                   item.result["score"], item.result["rate"])
            affected.add((user_id, item.metadata.variant))
            item.status = "attached"
        for affected_user, variant in affected:
            rating.refresh_player(db, affected_user, variant)
            statistics.rebuild_player(db, affected_user, variant)
        for item in selected:
            if item.run_id:
                leaderboards.refresh_run(db, item.run_id, now=now)
        summary = summarize(plan)
        db.execute("""INSERT INTO human_bulk_replay_import_batches
            (id,user_id,input_root,tool_version,created_at,summary_json)
            VALUES(?,?,?,?,?,?)""", (batch_id, user_id, str(input_root), TOOL_VERSION, now,
                                      json.dumps(summary, separators=(",", ":"))))
        for item in plan:
            metadata = item.metadata
            db.execute("""INSERT INTO human_bulk_replay_import_items
                (batch_id,relative_path,run_id,replay_source,status,archive_crc,archive_size,note)
                VALUES(?,?,?,?,?,?,?,?)""", (
                batch_id, metadata.relative_path if metadata else item.path.name,
                item.run_id, metadata.source if metadata else "unknown", item.status,
                zlib.crc32(item.normalized) if item.normalized else None,
                len(item.normalized) if item.normalized else None, item.detail,
            ))
    return {"batch_id": batch_id, **summarize(plan)}


def summarize(plan: list[PlannedReplay]) -> dict:
    counts: dict[str, int] = {}
    for item in plan:
        counts[item.status] = counts.get(item.status, 0) + 1
    return {"total": len(plan), "counts": dict(sorted(counts.items()))}


def resolve_user_id(username: str | None, user_id: int | None) -> int:
    if user_id is not None:
        return int(user_id)
    if not username:
        raise ValueError("username_or_user_id_required")
    from backend.human_play.service import player_id_for_name
    return int(player_id_for_name(username))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attach packaged external replays to existing Play history records.")
    parser.add_argument("--input", required=True, type=Path, help="Directory to scan recursively")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--username")
    identity.add_argument("--user-id", type=int)
    parser.add_argument("--timezone", default="Asia/Shanghai",
                        help="Timezone used by dates embedded in filenames")
    parser.add_argument("--overrides", type=Path,
                        help="JSON map: relative filename -> exact run id, or null to skip")
    parser.add_argument("--report", type=Path,
                        help="Write the full plan/result as JSON")
    parser.add_argument("--apply", action="store_true",
                        help="Actually attach uniquely matched replays; default is dry-run")
    parser.add_argument("--create-missing", action="store_true",
                        help="Create audited approved manual archives when no score record exists")
    parser.add_argument("--operator-id", type=int,
                        help="Approving operator recorded for --create-missing")
    parser.add_argument("--require-all", action="store_true",
                        help="Fail unless every non-PKU input is matched/already attached")
    args = parser.parse_args(argv)
    if not args.input.is_dir():
        parser.error("--input must be an existing directory")
    try:
        resolved_user_id = resolve_user_id(args.username, args.user_id)
        plan = build_plan(args.input, user_id=resolved_user_id,
                          timezone_name=args.timezone,
                          overrides=load_overrides(args.overrides),
                          create_missing=args.create_missing)
    except Exception as exc:
        parser.error(str(exc))
    blocking = [item for item in plan if item.metadata and item.metadata.source != "pku"
                and item.status not in {"matched", "create_missing", "already_attached",
                                        "duplicate_input"}]
    result = {"mode": "apply" if args.apply else "dry-run", "user_id": resolved_user_id,
              "input": str(args.input.resolve()), **summarize(plan),
              "files": [item.payload() for item in plan]}
    if args.require_all and blocking:
        result["error"] = "unresolved_inputs"
    elif args.apply:
        result["apply"] = apply_plan(plan, user_id=resolved_user_id,
                                     input_root=args.input.resolve(),
                                     operator_id=args.operator_id,
                                     timezone_name=args.timezone)
        result.update(summarize(plan))
        result["files"] = [item.payload() for item in plan]
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    return 2 if args.require_all and blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())
