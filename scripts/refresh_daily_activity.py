"""Idempotently backfill daily active accounts from existing action records.

Run periodically; authenticated traffic is also recorded by shared middleware.
Historical browsing without action records cannot be recovered.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from backend.auth.daily_activity import BEIJING
from backend.auth.db import auth_db, get_auth_db_path


# Use only events caused by a person. Settlement, grant and award rows are not visits.
AUTH_SOURCES = (
    ('sessions', 'created_at', 'user_id', 'recorded', None),
    ('usage_events', 'created_at', 'user_id', 'main', None),
    ('token_ledger', 'created_at', 'user_id', 'recorded',
     "event_type IN ('reserve', 'consume', 'room_prediction_stake', "
     "'room_prediction_target_stake', 'live_red_send')"),
    ('gamer_ranked_runs', 'started_at', 'user_id', 'main', None),
    ('minigame_ranked_runs', 'started_at', 'user_id', 'main', None),
    ('battle_members', 'joined_at', 'user_id', 'main', None),
    ('battle_chat_messages', 'created_at', 'user_id', 'main', None),
    ('analysis_history_jobs', 'created_at', 'user_id', 'main', None),
    ('live_gift_orders', 'created_at', 'user_id', 'live', None),
    ('live_lucky_entries', 'joined_at', 'user_id', 'live', None),
    ('live_red_claims', 'created_at', 'user_id', 'live', None),
)

COMPETITION_SOURCES = (
    ('competition_commands', 'created_at', 'user_id'),
    ('tournament_enrollment_audit', 'created_at', 'actor_user_id'),
    ('tournament_roster_audit', 'created_at', 'actor_user_id'),
    ('practice_bests', 'achieved_at', 'user_id'),
)


def _cutoff_utc(first_day: date) -> datetime:
    return datetime.combine(first_day, datetime.min.time(), BEIJING).astimezone(timezone.utc)


def refresh(*, first_day: date, last_day: date, play_db: Path | None = None,
            competition_db: Path | None = None) -> dict:
    if first_day > last_day:
        raise ValueError('first_day must not exceed last_day')
    start = _cutoff_utc(first_day)
    end = _cutoff_utc(last_day + timedelta(days=1))
    activity = set()
    with closing(sqlite3.connect(
        get_auth_db_path().resolve().as_uri() + '?mode=ro', uri=True
    )) as source:
        source.row_factory = sqlite3.Row
        valid_users = {row['id'] for row in source.execute('SELECT id FROM users')}
        available_tables = {
            row['name'] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table, timestamp, actor, site, condition in AUTH_SOURCES:
            if table not in available_tables:
                continue
            where = f'{timestamp} >= ? AND {timestamp} < ? AND {actor} IS NOT NULL'
            if condition:
                where += f' AND {condition}'
            entries = source.execute(
                f"SELECT DISTINCT date({timestamp}, '+8 hours'), {actor} "
                f'FROM {table} WHERE {where}',
                (start.isoformat(), end.isoformat()),
            )
            activity.update((day, user_id, site) for day, user_id in entries
                            if day and user_id in valid_users)

    path = Path(play_db) if play_db is not None else Path(
        os.getenv('HUMAN_PLAY_DB') or '/var/lib/2048tables/play/human.sqlite3')
    if play_db is not None and not path.is_file():
        raise FileNotFoundError(path)
    if path.is_file():
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as play:
            play.row_factory = sqlite3.Row
            entries = play.execute('''
                SELECT date(created, 'unixepoch', '+8 hours'), user_id
                FROM human_runs WHERE created >= ? AND created < ?
                UNION
                SELECT date(chunks.received, 'unixepoch', '+8 hours'), runs.user_id
                FROM human_chunks AS chunks
                JOIN human_runs AS runs ON runs.id = chunks.run_id
                WHERE chunks.received >= ? AND chunks.received < ?
            ''', (start.timestamp(), end.timestamp(), start.timestamp(), end.timestamp()))
            activity.update((day, user_id, 'play') for day, user_id in entries
                            if day and user_id in valid_users)

    competition_path = Path(competition_db) if competition_db is not None else (
        Path(os.environ['COMPETITION_DB']) if os.getenv('COMPETITION_DB') else None)
    if competition_db is not None and not competition_path.is_file():
        raise FileNotFoundError(competition_path)
    if competition_path is not None and competition_path.is_file():
        with closing(sqlite3.connect(competition_path.resolve().as_uri() + '?mode=ro', uri=True)) as source:
            tables = {row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table, timestamp, actor in COMPETITION_SOURCES:
                if table not in tables:
                    continue
                entries = source.execute(
                    f"SELECT DISTINCT date({timestamp}, '+8 hours'), {actor} FROM {table} "
                    f"WHERE julianday({timestamp}) >= julianday(?) "
                    f"AND julianday({timestamp}) < julianday(?) AND {actor} IS NOT NULL",
                    (start.isoformat(), end.isoformat()),
                )
                activity.update((day, user_id, 'tournament') for day, user_id in entries
                                if day and user_id in valid_users)

    with auth_db() as db:
        before = db.total_changes
        db.executemany(
            'INSERT OR IGNORE INTO daily_user_activity(day, user_id, site) VALUES(?, ?, ?)',
            sorted(activity),
        )
        added = db.total_changes - before
        totals = db.execute('''
            SELECT day, COUNT(DISTINCT user_id) AS active_accounts
            FROM daily_user_activity WHERE day >= ? AND day <= ?
            GROUP BY day ORDER BY day
        ''', (first_day.isoformat(), last_day.isoformat())).fetchall()
    return {'added': added,
            'days': [dict(row) for row in totals]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=14)
    parser.add_argument('--since', type=date.fromisoformat)
    parser.add_argument('--through', type=date.fromisoformat)
    parser.add_argument('--play-db', type=Path)
    parser.add_argument('--competition-db', type=Path)
    args = parser.parse_args()
    last_day = args.through or datetime.now(BEIJING).date()
    first_day = args.since or last_day - timedelta(days=max(1, args.days) - 1)
    print(json.dumps(refresh(first_day=first_day, last_day=last_day,
                             play_db=args.play_db, competition_db=args.competition_db), ensure_ascii=False))


if __name__ == '__main__':
    main()
