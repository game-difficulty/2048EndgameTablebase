"""Account activity shared by the main, Play and Live sites."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from backend.auth.db import auth_db


BEIJING = timezone(timedelta(hours=8))
SITES = frozenset({'main', 'play', 'live'})


def activity_day(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(BEIJING).date().isoformat()


def site_from_host(host: str) -> str:
    hostname = str(host or '').split(':', 1)[0].lower()
    if hostname == 'play.2048tables.online':
        return 'play'
    if hostname == 'live.2048tables.online':
        return 'live'
    return 'main'


def record_daily_visit(user_id: int, site: str, *, now: datetime | None = None) -> None:
    if site not in SITES:
        raise ValueError(f'unknown activity site: {site}')
    day = activity_day(now)
    with auth_db() as db:
        db.execute(
            'INSERT OR IGNORE INTO daily_user_activity(day, user_id, site) VALUES(?, ?, ?)',
            (day, int(user_id), site),
        )
