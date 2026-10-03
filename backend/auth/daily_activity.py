"""Account activity shared by all public sites."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import OrderedDict
from threading import Lock

from backend.auth.db import auth_db, get_auth_db_path


BEIJING = timezone(timedelta(hours=8))
SITES = frozenset({'main', 'play', 'live', 'tournament', 'tables'})
_recorded = OrderedDict()
_record_lock = Lock()
_CACHE_LIMIT = 16384


def activity_day(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(BEIJING).date().isoformat()


def site_from_host(host: str) -> str:
    hostname = str(host or '').split(':', 1)[0].lower()
    if hostname == 'play.2048tables.online':
        return 'play'
    if hostname == 'live.2048tables.online':
        return 'live'
    if hostname == 'tournament.2048tables.online':
        return 'tournament'
    if hostname == 'tables.2048tables.online':
        return 'tables'
    return 'main'


def bind_activity_account(connection, user_id: int) -> None:
    """Only call after authenticating a real account, never a client-supplied ID."""
    connection.state.daily_activity_user_id = int(user_id)


def _cache_key(user_id: int, site: str, day: str) -> tuple:
    return (str(get_auth_db_path().resolve()), day, int(user_id), site)


def visit_is_recorded(user_id: int, site: str, day: str) -> bool:
    key = _cache_key(user_id, site, day)
    with _record_lock:
        return key in _recorded


def record_daily_visit(user_id: int, site: str, *, now: datetime | None = None) -> None:
    if site not in SITES:
        raise ValueError(f'unknown activity site: {site}')
    day = activity_day(now)
    key = _cache_key(user_id, site, day)
    with _record_lock:
        if key in _recorded:
            return
    with auth_db() as db:
        db.execute(
            'INSERT OR IGNORE INTO daily_user_activity(day, user_id, site) VALUES(?, ?, ?)',
            (day, int(user_id), site),
        )
    with _record_lock:
        # Cache only committed writes. Different processes still deduplicate in SQL.
        _recorded[key] = None
        while len(_recorded) > _CACHE_LIMIT:
            _recorded.popitem(last=False)
