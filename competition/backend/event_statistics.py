"""Read-only Play results integration. No replay blobs or game simulation."""
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from contextlib import closing
from datetime import datetime, timezone

from .errors import CompetitionError


def top_rating(variant, board_sums):
    """Pinned 3x3 formula; Play is deployed separately, not a runtime dependency."""
    if variant != '3x3':
        raise ValueError('Unsupported event rating variant')
    values = list(board_sums)
    return 800 * math.log2(sum(values) / len(values)) - 5468


def readonly(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    db.row_factory = sqlite3.Row
    return db


def accounts(user_ids):
    from backend.auth.db import get_auth_db_path
    try:
        with closing(readonly(get_auth_db_path())) as db:
            rows = db.execute(f"SELECT id,display_name FROM users WHERE id IN ({','.join('?' for _ in user_ids)}) AND status='active'", user_ids)
            return {row['id']: row['display_name'] or f"User {row['id']}" for row in rows}
    except sqlite3.Error as exc:
        raise CompetitionError('EVENT_SOURCE_UNAVAILABLE', '账号数据源暂不可用，未导入任何名单。', 503) from exc


def accounts_by_username(usernames):
    """Resolve current usernames only, using the main site's normalization rules."""
    from backend.auth.db import get_auth_db_path
    from backend.profile.validation import canonical_display_name_key
    keys = sorted({canonical_display_name_key(name) for name in usernames})
    if not keys:
        return {}
    try:
        with closing(readonly(get_auth_db_path())) as db:
            rows = db.execute(f"SELECT id,display_name FROM users WHERE display_name_key IN ({','.join('?' for _ in keys)}) AND status='active'", keys)
            matches = {}
            for row in rows:
                key = canonical_display_name_key(row['display_name'])
                if key in keys:
                    matches.setdefault(key, []).append(row['id'])
            return matches
    except sqlite3.Error as exc:
        raise CompetitionError('EVENT_SOURCE_UNAVAILABLE', '账号数据源暂不可用，未导入任何名单。', 503) from exc


def play_results(user_ids, start, end):
    from backend.auth.db import get_auth_db_path
    if not user_ids:
        return {}
    try:
        path = Path(os.environ.get('HUMAN_PLAY_DB') or get_auth_db_path().with_name('human-play.sqlite3'))
        with closing(readonly(path)) as db:
            db.execute('BEGIN')
            # Same native eligibility as Play ranking; imported/manual/Verse runs are never eligible here.
            result = {}
            for uid in user_ids:
                predicate = """user_id=? AND variant='3x3' AND status='sealed'
                    AND source='native' AND visible=1 AND eligibility='eligible'
                    AND (reason='game_over' OR id IN (SELECT run_id FROM human_rank_approvals))
                    AND created>=? AND ended<? AND ended>=created"""
                args = (uid, start, end)
                count = db.execute(f'SELECT COUNT(*) FROM human_runs WHERE {predicate}', args).fetchone()[0]
                rows = db.execute(f"""SELECT id,created,ended,json_extract(state,'$.score') AS score,
                    json_extract(state,'$.board') AS board FROM human_runs WHERE {predicate}
                    ORDER BY score DESC,ended DESC,id DESC LIMIT 5""", args)
                games = []
                for row in rows:
                    board = json.loads(row['board'])
                    games.append({'id': row['id'], 'score': row['score'], 'board_sum': sum(board),
                                  'started_at': row['created'], 'ended_at': row['ended']})
                result[uid] = {'completed_games': count, 'games': games}
            return result
    except (sqlite3.Error, ValueError, TypeError) as exc:
        raise CompetitionError('EVENT_SOURCE_UNAVAILABLE', '对局数据源暂不可用，请稍后刷新；这不代表选手成绩为零。', 503) from exc


class EventStatistics:
    def __init__(self, catalog, *, account_reader=accounts, result_reader=play_results):
        self.catalog = catalog
        self.database = catalog.database
        self.account_reader = account_reader
        self.result_reader = result_reader
        self._cache = {}
        self._cache_lock = threading.Lock()

    def _config(self, db, slug):
        event = self.catalog._event(db, slug)
        if event['format_key'] != 'team-top5-3x3-v1':
            raise CompetitionError('EVENT_FORMAT_MISMATCH', '该赛事不支持积分统计。', 409)
        return event, db.execute('SELECT * FROM tournament_statistics_config WHERE event_slug=?', (slug,)).fetchone()

    def import_roster(self, slug, principal, *, entries, revision, dry_run=True):
        from .event_enrollment import EventEnrollment
        return EventEnrollment(self.catalog, account_reader=self.account_reader,
                               username_reader=self.catalog.enrollment.username_reader).import_roster(
            slug, principal, entries=entries, revision=revision, dry_run=dry_run)

    def standings(self, slug):
        # Public polling shares a short cache; never scan the same twenty histories per spectator.
        with self._cache_lock:
            with self.database.transaction() as db:
                revision = self.catalog.enrollment.config(db, slug)['revision']
            cached = self._cache.get(slug)
            if cached and cached[1]['roster_revision'] == revision and time.monotonic() - cached[0] < 10:
                return cached[1]
            result = self._standings(slug)
            self._cache[slug] = (time.monotonic(), result)
            return result

    def _standings(self, slug):
        now = datetime.now(timezone.utc)
        with self.database.transaction() as db:
            _, config = self._config(db, slug)
            config = dict(config)
            enrollment = self.catalog.enrollment.config(db, slug)
            config['roster_revision'] = enrollment['revision']
            config['roster_locked'] = bool(enrollment['roster_locked'])
            roster = [dict(row) for row in db.execute('''SELECT e.user_id,e.display_name,e.is_external,t.name AS team_name
                FROM tournament_entrants e LEFT JOIN tournament_teams t ON t.id=e.team_id
                WHERE e.event_slug=? ORDER BY e.user_id''', (slug,))]
        start = datetime.fromisoformat(config['starts_at']).timestamp()
        end = datetime.fromisoformat(config['ends_at']).timestamp()
        data = self.result_reader([row['user_id'] for row in roster], start, min(end, now.timestamp()))
        players = []
        for member in roster:
            record = data.get(member['user_id'], {'completed_games': 0, 'games': []})
            games = record['games']
            sums = [game['board_sum'] for game in games]
            players.append({**member, **record, 'selected_games': len(games),
                            'board_sum': sum(sums), 'average_board_sum': sum(sums) / 5,
                            'rating': top_rating('3x3', sums + [0] * (5-len(sums))) if sum(sums) > 0 else 0,
                            'complete': len(games) == 5})
        players.sort(key=lambda p: (-(p['rating'] if p['rating'] is not None else -1e9), p['user_id']))
        teams = []
        for name in sorted({p['team_name'] for p in players if p['team_name']}):
            members = [p for p in players if p['team_name'] == name]
            teams.append({'name': name, 'players': members, 'completed_games': sum(p['completed_games'] for p in members),
                          'selected_games': sum(p['selected_games'] for p in members),
                          'board_sum': sum(p['board_sum'] for p in members),
                          'rating': sum(p['rating'] for p in members),
                          'complete': all(p['complete'] for p in members)})
        teams.sort(key=lambda t: (-(t['rating'] if t['rating'] is not None else -1e9), t['name']))
        return {**config, 'phase': 'upcoming' if now.timestamp() < start else 'ended' if now.timestamp() >= end else 'active',
                'as_of': now.isoformat(), 'players': players, 'teams': teams,
                'unassigned_count': sum(not p['team_name'] for p in players),
                'rating_note': '按得分选最佳五局，不足五局以零补齐，再以五局平均盘面和计算个人 rating；零有效局时 rating 为零。团队盘面和与 rating 分别为队员值之和。'}
