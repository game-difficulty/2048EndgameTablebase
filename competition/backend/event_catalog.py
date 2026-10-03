"""Event directory above rooms; never derives event membership from a room name."""
import re
import sqlite3
from datetime import datetime, timezone

from .errors import CompetitionError
from .event_formats import FORMATS
from .event_statistics import EventStatistics
from .event_enrollment import EventEnrollment


class EventCatalog:
    def __init__(self, rooms):
        self.rooms = rooms
        self.database = rooms.database
        self.statistics = EventStatistics(self)
        self.enrollment = EventEnrollment(self)

    def _manager(self, event, principal):
        return bool(principal and (self.rooms._is_platform_organizer(principal)
                    or event['owner_user_id'] == principal.user_id))

    def _event(self, db, slug):
        row = db.execute('SELECT * FROM tournament_events WHERE slug=?', (slug,)).fetchone()
        if not row:
            raise CompetitionError('EVENT_NOT_FOUND', '赛事不存在。', 404)
        return row

    def assign_organizer(self, slug, principal, *, user_id, dry_run=True):
        if principal.site_role.lower() != 'admin':
            raise CompetitionError('EVENT_OWNER_ADMIN_REQUIRED', '仅站长可指定赛事举办方。', 403)
        with self.database.transaction(immediate=True) as db:
            event = self._event(db, slug)
            users = self.enrollment.account_reader([user_id])
            if user_id not in users:
                raise CompetitionError('EVENT_OWNER_NOT_FOUND', '找不到该用户或账号不可用。', 404)
            organizer = {'user_id': user_id, 'display_name': users[user_id]}
            if dry_run:
                return {'organizer': organizer}
            db.execute('UPDATE tournament_events SET owner_user_id=? WHERE slug=?', (user_id, slug))
            self.enrollment._audit(db, slug, principal, 'assign_organizer', {
                'previous_user_id': event['owner_user_id'], 'user_id': user_id})
        return {'event': self.detail(slug, principal), 'organizer': organizer}

    def _view(self, db, row, principal):
        result = {key: row[key] for key in ('slug', 'name', 'description', 'rules', 'status', 'format_key')} | {
            'can_manage': self._manager(row, principal),
            'can_assign_organizer': bool(principal and principal.site_role.lower() == 'admin'),
            'organizer': {'user_id': row['owner_user_id']} if row['owner_user_id'] else None,
            'room_count': db.execute('SELECT COUNT(*) FROM tournament_room_links WHERE event_slug=?',
                                     (row['slug'],)).fetchone()[0],
            'entry_kind': 'team', **FORMATS[row['format_key']],
        }
        result['team_size'] = self.enrollment.config(db, row['slug'])['team_size']
        if result['capabilities']['rooms']:
            result['label'] = '团队选 Ban 赛事'
        if result['capabilities']['statistics']:
            window = db.execute('SELECT starts_at,ends_at FROM tournament_statistics_config WHERE event_slug=?', (row['slug'],)).fetchone()
            now = datetime.now(timezone.utc)
            result['statistics_window'] = dict(window)
            result['status'] = ('preparing' if now < datetime.fromisoformat(window['starts_at']) else
                                'finished' if now >= datetime.fromisoformat(window['ends_at']) else 'active')
        return result

    def list(self, principal=None):
        with self.database.transaction() as db:
            return [self._view(db, row, principal) for row in db.execute(
                'SELECT * FROM tournament_events ORDER BY created_at DESC, slug')]

    def room_event(self, db, competition_id):
        row = db.execute('''SELECT e.slug,e.name FROM tournament_events e JOIN tournament_room_links l
            ON l.event_slug=e.slug WHERE l.competition_id=?''', (competition_id,)).fetchone()
        return dict(row) if row else None

    def update(self, slug, principal, *, name, description, rules, status):
        name = ' '.join(name.split())
        if not 2 <= len(name) <= 100 or status not in ('preparing', 'active', 'finished'):
            raise CompetitionError('INVALID_EVENT', '请检查赛事名称及状态。')
        with self.database.transaction(immediate=True) as db:
            event = self._event(db, slug)
            if not self._manager(event, principal):
                raise CompetitionError('EVENT_MANAGER_REQUIRED', '无权管理该赛事。', 403)
            db.execute('UPDATE tournament_events SET name=?,description=?,rules=?,status=? WHERE slug=?',
                       (name, description, rules, status, slug))
        return self.detail(slug, principal)

    def detail(self, slug, principal=None):
        with self.database.transaction() as db:
            event = self._event(db, slug)
            result = self._view(db, event, principal)
            result['rooms'] = []
            result['record_candidates'] = []
            for room in db.execute('''SELECT c.* FROM competitions c JOIN tournament_room_links l
                ON l.competition_id=c.id LEFT JOIN competition_schedule s ON s.competition_id=c.id
                WHERE l.event_slug=? ORDER BY COALESCE(s.starts_at,c.created_at),c.id''', (slug,)):
                admitted = bool(principal and not db.execute(
                    'SELECT 1 FROM competition_expulsions WHERE competition_id=? AND user_id=?',
                    (room['id'], principal.user_id)).fetchone())
                may_enter = admitted and (self.rooms._is_platform_organizer(principal)
                    or room['created_by_user_id'] == principal.user_id or db.execute('''
                    SELECT 1 FROM competition_seats WHERE competition_id=? AND user_id=?
                    UNION SELECT 1 FROM competition_staff WHERE competition_id=? AND user_id=?
                    UNION SELECT 1 FROM competition_scheduled_players WHERE competition_id=? AND user_id=?''',
                    (room['id'], principal.user_id, room['id'], principal.user_id,room['id'],principal.user_id)).fetchone())
                # Directory is public; never expose join codes or private room snapshots to spectators.
                result['rooms'].append({'name': room['name'], 'status': room['status'],
                    'schedule': self.rooms.schedule.view(db, room['id']),
                    'series_score': dict(db.execute('SELECT yellow_wins AS yellow,white_wins AS white FROM competition_match_control WHERE competition_id=?', (room['id'],)).fetchone() or {}),
                    'room_code': room['room_code'] if may_enter else None,
                    'public_key': room['public_key'] if room['live_started_at'] else None})
                if room['status']=='FINISHED':
                    games=[dict(r) for r in db.execute('SELECT * FROM competition_game_results WHERE competition_id=?',(room['id'],))]
                    self.rooms._attach_result_timings(db,room['id'],games)
                    for game in games:
                        side=game.get('record_eligible_side')
                        if not side: continue
                        session=db.execute('SELECT s.project_ref,s.rules_version,s.player_user_id,p.name FROM competition_game_sessions s JOIN competition_projects p ON p.competition_id=s.competition_id AND p.project_key=s.project_key WHERE s.competition_id=? AND s.game_key=? AND s.side=?',(room['id'],game['game_key'],side)).fetchone()
                        if session:
                            result['record_candidates'].append({**dict(session),'game_key':game['game_key'],'score':game[f'{side}_score'],'elapsed_ms':game.get(f'{side}_elapsed_ms'),'public_key':room['public_key']})
            return result

    def create(self, principal, *, slug, name, description='', rules='', team_size=3):
        if type(team_size) is not int or not 1 <= team_size <= 16:
            raise CompetitionError('INVALID_EVENT', '每队人数须为 1 至 16。')
        if not self.rooms._is_platform_organizer(principal):
            raise CompetitionError('EVENT_MANAGER_REQUIRED', '仅赛事管理员可创建赛事。', 403)
        if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', slug) or not 2 <= len(slug) <= 64:
            raise CompetitionError('INVALID_EVENT_SLUG', '赛事地址须为 2–64 位小写英文、数字或连字符。')
        name = ' '.join(name.split())
        if not 2 <= len(name) <= 100:
            raise CompetitionError('INVALID_NAME', '赛事名称须为 2–100 个字符。')
        with self.database.transaction(immediate=True) as db:
            try:
                db.execute('''INSERT INTO tournament_events
                    (slug,name,description,rules,owner_user_id,created_at) VALUES(?,?,?,?,?,?)''',
                    (slug, name, description, rules, principal.user_id, datetime.now(timezone.utc).isoformat()))
                db.execute('INSERT INTO tournament_enrollment_config(event_slug,mode,team_size) VALUES(?,\'self_team\',?)', (slug, team_size))
            except sqlite3.IntegrityError as exc:
                raise CompetitionError('EVENT_SLUG_TAKEN', '该赛事地址已存在。', 409) from exc
        return self.detail(slug, principal)

    def link_in_transaction(self, db, slug, room, principal):
        self.rooms._reject_duel_official(db, room)
        event = self._event(db, slug)
        if not FORMATS[event['format_key']]['capabilities']['rooms']:
            raise CompetitionError('EVENT_FORMAT_MISMATCH', '统计型赛事不使用对战房间。', 409)
        if not self._manager(event, principal):
            raise CompetitionError('EVENT_MANAGER_REQUIRED', '无权管理该赛事。', 403)
        if not (self.rooms._is_platform_organizer(principal) or room['created_by_user_id'] == principal.user_id):
            raise CompetitionError('ROOM_OWNER_REQUIRED', '只能关联自建房间。', 403)
        existing = db.execute('SELECT event_slug FROM tournament_room_links WHERE competition_id=?',
                              (room['id'],)).fetchone()
        if existing:
            if existing['event_slug'] == slug:
                return
            raise CompetitionError('ROOM_EVENT_CONFLICT', '该房间已属于其他赛事。', 409)
        if event['status'] == 'finished':
            raise CompetitionError('EVENT_FINISHED', '已结束的赛事不能新增房间。', 409)
        db.execute('INSERT INTO tournament_room_links VALUES(?,?,?,?)',
                   (room['id'], slug, principal.user_id, datetime.now(timezone.utc).isoformat()))
        self.rooms._append_event(db, room['id'], 'competition.event_linked', principal.user_id, {'event_slug': slug})
        self.rooms._touch(db, room['id'])

    def link(self, slug, code, principal):
        with self.database.transaction(immediate=True) as db:
            room = self.rooms._room_row(db, code)
            self.link_in_transaction(db, slug, room, principal)
        return self.detail(slug, principal)
