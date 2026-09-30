"""Frozen room rosters, scheduled starts and one-time attendance adjudication."""
from datetime import datetime, timedelta, timezone

from .errors import CompetitionError


class EventSchedule:
    def __init__(self, rooms):
        self.rooms = rooms

    def bind(self, db, room, slug, yellow_team_id, white_team_id, starts_at):
        if not slug or not yellow_team_id or not white_team_id or yellow_team_id == white_team_id:
            raise CompetitionError('INVALID_SCHEDULE', '请从所属赛事选择两支不同的锁定队伍。')
        try:
            start = datetime.fromisoformat(starts_at.replace('Z', '+00:00'))
            if start.tzinfo is None or start <= datetime.now(timezone.utc):
                raise ValueError()
        except (ValueError, AttributeError):
            raise CompetitionError('INVALID_SCHEDULE_TIME', '开战时间须为未来时间，并包含时区。')
        config = self.rooms.events.enrollment.config(db, slug)
        if not config['roster_locked'] or config['team_size'] != 3 or config['mode'] == 'solo':
            raise CompetitionError('ROSTER_NOT_LOCKED', '请先锁定三人团队最终名单，再编排对战。', 409)
        cid = room['id']
        teams = []
        for side, tid in [('yellow', yellow_team_id), ('white', white_team_id)]:
            team = db.execute('SELECT * FROM tournament_teams WHERE event_slug=? AND id=?', (slug, tid)).fetchone()
            if not team:
                raise CompetitionError('INVALID_SCHEDULE_TEAM', '队伍不属于该赛事。')
            members = db.execute('SELECT * FROM tournament_entrants WHERE event_slug=? AND team_id=? ORDER BY user_id', (slug, tid)).fetchall()
            members = sorted(members, key=lambda p: p['user_id'] != team['captain_user_id'])
            if len(members) != 3 or members[0]['user_id'] != team['captain_user_id']:
                raise CompetitionError('INVALID_SCHEDULE_TEAM', '队伍须有三位成员及一位队长。')
            for position, member in enumerate(members, 1):
                db.execute('INSERT INTO competition_scheduled_players VALUES(?,?,?,?,?,NULL)',
                           (cid, member['user_id'], side, position, member['display_name']))
                db.execute('INSERT INTO competition_seats VALUES(?,?,?,?,?,?)',
                           (cid, side, position, member['user_id'], member['display_name'], datetime.now(timezone.utc).isoformat()))
            teams.append(team)
        db.execute('INSERT INTO competition_schedule(competition_id,starts_at,roster_revision,yellow_team_id,white_team_id,yellow_name,white_name) VALUES(?,?,?,?,?,?,?)',
                   (cid, start.astimezone(timezone.utc).isoformat(), config['revision'], yellow_team_id, white_team_id, teams[0]['name'], teams[1]['name']))
        self.rooms._append_event(db, cid, 'schedule.created', room['created_by_user_id'],
                                 {'starts_at': start.isoformat(), 'roster_revision': config['revision'], 'yellow_team_id': yellow_team_id, 'white_team_id': white_team_id})
        self.rooms._touch(db, cid, status='READY_CHECK')

    def view(self, db, cid):
        row = db.execute('SELECT * FROM competition_schedule WHERE competition_id=?', (cid,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result['late_at'] = (datetime.fromisoformat(row['starts_at']) + timedelta(minutes=10)).isoformat()
        result['players'] = [dict(p) for p in db.execute('SELECT user_id,side,position,display_name,arrived_at FROM competition_scheduled_players WHERE competition_id=? ORDER BY side,position', (cid,))]
        return result

    def check_in(self, code, principal):
        now = datetime.now(timezone.utc)
        with self.rooms.database.transaction(immediate=True) as db:
            room = self.rooms._room_row(db, code)
            self.rooms._ensure_admitted(db, room['id'], principal)
            # Adjudicate BEFORE accepting an arrival after the grace period.
            room, _ = self.settle(db, room, now)
            if room['status'] in ('SEATING', 'READY_CHECK'):
                member = db.execute('SELECT * FROM competition_scheduled_players WHERE competition_id=? AND user_id=?', (room['id'], principal.user_id)).fetchone()
                if member:
                    db.execute('INSERT OR IGNORE INTO competition_seats VALUES(?,?,?,?,?,?)',
                               (room['id'], member['side'], member['position'], principal.user_id, member['display_name'], now.isoformat()))
                cursor = db.execute('UPDATE competition_scheduled_players SET arrived_at=? WHERE competition_id=? AND user_id=? AND arrived_at IS NULL',
                                    (now.isoformat(), room['id'], principal.user_id))
                if cursor.rowcount:
                    self.rooms._append_event(db, room['id'], 'schedule.arrived', principal.user_id, {})
                    full = db.execute('SELECT COUNT(*) FROM competition_seats WHERE competition_id=?', (room['id'],)).fetchone()[0] == 6
                    self.rooms._touch(db, room['id'], status='READY_CHECK' if full else 'SEATING')
            return self.rooms._snapshot(db, self.rooms._room_row(db, code), principal)

    def may_draw(self, db, cid, now):
        schedule = self.view(db, cid)
        return not schedule or (not schedule['exception'] and now >= datetime.fromisoformat(schedule['starts_at'])
                                and all(p['arrived_at'] for p in schedule['players']))

    def settle(self, db, room, now):
        schedule = self.view(db, room['id'])
        if not schedule or room['status'] not in ('SEATING', 'READY_CHECK'):
            return room, False
        cid = room['id']
        changed = False
        if not schedule['attendance_resolved'] and now > datetime.fromisoformat(schedule['late_at']):
            deadline = datetime.fromisoformat(schedule['late_at'])
            complete = {side: all(p['arrived_at'] and datetime.fromisoformat(p['arrived_at']) <= deadline
                                 for p in schedule['players'] if p['side'] == side) for side in ('yellow', 'white')}
            db.execute('UPDATE competition_schedule SET attendance_resolved=1 WHERE competition_id=?', (cid,))
            if not any(complete.values()):
                db.execute("UPDATE competition_schedule SET exception='both_late' WHERE competition_id=?", (cid,))
                self.rooms._append_event(db, cid, 'schedule.both_late', None, {})
            elif not all(complete.values()):
                winner = next(side for side, arrived in complete.items() if arrived)
                self._forfeit(db, room, winner, now)
                return self.rooms._room_row(db, room['room_code']), True
            self.rooms._touch(db, cid)
            changed = True
        if self.may_draw(db, cid, now) and db.execute('SELECT COUNT(*) FROM competition_team_readiness WHERE competition_id=?', (cid,)).fetchone()[0] == 2:
            self.rooms._initialize_draw(db, cid, now=now)
            self.rooms._append_event(db, cid, 'competition.status_changed', None, {'from': room['status'], 'to': 'DRAW'})
            self.rooms._touch(db, cid, status='DRAW')
            changed = True
        return self.rooms._room_row(db, room['room_code']), changed

    def _forfeit(self, db, room, winner, now):
        cid, stamp = room['id'], now.isoformat()
        db.execute('UPDATE competition_schedule SET exception=? WHERE competition_id=?',
                   ('white_late' if winner == 'yellow' else 'yellow_late', cid))
        db.execute('''INSERT INTO competition_match_control(competition_id,current_game_key,phase_token,yellow_wins,white_wins,winner_side,finish_reason,created_at,updated_at)
            VALUES(?,'A',?,?,?,?,'late_forfeit',?,?)''',
            (cid, self.rooms._new_phase_token(), 3 if winner == 'yellow' else 0, 3 if winner == 'white' else 0, winner, stamp, stamp))
        db.execute('INSERT INTO competition_suspensions(competition_id,active,updated_at) VALUES(?,0,?)', (cid, stamp))
        for game in 'ABC':
            db.execute('''INSERT INTO competition_game_results(competition_id,game_key,yellow_score,white_score,winner_side,reason,result_revision,published_at)
                VALUES(?,?,0,0,?,'late_forfeit',1,?)''', (cid, game, winner, stamp))
        self.rooms._append_event(db, cid, 'match.finished', None, {'winner_side': winner, 'reason': 'late_forfeit'})
        self.rooms._touch(db, cid, status='FINISHED')
