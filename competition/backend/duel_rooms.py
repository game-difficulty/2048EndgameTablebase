"""Public authenticated 1v1 rooms; no event or official privileges."""
from datetime import datetime, timedelta, timezone
import uuid

from .errors import CompetitionError
from .room_flow import fixed_rules


class DuelRooms:
    WAIT_MINUTES = 30
    CREATE_COOLDOWN_SECONDS = 60
    HOURLY_LIMIT = 10

    def __init__(self, rooms):
        self.rooms = rooms

    def catalog(self):
        # Registration order declares the current version; older registered
        # versions remain available to already-frozen rooms only.
        return [d.snapshot() for d in self.rooms.project_registry.current_descriptors()]

    def create(self, principal, *, name, projects, command_id, clock_seconds=1800, _challenge=None):
        r = self.rooms
        if principal.user_id <= 0:
            raise CompetitionError('LOGIN_REQUIRED', '请先登录。', 401)
        command_id = r._normalize_command_id(command_id)
        name = ' '.join(str(name).split())
        if not 2 <= len(name) <= 100:
            raise CompetitionError('INVALID_NAME', '房间名称须为 2 至 100 字。')
        if not isinstance(projects, list) or any(not isinstance(p, str) for p in projects) or len(set(projects)) != len(projects):
            raise CompetitionError('INVALID_PROJECT_POOL', '请选择不重复的已注册项目。')
        rules = fixed_rules(len(projects), clock_seconds)
        catalog = {p['project_ref']: p for p in self.catalog()}
        if _challenge is None and any(p not in catalog for p in projects):
            raise CompetitionError('PROJECT_ADAPTER_UNAVAILABLE', '项目不在服务端可用目录中。', 409)
        if _challenge is None:
            pool = r._normalize_projects([dict(key=p, name=catalog[p]['display_name'], project_ref=p,
                     adapter_rules_version=catalog[p]['rules_version'], rules_version=catalog[p]['rules_version']) for p in projects])
        else:
            from .projects.target_challenge import pool as challenge_pool
            pool = challenge_pool(_challenge)
            rules.update(workflow='time_attack', version='time-attack-v1')
        now = datetime.now(timezone.utc)
        with r.database.transaction(immediate=True) as db:
            previous = db.execute('SELECT c.* FROM competitions c JOIN competition_duel_rooms d ON d.competition_id=c.id WHERE d.owner_user_id=? AND d.command_id=?',
                                  (principal.user_id, command_id)).fetchone()
            if previous:
                if r.room_flow.kind(db, previous['id']) != ('time_attack' if _challenge else 'duel'):
                    raise CompetitionError('COMMAND_ID_REUSED', 'This creation command belongs to another room type.', 409)
                return r._snapshot(db, previous, principal)
            # Expiration is based on creation/phase entry, not GETs or heartbeats.
            for row in db.execute("SELECT c.* FROM competitions c JOIN competition_duel_rooms d ON d.competition_id=c.id WHERE c.status NOT IN ('FINISHED','CANCELLED') AND d.expires_at<=?", (now.isoformat(),)).fetchall():
                self.expire(db, row, now)
            active = db.execute("""SELECT 1 FROM competitions c JOIN competition_duel_rooms d ON d.competition_id=c.id
                WHERE c.status NOT IN ('FINISHED','CANCELLED') AND
                (d.owner_user_id=? OR EXISTS(SELECT 1 FROM competition_seats s WHERE s.competition_id=c.id AND s.user_id=?))""",
                (principal.user_id, principal.user_id)).fetchone()
            if active:
                raise CompetitionError('DUEL_ACTIVE_ROOM', '请先完成或关闭当前自由对决房间。', 409)
            recent = db.execute('SELECT created_at FROM competition_duel_rooms WHERE owner_user_id=? AND created_at>? ORDER BY created_at DESC',
                                (principal.user_id, (now-timedelta(hours=1)).isoformat())).fetchall()
            if len(recent) >= self.HOURLY_LIMIT or (recent and now-datetime.fromisoformat(recent[0]['created_at']) < timedelta(seconds=self.CREATE_COOLDOWN_SECONDS)):
                raise CompetitionError('DUEL_CREATE_LIMIT', '创建过于频繁，请稍后再试。', 429)
            cid = uuid.uuid4().hex
            for attempt in range(12):
                code = r._new_room_code()
                if not db.execute('SELECT 1 FROM competitions WHERE room_code=?', (code,)).fetchone():
                    break
            else:
                raise CompetitionError('ROOM_CODE_EXHAUSTED', '暂时无法分配房间码。', 503)
            room = r._create_room_records(db, principal, cid=cid, code=code, name=name, projects=pool,
                                          rules=rules, configurable=True, grant_organizer=False, now=now.isoformat())
            db.execute('INSERT INTO competition_duel_rooms VALUES(?,?,?,?,?)',
                       (cid, principal.user_id, command_id, now.isoformat(), (now+timedelta(minutes=self.WAIT_MINUTES)).isoformat()))
            r.room_flow.freeze(db, cid, projects)
            if _challenge is not None:
                import json
                db.execute('INSERT INTO competition_time_attack(competition_id,configuration_json) VALUES(?,?)',
                           (cid, json.dumps(_challenge)))
            db.execute('INSERT INTO competition_seats(competition_id,side,position,user_id,display_name_snapshot,seated_at) VALUES(?,\'yellow\',1,?,?,?)',
                       (cid, principal.user_id, principal.display_name, now.isoformat()))
            return r._snapshot(db, room, principal)

    def expire(self, db, room, now):
        if room['status'] in ('FINISHED', 'CANCELLED'):
            return room, False
        meta = db.execute('SELECT expires_at FROM competition_duel_rooms WHERE competition_id=?', (room['id'],)).fetchone()
        waiting = room['status'] in ('SEATING', 'READY_CHECK') or room['status'].endswith('_READY')
        if waiting and meta and now >= datetime.fromisoformat(meta['expires_at']):
            self.rooms._append_event(db, room['id'], 'duel.expired', None, {})
            self.rooms._touch(db, room['id'], status='CANCELLED')
            return self.rooms._room_row(db, room['room_code']), True
        return room, False

    def reset_wait(self, db, cid, now):
        db.execute('UPDATE competition_duel_rooms SET expires_at=? WHERE competition_id=?',
                   ((now+timedelta(minutes=self.WAIT_MINUTES)).isoformat(), cid))

    def admit(self, db, room, principal):
        other = db.execute("""SELECT 1 FROM competitions c JOIN competition_duel_rooms d ON d.competition_id=c.id
            WHERE c.id!=? AND c.status NOT IN ('FINISHED','CANCELLED') AND
            (d.owner_user_id=? OR EXISTS(SELECT 1 FROM competition_seats s WHERE s.competition_id=c.id AND s.user_id=?))""",
            (room['id'], principal.user_id, principal.user_id)).fetchone()
        if other:
            raise CompetitionError('DUEL_ACTIVE_ROOM', '请先完成或关闭当前自由对决房间。', 409)
