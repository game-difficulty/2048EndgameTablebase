"""Reusable round-robin fixtures above official rooms, not a second room workflow.

Groups, roster and format are frozen by the organizer. Captains can only book
their own fixtures; room ownership always stays with an event organizer.
"""
from datetime import datetime, timedelta, timezone
import json
import uuid

from .errors import CompetitionError
from .event_formats import FORMATS
from .projects import tournament_project_catalog
from .room_rules import normalize_rules


def fail(code, message, status=409):
    raise CompetitionError(code, message, status)


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def future_time(value):
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None or result <= datetime.now(timezone.utc):
            raise ValueError()
        return result.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError, AttributeError):
        fail('INVALID_SCHEDULE_TIME', '开战时间须为未来时间，并包含时区。', 400)


def round_robin(team_ids):
    """Circle method; odd groups have a bye, never a fictitious opponent."""
    ring = list(team_ids)
    if len(ring) % 2:
        ring.append(None)
    for number in range(1, len(ring)):
        for index in range(len(ring) // 2):
            a, b = ring[index], ring[-index-1]
            if a is not None and b is not None:
                # Alternate display orientation; BP first side is still drawn in-room.
                yield number, (a if number % 2 else b), (b if number % 2 else a)
        ring = [ring[0], ring[-1], *ring[1:-1]]


class EventFixtures:
    def __init__(self, rooms):
        self.rooms = rooms

    def project_options(self):
        return tournament_project_catalog()

    def _teams(self, db, slug):
        teams = []
        for row in db.execute('SELECT * FROM tournament_teams WHERE event_slug=? ORDER BY name,id', (slug,)):
            members = [dict(p) for p in db.execute('''SELECT e.user_id,e.display_name,p.position
                FROM tournament_entrants e LEFT JOIN tournament_roster_positions p
                ON p.event_slug=e.event_slug AND p.user_id=e.user_id
                WHERE e.event_slug=? AND e.team_id=? ORDER BY p.position''', (slug, row['id']))]
            teams.append(dict(id=row['id'], name=row['name'], captain_user_id=row['captain_user_id'], members=members))
        return teams

    def _require_event(self, db, slug, principal, *, manager=False):
        event = self.rooms.events._event(db, slug)
        if not FORMATS[event['format_key']]['capabilities']['rooms']:
            fail('EVENT_FORMAT_MISMATCH', '统计型赛事不使用对战房间。')
        if manager and not self.rooms.events._manager(event, principal):
            fail('EVENT_MANAGER_REQUIRED', '无权管理该赛事。', 403)
        return event

    def _open(self, event):
        if event['status'] == 'finished':
            fail('EVENT_FINISHED', '赛事已结束，不能预约或修改赛程。')

    def create_stage(self, slug, principal, *, name, groups, command_id, projects=None, rules=None):
        r = self.rooms
        command_id = r._normalize_command_id(command_id)
        name = ' '.join(str(name).split())
        if not 2 <= len(name) <= 60:
            fail('INVALID_FIXTURE_STAGE', '阶段名称须为 2 至 60 字。', 400)
        request = encoded(dict(name=name, groups=groups, projects=projects, rules=rules))
        with r.database.transaction(immediate=True) as db:
            event = self._require_event(db, slug, principal, manager=True)
            previous = db.execute('SELECT request_json FROM tournament_fixture_stages WHERE event_slug=? AND command_id=?', (slug, command_id)).fetchone()
            if previous:
                if previous[0] != request:
                    fail('COMMAND_ID_REUSED', '这次操作标识已用于不同内容。')
                return self._view(db, slug, principal)
            self._open(event)
            config = r.events.enrollment.config(db, slug)
            if not config['roster_locked'] or config['mode'] == 'solo':
                fail('ROSTER_NOT_LOCKED', '请先锁定最终团队名单，再生成小组对阵。')
            if not isinstance(groups, list) or not 1 <= len(groups) <= 16:
                fail('INVALID_FIXTURE_STAGE', '请设置 1 至 16 个小组。', 400)
            available = {t['id']: t for t in self._teams(db, slug)}
            frozen, seen, names = [], set(), set()
            for group in groups:
                label = str(group.get('name', '')).strip()
                ids = group.get('team_ids', [])
                if not label or len(label) > 40 or label.casefold() in names or not isinstance(ids, list) or not 2 <= len(ids) <= 16:
                    fail('INVALID_FIXTURE_STAGE', '小组名称不能重复，每组须有 2 至 16 支队伍。', 400)
                names.add(label.casefold())
                selected = []
                for tid in ids:
                    if tid not in available or tid in seen:
                        fail('INVALID_FIXTURE_STAGE', '队伍必须属于本赛事，且只能分入一个小组。', 400)
                    team = available[tid]
                    if len(team['members']) != config['team_size'] or [m['position'] for m in team['members']] != list(range(1, config['team_size']+1)) or team['members'][0]['user_id'] != team['captain_user_id']:
                        fail('INVALID_SCHEDULE_TEAM', '请设置完整队伍及连续队内序号，1 号为队长。')
                    seen.add(tid)
                    selected.append(team)
                frozen.append(dict(name=label, teams=selected))
            if len(seen) > 64:
                fail('INVALID_FIXTURE_STAGE', '单个阶段最多支持 64 支队伍。', 400)
            options = {p['project_ref']: p for p in self.project_options()}
            refs = list(options) if projects is None else projects
            if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != len(refs) or any(ref not in options for ref in refs):
                fail('INVALID_PROJECT_POOL', '请选择不重复的服务端项目池项目。', 400)
            pool = r._normalize_projects([options[ref] for ref in refs])
            raw_rules = dict(rules or {})
            raw_rules.setdefault('team_size', config['team_size'])
            normalized = normalize_rules(raw_rules, pool_size=len(pool), team_clock_ms=r.team_clock_ms,
                                         draft_seconds=r.draft_turn_seconds, lineup_seconds=r.lineup_seconds)
            if normalized['team_size'] != config['team_size']:
                fail('INVALID_ROOM_RULES', '阶段每队人数须与锁定名单一致。', 400)
            if db.execute('SELECT 1 FROM tournament_fixture_stages WHERE event_slug=? AND name=?', (slug, name)).fetchone():
                fail('FIXTURE_STAGE_EXISTS', '该阶段名称已存在。')
            sid = uuid.uuid4().hex
            db.execute('''INSERT INTO tournament_fixture_stages
                VALUES(?,?,?,?,?,?,?,?,?,?)''', (sid, slug, name, encoded(frozen), encoded(pool), encoded(normalized),
                    event['owner_user_id'] or principal.user_id, datetime.now(timezone.utc).isoformat(), command_id, request))
            for group in frozen:
                for number, yellow, white in round_robin([t['id'] for t in group['teams']]):
                    db.execute('''INSERT INTO tournament_fixtures
                        (id,stage_id,group_name,round_number,yellow_team_id,white_team_id) VALUES(?,?,?,?,?,?)''',
                        (uuid.uuid4().hex, sid, group['name'], number, yellow, white))
            return self._view(db, slug, principal)

    def _participants(self, stage, fixture):
        teams = {t['id']: t for g in json.loads(stage['groups_json']) for t in g['teams']}
        return teams[fixture['yellow_team_id']], teams[fixture['white_team_id']]

    def _validate_roster(self, db, slug, teams):
        config = self.rooms.events.enrollment.config(db, slug)
        if not config['roster_locked']:
            fail('ROSTER_NOT_LOCKED', '请先锁定最终团队名单。')
        current = {t['id']: t for t in self._teams(db, slug)}
        for team in teams:
            actual = current.get(team['id'])
            identity = lambda t: (t['captain_user_id'], [(m['user_id'], m['position']) for m in t['members']])
            if not actual or identity(actual) != identity(team):
                fail('FIXTURE_ROSTER_CHANGED', '参赛人员与生成对阵时不一致，请联系举办方处理。')

    def _conflicts(self, db, teams, start, *, exclude=None):
        ids = {t['id'] for t in teams}
        moment = datetime.fromisoformat(start)
        warnings = []
        for row in db.execute('''SELECT s.*,c.status FROM competition_schedule s JOIN competitions c ON c.id=s.competition_id
            WHERE c.status NOT IN ('FINISHED','CANCELLED')'''):
            if row['competition_id'] == exclude or not ids.intersection((row['yellow_team_id'], row['white_team_id'])):
                continue
            distance = abs(datetime.fromisoformat(row['starts_at']) - moment)
            if distance == timedelta(0):
                fail('FIXTURE_TIME_CONFLICT', '同一队伍不能在同一时间安排两场比赛。')
            if distance < timedelta(hours=1):
                warnings.append('相邻场次间隔不足一小时，请确认前一场能够结束；系统不保证比赛时长。')
        return list(dict.fromkeys(warnings))

    def _editable(self, db, fixture):
        if not fixture['competition_id']:
            return
        room = db.execute('SELECT * FROM competitions WHERE id=?', (fixture['competition_id'],)).fetchone()
        schedule = self.rooms.schedule.view(db, room['id'])
        if room['status'] not in ('SEATING', 'READY_CHECK') or datetime.now(timezone.utc) >= datetime.fromisoformat(schedule['starts_at']):
            fail('FIXTURE_SCHEDULE_CLOSED', '开赛时间已到或比赛流程已开始，不能改期。')

    def _reschedule(self, db, fixture, start, principal):
        cid = fixture['competition_id']
        db.execute('UPDATE competition_schedule SET starts_at=?,attendance_resolved=0,exception=NULL WHERE competition_id=?', (start, cid))
        # Arrivals and seats stay; readiness must be explicitly renewed for the new time.
        db.execute('DELETE FROM competition_team_readiness WHERE competition_id=?', (cid,))
        self.rooms._append_event(db, cid, 'schedule.rescheduled', principal.user_id, {'starts_at': start, 'fixture_id': fixture['id']})
        self.rooms._touch(db, cid)

    def action(self, slug, fixture_id, principal, *, action, revision, command_id, starts_at=None, room_code=None):
        r = self.rooms
        command_id = r._normalize_command_id(command_id)
        request = encoded(dict(action=action, revision=revision, starts_at=starts_at, room_code=room_code))
        warnings, changed_code = [], None
        with r.database.transaction(immediate=True) as db:
            event = self._require_event(db, slug, principal)
            fixture = db.execute('''SELECT f.* FROM tournament_fixtures f JOIN tournament_fixture_stages s
                ON s.id=f.stage_id WHERE f.id=? AND s.event_slug=?''', (fixture_id, slug)).fetchone()
            if not fixture:
                fail('FIXTURE_NOT_FOUND', '找不到该赛事对阵。', 404)
            stage = db.execute('SELECT * FROM tournament_fixture_stages WHERE id=?', (fixture['stage_id'],)).fetchone()
            teams = self._participants(stage, fixture)
            manager = r.events._manager(event, principal)
            captains = [t['captain_user_id'] for t in teams]
            if not manager and principal.user_id not in captains:
                fail('FIXTURE_CAPTAIN_REQUIRED', '仅该场双方队长和举办方可操作时刻表。', 403)
            if fixture['competition_id'] and not manager:
                r._ensure_admitted(db, fixture['competition_id'], principal)
            previous = db.execute('SELECT * FROM tournament_fixture_commands WHERE fixture_id=? AND command_id=?', (fixture_id, command_id)).fetchone()
            if previous:
                if previous['actor_user_id'] != principal.user_id or previous['request_json'] != request:
                    fail('COMMAND_ID_REUSED', '这次操作标识已用于不同内容。')
                return {'schedule': self._view(db, slug, principal), 'warnings': []}
            self._open(event)
            if fixture['revision'] != revision:
                fail('FIXTURE_CHANGED', '对阵已更新，请刷新后重试。')
            if action == 'bind':
                if not manager:
                    fail('EVENT_MANAGER_REQUIRED', '只有举办方可绑定已有房间。', 403)
                if fixture['competition_id']:
                    old = db.execute('SELECT status FROM competitions WHERE id=?', (fixture['competition_id'],)).fetchone()
                    if old['status'] != 'CANCELLED':
                        fail('FIXTURE_ROOM_EXISTS', '该对阵已有有效房间。')
                room = r._room_row(db, room_code)
                r._reject_duel_official(db, room)
                schedule = r.schedule.view(db, room['id'])
                if not schedule or {schedule['yellow_team_id'], schedule['white_team_id']} != {t['id'] for t in teams}:
                    fail('FIXTURE_ROOM_MISMATCH', '请选择已绑定该场两队固定名单的正式房间。')
                for side in ('yellow', 'white'):
                    team = next(t for t in teams if t['id'] == schedule[f'{side}_team_id'])
                    players = sorted((p for p in schedule['players'] if p['side'] == side), key=lambda p: p['position'])
                    if [(p['user_id'], p['position']) for p in players] != [(p['user_id'], p['position']) for p in team['members']]:
                        fail('FIXTURE_ROOM_MISMATCH', '房间固定名单与该场对阵的参赛人员不一致。')
                if db.execute('SELECT 1 FROM tournament_fixtures WHERE competition_id=?', (room['id'],)).fetchone():
                    fail('FIXTURE_ROOM_EXISTS', '该房间已绑定其他对阵。')
                r.events.link_in_transaction(db, slug, room, principal)
                db.execute('UPDATE tournament_fixtures SET competition_id=? WHERE id=?', (room['id'], fixture_id))
                changed_code = room['room_code']
                r._append_event(db, room['id'], 'schedule.fixture_bound', principal.user_id, {'fixture_id': fixture_id})
            else:
                self._editable(db, fixture)
                if action == 'schedule':
                    self._validate_roster(db, slug, teams)
                    start = future_time(starts_at)
                    warnings = self._conflicts(db, teams, start, exclude=fixture['competition_id'])
                    if not fixture['competition_id']:
                        cid, stamp = uuid.uuid4().hex, datetime.now(timezone.utc).isoformat()
                        for _ in range(12):
                            code = r._new_room_code()
                            if not db.execute('SELECT 1 FROM competitions WHERE room_code=?', (code,)).fetchone():
                                break
                        else:
                            fail('ROOM_CODE_EXHAUSTED', '暂时无法分配房间码，请稍后重试。', 503)
                        room = r._create_room_records(db, principal, cid=cid, code=code,
                            name=f"{stage['name']} · {teams[0]['name']} vs {teams[1]['name']}"[:100],
                            projects=json.loads(stage['projects_json']), rules=json.loads(stage['rules_json']),
                            configurable=True, grant_organizer=True, now=stamp,
                            owner_user_id=event['owner_user_id'] or stage['organizer_user_id'])
                        # This link is authorized by the fixture's captain policy, not general room creation.
                        db.execute('INSERT INTO tournament_room_links VALUES(?,?,?,?)', (cid, slug, principal.user_id, stamp))
                        r.schedule.bind(db, room, slug, teams[0]['id'], teams[1]['id'], start)
                        r._append_event(db, cid, 'schedule.booked', principal.user_id, {'fixture_id': fixture_id, 'starts_at': start})
                        db.execute('UPDATE tournament_fixtures SET competition_id=? WHERE id=?', (cid, fixture_id))
                        changed_code = code
                    elif manager:
                        self._reschedule(db, fixture, start, principal)
                        db.execute('UPDATE tournament_fixtures SET proposed_at=NULL,proposed_by_user_id=NULL WHERE id=?', (fixture_id,))
                    else:
                        if fixture['proposed_at'] and fixture['proposed_by_user_id'] != principal.user_id:
                            fail('FIXTURE_PROPOSAL_EXISTS', '对方已提出改期，请先确认或拒绝。')
                        db.execute('UPDATE tournament_fixtures SET proposed_at=?,proposed_by_user_id=? WHERE id=?', (start, principal.user_id, fixture_id))
                elif action in ('confirm', 'reject', 'cancel_proposal'):
                    if not fixture['proposed_at']:
                        fail('FIXTURE_NO_PROPOSAL', '当前没有待确认的改期。')
                    proposer = fixture['proposed_by_user_id']
                    if not manager and ((action == 'cancel_proposal' and principal.user_id != proposer) or (action != 'cancel_proposal' and principal.user_id == proposer)):
                        fail('FIXTURE_OTHER_CAPTAIN_REQUIRED', '改期须由另一方队长确认或拒绝，提议方只能撤回。', 403)
                    if action == 'confirm':
                        self._validate_roster(db, slug, teams)
                        start = future_time(fixture['proposed_at'])
                        warnings = self._conflicts(db, teams, start, exclude=fixture['competition_id'])
                        self._reschedule(db, fixture, start, principal)
                    db.execute('UPDATE tournament_fixtures SET proposed_at=NULL,proposed_by_user_id=NULL WHERE id=?', (fixture_id,))
                else:
                    fail('INVALID_FIXTURE_ACTION', '无效的时刻表操作。', 400)
                if fixture['competition_id']:
                    changed_code = db.execute('SELECT room_code FROM competitions WHERE id=?', (fixture['competition_id'],)).fetchone()[0]
            db.execute('UPDATE tournament_fixtures SET revision=revision+1 WHERE id=?', (fixture_id,))
            db.execute('INSERT INTO tournament_fixture_commands VALUES(?,?,?,?,?)', (fixture_id, command_id, principal.user_id, request, datetime.now(timezone.utc).isoformat()))
            return {'schedule': self._view(db, slug, principal), 'warnings': warnings, '_changed_room_code': changed_code}

    def snapshot(self, slug, principal=None):
        with self.rooms.database.transaction() as db:
            self._require_event(db, slug, principal)
            return self._view(db, slug, principal)

    def _view(self, db, slug, principal):
        r = self.rooms
        event = r.events._event(db, slug)
        manager = r.events._manager(event, principal)
        config = r.events.enrollment.config(db, slug)
        result = dict(stages=[], teams=self._teams(db, slug), can_manage=manager,
                      roster_locked=bool(config['roster_locked']), team_size=config['team_size'],
                      projects=self.project_options(), event_finished=event['status'] == 'finished')
        now = datetime.now(timezone.utc)
        for stage in db.execute('SELECT * FROM tournament_fixture_stages WHERE event_slug=? ORDER BY created_at,id', (slug,)):
            entry = dict(id=stage['id'], name=stage['name'], groups=json.loads(stage['groups_json']),
                         rules=json.loads(stage['rules_json']), projects=json.loads(stage['projects_json']), fixtures=[])
            for f in db.execute('SELECT * FROM tournament_fixtures WHERE stage_id=? ORDER BY group_name,round_number,id', (stage['id'],)):
                teams = self._participants(stage, f)
                uid = principal.user_id if principal else None
                own = uid in [t['captain_user_id'] for t in teams]
                member = any(uid == m['user_id'] for t in teams for m in t['members'])
                room = db.execute('SELECT * FROM competitions WHERE id=?', (f['competition_id'],)).fetchone() if f['competition_id'] else None
                schedule = r.schedule.view(db, room['id']) if room else None
                admitted = bool(principal and (not room or not db.execute('SELECT 1 FROM competition_expulsions WHERE competition_id=? AND user_id=?', (room['id'], uid)).fetchone()))
                editable = not result['event_finished'] and (not room or (room['status'] in ('SEATING','READY_CHECK') and now < datetime.fromisoformat(schedule['starts_at'])))
                score = dict(db.execute('SELECT yellow_wins AS yellow,white_wins AS white FROM competition_match_control WHERE competition_id=?', (room['id'],)).fetchone() or {}) if room else {}
                if score and schedule['yellow_team_id'] != f['yellow_team_id']:
                    score = dict(yellow=score['white'], white=score['yellow'])
                entry['fixtures'].append(dict(id=f['id'], group_name=f['group_name'], round_number=f['round_number'],
                    yellow_team_id=f['yellow_team_id'], white_team_id=f['white_team_id'],
                    yellow_name=teams[0]['name'], white_name=teams[1]['name'], revision=f['revision'],
                    game_count=r._rules(db, room['id'])['game_count'] if room else entry['rules']['game_count'],
                    is_mine=member, starts_at=schedule['starts_at'] if schedule else None,
                    status=room['status'] if room else 'UNSCHEDULED', exception=schedule['exception'] if schedule else None,
                    series_score=score, proposed_at=f['proposed_at'] if editable else None,
                    proposed_by_me=uid is not None and uid == f['proposed_by_user_id'],
                    can_schedule=bool(admitted and (own or manager) and editable and config['roster_locked']),
                    can_confirm=bool(admitted and editable and f['proposed_at'] and (manager or (own and uid != f['proposed_by_user_id']))),
                    can_withdraw=bool(admitted and editable and f['proposed_at'] and (manager or uid == f['proposed_by_user_id'])),
                    can_bind=bool(manager and not result['event_finished'] and (not room or room['status']=='CANCELLED')),
                    room_code=room['room_code'] if room and admitted and (manager or member) else None,
                    public_key=room['public_key'] if room and room['live_started_at'] else None))
            result['stages'].append(entry)
        return result
