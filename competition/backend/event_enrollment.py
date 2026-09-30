"""Event-scoped entries and teams. None of these roles grant room-management rights."""
import json
import uuid
from datetime import datetime, timezone

from .errors import CompetitionError
from .event_statistics import accounts


def fail(message, code='ENROLLMENT_INVALID', status=409):
    raise CompetitionError(code, message, status)


class EventEnrollment:
    def __init__(self, catalog, account_reader=accounts):
        self.catalog, self.database, self.account_reader = catalog, catalog.database, account_reader

    def config(self, db, slug):
        self.catalog._event(db, slug)
        return dict(db.execute('SELECT * FROM tournament_enrollment_config WHERE event_slug=?', (slug,)).fetchone())

    def _manager(self, db, slug, principal):
        if not self.catalog._manager(self.catalog._event(db, slug), principal):
            fail('只有本赛事举办方或赛事管理员可以操作。', 'EVENT_MANAGER_REQUIRED', 403)

    def _editable(self, config, revision):
        if config['revision'] != revision:
            fail('名单或设置已更新，请刷新后重试。', 'ROSTER_CHANGED')
        if config['roster_locked']:
            fail('最终名单已锁定，需要管理员填写原因解锁。', 'ROSTER_LOCKED')

    def _registration(self, config):
        if not config['registration_open'] or config['registration_locked']:
            fail('当前不接受报名、退出或自主组队。', 'REGISTRATION_CLOSED')

    def _audit(self, db, slug, principal, action, payload):
        db.execute('UPDATE tournament_enrollment_config SET revision=revision+1 WHERE event_slug=?', (slug,))
        revision = self.config(db, slug)['revision']
        db.execute('INSERT INTO tournament_enrollment_audit VALUES(?,?,?,?,?,?)',
                   (slug, revision, principal.user_id, action, json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()))

    def _team(self, db, slug, team_id):
        row = db.execute('SELECT * FROM tournament_teams WHERE event_slug=? AND id=?', (slug, team_id)).fetchone()
        if not row:
            fail('找不到本赛事中的队伍。')
        return row

    def _entrant(self, db, slug, uid):
        return db.execute('SELECT * FROM tournament_entrants WHERE event_slug=? AND user_id=?', (slug, uid)).fetchone()

    def _add(self, db, slug, principal, config):
        if self._entrant(db, slug, principal.user_id):
            return
        count = db.execute('SELECT COUNT(*) FROM tournament_entrants WHERE event_slug=?', (slug,)).fetchone()[0]
        if config['capacity'] and count >= config['capacity']:
            fail('报名名额已满。')
        db.execute('INSERT INTO tournament_entrants VALUES(?,?,?,NULL,0,\'self\')', (slug, principal.user_id, principal.display_name))

    def snapshot(self, slug, principal=None):
        with self.database.transaction() as db:
            return self._snapshot(db, slug, principal)

    def _snapshot(self, db, slug, principal):
        config = self.config(db, slug)
        entries = [dict(row) for row in db.execute('SELECT * FROM tournament_entrants WHERE event_slug=? ORDER BY user_id', (slug,))]
        positions = dict(db.execute('SELECT user_id,position FROM tournament_roster_positions WHERE event_slug=?', (slug,)))
        for entry in entries:
            entry['position'] = positions.get(entry['user_id'])
        teams = [dict(row) for row in db.execute('SELECT * FROM tournament_teams WHERE event_slug=? ORDER BY name', (slug,))]
        uid = principal.user_id if principal else None
        invitations = []
        if uid:
            invitations = [dict(row) for row in db.execute('''SELECT i.*,t.name AS team_name,t.captain_user_id
                FROM tournament_team_invites i JOIN tournament_teams t ON t.id=i.team_id
                WHERE t.event_slug=? AND (i.user_id=? OR t.captain_user_id=?)''', (slug, uid, uid))]
        manager = self.catalog._manager(self.catalog._event(db, slug), principal)
        return {**config, 'entries': entries, 'teams': teams, 'invitations': invitations,
                'fixed_policy': self.catalog._event(db, slug)['format_key'] == 'team-top5-3x3-v1',
                'me': {'user_id': uid, 'can_manage': manager},
                'audit': [dict(row) for row in db.execute('''SELECT revision,actor_user_id,action,payload_json,created_at
                    FROM tournament_enrollment_audit WHERE event_slug=? ORDER BY revision DESC LIMIT 20''', (slug,))] if manager else []}

    def action(self, slug, principal, *, action, revision, **payload):
        with self.database.transaction(immediate=True) as db:
            config = self.config(db, slug)
            if action == 'unlock':
                self._manager(db, slug, principal)
                if config['revision'] != revision:
                    fail('名单已更新，请刷新后重试。', 'ROSTER_CHANGED')
                reason = str(payload.get('reason') or '').strip()
                if len(reason) < 4:
                    fail('请填写至少四个字符的解锁原因。')
                db.execute('UPDATE tournament_enrollment_config SET roster_locked=0,registration_locked=0,registration_open=0 WHERE event_slug=?', (slug,))
            else:
                self._editable(config, revision)
                if action in ('settings', 'lock_registration', 'lock_roster'):
                    self._manager(db, slug, principal)
                    if action == 'settings':
                        mode = payload.get('mode') or config['mode']
                        capacity = payload.get('capacity', config['capacity'])
                        if mode not in ('solo','self_team','organizer_team') or type(capacity) is not int or not 0 <= capacity <= 10000:
                            fail('报名方式或人数上限无效。')
                        if self.catalog._event(db, slug)['format_key'] == 'team-top5-3x3-v1' and (mode != 'organizer_team' or capacity != 20):
                            fail('14360杯采用举办方分队，固定二十人。')
                        count = db.execute('SELECT COUNT(*) FROM tournament_entrants WHERE event_slug=?', (slug,)).fetchone()[0]
                        if count and mode != config['mode']:
                            fail('已有报名，不能直接切换报名方式。')
                        if capacity and count > capacity:
                            fail('人数上限不能少于已有报名人数。')
                        if config['registration_locked']:
                            fail('报名已锁定，请先填写原因解锁。')
                        db.execute('UPDATE tournament_enrollment_config SET mode=?,capacity=?,registration_open=? WHERE event_slug=?',
                                   (mode, capacity, bool(payload.get('registration_open')), slug))
                    elif action == 'lock_registration':
                        db.execute('UPDATE tournament_enrollment_config SET registration_open=0,registration_locked=1 WHERE event_slug=?', (slug,))
                    else:
                        self._validate_final(db, slug, config)
                        frozen = self._snapshot(db, slug, principal)
                        payload = {'entries': frozen['entries'], 'teams': frozen['teams']}
                        db.execute('UPDATE tournament_enrollment_config SET registration_open=0,registration_locked=1,roster_locked=1 WHERE event_slug=?', (slug,))
                        db.execute('DELETE FROM tournament_team_invites WHERE team_id IN (SELECT id FROM tournament_teams WHERE event_slug=?)', (slug,))
                else:
                    if action not in ('submit_team','unsubmit_team'):
                        self._registration(config)
                    self._player_action(db, slug, principal, config, action, payload)
            self._audit(db, slug, principal, action, payload)
            return self._snapshot(db, slug, principal)

    def _player_action(self, db, slug, principal, config, action, payload):
        uid = principal.user_id
        member = self._entrant(db, slug, uid)
        if action == 'signup':
            self._add(db, slug, principal, config)
            return
        if action == 'withdraw':
            if member and member['team_id'] and config['mode'] == 'self_team':
                fail('请先离开或解散队伍，再退出报名。')
            if member and member['team_id']:
                db.execute('UPDATE tournament_teams SET submitted=0,captain_user_id=CASE WHEN captain_user_id=? THEN NULL ELSE captain_user_id END WHERE id=?', (uid, member['team_id']))
            db.execute('DELETE FROM tournament_entrants WHERE event_slug=? AND user_id=?', (slug, uid))
            db.execute('DELETE FROM tournament_team_invites WHERE user_id=? AND team_id IN (SELECT id FROM tournament_teams WHERE event_slug=?)', (uid, slug))
            return
        if config['mode'] != 'self_team':
            fail('本赛事不开放自由组队。')
        if action == 'create_team':
            if member and member['team_id']:
                fail('你已加入队伍。')
            name = ' '.join(str(payload.get('name') or '').split())
            if not 1 <= len(name) <= 40:
                fail('队名须为 1–40 个字符。')
            if db.execute('SELECT 1 FROM tournament_teams WHERE event_slug=? AND name=?', (slug, name)).fetchone():
                fail('本赛事已有同名队伍。')
            self._add(db, slug, principal, config)
            team_id = uuid.uuid4().hex
            db.execute('INSERT INTO tournament_teams VALUES(?,?,?,?,0)', (team_id, slug, name, uid))
            db.execute('UPDATE tournament_entrants SET team_id=? WHERE event_slug=? AND user_id=?', (team_id, slug, uid))
            db.execute('DELETE FROM tournament_team_invites WHERE user_id=? AND team_id IN (SELECT id FROM tournament_teams WHERE event_slug=?)', (uid, slug))
            return
        team = self._team(db, slug, payload.get('team_id'))
        is_captain = team['captain_user_id'] == uid
        if action in ('invite','cancel_invite','submit_team','unsubmit_team','disband_team') and not is_captain:
            fail('只有该队队长可以操作。', status=403)
        if action == 'invite':
            if team['submitted']:
                fail('请先撤回队伍报名再邀请。')
            target = payload.get('user_id')
            names = self.account_reader([target]) if type(target) is int and target > 0 else {}
            if target not in names:
                fail('找不到有效的 Table 用户 ID。')
            existing = self._entrant(db, slug, target)
            if existing and existing['team_id']:
                fail('该选手已加入队伍。')
            if db.execute('SELECT COUNT(*) FROM tournament_entrants WHERE team_id=?', (team['id'],)).fetchone()[0] >= config['team_size']:
                fail('队伍已经满员。')
            db.execute('INSERT OR IGNORE INTO tournament_team_invites VALUES(?,?,?)', (team['id'], target, names[target]))
        elif action in ('accept_invite','decline_invite'):
            if not db.execute('SELECT 1 FROM tournament_team_invites WHERE team_id=? AND user_id=?', (team['id'], uid)).fetchone():
                fail('邀请不存在或已经失效。')
            if action == 'accept_invite':
                if member and member['team_id']:
                    fail('你已加入其他队伍。')
                if team['submitted'] or db.execute('SELECT COUNT(*) FROM tournament_entrants WHERE team_id=?', (team['id'],)).fetchone()[0] >= config['team_size']:
                    fail('队伍已提交或已满员。')
                self._add(db, slug, principal, config)
                db.execute('UPDATE tournament_entrants SET team_id=? WHERE event_slug=? AND user_id=?', (team['id'], slug, uid))
                db.execute('DELETE FROM tournament_team_invites WHERE user_id=? AND team_id IN (SELECT id FROM tournament_teams WHERE event_slug=?)', (uid, slug))
            else:
                db.execute('DELETE FROM tournament_team_invites WHERE team_id=? AND user_id=?', (team['id'], uid))
        elif action == 'cancel_invite':
            db.execute('DELETE FROM tournament_team_invites WHERE team_id=? AND user_id=?', (team['id'], payload.get('user_id')))
        elif action in ('submit_team','unsubmit_team'):
            if action == 'submit_team' and db.execute('SELECT COUNT(*) FROM tournament_entrants WHERE team_id=?', (team['id'],)).fetchone()[0] != config['team_size']:
                fail('队伍人数不足，不能提交报名。')
            db.execute('UPDATE tournament_teams SET submitted=? WHERE id=?', (action == 'submit_team', team['id']))
        elif action == 'leave_team':
            if not member or member['team_id'] != team['id'] or is_captain:
                fail('队长请使用解散队伍；其他用户只能离开自己的队伍。')
            db.execute('UPDATE tournament_entrants SET team_id=NULL WHERE event_slug=? AND user_id=?', (slug, uid))
            db.execute('UPDATE tournament_teams SET submitted=0 WHERE id=?', (team['id'],))
        elif action == 'disband_team':
            db.execute('UPDATE tournament_entrants SET team_id=NULL WHERE team_id=?', (team['id'],))
            db.execute('DELETE FROM tournament_team_invites WHERE team_id=?', (team['id'],))
            db.execute('DELETE FROM tournament_teams WHERE id=?', (team['id'],))
        else:
            fail('不支持的报名操作。')

    def _validate_final(self, db, slug, config):
        state = self._snapshot(db, slug, None)
        entries, teams = state['entries'], state['teams']
        if not entries:
            fail('名单为空，不能锁定。')
        if config['mode'] != 'solo':
            if any(not e['team_id'] for e in entries):
                fail('还有未分组选手，暂不能锁定最终名单。')
            for team in teams:
                members = [e for e in entries if e['team_id'] == team['id']]
                if len(members) != config['team_size']:
                    fail('每支队伍都必须达到规定人数。')
                if config['mode'] == 'self_team' and not team['submitted']:
                    fail('所有队伍需由队长提交报名后才能锁定。')
                if self.catalog._event(db, slug)['format_key'] == 'team-draft-v1' and not team['captain_user_id']:
                    fail('团队对战名单中的每队需指定一位队长。')
                if self.catalog._event(db,slug)['format_key']=='team-draft-v1' and sorted(e.get('position') or 0 for e in members)!=[1,2,3]:
                    fail('请由举办方导入或编辑各队明确的 1、2、3 号位后再锁定。')
        if self.catalog._event(db, slug)['format_key'] == 'team-top5-3x3-v1':
            if len(entries) != 20 or len(teams) != 4 or sum(e['is_external'] for e in entries) != 1:
                fail('14360杯最终名单须为四队各五人，并标记一位外援。')

    def import_roster(self, slug, principal, *, entries, revision, dry_run=True):
        with self.database.transaction() as db:
            self._manager(db, slug, principal)
            self._editable(self.config(db, slug), revision)
        ids = [entry['user_id'] for entry in entries]
        if not ids or len(ids) != len(set(ids)) or any(type(uid) is not int or uid <= 0 for uid in ids):
            fail('名单必须包含不重复的有效用户 ID。')
        names = self.account_reader(ids)
        if set(ids) - names.keys():
            fail(f'找不到有效的 Table 账号：{sorted(set(ids) - names.keys())}。')
        resolved = [{'user_id': e['user_id'], 'display_name': names[e['user_id']], 'team_name': ' '.join(str(e.get('team_name') or '').split()),
                     'is_external': bool(e.get('is_external')), 'captain': bool(e.get('captain')), 'position': e.get('position')} for e in entries]
        with self.database.transaction(immediate=True) as db:
            self._manager(db, slug, principal)
            config = self.config(db, slug)
            self._editable(config, revision)
            if config['capacity'] and len(entries) > config['capacity']:
                fail('导入人数超过赛事名额。')
            previous = {r['user_id'] for r in db.execute('SELECT user_id FROM tournament_entrants WHERE event_slug=?', (slug,))}
            if config['registration_locked'] and set(ids) != previous:
                fail('参赛人员已锁定，只能调整现有人员的分组。增减人员需先解锁。')
            groups = {}
            for e in resolved:
                if len(e['team_name']) > 40:
                    fail('队伍名称不能超过四十个字符。')
                if config['mode'] == 'solo' and (e['team_name'] or e['captain']):
                    fail('单人报名不能填写队伍或队长。')
                if e['captain'] and not e['team_name']:
                    fail('未分组的选手不能标记为队长。')
                if e['team_name']:
                    groups.setdefault(e['team_name'], []).append(e)
            for group in groups.values():
                positions = [e['position'] for e in group if e['position'] is not None]
                if positions and (len(positions) != len(group) or len(set(positions)) != len(positions)
                                  or any(e['captain'] != (e['position'] == 1) for e in group)):
                    fail('队内序号须完整填写且不重复，1 号位必须为队长。')
                if len(group) > config['team_size'] or sum(e['captain'] for e in group) > 1:
                    fail('队伍人数超限，或同队有多个队长。')
                if config['mode'] == 'self_team' and sum(e['captain'] for e in group) != 1:
                    fail('自由组队名单中，每队必须指定一名队长。')
            if dry_run:
                return {'entries': resolved, 'revision': revision, 'saved': False}
            db.execute('DELETE FROM tournament_team_invites WHERE team_id IN (SELECT id FROM tournament_teams WHERE event_slug=?)', (slug,))
            db.execute('DELETE FROM tournament_entrants WHERE event_slug=?', (slug,))
            db.execute('DELETE FROM tournament_roster_positions WHERE event_slug=?', (slug,))
            db.execute('DELETE FROM tournament_teams WHERE event_slug=?', (slug,))
            team_ids = {}
            for name, group in groups.items():
                tid = uuid.uuid4().hex
                team_ids[name] = tid
                captain = next((e['user_id'] for e in group if e['captain']), None)
                db.execute('INSERT INTO tournament_teams VALUES(?,?,?,?,0)', (tid, slug, name, captain))
            db.executemany('INSERT INTO tournament_entrants VALUES(?,?,?,?,?,\'imported\')',
                          [(slug, e['user_id'], e['display_name'], team_ids.get(e['team_name']), e['is_external']) for e in resolved])
            self._audit(db, slug, principal, 'import_roster', {'entries': resolved})
            db.executemany('INSERT INTO tournament_roster_positions VALUES(?,?,?)',
                           [(slug,e['user_id'],e['position']) for e in resolved if e['position'] is not None])
            return {'entries': resolved, 'revision': revision+1, 'saved': True}
