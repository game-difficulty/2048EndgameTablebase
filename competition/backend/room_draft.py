"""Persistent ordered BP steps. Project keys are opaque to the room workflow."""
import hashlib
import hmac
import json
from datetime import timedelta

from .errors import CompetitionError


class RoomDraft:
    def __init__(self, rooms):
        self.rooms = rooms

    def state(self, db, cid):
        row = db.execute('SELECT * FROM competition_draft_steps WHERE competition_id=?', (cid,)).fetchone()
        if not row:
            return None
        return {'step_index': row['step_index'], **{k: json.loads(row[k+'_json']) for k in ('selected', 'banned', 'history')}}

    def save(self, db, cid, state):
        db.execute('UPDATE competition_draft_steps SET step_index=?,selected_json=?,banned_json=?,history_json=? WHERE competition_id=?',
                   (state['step_index'], *(json.dumps(state[k], ensure_ascii=False) for k in ('selected', 'banned', 'history')), cid))

    def available(self, db, cid):
        state = self.state(db, cid)
        excluded = set(state['selected'] + state['banned'])
        return [p for p in self.rooms._project_keys(db, cid) if p not in excluded]

    def actor(self, first, step):
        return first if step['actor'] == 'first' else ('white' if first == 'yellow' else 'yellow')

    def set_phase(self, db, room, status, now, seconds, side=None):
        deadline = (now + timedelta(seconds=seconds)).isoformat()
        db.execute('UPDATE competition_drafts SET phase_token=?,phase_started_at=?,yellow_deadline_at=?,white_deadline_at=?,updated_at=? WHERE competition_id=?',
                   (self.rooms._new_phase_token(), now.isoformat(), deadline if side in (None, 'yellow') else None,
                    deadline if side in (None, 'white') else None, now.isoformat(), room['id']))
        self.rooms._append_event(db, room['id'], 'draft.phase_started', None, {'phase': status, 'active_side': side, 'deadline_at': deadline})
        self.rooms._touch(db, room['id'], status=status)
        return self.rooms._room_row(db, room['room_code'])

    def enter(self, db, room, now):
        cid = room['id']
        db.execute('INSERT OR IGNORE INTO competition_draft_steps(competition_id) VALUES(?)', (cid,))
        db.execute('INSERT OR IGNORE INTO competition_prediction_windows(competition_id,opened_at,minimum_until) VALUES(?,?,?)',
                   (cid, now.isoformat(), (now+timedelta(seconds=60)).isoformat()))
        state, rules = self.state(db, cid), self.rooms._rules(db, cid)
        draft = self.rooms._draft_row(db, cid)
        if state['step_index'] < len(rules['steps']):
            step = rules['steps'][state['step_index']]
            return self.set_phase(db, room, 'DRAFT_STEP', now, rules['draft_seconds'], self.actor(draft['first_side'], step))
        if rules['final_selection'] == 'blind':
            return self.set_phase(db, room, 'BLIND_PICK', now, rules['draft_seconds'])
        choices = self.available(db, cid)
        return self.finish(db, room, self.draw(draft, choices, 'remaining-pool'), now)

    @staticmethod
    def draw(draft, choices, label):
        seed = bytes.fromhex(draft['random_seed_hex'])
        # Rejection sampling avoids modulo bias; label includes the ordered pool.
        bound = 2**256 - (2**256 % len(choices))
        nonce = 0
        while True:
            message = json.dumps([label, choices, nonce], separators=(',', ':')).encode()
            value = int.from_bytes(hmac.new(seed, message, hashlib.sha256).digest(), 'big')
            if value < bound:
                return choices[value % len(choices)]
            nonce += 1

    def submit(self, db, room, picks, bans, side, actor, now, automatic=False):
        cid = room['id']; state = self.state(db, cid); rules = self.rooms._rules(db, cid)
        if room['status'] != 'DRAFT_STEP' or not state or state['step_index'] >= len(rules['steps']):
            raise CompetitionError('INVALID_DRAFT_PHASE', '当前不是选禁阶段。', 409)
        step = rules['steps'][state['step_index']]
        if side != self.actor(self.rooms._draft_row(db, cid)['first_side'], step):
            raise CompetitionError('NOT_YOUR_TURN', '当前不是本方的选禁回合。', 403)
        if not isinstance(picks, list) or not isinstance(bans, list) or any(not isinstance(v, str) for v in picks+bans):
            raise CompetitionError('INVALID_DRAFT_SELECTION', '请选择有效的项目。')
        if len(picks) != step['picks'] or len(bans) != step['bans'] or len(set(picks+bans)) != len(picks+bans) or not set(picks+bans) <= set(self.available(db, cid)):
            raise CompetitionError('INVALID_DRAFT_SELECTION', '选禁数量不符，或项目已被选择、禁用。')
        item = {'step_index': state['step_index'], 'side': side, 'picks': picks, 'bans': bans, 'automatic': automatic}
        state['selected'].extend(picks); state['banned'].extend(bans); state['history'].append(item); state['step_index'] += 1
        self.save(db, cid, state)
        self.rooms._append_event(db, cid, 'draft.step_submitted', actor, item)
        return self.enter(db, room, now)

    def timeout(self, db, room, now):
        draft = self.rooms._draft_row(db, room['id'])
        deadlines = [draft[k] for k in ('yellow_deadline_at', 'white_deadline_at') if draft[k]]
        if not deadlines or now.isoformat() < max(deadlines):
            return room, False
        rules = self.rooms._rules(db, room['id']); state = self.state(db, room['id']); step = rules['steps'][state['step_index']]
        pool = self.available(db, room['id'])
        # Same deterministic pool-order fallback as the legacy room.
        return self.submit(db, room, pool[:step['picks']], pool[step['picks']:step['picks']+step['bans']],
                           self.actor(draft['first_side'], step), None, now, True), True

    def finish(self, db, room, selected, now):
        state = self.state(db, room['id'])
        state['selected'].append(selected)
        state['history'].append({'step_index': state['step_index'], 'side': 'random', 'picks': [selected], 'bans': [], 'automatic': True})
        self.save(db, room['id'], state)
        self.rooms._append_event(db, room['id'], 'draft.final_selected', None, {'project_key': selected})
        return self.set_phase(db, room, 'C_DRAW', now, self.rooms.c_draw_reveal_seconds)

    def view(self, db, cid, draft, status):
        state = self.state(db, cid)
        if state is None:
            state = {'step_index': 0, 'selected': [], 'banned': [], 'history': []}
        rules = self.rooms._rules(db, cid)
        step = rules['steps'][state['step_index']] if state['step_index'] < len(rules['steps']) else None
        complete = len(state['selected']) == rules['game_count']
        return {**state, 'steps': rules['steps'], 'current_step': step if status == 'DRAFT_STEP' else None,
                'active_side': self.actor(draft['first_side'], step) if step and status == 'DRAFT_STEP' else None,
                'final_selection': rules['final_selection'], 'complete': complete,
                'seed_hex': draft['random_seed_hex'] if status == 'FINISHED' else None}
