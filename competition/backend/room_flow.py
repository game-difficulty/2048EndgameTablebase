"""Small shared entry points for draft and fixed-sequence match plans."""
import json
import secrets

from .room_rules import MAX_GAMES
from .errors import CompetitionError


def fixed_rules(count, clock_seconds):
    if not 1 <= count <= MAX_GAMES:
        raise CompetitionError('INVALID_PROJECT_POOL', f'请选择 1 至 {MAX_GAMES} 个项目。')
    if type(clock_seconds) is not int or not 30 <= clock_seconds <= 86400:
        raise CompetitionError('INVALID_ROOM_RULES', '总用时须为 30 至 86400 秒。')
    return dict(version='fixed-sequence-v1', workflow='fixed_sequence',
                team_size=1, game_count=count, game_keys=[chr(65+i) for i in range(count)],
                series_mode='all', wins_required=None, lineup_policy='free', steps=[],
                team_clock_seconds=clock_seconds, draft_seconds=0, lineup_seconds=0,
                final_selection=None, auto_ready=False)


class RoomFlow:
    def __init__(self, rooms):
        self.rooms = rooms

    def is_duel(self, db, cid):
        return bool(db.execute('SELECT 1 FROM competition_duel_rooms WHERE competition_id=?', (cid,)).fetchone())

    def kind(self, db, cid):
        if db.execute('SELECT 1 FROM competition_time_attack WHERE competition_id=?', (cid,)).fetchone():
            return 'time_attack'
        return 'duel' if self.is_duel(db, cid) else 'competition'

    def freeze(self, db, cid, keys):
        db.execute('INSERT INTO competition_fixed_series VALUES(?,?,?)',
                   (cid, secrets.token_hex(32), json.dumps(keys)))

    def plan(self, db, cid):
        fixed = db.execute('SELECT * FROM competition_fixed_series WHERE competition_id=?', (cid,)).fetchone()
        if fixed:
            return dict(zip(self.rooms._game_keys(db, cid), json.loads(fixed['projects_json']))), fixed['seed_hex']
        row = db.execute('SELECT 1 FROM competition_drafts WHERE competition_id=?', (cid,)).fetchone()
        if not row:
            return {}, None
        draft = self.rooms._draft_row(db, cid)
        return {key: draft[f'project_{key.lower()}'] for key in self.rooms._game_keys(db, cid)}, draft['random_seed_hex']

    def begin(self, db, room, now):
        """Called exactly once, in the transaction accepting the second ready."""
        cid = room['id']
        if self.kind(db, cid) == 'time_attack':
            return self.rooms.time_attacks.begin(db, room, now)
        if self.rooms._rules(db, cid).get('workflow') != 'fixed_sequence':
            self.rooms._initialize_draw(db, cid, now=now)
            self.rooms._touch(db, cid, status='DRAW')
            return self.rooms._room_row(db, room['room_code'])
        # No synthetic BP or secret-lineup state. Each side has one player.
        assignments = {key: 1 for key in self.rooms._game_keys(db, cid)}
        for side in ('yellow', 'white'):
            self.rooms._insert_lineup(db, room, side=side, assignments=assignments,
                                      actor_user_id=None, automatic=True, now=now)
        room = self.rooms._finalize_lineups(db, room, now=now)
        db.execute('''UPDATE competition_game_readiness SET
            player_ready_by_user_id=(SELECT user_id FROM competition_seats s WHERE s.competition_id=? AND s.side=competition_game_readiness.side),
            captain_ready_by_user_id=(SELECT user_id FROM competition_seats s WHERE s.competition_id=? AND s.side=competition_game_readiness.side),
            player_ready_at=?, captain_ready_at=? WHERE competition_id=? AND game_key='A' ''',
            (cid, cid, now.isoformat(), now.isoformat(), cid))
        return self.rooms._start_game_in_transaction(db, room, game_key='A', now=now, actor_user_id=None)

    def ready_holds(self, db, cid, key, now):
        if self.rooms._rules(db, cid).get('auto_ready', True):
            self.rooms._set_hold(db, cid, f'GAME_{key}_READY', now, self.rooms.ready_preview_seconds)
            self.rooms._set_hold(db, cid, f'GAME_{key}_READY_TIMEOUT', now, self.rooms._flow(db, cid)['ready_seconds'])
