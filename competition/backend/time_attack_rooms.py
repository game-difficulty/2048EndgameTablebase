"""Public fixed-window PB matches. Attempt completion never completes the room."""
from datetime import datetime, timedelta, timezone
import json
import secrets
import uuid

from backend.human_play import engine
from .errors import CompetitionError
from .projects.target_challenge import configuration, reached, compare_best, PROJECT_REF, VERSION


class TimeAttackRooms:
    def __init__(self, rooms):
        self.rooms = rooms

    def create(self, principal, *, name, variant, target_kind, target_value, clock_seconds, command_id):
        config = configuration(variant, target_kind, target_value)
        return self.rooms.duels.create(principal, name=name, projects=[PROJECT_REF],
                                      clock_seconds=clock_seconds, command_id=command_id, _challenge=config)

    def _meta(self, db, cid):
        meta = db.execute('SELECT * FROM competition_time_attack WHERE competition_id=?', (cid,)).fetchone()
        if meta is None:
            raise CompetitionError('TIME_ATTACK_REQUIRED', 'This is not a time attack room.', 409)
        return meta

    def _new_attempt(self, db, cid, side, config, now):
        number = db.execute('SELECT COALESCE(MAX(number),0)+1 FROM competition_time_attempts WHERE competition_id=? AND side=?', (cid, side)).fetchone()[0]
        aid, seed = uuid.uuid4().hex, secrets.token_hex(16)
        state = engine.initial(aid, config['variant'], seed)
        db.execute('INSERT INTO competition_time_attempts(id,competition_id,side,number,started_at,seed,state_json) VALUES(?,?,?,?,?,?,?)',
                   (aid, cid, side, number, now.isoformat(), seed, json.dumps(state)))

    def begin(self, db, room, now):
        cid = room['id']
        meta = self._meta(db, cid)
        if meta['started_at']:
            return room
        config = json.loads(meta['configuration_json'])
        deadline = now + timedelta(seconds=self.rooms._rules(db, cid)['team_clock_seconds'])
        db.execute('UPDATE competition_time_attack SET started_at=?,deadline_at=? WHERE competition_id=?',
                   (now.isoformat(), deadline.isoformat(), cid))
        for side in ('yellow', 'white'):
            self._new_attempt(db, cid, side, config, now)
        self.rooms._append_event(db, cid, 'time_attack.started', None, {'deadline_at': deadline.isoformat()})
        self.rooms._touch(db, cid, status='GAME_A_PLAYING')
        return self.rooms._room_row(db, room['room_code'])

    def settle(self, db, room, now):
        if room['status'] in ('FINISHED', 'CANCELLED'):
            return room, False
        meta = self._meta(db, room['id'])
        if not meta['deadline_at']:
            return self.rooms.duels.expire(db, room, now)
        if now < datetime.fromisoformat(meta['deadline_at']):
            return room, False
        best = [db.execute('SELECT MIN(pb_ms) FROM competition_time_attempts WHERE competition_id=? AND side=?',
                           (room['id'], side)).fetchone()[0] for side in ('yellow', 'white')]
        winner = compare_best(*best)
        db.execute('UPDATE competition_time_attack SET winner_side=? WHERE competition_id=?', (winner, room['id']))
        db.execute("UPDATE competition_time_attempts SET status='expired' WHERE competition_id=? AND status='playing'", (room['id'],))
        self.rooms._append_event(db, room['id'], 'time_attack.finished', None, {'winner_side': winner, 'best_ms': best})
        self.rooms._touch(db, room['id'], status='FINISHED')
        return self.rooms._room_row(db, room['room_code']), True

    def command(self, code, principal, *, attempt_id, command_id, action, base_sequence=0, events=None):
        r = self.rooms
        command_id = r._normalize_command_id(command_id)
        with r.database.transaction(immediate=True) as db:
            # Receipt/admission time is sampled inside the serialized transaction.
            now = datetime.now(timezone.utc)
            room = r._room_row(db, code)
            meta = self._meta(db, room['id'])
            r._ensure_admitted(db, room['id'], principal)
            seat = db.execute('SELECT side FROM competition_seats WHERE competition_id=? AND user_id=?', (room['id'], principal.user_id)).fetchone()
            if not seat:
                raise CompetitionError('ACTIVE_PLAYER_REQUIRED', 'Only seated players may play.', 403)
            room, _ = self.settle(db, room, now)
            # Commit deadline catch-up even when a late command is refused.
            if room['status'] != 'GAME_A_PLAYING':
                return r._snapshot(db, room, principal)
            label = f'time_attack.{action}'
            if r._check_command(db, room['id'], principal, command_id, label):
                return r._snapshot(db, room, principal)
            attempt = db.execute('SELECT * FROM competition_time_attempts WHERE competition_id=? AND side=? ORDER BY number DESC LIMIT 1', (room['id'], seat['side'])).fetchone()
            if attempt is None or attempt['id'] != attempt_id:
                raise CompetitionError('STALE_ATTEMPT', 'Reload the current attempt.', 409)
            config = json.loads(meta['configuration_json'])
            if config['rules_version'] != VERSION:
                raise CompetitionError('PROJECT_ADAPTER_UNAVAILABLE', 'Unsupported frozen challenge version.', 409)
            elapsed = max(0, int((now-datetime.fromisoformat(attempt['started_at'])).total_seconds()*1000))
            if action == 'restart':
                if elapsed < 500:
                    raise CompetitionError('TIME_ATTACK_RESTART_LIMIT', 'Wait half a second before restarting.', 429)
                db.execute("UPDATE competition_time_attempts SET status='abandoned' WHERE id=? AND status='playing'", (attempt_id,))
                self._new_attempt(db, room['id'], seat['side'], config, now)
            elif action == 'submit':
                state = json.loads(attempt['state_json'])
                if (type(base_sequence) is not int or base_sequence != state['seq']
                        or attempt['status'] != 'playing'):
                    raise CompetitionError('STALE_ATTEMPT', 'Reload the current attempt.', 409)
                if (not isinstance(events, list) or not 1 <= len(events) <= 64 or any(
                    not isinstance(e, list) or len(e) != 2 or type(e[0]) is not int or type(e[1]) is not int
                    or not 0 <= e[0] < 128 or not 0 <= e[1] <= 86400000 for e in events)):
                    raise CompetitionError('INVALID_ATTEMPT', 'Invalid move batch.')
                raw = b''
                status, pb = 'playing', None
                try:
                    for event in events:
                        if status != 'playing':
                            raise ValueError('moves_after_completion')
                        encoded = engine.EVENT.pack(*event)
                        state = engine.advance(state, config['variant'], encoded)
                        raw += encoded
                        if reached(config, state['board']):
                            status, pb = 'reached', elapsed
                        elif (config['target_kind'] == 'board_sum' and sum(state['board']) > config['target_value']):
                            status = 'overshot'
                        elif engine.game_over(state['board'], *engine.VARIANTS[config['variant']]):
                            status = 'no_moves'
                    if state['elapsed'] > elapsed + 1000:
                        raise ValueError('future_time')
                except (ValueError, OverflowError) as exc:
                    raise CompetitionError('INVALID_ATTEMPT', 'The move record failed verification.') from exc
                # PB is server elapsed, not the player's claimed stopwatch.
                # This includes transport latency but cannot be reduced by forged deltas.
                db.execute('UPDATE competition_time_attempts SET state_json=?,status=?,pb_ms=?,events=? WHERE id=?',
                           (json.dumps(state), status, pb, bytes(attempt['events'])+raw, attempt_id))
            else:
                raise CompetitionError('INVALID_ATTEMPT', 'Unknown attempt action.')
            r._record_command(db, room['id'], principal, command_id, label)
            r._touch(db, room['id'])
            return r._snapshot(db, r._room_row(db, code), principal)

    def view(self, db, room, principal, now):
        meta = db.execute('SELECT * FROM competition_time_attack WHERE competition_id=?', (room['id'],)).fetchone()
        if not meta:
            return None
        config = json.loads(meta['configuration_json'])
        players = {}
        for side in ('yellow', 'white'):
            seat = db.execute('SELECT user_id FROM competition_seats WHERE competition_id=? AND side=?', (room['id'], side)).fetchone()
            row = db.execute('SELECT * FROM competition_time_attempts WHERE competition_id=? AND side=? ORDER BY number DESC LIMIT 1', (room['id'], side)).fetchone()
            best = db.execute('SELECT id,pb_ms,number FROM competition_time_attempts WHERE competition_id=? AND side=? AND pb_ms IS NOT NULL ORDER BY pb_ms,number LIMIT 1', (room['id'], side)).fetchone()
            completed = db.execute('SELECT COUNT(*) FROM competition_time_attempts WHERE competition_id=? AND side=? AND pb_ms IS NOT NULL', (room['id'], side)).fetchone()[0]
            attempt = None
            if row:
                state = json.loads(row['state_json'])
                attempt = dict(id=row['id'], number=row['number'], status=row['status'], started_at=row['started_at'],
                               board=state['board'], sequence=state['seq'], score=state['score'], pb_ms=row['pb_ms'])
                if seat and seat['user_id'] == principal.user_id:
                    attempt['state'] = state
                    attempt['seed'] = row['seed']
            players[side] = dict(attempt=attempt, best=dict(best) if best else None, completed=completed)
        return dict(configuration=config, started_at=meta['started_at'], deadline_at=meta['deadline_at'],
                    winner_side=meta['winner_side'], players=players, timing='server-confirmed-ms')

    def best_record(self, code, principal, side):
        if side not in ('yellow', 'white'):
            raise CompetitionError('INVALID_SIDE', 'Unknown side.')
        with self.rooms.database.transaction() as db:
            room = self.rooms._room_row(db, code)
            self.rooms._ensure_admitted(db, room['id'], principal)
            config = json.loads(self._meta(db, room['id'])['configuration_json'])
            row = db.execute('SELECT * FROM competition_time_attempts WHERE competition_id=? AND side=? AND pb_ms IS NOT NULL ORDER BY pb_ms,number LIMIT 1', (room['id'], side)).fetchone()
            if row is None:
                raise CompetitionError('NO_VALID_ATTEMPT', 'There is no verified PB yet.', 404)
            # Never expose an active attempt's RNG to another player.
            return dict(id=row['id'], number=row['number'], pb_ms=row['pb_ms'], configuration=config,
                        header=dict(run_id=row['id'], variant=config['variant'], seed=row['seed']),
                        events=list(engine.EVENT.iter_unpack(bytes(row['events']))))
