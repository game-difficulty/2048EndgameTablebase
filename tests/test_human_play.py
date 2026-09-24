import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.auth.db import init_auth_db, auth_db
from backend.auth.service import create_session, iso
from backend.gamer_ranked.prng import Xoshiro128StarStar
from backend.human_play import admin, engine, service
from backend.human_play.store import init_db, database
from backend.human_play.routes import router

SEED = '00000001000000020000000300000004'
BROWSER = '11111111-1111-1111-1111-111111111111'
OTHER_BROWSER = '22222222-2222-2222-2222-222222222222'
WRITER = 'test-writer-0000000000001'


def next_event(state, variant, threshold=None):
    for direction in range(4):
        moved, _ = engine.move(state['board'], *engine.VARIANTS[variant], direction)
        if moved != state['board']:
            rng = Xoshiro128StarStar(list(state['rng']))
            index, value = engine.spawn(moved, rng)
            event = engine.EVENT.pack(direction | (index << 2) | (64 if value == 4 else 0), 10 if state['seq'] else 0)
            return event, engine.advance(state, variant, event, threshold)
    return None, state


class HumanPlayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'CLOUD_AUTH_DB': self.temp.name + '/auth.db',
            'HUMAN_PLAY_DB': self.temp.name + '/human.db',
            'HUMAN_PLAY_THRESHOLDS': json.dumps({key: 10000000 for key in engine.VARIANTS})})
        self.env.start(); init_auth_db(); init_db()
        with auth_db() as db:
            for uid in (1, 2):
                db.execute('INSERT INTO users(id,email,password_hash,display_name,created_at,updated_at) VALUES(?,?,?,?,?,?)',
                           (uid, f'{uid}@test.invalid', '!disabled', f'Player {uid}', iso(), iso()))
            self.token = create_session(db, 1)[0]
        app = FastAPI(); app.include_router(router)
        self.client = TestClient(app); self.client.headers['Authorization'] = 'Bearer ' + self.token
        self.number = 0

    def tearDown(self):
        self.client.close(); self.env.stop(); self.temp.cleanup()

    def new(self, variant='4x4', browser=BROWSER, threshold=None):
        self.number += 1
        with patch.dict(os.environ, {'HUMAN_PLAY_THRESHOLDS': json.dumps({variant: threshold if threshold is not None else 10000000})}):
            return service.create(1, browser, variant, f'test-request-{self.number:016d}', WRITER)

    def records(self, run, until=None, count=None):
        state = engine.initial(run['run_id'], run['variant'], run['seed'])
        raw = b''; states = [state]
        while count is None or state['seq'] < count:
            event, next_state = next_event(state, run['variant'], run['threshold'])
            if not event: break
            raw += event; state = next_state; states.append(state)
            if until and until(state): break
        return raw, state, states

    def send(self, run, raw, action='seal', reason='game_over', start=0, states=None, local_seq=None, prefix=None, writer=WRITER):
        initial = engine.initial(run['run_id'], run['variant'], run['seed'])
        return service.submit(1, BROWSER, run['run_id'], action=action, writer=writer, epoch=run.get('epoch',1),
            start=start, prefix_hash=prefix or (states[start]['hash'] if states else initial['hash']),
            local_seq=local_seq if local_seq is not None else start + len(raw)//5, data=raw, reason=reason)

    def test_rectangular_movement_and_single_merge(self):
        board = [2,2,2,2, 0,0,0,0, 4,0,4,0]
        moved, score = engine.move(board, 3, 4, 3)
        self.assertEqual(moved, [4,4,0,0, 0,0,0,0, 8,0,0,0]); self.assertEqual(score,16)
        self.assertEqual(engine.move([32768,32768,0,0,0,0,0,0],2,4,3)[0][0],65536)

    def test_four_slots_and_other_browser_are_independent(self):
        ids = {self.new(v)['run_id'] for v in engine.VARIANTS}
        ids.add(self.new('4x4', OTHER_BROWSER)['run_id']); self.assertEqual(len(ids),5)
        with self.assertRaisesRegex(service.RunError,'slot_exists'): self.new('4x4')

    def test_idempotent_creation(self):
        a = service.create(1,BROWSER,'3x4','idempotent-request-0001',WRITER)
        b = service.create(1,BROWSER,'3x4','idempotent-request-0001',WRITER)
        self.assertEqual(a['run_id'],b['run_id']); self.assertEqual(a['seed'],b['seed'])

    def test_cross_browser_access_and_active_replay_forbidden(self):
        run = self.new()
        with self.assertRaisesRegex(service.RunError,'run_not_found'): service.status(1,OTHER_BROWSER,run['run_id'])
        with self.assertRaisesRegex(service.RunError,'replay_not_found'): service.replay(run['run_id'],1)
        self.assertNotIn('seed', service.status(1,BROWSER,run['run_id']))
        self.assertNotIn('board', service.status(1,BROWSER,run['run_id']))

    def test_low_score_not_uploaded_until_final_and_binary_archive(self):
        run = self.new('2x4'); raw, state, _ = self.records(run)
        with database() as db: self.assertEqual(db.execute('SELECT count(*) FROM human_chunks').fetchone()[0],0)
        result = self.send(run,raw); self.assertEqual(result['status'],'sealed')
        archive = service.replay(run['run_id'],2)
        binary = gzip.decompress(archive); self.assertEqual(binary[:4],b'HPR2')
        self.assertEqual(engine.parse_replay(binary)[1],raw)
        self.assertEqual(service.leaderboard('2x4')['entries'][0]['score'],state['score'])

    def test_restarted_history_is_private_and_retained(self):
        run = self.new(); raw,_,_ = self.records(run,count=3)
        self.send(run,raw,reason='restarted')
        self.assertEqual(len(service.history(1,1)['entries']),1)
        self.assertEqual(len(service.history(1,2)['entries']),0)
        with self.assertRaisesRegex(service.RunError,'replay_not_found'): service.replay(run['run_id'],2)
        self.assertEqual(service.leaderboard('4x4')['entries'],[])

    def approve(self, run, state):
        return admin.review(run['run_id'], approved=True, operator='site-owner', note='Lost local save; retained prefix reviewed',
                            expected_seq=state['seq'], expected_hash=state['hash'])

    def test_manual_checkpoint_ranking_and_revocation(self):
        run, raw, state, states, _ = self.monitored()
        self.assertFalse(engine.game_over(state['board'], *engine.VARIANTS[run['variant']]))
        self.assertEqual(service.leaderboard('4x4')['entries'], [])
        with database() as db:
            db.execute('UPDATE human_chunks SET received=1 WHERE run_id=?', (run['run_id'],))
        result = self.approve(run, state)
        self.assertEqual((result['status'], result['reason'], result['ended_at']), ('sealed', 'interrupted', 1))
        self.assertEqual(service.leaderboard('4x4')['entries'][0]['score'], state['score'])
        self.assertEqual(service.leaderboard('4x4', 'week')['entries'], [])  # Approval cannot move an old score into this week.
        history = service.history(1, 2)
        self.assertEqual(history['bests']['4x4'], state['score'])
        self.assertEqual(history['stats']['completed'], 0)
        binary = gzip.decompress(service.replay(run['run_id'], 2))
        size = struct.unpack_from('<I', binary, 4)[0]
        self.assertEqual(engine.parse_replay(binary)[1], raw)
        self.assertEqual(json.loads(binary[8:8 + size])['reason'], 'interrupted')
        status = service.status(1, BROWSER, run['run_id'])
        self.assertNotIn('board', status)
        self.assertEqual(status['seq'], state['seq'])
        event, _ = next_event(state, run['variant'], run['threshold'])
        with self.assertRaisesRegex(service.RunError, 'run_sealed'):
            self.send(run, event, action='append', start=state['seq'], states=states)
        revoked = admin.review(run['run_id'], approved=False, operator='site-owner', note='Approval withdrawn')
        self.assertEqual(len(revoked['ranking_reviews']), 2)
        self.assertEqual(service.leaderboard('4x4')['entries'], [])
        self.assertEqual(service.history(1, 2)['bests'], {})
        with self.assertRaisesRegex(service.RunError, 'replay_not_found'):
            service.replay(run['run_id'], 2)
        self.assertEqual(gzip.decompress(service.replay(run['run_id'], 1)), binary)

    def test_manual_review_requires_unchanged_verified_evidence(self):
        run, _, state, _, _ = self.monitored()
        stale = dict(state, seq=state['seq'] - 1)
        with self.assertRaisesRegex(service.RunError, 'review_progress_changed'):
            self.approve(run, stale)
        with database() as db:
            db.execute("UPDATE human_chunks SET digest='broken' WHERE run_id=?", (run['run_id'],))
        with self.assertRaisesRegex(service.RunError, 'retained_replay_invalid'):
            self.approve(run, state)
        self.assertEqual(service.status(1, BROWSER, run['run_id'])['status'], 'active')
        self.assertFalse(admin.inspect(run['run_id'])['manually_approved'])

    def test_manual_review_of_existing_archive_preserves_bytes(self):
        run = self.new(); raw, state, _ = self.records(run, count=5)
        self.send(run, raw, reason='restarted')
        before = service.replay(run['run_id'], 1)
        self.approve(run, state)
        self.assertEqual(service.replay(run['run_id'], 2), before)
        self.assertEqual(service.history(1, 2)['entries'][0]['reason'], 'restarted')
        with database() as db:
            damaged = bytearray(before)
            damaged[-8] ^= 1  # Corrupt gzip CRC32; no separate SHA-256 is needed.
            db.execute("UPDATE human_runs SET archive=? WHERE id=?", (bytes(damaged), run['run_id']))
        admin.review(run['run_id'], approved=False, operator='site-owner', note='Rechecking')
        with self.assertRaisesRegex(service.RunError, 'retained_replay_invalid'):
            self.approve(run, state)

    def test_legacy_archive_digest_migration_preserves_replay(self):
        run = self.new(); raw, _, _ = self.records(run, count=5)
        self.send(run, raw, reason='restarted')
        before = service.replay(run['run_id'], 1)
        with database() as db:
            db.execute('ALTER TABLE human_runs ADD COLUMN archive_hash TEXT')
            db.execute("UPDATE human_runs SET archive_hash='legacy' WHERE id=?", (run['run_id'],))
        init_db(); init_db()
        with database() as db:
            self.assertNotIn('archive_hash', {r['name'] for r in db.execute('PRAGMA table_info(human_runs)')})
        self.assertEqual(service.replay(run['run_id'], 1), before)
        response = self.client.get(f"/api/human/replays/{run['run_id']}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, gzip.decompress(before))
        self.assertNotIn('X-Replay-SHA256', response.headers)

    def test_manual_review_cannot_bypass_disqualification_or_empty_record(self):
        run = self.new()
        state = engine.initial(run['run_id'], run['variant'], run['seed'])
        with self.assertRaisesRegex(service.RunError, 'retained_replay_invalid'):
            self.approve(run, state)
        with database() as db:
            db.execute("UPDATE human_runs SET eligibility='disqualified' WHERE id=?", (run['run_id'],))
        with self.assertRaisesRegex(service.RunError, 'run_disqualified'):
            self.approve(run, state)
        response = self.client.post(f"/api/human/runs/{run['run_id']}/approve", json={})
        self.assertEqual(response.status_code, 404)

    def monitored(self):
        run = self.new(threshold=8)
        raw,state,states = self.records(run, until=lambda s:s['score'] > 8)
        result = self.send(run,raw,action='monitor')
        return run,raw,state,states,result

    def test_threshold_is_strict_and_first_over_required(self):
        run = self.new(threshold=8)
        raw,state,states = self.records(run, until=lambda s:s['score'] > 8)
        prior = states[-2]
        with self.assertRaisesRegex(service.RunError,'below_threshold'):
            self.send(run,raw[:-5],action='monitor')
        self.assertLessEqual(prior['score'],8)
        self.assertTrue(self.send(run,raw,action='monitor')['monitored'])

    def test_cannot_skip_high_score_monitoring_and_submit_later(self):
        run = self.new('2x4',threshold=0); raw,_,_ = self.records(run)
        with self.assertRaisesRegex(service.RunError,'monitoring_required'): self.send(run,raw)

    def test_reentry_local_behind_rejected_without_replacing_server(self):
        run,raw,state,states,result = self.monitored()
        with self.assertRaisesRegex(service.RunError,'rollback_detected'):
            self.send(run,b'',action='reentry',start=state['seq'],states=states,local_seq=state['seq']-1)
        after=service.status(1,BROWSER,run['run_id'])
        self.assertEqual(after['seq'],state['seq']); self.assertEqual(after['eligibility'],'disqualified')

    def test_reentry_same_prefix_and_ahead_both_pass(self):
        run,raw,state,states,_ = self.monitored()
        self.send(run,b'',action='reentry',start=state['seq'],states=states)
        event,next_state=next_event(state,run['variant'],run['threshold'])
        result=self.send(run,event,action='reentry',start=state['seq'],states=states)
        self.assertEqual(result['seq'],next_state['seq']); self.assertNotIn('board',result)

    def test_same_count_wrong_prefix_is_disqualified(self):
        run,_,state,states,_=self.monitored()
        with self.assertRaisesRegex(service.RunError,'prefix_conflict'):
            self.send(run,b'',action='reentry',start=state['seq'],states=states,prefix='0'*64)

    def test_ack_loss_exact_upload_retry_is_idempotent(self):
        run,raw,state,states,_=self.monitored()
        result=self.send(run,raw,action='monitor')
        self.assertEqual(result['seq'],state['seq'])
        with database() as db:self.assertEqual(db.execute('SELECT count(*) FROM human_chunks').fetchone()[0],1)

    def test_expired_permit_blocks_append_until_reentry(self):
        run,_,state,states,_=self.monitored()
        with database() as db:db.execute('UPDATE human_runs SET permit_until=0 WHERE id=?',(run['run_id'],))
        event,_=next_event(state,run['variant'],run['threshold'])
        with self.assertRaisesRegex(service.RunError,'reentry_required'):
            self.send(run,event,action='append',start=state['seq'],states=states)
        self.assertTrue(self.send(run,event,action='reentry',start=state['seq'],states=states)['monitored'])

    def test_old_writer_rejected_after_same_browser_claim(self):
        run=self.new(); claimed=service.claim_low(1,BROWSER,run['run_id'],'new-writer-0000000000001',1)
        self.assertEqual(claimed['epoch'],2)
        with self.assertRaisesRegex(service.RunError,'writer_changed'):
            self.send(run,b'',reason='restarted')

    def test_replacing_one_variant_keeps_other_variant(self):
        a=self.new('4x4'); b=self.new('3x4')
        service.create(1,BROWSER,'4x4','new-replacement-00001',WRITER,a['run_id'])
        self.assertEqual(service.status(1,BROWSER,a['run_id'])['status'],'pending_archive')
        self.assertEqual(service.status(1,BROWSER,b['run_id'])['status'],'active')

    def test_retired_run_can_archive_missing_first_checkpoint_but_cannot_resume(self):
        run = self.new(threshold=8)
        raw, state, states = self.records(run, until=lambda s: s['score'] > 8)
        event, final = next_event(state, run['variant'], run['threshold'])
        service.create(1, BROWSER, '4x4', 'replacement-checkpoint-001', WRITER, run['run_id'])
        result = self.send(run, raw, action='monitor')
        self.assertEqual(result['status'], 'pending_archive')
        with self.assertRaisesRegex(service.RunError, 'run_inactive'):
            self.send(run, event, action='append', start=state['seq'], states=states)
        with self.assertRaisesRegex(service.RunError, 'run_inactive'):
            self.send(run, event, action='reentry', start=state['seq'], states=states)
        sealed = self.send(run, event, reason='restarted', start=state['seq'], states=states)
        self.assertEqual(sealed['status'], 'sealed')
        self.assertEqual(sealed['seq'], final['seq'])

    def test_api_auth_origin_and_binary_limits(self):
        self.assertEqual(self.client.post('/api/human/runs',headers={'Origin':'https://untrusted.invalid'},json={}).status_code,403)
        self.assertEqual(TestClient(self.client.app).get('/api/human/runs/no/status').status_code,401)
        run=self.new()
        response=self.client.post(f"/api/human/runs/{run['run_id']}/seal",content=b'bad')
        self.assertEqual(response.status_code,415)
        self.assertEqual(self.client.get(f"/api/human/runs/{run['run_id']}/status",headers={'X-Human-Browser':OTHER_BROWSER}).status_code,404)

    def test_frontend_backend_determinism_all_variants(self):
        root=Path(__file__).resolve().parents[1]
        script="""import {VARIANTS,initialState,initialHash,nextMove,eventHash} from './src/human/engine.js';
        const results=[]; for(const variant of Object.keys(VARIANTS)) {
          let s={...initialState('fixture',variant,'00000001000000020000000300000004'),variant};
          let hash=await initialHash('fixture',variant,'00000001000000020000000300000004'); const events=[];
          for(let i=0;i<80;i++){let n;for(let d=0;d<4;d++){n=nextMove(s,d,i?10:0);if(n)break;}if(!n)break;
            hash=await eventHash(hash,n.event);events.push(n.event);s=n.state;}
          results.push({variant,state:s,events,hash});}console.log(JSON.stringify(results));"""
        output=subprocess.check_output(['node','--input-type=module','-e',script],cwd=root/'frontend',text=True)
        for fixture in json.loads(output):
            raw=b''.join(engine.EVENT.pack(*event) for event in fixture['events'])
            actual=engine.advance(engine.initial('fixture',fixture['variant'],SEED),fixture['variant'],raw)
            for key in ('board','rng','seq','score','elapsed'):self.assertEqual(actual[key],fixture['state'][key],(fixture['variant'],key))
            self.assertEqual(actual['hash'],fixture['hash'])


if __name__ == '__main__': unittest.main()
