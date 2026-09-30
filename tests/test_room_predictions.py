import asyncio
import os
import random
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch
from fastapi import HTTPException
from backend.auth.db import auth_db, init_auth_db
from backend.room_activities import predictions as p
from backend.room_activities.runtime import RoomActivities


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        env = patch.dict(os.environ, CLOUD_AUTH_DB=str(Path(self.directory.name)/'auth.db'))
        env.start(); self.addCleanup(env.stop)
        init_auth_db(); p.init_schema()
        with auth_db() as db:
            for uid in range(1,6):
                db.execute('''INSERT INTO users(id,email,email_identity,password_hash,display_name,created_at,updated_at)
                    VALUES(?,?,?,?,?,'now','now')''',(uid,f'{uid}@test.invalid',f'{uid}@test.invalid','test',str(uid)))
                db.execute("INSERT INTO token_accounts(user_id,bonus_balance_units,paid_balance_units,created_at,updated_at) VALUES(?,777000,20000000,'now','now')",(uid,))
        self.batch=dict(id=str(uuid.uuid4()),started_at=100,deadline=280,
                        participants=[dict(id=name,name=name) for name in ('lume','clari','vero')])
        p.ensure_market('room',self.batch)

    def bet(self,uid,option,amount=100,request_id=None,now=110,available=True):
        return p.place('room',uid,dict(market_id=self.batch['id'],option_id=option,amount=amount,
                                      request_id=request_id or str(uuid.uuid4())),available,now)

    def balances(self):
        with auth_db() as db:
            return [tuple(r) for r in db.execute('SELECT paid_balance_units,bonus_balance_units FROM token_accounts ORDER BY user_id')]

    def target_bet(self, uid=1, option='lume', amount=100, request_id=None, now=110, available=True):
        return p.place('room',uid,dict(kind='target65536',market_id=self.batch['id'],option_id=option,
            amount=amount,request_id=request_id or str(uuid.uuid4())),available,now)

    def test_amount_options_are_enforced_for_main_and_target_stakes(self):
        self.assertEqual(p.listing('room')['amounts'], [100, 1000, 5000, 10000])
        with self.assertRaises(HTTPException):
            self.bet(1, 'lume', 500)
        self.bet(1, 'lume', 10000)
        self.target_bet(amount=10000)
        self.assertEqual(self.balances()[0][0], 0)

    def test_target_win_pays_eight_times_without_principal_even_when_main_has_no_winner(self):
        self.bet(1,'lume');self.target_bet();self.target_bet(amount=1000)
        public=p.listing('room',now=120)['market']
        self.assertEqual(public['pool_units'],100000)
        self.assertNotIn('mine',public['target_bet'])
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        self.assertEqual(self.balances()[0][0],27600000)
        p.settle('room',self.batch['id'],['vero'],now=400)
        self.assertEqual(self.balances()[0],(27700000,777000))
        detail=p.listing('room',1,now=401)
        result=dict(principal=0,profit=8800000,refund=0,loss=0)
        self.assertEqual(detail['market']['target_bet']['result'],result)
        self.assertEqual(detail['recent'][0]['target_bet'],result)
        self.assertEqual(detail['recent'][0]['stake_units'],1200000)
        self.assertEqual(detail['market']['result']['refund'],100000)

    def test_target_miss_does_not_change_main_pool_settlement(self):
        self.bet(1,'lume');self.bet(2,'clari',1000);self.target_bet()
        p.record_targets('room',self.batch['id'],{'lume':'missed'})
        p.settle('room',self.batch['id'],['lume'],now=400)
        self.assertEqual(self.balances()[:2],[(20000000,777000),(19900000,777000)])
        self.assertEqual(p.listing('room',1)['market']['target_bet']['result'],dict(principal=0,profit=0,refund=0,loss=100000))

    def test_recent_combines_both_stakes_and_negative_profit_with_five_latest_starts(self):
        for i in range(7):
            self.batch.update(id=str(uuid.uuid4()),started_at=100+i*1000,deadline=700+i*1000)
            p.ensure_market('room',self.batch)
            self.bet(1,'lume',1000,now=self.batch['started_at']+1)
            self.bet(2,'clari',100,now=self.batch['started_at']+1)
            self.target_bet(amount=1000,now=self.batch['started_at']+1)
            p.record_targets('room',self.batch['id'],{'lume':'missed'})
            p.settle('room',self.batch['id'],['lume'],now=self.batch['deadline']+1)
        recent=p.listing('room',1)['recent']
        self.assertEqual(len(recent),5)
        self.assertEqual([r['started_at'] for r in recent],[6100,5100,4100,3100,2100])
        self.assertTrue(all(r['stake_units']==2000000 and r['net_profit_units']==-900000 for r in recent))

    def test_combo_upgrades_to_50_times_without_principal_once_without_stacking(self):
        self.bet(1,'lume');self.target_bet()
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        p.record_targets('room',self.batch['id'],{'lume':'reached_combo'})
        # Stale facts and duplicate deliveries cannot downgrade the best tier.
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        p.record_targets('room',self.batch['id'],{'lume':'reached_combo'})
        detail=p.listing('room',1,now=120)['market']['target_bet']
        self.assertEqual(detail['outcomes'],{'lume':'reached_combo'})
        self.assertEqual((detail['bonus_target'],detail['bonus_reward_multiplier']),(32768,50))
        self.assertEqual(detail['result'],dict(principal=0,profit=5000000,refund=0,loss=0))
        with self.assertRaises(HTTPException):self.target_bet()
        with ThreadPoolExecutor(max_workers=4) as executor:
            done=list(executor.map(lambda _:p.settle('room',self.batch['id'],['vero'],now=400),range(4)))
        self.assertEqual(sum(done),1)
        self.assertEqual(self.balances()[0],(24900000,777000))
        self.assertEqual(p.listing('room',1)['market']['target_bet']['result'],
                         dict(principal=0,profit=5000000,refund=0,loss=0))

    def test_combo_direct_hit_and_technical_void_retains_paid_reward(self):
        self.bet(1,'lume');self.target_bet(amount=5000)
        p.record_targets('room',self.batch['id'],{'lume':'reached_combo'})
        p.settle('room',self.batch['id'],void=True)
        self.assertEqual(self.balances()[0],(265000000,777000))
        self.assertEqual(p.listing('room',1)['market']['target_bet']['result'],
                         dict(principal=0,profit=250000000,refund=0,loss=0))

    def test_additive_migration_keeps_old_outcomes_and_settled_balances(self):
        self.bet(1,'lume');self.target_bet()
        with auth_db() as db:
            db.execute('DROP TABLE room_prediction_target_outcomes')
            db.execute("CREATE TABLE room_prediction_target_outcomes (market_id TEXT NOT NULL, option_id TEXT NOT NULL, outcome TEXT NOT NULL CHECK(outcome IN ('reached','missed')), PRIMARY KEY(market_id,option_id))")
            db.execute('INSERT INTO room_prediction_target_outcomes VALUES(?,?,?)',(self.batch['id'],'lume','reached'))
        p.init_schema();p.init_schema()
        self.assertEqual(p.listing('room')['market']['target_bet']['outcomes'],{'lume':'reached'})
        p.settle('room',self.batch['id'],['vero'])
        balance=self.balances()
        p.record_targets('room',self.batch['id'],{'lume':'reached_combo'})
        self.assertFalse(p.settle('room',self.batch['id'],['vero']))
        self.assertEqual(self.balances(),balance)
        self.assertEqual(p.listing('room',1)['market']['target_bet']['result']['profit'],800000)

    def test_unknown_unsettled_rules_are_rejected_instead_of_silently_repriced(self):
        self.bet(1,'lume');self.target_bet()
        before=self.balances()
        with auth_db() as db:
            db.execute('UPDATE room_prediction_markets SET rules=? WHERE id=?',
                       ('obsolete',self.batch['id']))
        with self.assertRaisesRegex(ValueError,'unknown_prediction_rules'):
            p.settle('room',self.batch['id'],['lume'])
        self.assertEqual(self.balances(),before)
        with auth_db() as db:
            db.execute('UPDATE room_prediction_markets SET rules=?,target_rules=? WHERE id=?',
                       (p.RULE_VERSION,'obsolete',self.batch['id']))
        with self.assertRaisesRegex(ValueError,'unknown_prediction_target_rules'):
            p.record_targets('room',self.batch['id'],{'lume':'reached'})
        self.assertEqual(self.balances(),before)

    def test_target_requires_main_same_player_open_unresolved_and_permanent_balance(self):
        with self.assertRaises(HTTPException) as missing:self.target_bet()
        self.assertEqual(missing.exception.detail,'prediction_main_required')
        self.bet(1,'lume')
        with self.assertRaises(HTTPException) as switching:self.target_bet(option='clari')
        self.assertEqual(switching.exception.detail,'prediction_cannot_switch')
        for when,available in [(280,True),(110,False),(110,lambda:False)]:
            with self.assertRaises(HTTPException):self.target_bet(now=when,available=available)
        for amount in [True,100.0,-100,101]:
            with self.assertRaises(HTTPException):self.target_bet(amount=amount)
        with auth_db() as db:db.execute('UPDATE token_accounts SET paid_balance_units=99000 WHERE user_id=1')
        with self.assertRaises(HTTPException) as insufficient:self.target_bet()
        self.assertEqual(insufficient.exception.status_code,402)
        self.assertEqual(self.balances()[0],(99000,777000))
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        with self.assertRaises(HTTPException) as reached:self.target_bet()
        self.assertEqual(reached.exception.detail,'prediction_target_closed')
        self.bet(2,'clari')
        p.record_targets('room',self.batch['id'],{'clari':'missed'})
        with self.assertRaises(HTTPException) as ended:self.target_bet(uid=2,option='clari')
        self.assertEqual(ended.exception.detail,'prediction_target_closed')

    def test_target_concurrent_retry_and_cross_kind_request_conflict(self):
        main_key=str(uuid.uuid4());self.bet(1,'lume',request_id=main_key)
        with self.assertRaises(HTTPException) as conflict:self.target_bet(request_id=main_key)
        self.assertEqual(conflict.exception.detail,'prediction_request_conflict')
        key=str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _:self.target_bet(request_id=key),range(4)))
        with self.assertRaises(HTTPException):self.bet(1,'lume',request_id=key)
        with self.assertRaises(HTTPException):self.target_bet(amount=1000,request_id=key)
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        p.close('room',self.batch['id'])
        self.target_bet(request_id=key,now=400,available=False)
        self.assertEqual(self.balances()[0][0],20600000)
        with ThreadPoolExecutor(max_workers=4) as executor:
            done=list(executor.map(lambda _:p.settle('room',self.batch['id'],['lume'],now=400),range(4)))
        self.assertEqual(sum(done),1)
        self.assertEqual(self.balances()[0][0],20700000)

    def test_target_void_retains_paid_reward_and_refunds_unpaid_stakes(self):
        self.bet(1,'lume',1000);self.target_bet(amount=1000)
        self.bet(2,'clari',1000);self.target_bet(uid=2,option='clari',amount=1000)
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        p.settle('room',self.batch['id'],void=True,now=400)
        self.assertEqual(self.balances()[:2],[(27000000,777000),(20000000,777000)])
        self.assertEqual(p.listing('room',1)['market']['target_bet']['result'],dict(principal=0,profit=8000000,refund=0,loss=0))
        self.assertEqual(p.listing('room',2)['market']['target_bet']['result'],dict(principal=0,profit=0,refund=1000000,loss=0))

    def test_target_missing_facts_or_credit_failure_roll_back_both_settlements(self):
        self.bet(1,'lume');self.bet(2,'clari');self.target_bet()
        before=self.balances()
        with self.assertRaisesRegex(ValueError,'prediction_target_result_missing'):
            p.settle('room',self.batch['id'],['lume'])
        self.assertEqual(self.balances(),before)
        self.assertIsNone(p.listing('room',1)['market']['result'])
        original=p._credit
        def fail(db,uid,delta,event,*args):
            original(db,uid,delta,event,*args)
            if event=='room_prediction_target_settlement':raise RuntimeError('target credit failed')
        with patch.object(p,'_credit',side_effect=fail),self.assertRaises(RuntimeError):
            p.record_targets('room',self.batch['id'],{'lume':'reached'})
        self.assertEqual(self.balances(),before)
        self.assertEqual(p.listing('room')['market']['target_bet']['outcomes'],{})
        self.assertEqual(p.announcement_events('room'),[])
        self.assertIsNone(p.listing('room',1)['market']['target_bet']['result'])
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        self.assertTrue(p.settle('room',self.batch['id'],['lume']))

    def test_immediate_tiers_concurrent_replay_pay_only_difference_and_emit_once(self):
        self.bet(1,'lume');self.target_bet()
        for outcome,balance in [('reached',20600000),('reached_combo',24800000)]:
            with ThreadPoolExecutor(max_workers=5) as executor:
                list(executor.map(lambda _:p.record_targets('room',self.batch['id'],{'lume':outcome},now=120),range(10)))
            self.assertEqual(self.balances()[0][0],balance)
        events=p.announcement_events('room',True)
        self.assertEqual([e['total_units'] for e in events],[800000,5000000])
        self.assertEqual([e['tier'] for e in events],['65k','65k+32k'])
        with auth_db() as db:
            paid=[r[0] for r in db.execute("SELECT paid_delta_units FROM token_ledger WHERE event_type='room_prediction_target_settlement' ORDER BY id")]
        self.assertEqual(paid,[800000,4200000])
        p.init_schema()
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        self.assertEqual(p.announcement_events('room',True),events)
        p.announcements_delivered('other',[e['id'] for e in events])
        self.assertEqual(p.announcement_events('room',True),events)
        p.announcements_delivered('room',[e['id'] for e in events])
        self.assertEqual(p.announcement_events('room',True),[])
        self.assertEqual(p.announcement_events('room'),events)
        self.assertEqual(p.announcement_events('other'),[])

    def test_announcement_first_side_stake_order_topups_and_aggregate_all_recipients(self):
        for uid in (4,2,5,1):
            self.bet(uid,'lume');self.target_bet(uid=uid)
        self.target_bet(uid=4,amount=1000)
        self.bet(3,'clari')  # Main-only bets never appear in side-bet announcements.
        p.record_targets('room',self.batch['id'],{'lume':'reached_combo','clari':'reached','vero':'reached_combo'},now=120)
        events=p.announcement_events('room')
        self.assertEqual(len(events),2)
        for event in events:
            self.assertEqual(event['names'],['4','2','5'])
            self.assertEqual(event['recipient_count'],4)
            self.assertEqual(event['option_id'],'lume')
        self.assertEqual([e['total_units'] for e in events],[11200000,70000000])

    def test_all_recipient_credits_and_announcement_roll_back_together(self):
        for uid in (1,2):self.bet(uid,'lume');self.target_bet(uid=uid)
        before=self.balances();original=p._credit
        def fail(db,uid,*args):
            original(db,uid,*args)
            if uid==2:raise RuntimeError('mid payout')
        with patch.object(p,'_credit',side_effect=fail),self.assertRaises(RuntimeError):
            p.record_targets('room',self.batch['id'],{'lume':'reached_combo'})
        self.assertEqual(self.balances(),before)
        self.assertEqual(p.announcement_events('room'),[])
        p.record_targets('room',self.batch['id'],{'lume':'reached_combo'})
        self.assertEqual([r[0] for r in self.balances()[:2]],[24800000,24800000])

    def test_committed_reward_outbox_retries_broadcast_failure_without_repaying(self):
        self.bet(1,'lume');self.target_bet()
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        before=self.balances()
        def broken(event):raise RuntimeError('transport down')
        hub=SimpleNamespace(room=SimpleNamespace(id='room'),chat=deque(maxlen=100),broadcast=broken)
        activities=RoomActivities(hub)
        with self.assertRaises(RuntimeError):asyncio.run(activities.flush_announcements())
        self.assertEqual(len(p.announcement_events('room',True)),1)
        sent=[];hub.broadcast=sent.append
        asyncio.run(activities.flush_announcements());asyncio.run(activities.flush_announcements())
        self.assertEqual(len(sent),1)
        self.assertEqual(len(hub.chat),1)
        self.assertEqual(p.announcement_events('room',True),[])
        self.assertEqual(self.balances(),before)

    def test_target_facts_are_room_scoped_and_immutable(self):
        p.record_targets('other',self.batch['id'],{'lume':'reached'})
        self.assertEqual(p.listing('room')['market']['target_bet']['outcomes'],{})
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        p.record_targets('room',self.batch['id'],{'lume':'reached'})
        with self.assertRaises(ValueError):p.record_targets('room',self.batch['id'],{'lume':'missed'})
        self.assertEqual(p.listing('room')['market']['target_bet']['outcomes'],{'lume':'reached'})

    def test_example_large_opponent_refund_and_idempotent_settlement(self):
        self.bet(1,'lume');self.bet(2,'clari',5000);self.bet(2,'clari',5000)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results=list(executor.map(lambda _:p.settle('room',self.batch['id'],['lume'],now=400),range(4)))
        self.assertEqual(sum(results),1)
        self.assertEqual(self.balances()[:2],[(20100000,777000),(19900000,777000)])
        detail=p.listing('room',1,now=401)
        self.assertEqual(detail['market']['result'],dict(principal=100000,profit=100000,refund=0,loss=0))
        self.assertEqual(p.listing('room',2,now=401)['market']['result']['refund'],9900000)

    def test_tied_winners_share_only_third_option_proportionally(self):
        self.bet(1,'lume');self.bet(2,'clari',1000)
        for _ in range(3):self.bet(3,'vero')
        p.settle('room',self.batch['id'],['lume','clari'],now=400)
        self.assertEqual([r[0] for r in self.balances()[:3]],[20000000,20300000,19700000])

    def test_option_pool_matching_removes_account_split_advantage(self):
        single = [
            dict(user_id=1, option_id='a', units=100),
            dict(user_id=2, option_id='b', units=10000),
        ]
        split = [dict(user_id=1, option_id='a', units=100)] + [
            dict(user_id=uid, option_id='b', units=100) for uid in range(2, 102)
        ]
        self.assertEqual(p.allocate(single, {'a'})[1]['profit'], 100)
        self.assertEqual(p.allocate(split, {'a'})[1]['profit'], 100)
        self.assertEqual(sum(part['loss'] for uid, part in p.allocate(split, {'a'}).items() if uid != 1), 100)

    def test_equal_probability_options_give_each_stake_zero_expected_net(self):
        stakes = [
            dict(user_id=1, option_id='a', units=100),
            dict(user_id=2, option_id='b', units=100),
            dict(user_id=3, option_id='b', units=100),
        ]
        outcomes = [p.allocate(stakes, {winner}) for winner in 'abc']
        for uid in (1, 2, 3):
            self.assertEqual(sum(result[uid]['profit'] - result[uid]['loss'] for result in outcomes), 0)

        # Per-option rounding must mirror the loss allocation exactly. Combining
        # both winning pools before rounding would favor one of the equal stakes.
        tiny = [
            dict(user_id=1, option_id='a', units=1),
            dict(user_id=2, option_id='a', units=1),
            dict(user_id=3, option_id='b', units=1),
            dict(user_id=4, option_id='c', units=1),
        ]
        outcomes = [p.allocate(tiny, {winner}) for winner in 'abc']
        for uid in range(1, 5):
            self.assertEqual(sum(result[uid]['profit'] - result[uid]['loss'] for result in outcomes), 0)

    def test_tied_winner_coalition_charges_each_losing_option_once(self):
        stakes = [
            dict(user_id=1, option_id='a', units=100),
            dict(user_id=2, option_id='b', units=500),
        ] + [dict(user_id=uid, option_id='c', units=100) for uid in range(3, 13)]
        result = p.allocate(stakes, {'a', 'b'})
        self.assertEqual(result[1]['profit'], 100)
        self.assertEqual(result[2]['profit'], 500)
        self.assertEqual(sum(result[uid]['loss'] for uid in range(3, 13)), 600)
        self.assertEqual(sum(result[uid]['refund'] for uid in range(3, 13)), 400)

    def test_pairwise_matching_is_zero_ev_with_equal_marginals_and_ties(self):
        stakes = [
            dict(user_id=1, option_id='a', units=100),
            dict(user_id=2, option_id='b', units=900),
            dict(user_id=3, option_id='c', units=100),
        ]
        # Every non-empty winner set occurs once. Each option therefore has the
        # same hit rate, including the three two-way ties and the all-way tie.
        outcomes = [p.allocate(stakes, winners) for winners in (
            {'a'}, {'b'}, {'c'}, {'a','b'}, {'a','c'}, {'b','c'}, {'a','b','c'},
        )]
        for uid in (1, 2, 3):
            self.assertEqual(sum(result[uid]['profit'] - result[uid]['loss']
                                 for result in outcomes), 0)

    def test_pairwise_account_slices_never_use_one_stake_twice(self):
        stakes = [
            dict(user_id=1, option_id='a', units=1),
            dict(user_id=2, option_id='a', units=1),
            dict(user_id=3, option_id='b', units=1),
            dict(user_id=4, option_id='c', units=1),
        ]
        for winners in ({'b'}, {'b','c'}):
            result = p.allocate(stakes, winners)
            for stake in stakes:
                self.assertLessEqual(result[stake['user_id']]['loss'], stake['units'])
                self.assertGreaterEqual(result[stake['user_id']]['refund'], 0)

    def test_all_tie_no_winner_and_technical_void_refund_everything(self):
        for winners,void in [(['lume','clari','vero'],False),(['vero'],False),([],True)]:
            self.batch['id']=str(uuid.uuid4());p.ensure_market('room',self.batch)
            self.bet(1,'lume',1000);self.bet(2,'clari',1000)
            before=self.balances()
            p.settle('room',self.batch['id'],winners,void,now=400)
            after=self.balances()
            self.assertEqual(after[0][0]-before[0][0],1000000)
            self.assertEqual(after[1][0]-before[1][0],1000000)

    def test_retry_after_close_is_success_but_changed_body_is_rejected(self):
        key=str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _:self.bet(1,'lume',request_id=key),range(4)))
        p.close('room',self.batch['id'])
        self.bet(1,'lume',request_id=key,now=400,available=False)
        self.assertEqual(self.balances()[0][0],19900000)
        with self.assertRaises(HTTPException) as conflict:self.bet(1,'clari',request_id=key)
        self.assertEqual(conflict.exception.detail,'prediction_request_conflict')
        with self.assertRaises(HTTPException):self.bet(1,'lume')

    def test_deadline_pause_switch_and_permanent_only(self):
        for when,available in [(280,True),(110,False),(110,lambda:False)]:
            with self.assertRaises(HTTPException):self.bet(1,'lume',now=when,available=available)
        self.bet(1,'lume')
        with self.assertRaises(HTTPException):self.bet(1,'clari')
        with auth_db() as db:db.execute('UPDATE token_accounts SET paid_balance_units=99000 WHERE user_id=2')
        with self.assertRaises(HTTPException) as insufficient:self.bet(2,'lume')
        self.assertEqual(insufficient.exception.status_code,402)
        self.assertEqual(self.balances()[1],(99000,777000))

    def test_room_isolation_and_public_state_contains_no_private_balances(self):
        self.bet(1,'lume')
        self.assertIsNone(p.listing('other')['market'])
        self.assertFalse(p.settle('other',self.batch['id'],['lume']))
        public=p.listing('room',now=120)
        self.assertNotIn('paid_balance_units',public)
        self.assertNotIn('mine',public['market'])
        self.assertNotIn('result',public['market'])
        self.assertEqual(public['market']['options'][0]['count'],1)

    def test_transaction_rolls_back_all_settlement_credits_on_failure(self):
        self.bet(1,'lume');self.bet(2,'clari')
        before=self.balances()
        original=p._credit
        def fail(db,uid,*args):
            original(db,uid,*args)
            raise RuntimeError('simulated crash')
        with patch.object(p,'_credit',side_effect=fail),self.assertRaises(RuntimeError):
            p.settle('room',self.batch['id'],['lume'])
        self.assertEqual(self.balances(),before)
        self.assertTrue(p.settle('room',self.batch['id'],['lume']))

    def test_random_conservation_caps_rounding_and_ties(self):
        rng=random.Random(12345)
        for _ in range(3000):
            stakes=[dict(user_id=i,option_id=rng.choice('abc'),units=rng.randint(1,100000)) for i in range(rng.randint(1,25))]
            winners=set(rng.sample(list('abc'),rng.randint(1,3)))
            result=p.allocate(stakes,winners)
            self.assertEqual(sum(s['units'] for s in stakes),sum(r['principal']+r['profit']+r['refund'] for r in result.values()))
            self.assertEqual(sum(r['profit'] for r in result.values()),sum(r['loss'] for r in result.values()))
            for s in stakes:
                self.assertLessEqual(result[s['user_id']]['profit'],s['units'])
                self.assertLessEqual(result[s['user_id']]['loss'],s['units'])
                self.assertGreaterEqual(result[s['user_id']]['refund'],0)
            self.assertEqual(result,p.allocate(list(reversed(stakes)),winners))

    def test_random_equal_probability_expected_net_is_exactly_zero(self):
        rng = random.Random(54321)
        for _ in range(1000):
            stakes = [dict(user_id=i, option_id=rng.choice('abc'), units=rng.randint(1, 100000))
                      for i in range(rng.randint(1, 25))]
            outcomes = [p.allocate(stakes, {winner}) for winner in 'abc']
            for stake in stakes:
                uid = stake['user_id']
                self.assertEqual(sum(result[uid]['profit'] - result[uid]['loss'] for result in outcomes), 0)

    def test_random_equal_marginals_with_all_tie_shapes_are_exactly_zero(self):
        rng = random.Random(98765)
        winner_sets = (
            {'a'}, {'b'}, {'c'},
            {'a', 'b'}, {'a', 'c'}, {'b', 'c'},
            {'a', 'b', 'c'},
        )
        # Uniformly weighting every non-empty winner set gives every option the
        # same 4/7 hit rate. This covers sole winners, two-way ties and a
        # three-way tie while varying both account splits and option totals.
        for _ in range(1000):
            stakes = [dict(user_id=i, option_id=rng.choice('abc'), units=rng.randint(1, 100000))
                      for i in range(rng.randint(1, 25))]
            outcomes = [p.allocate(stakes, winners) for winners in winner_sets]
            for stake in stakes:
                uid = stake['user_id']
                self.assertEqual(sum(result[uid]['profit'] - result[uid]['loss']
                                     for result in outcomes), 0)


if __name__=='__main__':unittest.main()
