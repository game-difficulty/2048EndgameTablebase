import json
import os
import tempfile
import time
import unittest
from datetime import timedelta
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.admin.routes import router
from backend.auth.db import auth_db, init_auth_db
from backend.auth.service import create_session, public_user, iso, utcnow
from backend.gifts.identity import supporter_level
from backend.human_play.store import database, init_db
from backend.quota.service import grant_weekly_tokens_if_due
from competition.backend.domain import Principal


class ModeratorPermissionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'CLOUD_AUTH_DB': self.temp.name + '/auth.db',
            'HUMAN_PLAY_DB': self.temp.name + '/human.db',
            'ADMIN_ALLOWED_IDENTITIES': 'owner@test.invalid',
        })
        self.env.start()
        init_auth_db()
        init_db()
        self.tokens = {}
        with auth_db() as db:
            for ident, role in [(1, 'user'), (2, 'moderator'), (3, 'moderator'),
                                (4, 'user'), (5, 'user'), (6, 'admin'), (7, 'organizer')]:
                email = 'owner@test.invalid' if ident == 1 else f'user{ident}@test.invalid'
                db.execute('''INSERT INTO users(id,email,password_hash,display_name,role,
                    registered_with_invite,created_at,updated_at) VALUES(?,?,'!',?,?,0,?,?)''',
                    (ident, email, f'Account {ident}', role, iso(), iso()))
                public_user(db.execute('SELECT * FROM users WHERE id=?', (ident,)).fetchone(), db=db)
                self.tokens[ident] = create_session(db, ident)[0]
            db.execute("UPDATE user_entitlements SET tier='supporter' WHERE user_id=5")
        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)
        self.actor(2)

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.temp.cleanup()

    def actor(self, ident):
        self.client.headers['Authorization'] = 'Bearer ' + self.tokens[ident]

    def test_permissions_and_public_identity(self):
        self.assertEqual(self.client.get('/api/admin/permissions').json(),
                         {'owner': False, 'moderate': True, 'user_id': 2})
        with auth_db() as db:
            user = public_user(db.execute('SELECT * FROM users WHERE id=2').fetchone(), db=db)
        self.assertTrue(user['management']['moderate'])
        self.assertFalse(user['management']['owner'])

    def test_regular_supporter_and_legacy_roles_do_not_gain_main_site_management(self):
        for ident in (4, 5, 6, 7):
            self.actor(ident)
            for path in ('permissions', 'users', 'overview', 'approval-transactions', 'profile-reviews'):
                with self.subTest(ident=ident, path=path):
                    self.assertEqual(self.client.get('/api/admin/' + path).status_code, 403)

    def test_restricted_get_and_write_endpoints_are_forbidden(self):
        for path in ('overview', 'live', 'tablebase-workers'):
            self.assertEqual(self.client.get('/api/admin/' + path).status_code, 403)
        for path, body in [('live', {'enabled': True}), ('live/batch/void', {}),
                           ('users/4/tokens', {'tokens': 10, 'set_supporter': True}),
                           ('users/4/moderator', {'enabled': True}),
                           ('users/4/managed-password', {'new_password': 'not-a-real-password'})]:
            with self.subTest(path=path):
                self.assertEqual(self.client.post('/api/admin/' + path, json=body).status_code, 403)

    def test_users_response_has_no_financial_or_global_statistics(self):
        response = self.client.get('/api/admin/users?page_size=2').json()
        self.assertEqual(set(response), {'users', 'users_page'})
        self.assertEqual(response['users_page'], {'page': 1, 'page_size': 2, 'has_more': True})
        for user in response['users']:
            self.assertFalse({'token_balance', 'sessions', 'usage_events', 'managed_test_account'} & set(user))
        last = self.client.get('/api/admin/users?page_size=2&page=4').json()
        self.assertFalse(last['users_page']['has_more'])

    def test_protected_status_targets_including_self(self):
        for ident in (1, 2, 3, 6):
            response = self.client.post(f'/api/admin/users/{ident}/status', json={'status': 'disabled'})
            self.assertIn(response.status_code, (400, 403))
        with auth_db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM users WHERE status='disabled'").fetchone()[0], 0)

    def test_disable_enable_normal_and_supporter_and_audit(self):
        for ident in (4, 5):
            for status in ('disabled', 'active'):
                response = self.client.post(f'/api/admin/users/{ident}/status', json={'status': status})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['user']['status'], status)
                self.assertNotIn('token_balance', response.json()['user'])
        with auth_db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM management_audit WHERE action='status'").fetchone()[0], 4)
            self.assertIsNotNone(db.execute('SELECT revoked_at FROM sessions WHERE user_id=4').fetchone()[0])

    def test_owner_appoint_revoke_is_immediate_and_preserves_paid_and_sponsorship(self):
        self.actor(1)
        with auth_db() as db:
            db.execute('UPDATE token_accounts SET paid_balance_units=123000 WHERE user_id=5')
        result = self.client.post('/api/admin/users/5/moderator', json={'enabled': True})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['user']['token_balance'], {'bonus': 131072, 'paid': 123, 'total': 131195})
        self.actor(5)
        self.assertEqual(self.client.get('/api/admin/permissions').status_code, 200)
        self.actor(1)
        self.assertEqual(self.client.post('/api/admin/users/5/moderator', json={'enabled': True}).status_code, 200)
        result = self.client.post('/api/admin/users/5/moderator', json={'enabled': False}).json()['user']
        self.assertEqual(result['role'], 'user')
        self.assertTrue(result['entitlements']['is_supporter'])
        self.assertEqual(result['token_balance']['bonus'], 32768)
        self.assertEqual(result['token_balance']['paid'], 123)
        self.actor(5)
        self.assertEqual(self.client.get('/api/admin/permissions').status_code, 403)
        with auth_db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM management_audit WHERE action='role'").fetchone()[0], 2)

    def test_role_endpoint_cannot_overwrite_owner_or_other_existing_roles(self):
        self.actor(1)
        for ident in (1, 6, 7):
            self.assertEqual(self.client.post(f'/api/admin/users/{ident}/moderator', json={'enabled': True}).status_code, 403)

    def test_weekly_cap_is_not_additive_and_paid_balance_is_unchanged(self):
        from backend.quota.routes import public_quota_rules
        self.assertEqual(public_quota_rules()['weekly_grants']['moderator'], 131072)
        with auth_db() as db:
            db.execute("UPDATE user_entitlements SET tier='supporter' WHERE user_id=2")
            db.execute('UPDATE token_accounts SET bonus_balance_units=1000, paid_balance_units=99000, last_weekly_grant_at=? WHERE user_id=2',
                       (iso(utcnow() - timedelta(days=8)),))
        balance = grant_weekly_tokens_if_due(2)
        self.assertEqual(balance['bonus'], 131072)
        self.assertEqual(balance['paid'], 99)
        self.assertEqual(grant_weekly_tokens_if_due(2), balance)

    def test_moderator_role_does_not_confer_competition_or_sponsor_privileges(self):
        self.assertFalse(Principal(user_id=2, display_name='Moderator', site_role='moderator').is_platform_organizer)
        self.assertEqual(supporter_level({'id': 2, 'role': 'moderator'}), 0)

    def test_actual_sponsor_remains_eligible_for_supporter_board_when_moderator(self):
        from backend.leaderboards.service import _supporter_rows
        self.actor(1)
        self.assertEqual(self.client.post('/api/admin/users/5/moderator', json={'enabled': True}).status_code, 200)
        with auth_db() as db:
            self.assertIn(5, {row['user_id'] for row in _supporter_rows(db)})
        self.actor(2)
        result = self.client.get('/api/admin/users?tier=moderator').json()
        self.assertEqual({row['id'] for row in result['users']}, {2, 3, 5})

    def add_claim(self, ident):
        with database() as db:
            db.execute('''INSERT INTO human_external_claims
                (id,user_id,provider,username,username_key,status,requested,updated,counts)
                VALUES(?,?,'verse',?,?,'pending',?,?,'{}')''',
                (ident, ident, f'player{ident}', f'player{ident}', time.time(), time.time()))

    def test_approval_lists_remove_actions_for_protected_users_and_writes_recheck(self):
        for ident in (1, 2, 3, 4):
            self.add_claim(ident)
        queue = self.client.get('/api/admin/approval-transactions').json()['transactions']
        self.assertEqual({row['user_id'] for row in queue if row['actions']}, {4})
        for ident in (1, 2, 3):
            for action, body in [('decision', {'approved': False, 'note': 'test'}), ('retry', {}), ('revoke', {})]:
                response = self.client.post(f'/api/admin/verse-claims/{ident}/{action}', json=body)
                self.assertEqual(response.status_code, 403, response.text)
        result = self.client.post('/api/admin/verse-claims/4/decision', json={'approved': False, 'note': 'Reviewed'})
        self.assertEqual(result.status_code, 200, result.text)

    def test_profile_review_and_bulk_skip_protected_users(self):
        with auth_db() as db:
            for ident in (1, 2, 3, 4):
                db.execute('''INSERT INTO user_profile_change_events
                    (id,user_id,change_type,old_value,new_value,ip_address,user_agent,created_at)
                    VALUES(?,?,'display_name','old',?,'','test',?)''', (ident, ident, f'Account {ident}', iso()))
                db.execute("INSERT INTO profile_change_reviews(event_id,status) VALUES(?,'pending')", (ident,))
        queue = self.client.get('/api/admin/profile-reviews').json()
        self.assertEqual(queue['pending_total'], 1)
        self.assertEqual({item['user_id'] for item in queue['items'] if item['can_manage']}, {4})
        for ident in (1, 2, 3):
            self.assertEqual(self.client.post(f'/api/admin/profile-reviews/{ident}', json={'action': 'revoke'}).status_code, 403)
        response = self.client.post('/api/admin/profile-reviews/approve-pending').json()
        self.assertEqual(response, {'updated': 1, 'skipped': 3})
        response = self.client.post('/api/admin/profile-reviews/4', json={'action': 'revoke'})
        self.assertEqual(response.status_code, 200, response.text)
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT display_name FROM users WHERE id=4').fetchone()[0], 'User4')

    def test_archive_and_assistance_permissions_cover_all_write_routes(self):
        with database() as db:
            for ident in (1, 2, 3, 4):
                db.execute('''INSERT INTO human_archive_applications
                    (id,user_id,variant,claimed_ended_at,claimed_score,status,replay_crc,replay_size,
                     moves,final_board_json,is_game_over,timing_summary_json,warning_flags_json,
                     original_filename,requested_at,updated_at)
                    VALUES(?,?,'4x4',1,100,'pending',1,100,10,?,1,'{}','[]','test.vrs',?,?)''',
                    (ident, ident, json.dumps([0] * 16), time.time(), time.time()))
                db.execute('''INSERT INTO human_runs
                    (id,user_id,browser,variant,request_id,seed,threshold,created,writer,state)
                    VALUES(?,?,'browser','4x4','request','seed',0,1,'writer','{}')''', (str(ident), ident))
                db.execute('''INSERT INTO human_assistance_reviews
                    (id,run_id,user_id,requested_at,updated_at,details) VALUES(?,?,?,1,1,'{}')''',
                    (ident, str(ident), ident))
        for ident in (1, 2, 3):
            for path in (f'archive-applications/{ident}/decision', f'archive-applications/{ident}/revoke',
                         f'assistance-reviews/{ident}/decision'):
                self.assertEqual(self.client.post('/api/admin/' + path,
                    json={'approved': False, 'note': 'review'}).status_code, 403)
        for kind in ('archive-applications', 'assistance-reviews'):
            response = self.client.post(f'/api/admin/{kind}/4/decision', json={'approved': False, 'note': 'review'})
            self.assertEqual(response.status_code, 200, response.text)

    def test_owner_retains_overview_and_can_revoke_moderator(self):
        self.actor(1)
        response = self.client.get('/api/admin/overview').json()
        self.assertIn('users_total', response['summary'])
        self.assertIn('token_activity', response)
        self.assertIn('token_balance', response['users'][0])
        self.assertEqual(self.client.post('/api/admin/users/2/moderator', json={'enabled': False}).status_code, 200)
        self.actor(2)
        self.assertEqual(self.client.get('/api/admin/users').status_code, 403)

    def test_cross_origin_role_write_is_rejected(self):
        self.actor(1)
        response = self.client.post('/api/admin/users/4/moderator', json={'enabled': True},
                                    headers={'Origin': 'https://untrusted.invalid'})
        self.assertEqual(response.status_code, 403)


if __name__ == '__main__':
    unittest.main()
