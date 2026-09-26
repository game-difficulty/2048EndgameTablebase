import os
import tempfile
import unittest
from pathlib import Path

from backend.auth.db import auth_db, init_auth_db
from backend.chat_moderation import blocked, visible_messages


class ChatModerationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.previous = os.environ.get('CLOUD_AUTH_DB')
        os.environ['CLOUD_AUTH_DB'] = str(Path(self.directory.name) / 'auth.sqlite3')
        init_auth_db()

    def tearDown(self):
        if self.previous is None:
            os.environ.pop('CLOUD_AUTH_DB', None)
        else:
            os.environ['CLOUD_AUTH_DB'] = self.previous
        self.directory.cleanup()

    def test_normalized_blocklist(self):
        self.assertTrue(blocked('出 售 雷 管'))
        self.assertTrue(blocked('成，人，论，坛'))
        self.assertTrue(blocked('办 假 证'))
        self.assertFalse(blocked('讨论 2048 的走法'))
        self.assertFalse(blocked('这局先合并小数再查表'))
        self.assertFalse(blocked('和弦、按摩、兼职都是普通词'))

    def test_disabled_user_history_is_hidden_by_id_only(self):
        with auth_db() as db:
            db.execute("INSERT INTO users(id,email,email_identity,password_hash,display_name,display_name_key,role,status,created_at,updated_at) VALUES (1,'a@a.cn','a@a.cn','x','Alice','alice','user','disabled','2026-01-01','2026-01-01')")
            db.execute("INSERT INTO users(id,email,email_identity,password_hash,display_name,display_name_key,role,status,created_at,updated_at) VALUES (2,'b@a.cn','b@a.cn','x','Bob','bob','user','active','2026-01-01','2026-01-01')")
        messages = [dict(type='chat', user_id=1, text='hidden'),
                    dict(type='chat', user_id=2, text='visible'),
                    dict(type='chat', guest=True, name='Alice', text='guest'),
                    dict(type='chat', name='Alice', text='legacy')]
        self.assertEqual([item['text'] for item in visible_messages(messages)], ['visible', 'guest', 'legacy'])


if __name__ == '__main__':
    unittest.main()
