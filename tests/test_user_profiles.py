from __future__ import annotations

from datetime import datetime, timedelta, timezone
import io
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from backend.auth.db import auth_db, init_auth_db
from backend.profile.service import (
    DisplayNameTakenError,
    ProfileCooldownError,
    public_profile,
    update_avatar,
    update_display_name,
)
from backend.profile.storage import process_avatar_bytes
from backend.profile.reviews import list_reviews, decide_review, review_all_pending


class UserProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.old_db = os.environ.get("CLOUD_AUTH_DB")
        self.old_avatar_root = os.environ.get("CLOUD_AVATAR_ROOT")
        os.environ["CLOUD_AUTH_DB"] = str(Path(self.tempdir.name) / "auth.sqlite3")
        os.environ["CLOUD_AVATAR_ROOT"] = str(Path(self.tempdir.name) / "avatars")
        init_auth_db()

    def tearDown(self) -> None:
        if self.old_db is None:
            os.environ.pop("CLOUD_AUTH_DB", None)
        else:
            os.environ["CLOUD_AUTH_DB"] = self.old_db
        if self.old_avatar_root is None:
            os.environ.pop("CLOUD_AVATAR_ROOT", None)
        else:
            os.environ["CLOUD_AVATAR_ROOT"] = self.old_avatar_root
        self.tempdir.cleanup()

    def _create_user(self, name: str, *, age_days: int = 31) -> int:
        created_at = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        with auth_db() as db:
            cursor = db.execute(
                """
                INSERT INTO users
                (email, email_identity, password_hash, display_name, display_name_key,
                 role, status, created_at, updated_at)
                VALUES (?, ?, 'hash', ?, ?, 'user', 'active', ?, ?)
                """,
                (
                    f"{name.casefold()}@example.com",
                    f"{name.casefold()}@example.com",
                    name,
                    name.casefold(),
                    created_at,
                    created_at,
                ),
            )
            user_id = int(cursor.lastrowid)
            db.execute(
                """
                INSERT INTO user_entitlements
                (user_id, tier, can_upload_avatar, created_at, updated_at)
                VALUES (?, 'free', 1, ?, ?)
                """,
                (user_id, created_at, created_at),
            )
            return user_id

    @staticmethod
    def _image_bytes(color: tuple[int, int, int]) -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (320, 180), color).save(output, format="PNG")
        return output.getvalue()

    def test_new_profile_changes_are_reviewable_and_reset_keeps_cooldown(self) -> None:
        user_id = self._create_user('ReviewTarget')
        self.assertTrue(update_display_name(user_id, 'ChangedName'))
        reviews = list_reviews()
        name_event = next(item for item in reviews['items'] if item['change_type'] == 'display_name')
        self.assertTrue(name_event['is_current'])
        self.assertEqual(decide_review(name_event['id'], 'revoke', user_id)['status'], 'revoked')
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT display_name FROM users WHERE id=?', (user_id,)).fetchone()[0], f'User{user_id}')
            self.assertIsNotNone(db.execute('SELECT display_name_changed_at FROM user_profiles WHERE user_id=?', (user_id,)).fetchone()[0])

        avatar = process_avatar_bytes(self._image_bytes((12, 34, 56)))
        self.assertTrue(update_avatar(user_id, avatar))
        avatar_event = next(item for item in list_reviews()['items'] if item['change_type'] == 'avatar')
        self.assertTrue(avatar_event['is_current'])
        self.assertEqual(decide_review(avatar_event['id'], 'keep', user_id)['status'], 'reviewed')
        self.assertEqual(decide_review(avatar_event['id'], 'revoke', user_id)['status'], 'revoked')
        with auth_db() as db:
            profile = db.execute('SELECT avatar_key, avatar_changed_at FROM user_profiles WHERE user_id=?', (user_id,)).fetchone()
            self.assertIsNone(profile['avatar_key'])
            self.assertIsNotNone(profile['avatar_changed_at'])

    def test_old_review_cannot_revoke_a_later_avatar(self) -> None:
        user_id = self._create_user('AvatarVersions')
        self.assertTrue(update_avatar(user_id, process_avatar_bytes(self._image_bytes((1, 2, 3)))))
        old_event = list_reviews()['items'][0]['id']
        with auth_db() as db:
            db.execute('UPDATE user_profiles SET avatar_changed_at=? WHERE user_id=?',
                       ((datetime.now(timezone.utc) - timedelta(days=31)).isoformat(), user_id))
        self.assertTrue(update_avatar(user_id, process_avatar_bytes(self._image_bytes((4, 5, 6)))))
        with auth_db() as db:
            current = db.execute('SELECT avatar_key FROM user_profiles WHERE user_id=?', (user_id,)).fetchone()[0]
        self.assertEqual(decide_review(old_event, 'revoke', user_id)['status'], 'superseded')
        with auth_db() as db:
            self.assertEqual(db.execute('SELECT avatar_key FROM user_profiles WHERE user_id=?', (user_id,)).fetchone()[0], current)

    def test_review_list_filters_and_sorts_by_change_time(self) -> None:
        first_user = self._create_user('ReviewFirst')
        second_user = self._create_user('ReviewSecond')
        self.assertTrue(update_display_name(first_user, 'FirstChanged'))
        self.assertTrue(update_display_name(second_user, 'SecondChanged'))
        self.assertTrue(update_avatar(first_user, process_avatar_bytes(self._image_bytes((14, 28, 42)))))

        all_items = list_reviews(status='all')['items']
        first_name = next(item for item in all_items if item['new_value'] == 'FirstChanged')
        with auth_db() as db:
            db.execute('UPDATE user_profile_change_events SET created_at=? WHERE id=?',
                       ('2030-01-01T00:00:00+00:00', first_name['id']))

        self.assertEqual(list_reviews(status='all')['items'][0]['id'], first_name['id'])
        names = list_reviews(status='all', change_type='display_name')['items']
        self.assertEqual(len(names), 2)
        self.assertTrue(all(item['change_type'] == 'display_name' for item in names))
        self.assertEqual([item['id'] for item in list_reviews(status='all', query='reviewfirst@example.com')['items']][:1],
                         [first_name['id']])
        self.assertEqual(list_reviews(status='pending', change_type='avatar')['total'], 1)
        self.assertEqual(decide_review(first_name['id'], 'keep', first_user)['status'], 'reviewed')
        self.assertEqual(list_reviews(status='reviewed', change_type='display_name')['items'][0]['id'], first_name['id'])

    def test_review_all_pending_only_updates_pending_records(self) -> None:
        users = [self._create_user(name) for name in ('BulkFirst', 'BulkSecond', 'BulkThird')]
        for user_id, name in zip(users, ('FirstChange', 'SecondChange', 'ThirdChange')):
            self.assertTrue(update_display_name(user_id, name))
        reviews = {item['user_id']: item for item in list_reviews(status='all')['items']}
        self.assertEqual(decide_review(reviews[users[0]]['id'], 'keep', users[0])['status'], 'reviewed')
        self.assertEqual(decide_review(reviews[users[1]]['id'], 'revoke', users[0])['status'], 'revoked')
        self.assertEqual(list_reviews(status='all')['pending_total'], 1)

        self.assertEqual(review_all_pending(users[0]), {'updated': 1})
        self.assertEqual(review_all_pending(users[0]), {'updated': 0})
        self.assertEqual(list_reviews(status='all')['pending_total'], 0)
        with auth_db() as db:
            rows = db.execute('''SELECT event_id, status, reviewed_by, reviewed_at
                FROM profile_change_reviews''').fetchall()
        by_event = {row['event_id']: row for row in rows}
        self.assertEqual(by_event[reviews[users[0]]['id']]['status'], 'reviewed')
        self.assertEqual(by_event[reviews[users[1]]['id']]['status'], 'revoked')
        third = by_event[reviews[users[2]]['id']]
        self.assertEqual(third['status'], 'reviewed')
        self.assertEqual(third['reviewed_by'], users[0])
        self.assertIsNotNone(third['reviewed_at'])

    def test_display_name_is_unique_and_starts_independent_cooldown(self) -> None:
        alice = self._create_user("Alice")
        self._create_user("Bob")

        self.assertTrue(update_display_name(alice, "New Alice"))
        with self.assertRaises(ProfileCooldownError):
            update_display_name(alice, "Another Alice")

        charlie = self._create_user("Charlie")
        with self.assertRaises(DisplayNameTakenError):
            update_display_name(charlie, "new alice")

        with auth_db() as db:
            entry = db.execute(
                "SELECT change_type, old_value, new_value FROM user_profile_change_events WHERE user_id = ?",
                (alice,),
            ).fetchone()
            self.assertEqual(dict(entry), {
                "change_type": "display_name",
                "old_value": "Alice",
                "new_value": "New Alice",
            })

    def test_new_account_username_cooldown_uses_created_at(self) -> None:
        user_id = self._create_user("Recent", age_days=1)
        profile = public_profile(user_id)

        self.assertFalse(profile["can_change_display_name"])
        self.assertIsNotNone(profile["display_name_change_available_at"])
        self.assertTrue(profile["can_change_avatar"])

    def test_avatar_is_reencoded_cached_and_cooldown_is_enforced(self) -> None:
        user_id = self._create_user("Avatar User", age_days=1)
        avatar = process_avatar_bytes(self._image_bytes((240, 40, 80)))

        self.assertTrue(update_avatar(user_id, avatar))
        profile = public_profile(user_id)
        self.assertTrue(profile["avatar_url"].startswith(f"/media/avatars/{user_id}/"))
        self.assertFalse(profile["can_change_avatar"])

        avatar_path = Path(os.environ["CLOUD_AVATAR_ROOT"]) / profile["avatar_url"].split("/media/avatars/", 1)[1]
        self.assertTrue(avatar_path.is_file())
        with Image.open(avatar_path) as stored:
            self.assertEqual(stored.size, (128, 128))
            self.assertEqual(stored.format, "WEBP")

        self.assertFalse(update_avatar(user_id, avatar))
        other = process_avatar_bytes(self._image_bytes((20, 120, 220)))
        with self.assertRaises(ProfileCooldownError):
            update_avatar(user_id, other)


if __name__ == "__main__":
    unittest.main()
