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
