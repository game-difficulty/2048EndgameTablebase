import unittest
from unittest.mock import patch

from backend.actions import Action
from backend.cloud_safety import cloud_payload_error, is_cloud_action_blocked


class CloudBoardOwnershipTests(unittest.TestCase):
    def test_legacy_trainer_and_tester_board_mutations_are_blocked(self):
        with patch.dict("os.environ", {"CLOUD_MODE": "1", "APP_MODE": "cloud"}):
            for action in (
                Action.GET_STATE,
                Action.TRAINER_MOVE,
                Action.TRAINER_MANUAL_SPAWN,
                Action.TRAINER_STEP,
                Action.TESTER_MOVE,
                Action.TESTER_SET_BOARD,
                Action.TESTER_SET_TEXT_VISIBLE,
                Action.SET_BOARD,
                Action.SET_CELL,
                Action.UNDO,
                Action.SET_SPAWN_MODE,
                Action.ROTATE,
            ):
                with self.subTest(action=action):
                    self.assertTrue(is_cloud_action_blocked(action))

    def test_stateless_query_actions_remain_available(self):
        with patch.dict("os.environ", {"CLOUD_MODE": "1", "APP_MODE": "cloud"}):
            self.assertFalse(is_cloud_action_blocked(Action.TABLEBASE_QUERY))
            self.assertFalse(is_cloud_action_blocked(Action.TRAINER_SPAWN_QUERY))

    def test_board_related_cloud_actions_require_client_owned_protocol(self):
        with patch.dict("os.environ", {"CLOUD_MODE": "1", "APP_MODE": "cloud"}):
            for action in (
                Action.TRAINER_SET_EMPTY_PATTERN,
                Action.TRAINER_SET_FILEPATH,
                Action.TRAINER_DEFAULT,
                Action.TESTER_GET_INIT,
                Action.TESTER_SELECT_PATTERN,
                Action.TESTER_RESET_RANDOM,
                Action.TABLEBASE_QUERY,
            ):
                with self.subTest(action=action):
                    self.assertIsNotNone(cloud_payload_error(action, {}))
                    self.assertIsNotNone(
                        cloud_payload_error(action, {"client_local_board": False})
                    )
                    self.assertIsNone(
                        cloud_payload_error(action, {"client_local_board": True})
                    )

    def test_desktop_mode_keeps_legacy_payloads_available(self):
        with patch.dict("os.environ", {"CLOUD_MODE": "0", "APP_MODE": "desktop"}):
            self.assertIsNone(cloud_payload_error(Action.TABLEBASE_QUERY, {}))


if __name__ == "__main__":
    unittest.main()
