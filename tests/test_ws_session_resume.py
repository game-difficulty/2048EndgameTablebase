import unittest

from backend.state import ConnectionManager
from backend.session import GameSession


class WebSocketSessionResumeTests(unittest.TestCase):
    def test_detached_session_restores_for_same_user_and_client(self):
        manager = ConnectionManager()
        first_websocket = object()
        session = GameSession("trainer_resume_test")
        session.user_id = 7
        session.auth_session_id = 71
        session.board_encoded = 0x1234
        manager.active_connections[first_websocket] = session

        manager.disconnect(first_websocket)

        next_websocket = object()
        manager.active_connections[next_websocket] = GameSession("trainer_resume_test")
        restored = manager.restore_detached_session(
            next_websocket,
            {"id": 7, "session_id": 72, "email": "user@example.com", "role": "user"},
        )

        self.assertIs(restored, session)
        self.assertIs(manager.active_connections[next_websocket], session)
        self.assertEqual(int(restored.board_encoded), 0x1234)

    def test_detached_session_does_not_restore_for_different_user(self):
        manager = ConnectionManager()
        first_websocket = object()
        session = GameSession("trainer_resume_test")
        session.user_id = 7
        session.board_encoded = 0x1234
        manager.active_connections[first_websocket] = session

        manager.disconnect(first_websocket)

        next_websocket = object()
        fresh = GameSession("trainer_resume_test")
        manager.active_connections[next_websocket] = fresh
        restored = manager.restore_detached_session(
            next_websocket,
            {"id": 8, "session_id": 81, "email": "other@example.com", "role": "user"},
        )

        self.assertIs(restored, fresh)
        self.assertIs(manager.active_connections[next_websocket], fresh)


if __name__ == "__main__":
    unittest.main()
