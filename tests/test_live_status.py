import time
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.live import routes


class LiveStatusTests(unittest.TestCase):
    def test_status_is_lightweight_and_tracks_publisher_state(self):
        runtime = routes.LiveHub()
        app = FastAPI()
        app.include_router(routes.router)
        with patch.object(routes, 'hub', runtime), patch.object(runtime, 'snapshot', side_effect=AssertionError('heavy snapshot')), TestClient(app) as client:
            def check(expected):
                response = client.get('/api/live/status')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'online': expected})
                self.assertEqual(response.headers['cache-control'], 'no-store')

            check(False)
            runtime.producer = Mock()
            runtime.last_seen = time.monotonic()
            check(False)
            runtime.producer_ready = True
            check(True)
            runtime.control['enabled'] = False
            check(False)
            runtime.control['enabled'] = True
            runtime.control_supported = True
            check(False)
            runtime.control_ack = runtime.control['revision']
            check(True)
            runtime.last_seen = time.monotonic() - 21
            check(False)
            runtime.producer = None
            check(False)
