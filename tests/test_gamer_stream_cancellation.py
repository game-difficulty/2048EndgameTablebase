import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import gamer_stream_window as windows
from backend.gamer_tablebase_stream import GamerStreamService


class TrackingEvent(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.waiters = 0

    async def wait(self):
        self.waiters += 1
        try:
            return await super().wait()
        finally:
            self.waiters -= 1


class StreamCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_event_completion_and_timeout_clean_up_waiter(self):
        event = TrackingEvent()
        with self.assertRaises(asyncio.TimeoutError):
            await windows.wait_stream_event(event, .01)
        self.assertEqual(event.waiters, 0)
        event.set()
        self.assertTrue(await windows.wait_stream_event(event, .1))
        self.assertEqual(event.waiters, 0)

    async def test_cancel_pending_event_cleans_up_waiter(self):
        event = TrackingEvent()
        task = asyncio.create_task(windows.wait_stream_event(event, 90))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.assertEqual(event.waiters, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(event.waiters, 0)

    async def test_stop_during_completed_attach_wait_does_not_wait_for_idle(self):
        # Exercise the scheduling turns that lose cancellation on Python 3.10.
        for turns in (1, 2, 3):
            with self.subTest(turns=turns):
                event = TrackingEvent()
                event.set()
                window = windows.StreamWindow(7)
                window.produced = 7
                continued = []

                async def produce():
                    await windows.wait_stream_event(event, 90)
                    continued.append(True)
                    await window.wait(8)

                service = GamerStreamService()
                task = asyncio.create_task(produce())
                service.routes['old'] = SimpleNamespace(task=task)
                try:
                    for _ in range(turns):
                        await asyncio.sleep(0)
                    # A swallowed cancellation waits for this full idle timeout.
                    with patch.object(windows, 'STREAM_IDLE_SECONDS', .3):
                        started = asyncio.get_running_loop().time()
                        await service.stop('old')
                        self.assertLess(asyncio.get_running_loop().time() - started, .15)
                    self.assertTrue(task.cancelled())
                    self.assertEqual(continued, [])
                    self.assertEqual(event.waiters, 0)
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    await service.close()

    async def test_cancel_concurrent_with_credit_does_not_start_more_work(self):
        window = windows.StreamWindow(0)
        window.produced = 0
        continued = []

        async def produce():
            await window.wait(1)
            continued.append(True)

        task = asyncio.create_task(produce())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        window.update(0, 1)
        # Let the event waiter complete, but cancel before the producer resumes.
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(continued, [])


if __name__ == '__main__':
    unittest.main()
