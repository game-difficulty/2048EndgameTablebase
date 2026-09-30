import asyncio
import sqlite3
import unittest
from unittest.mock import AsyncMock, Mock, patch

from backend.validation_worker import run_validation_loop


class ValidationWorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_failure_does_not_block_validation(self):
        cleanup = Mock(side_effect=sqlite3.OperationalError('database is locked'))
        process = Mock(return_value=True)
        with patch('backend.validation_worker.asyncio.sleep',
                   new=AsyncMock(side_effect=asyncio.CancelledError)), self.assertLogs(
                       'backend.validation_worker', level='ERROR'):
            with self.assertRaises(asyncio.CancelledError):
                await run_validation_loop('test', cleanup=cleanup,
                                          process=process, recover=Mock())
        process.assert_called_once()

    async def test_processing_and_recovery_failures_are_retried(self):
        events = []
        def process():
            events.append('process')
            if events.count('process') == 1:
                raise sqlite3.OperationalError('database is locked')
            return True
        def recover():
            events.append('recover')
            if events.count('recover') == 1:
                raise RuntimeError('recovery busy')
        sleep = AsyncMock(side_effect=[None, None, asyncio.CancelledError])
        with patch('backend.validation_worker.asyncio.sleep', new=sleep), self.assertLogs(
                'backend.validation_worker', level='ERROR'):
            with self.assertRaises(asyncio.CancelledError):
                await run_validation_loop('test', cleanup=Mock(),
                                          process=process, recover=recover)
        self.assertEqual(events, ['process', 'recover', 'recover', 'process'])
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 5, 0.05])

    async def test_cancellation_is_not_retried(self):
        recover = Mock()
        with self.assertRaises(asyncio.CancelledError):
            await run_validation_loop('test', cleanup=Mock(),
                                      process=Mock(side_effect=asyncio.CancelledError),
                                      recover=recover)
        recover.assert_not_called()
