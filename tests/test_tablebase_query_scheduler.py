import asyncio
import threading
import unittest
from unittest.mock import patch

import numpy as np

from backend.tablebase_query_service import (
    TablebaseLookupSpec,
    TablebaseQueryScheduler,
    TablebaseQueryOverloaded,
    TablebaseQuerySuperseded,
)
from engine_core.VBoardMover import encode_board


def board_with_tile(index: int, value: int) -> int:
    board = np.zeros((4, 4), dtype=np.int32)
    row, col = divmod(index, 4)
    board[row, col] = value
    return int(encode_board(board))


class RecordingReader:
    def __init__(self, *, blocked_board: int | None = None):
        self.blocked_board = blocked_board
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def move_on_dic(self, board, pattern, target, full_pattern):
        del pattern, target, full_pattern
        encoded = int(encode_board(np.asarray(board, dtype=np.int32)))
        self.calls.append(encoded)
        if encoded == self.blocked_board:
            self.started.set()
            if not self.release.wait(timeout=2):
                raise TimeoutError("blocked query was not released")
        return {
            "left": float(encoded & 0xFFFF) / 65535,
            "right": 0.5,
            "down": 0.4,
            "up": 0.3,
        }, "float64"


def make_spec(reader, board_encoded, *, version="catalog-test"):
    return TablebaseLookupSpec(
        board_encoded=board_encoded,
        pattern="L3",
        target="256",
        full_pattern="L3_256",
        use_variant=False,
        book_reader=reader,
        catalog_version=version,
    )


class TablebaseQuerySchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schedulers = []

    async def asyncTearDown(self):
        for scheduler in self.schedulers:
            await scheduler.close()

    def scheduler(self, workers=1):
        scheduler = TablebaseQueryScheduler(worker_count=workers)
        self.schedulers.append(scheduler)
        return scheduler

    async def test_same_query_is_computed_once_for_multiple_streams(self):
        scheduler = self.scheduler(workers=2)
        reader = RecordingReader()
        spec = make_spec(reader, board_with_tile(0, 2))
        first = await scheduler.submit(
            spec,
            stream_key="user-a:tester",
            supporter=False,
        )
        second = await scheduler.submit(
            spec,
            stream_key="user-b:trainer",
            supporter=True,
        )
        first_result, second_result = await asyncio.gather(first.wait(), second.wait())
        self.assertEqual(first_result.results, second_result.results)
        self.assertEqual(reader.calls, [spec.board_encoded])

    async def test_new_request_supersedes_older_pending_request_for_stream(self):
        scheduler = self.scheduler()
        blocker_board = board_with_tile(0, 2)
        old_board = board_with_tile(1, 4)
        new_board = board_with_tile(2, 8)
        reader = RecordingReader(blocked_board=blocker_board)

        blocker = await scheduler.submit(
            make_spec(reader, blocker_board),
            stream_key="blocker",
            supporter=False,
        )
        self.assertTrue(await asyncio.to_thread(reader.started.wait, 1))
        old = await scheduler.submit(
            make_spec(reader, old_board),
            stream_key="same-user:tester",
            supporter=False,
        )
        new = await scheduler.submit(
            make_spec(reader, new_board),
            stream_key="same-user:tester",
            supporter=False,
        )
        with self.assertRaises(TablebaseQuerySuperseded):
            await asyncio.wait_for(old.wait(), timeout=0.2)
        reader.release.set()
        await blocker.wait()
        await new.wait()
        self.assertNotIn(old_board, reader.calls)
        self.assertIn(new_board, reader.calls)

    async def test_foreground_request_promotes_same_pending_prefetch_job(self):
        scheduler = self.scheduler()
        blocker_board = board_with_tile(0, 2)
        target_board = board_with_tile(1, 4)
        reader = RecordingReader(blocked_board=blocker_board)

        blocker = await scheduler.submit(
            make_spec(reader, blocker_board),
            stream_key="blocker",
            supporter=False,
        )
        self.assertTrue(await asyncio.to_thread(reader.started.wait, 1))
        prefetch = await scheduler.submit(
            make_spec(reader, target_board),
            stream_key="same-user:trainer",
            supporter=False,
            lane="prefetch",
            supersede=False,
            generation=scheduler.current_generation("same-user:trainer"),
            allow_overload=True,
        )
        foreground = await scheduler.submit(
            make_spec(reader, target_board),
            stream_key="same-user:trainer",
            supporter=False,
            lane="foreground",
            supersede=True,
        )

        reader.release.set()
        await blocker.wait()
        with self.assertRaises(TablebaseQuerySuperseded):
            await prefetch.wait()
        result = await foreground.wait()
        self.assertEqual(result.board_encoded, target_board)
        self.assertEqual(reader.calls.count(target_board), 1)

    async def test_supporter_foreground_query_precedes_regular_query(self):
        scheduler = self.scheduler()
        blocker_board = board_with_tile(0, 2)
        regular_board = board_with_tile(1, 4)
        supporter_board = board_with_tile(2, 8)
        reader = RecordingReader(blocked_board=blocker_board)
        blocker = await scheduler.submit(
            make_spec(reader, blocker_board),
            stream_key="blocker",
            supporter=False,
        )
        self.assertTrue(await asyncio.to_thread(reader.started.wait, 1))
        regular = await scheduler.submit(
            make_spec(reader, regular_board),
            stream_key="regular",
            supporter=False,
        )
        supporter = await scheduler.submit(
            make_spec(reader, supporter_board),
            stream_key="supporter",
            supporter=True,
        )
        reader.release.set()
        await asyncio.gather(blocker.wait(), regular.wait(), supporter.wait())
        self.assertLess(reader.calls.index(supporter_board), reader.calls.index(regular_board))

    async def test_completed_result_is_served_from_memory_cache(self):
        scheduler = self.scheduler()
        reader = RecordingReader()
        spec = make_spec(reader, board_with_tile(0, 16))
        first = await scheduler.submit(
            spec,
            stream_key="user:tester",
            supporter=False,
        )
        await first.wait()
        second = await scheduler.submit(
            spec,
            stream_key="user:tester",
            supporter=False,
        )
        await second.wait()
        self.assertEqual(reader.calls, [spec.board_encoded])

    async def test_overload_rejects_regular_before_supporter(self):
        scheduler = self.scheduler()
        reader = RecordingReader()
        with patch.object(scheduler, "estimated_wait_seconds", return_value=2.0):
            with self.assertRaises(TablebaseQueryOverloaded):
                await scheduler.submit(
                    make_spec(reader, board_with_tile(0, 2)),
                    stream_key="regular-overload",
                    supporter=False,
                )
            supporter = await scheduler.submit(
                make_spec(reader, board_with_tile(1, 4)),
                stream_key="supporter-overload",
                supporter=True,
            )
        await supporter.wait()
        self.assertEqual(len(reader.calls), 1)

    async def test_overload_does_not_reject_cached_or_coalesced_work(self):
        scheduler = self.scheduler()
        blocker_board = board_with_tile(0, 2)
        cached_board = board_with_tile(1, 4)
        reader = RecordingReader(blocked_board=blocker_board)

        cached = await scheduler.submit(
            make_spec(reader, cached_board),
            stream_key="cache-primer",
            supporter=False,
        )
        await cached.wait()
        blocker = await scheduler.submit(
            make_spec(reader, blocker_board),
            stream_key="coalesced-a",
            supporter=False,
        )
        self.assertTrue(await asyncio.to_thread(reader.started.wait, 1))

        with patch.object(scheduler, "estimated_wait_seconds", return_value=10.0):
            cached_again = await scheduler.submit(
                make_spec(reader, cached_board),
                stream_key="cache-consumer",
                supporter=False,
            )
            coalesced = await scheduler.submit(
                make_spec(reader, blocker_board),
                stream_key="coalesced-b",
                supporter=False,
            )

        self.assertEqual((await cached_again.wait()).board_encoded, cached_board)
        reader.release.set()
        await asyncio.gather(blocker.wait(), coalesced.wait())
        self.assertEqual(reader.calls.count(cached_board), 1)
        self.assertEqual(reader.calls.count(blocker_board), 1)

    async def test_scheduler_can_restart_after_close(self):
        scheduler = self.scheduler()
        reader = RecordingReader()
        first_spec = make_spec(reader, board_with_tile(0, 2))
        first = await scheduler.submit(
            first_spec,
            stream_key="before-close",
            supporter=False,
        )
        await first.wait()
        await scheduler.close()
        self.schedulers.remove(scheduler)

        second_spec = make_spec(reader, board_with_tile(1, 4), version="catalog-next")
        second = await scheduler.submit(
            second_spec,
            stream_key="after-close",
            supporter=False,
        )
        await second.wait()
        await scheduler.close()



if __name__ == "__main__":
    unittest.main()
