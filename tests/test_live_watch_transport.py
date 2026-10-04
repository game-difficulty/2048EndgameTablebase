import asyncio
from unittest.mock import AsyncMock

from backend.live.watch_transport import serve_viewer


def test_slow_identity_never_blocks_first_snapshot_and_disconnect_cleans_tasks():
    async def run():
        socket = AsyncMock()
        hold = asyncio.Event()
        socket.receive_text.side_effect = lambda: hold.wait()
        sent = asyncio.Event()
        async def send(_):
            sent.set()
        socket.send_json.side_effect = send
        async def identify():
            await hold.wait()
        # A real pending coroutine rather than an AsyncMock-returned coroutine.
        async def receive():
            await hold.wait()
            return 'close'
        socket.receive_text.side_effect = receive
        queue = asyncio.Queue()
        queue.put_nowait({'type':'snapshot'})
        task = asyncio.create_task(serve_viewer(socket, queue, identify, lambda:None))
        await asyncio.wait_for(sent.wait(), 1)
        assert not task.done()
        hold.set()
        await asyncio.wait_for(task, 1)
        socket.close.assert_awaited()
    asyncio.run(run())


def test_failed_writer_wakes_idle_reader_immediately():
    async def run():
        socket = AsyncMock()
        async def idle():
            await asyncio.Event().wait()
        socket.receive_text.side_effect = idle
        socket.send_json.side_effect = RuntimeError('send failed')
        queue = asyncio.Queue()
        queue.put_nowait({'type':'snapshot'})
        await asyncio.wait_for(serve_viewer(socket, queue, idle, lambda:None), 1)
        socket.close.assert_awaited()
    asyncio.run(run())
