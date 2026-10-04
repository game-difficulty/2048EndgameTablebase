"""Public delivery is independent of identity/entrance enrichment."""
import asyncio
import contextlib
import logging
import json


class EncodedMessage(dict):
    """Encode shared public broadcasts once, not twice per spectator."""
    def __init__(self, message):
        super().__init__(message)
        self.wire_text = json.dumps(message, ensure_ascii=False, separators=(',', ':'))
        self.wire_bytes = len(self.wire_text.encode('utf-8'))


async def serve_viewer(ws, queue, identify, on_ping):
    async def send():
        while True:
            item = await queue.get()
            if item is None:
                await asyncio.wait_for(ws.close(code=1013, reason='slow_consumer'), 5)
                return
            if isinstance(item, EncodedMessage):
                await asyncio.wait_for(ws.send_text(item.wire_text), 5)
            else:
                method = ws.send_bytes if isinstance(item, bytes) else ws.send_json
                await asyncio.wait_for(method(item), 5)

    async def receive():
        while await asyncio.wait_for(ws.receive_text(), 90) == 'ping':
            on_ping()

    async def enrich():
        try:
            await asyncio.wait_for(identify(), 10)
        except Exception:
            logging.getLogger(__name__).debug('Live viewer identity enrichment unavailable', exc_info=True)

    writer = asyncio.create_task(send())
    reader = asyncio.create_task(receive())
    identity = asyncio.create_task(enrich())
    try:
        done, _ = await asyncio.wait((writer, reader), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                logging.getLogger(__name__).info('Live viewer transport ended: %s', type(task.exception()).__name__)
    finally:
        for task in (writer, reader, identity):
            task.cancel()
        await asyncio.gather(writer, reader, identity, return_exceptions=True)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(ws.close(), 2)
