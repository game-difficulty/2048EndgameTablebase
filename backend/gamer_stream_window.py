"""Bounded, cumulative credits shared by the cloud and table worker."""
import asyncio

MAX_WINDOW = 64
REMOTE_LOOKAHEAD = 64
STREAM_IDLE_SECONDS = 90
MAX_STREAM_STEPS = 100_000


class StreamWindow:
    def __init__(self, allow_through=7, *, max_window=MAX_WINDOW):
        self.max_window = max_window
        self.produced = -1
        self.consumed = -1
        self.allow_through = -1
        self.changed = asyncio.Event()
        self.update(-1, allow_through)

    def update(self, consumed, allow_through):
        if (type(consumed) is not int or type(allow_through) is not int
                or not -1 <= consumed <= self.produced
                or not consumed <= allow_through <= min(consumed + self.max_window, MAX_STREAM_STEPS - 1)):
            raise ValueError('Invalid stream credit')
        self.consumed = max(self.consumed, consumed)
        self.allow_through = max(self.allow_through, allow_through)
        self.changed.set()

    async def wait(self, index):
        while index > self.allow_through:
            self.changed.clear()
            await asyncio.wait_for(self.changed.wait(), STREAM_IDLE_SECONDS)
