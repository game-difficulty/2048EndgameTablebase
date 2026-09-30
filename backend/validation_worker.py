"""Resilient single-consumer loop for ranked replay validation."""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)


async def run_validation_loop(name, *, cleanup, process, recover):
    next_cleanup = 0.0
    recovery_needed = False
    while True:
        if time.monotonic() >= next_cleanup:
            try:
                await asyncio.to_thread(cleanup)
            except Exception:
                logger.exception('%s validation cleanup failed; retrying in 60s', name)
                next_cleanup = time.monotonic() + 60
            else:
                next_cleanup = time.monotonic() + 3600
        try:
            if recovery_needed:
                # Only this loop consumes its queue; no validation is in flight.
                await asyncio.to_thread(recover)
                recovery_needed = False
            processed = await asyncio.to_thread(process)
        except Exception:
            logger.exception('%s validation worker failed; recovering in 5s', name)
            recovery_needed = True
            await asyncio.sleep(5)
        else:
            await asyncio.sleep(0.05 if processed else 1.0)
