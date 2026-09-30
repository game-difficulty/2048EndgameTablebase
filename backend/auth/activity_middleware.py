"""Record authenticated HTTP and WebSocket activity without endpoint-specific hooks."""

import asyncio
import logging
import time

from .daily_activity import activity_day, record_daily_visit, site_from_host, visit_is_recorded


logger = logging.getLogger(__name__)


class DailyActivityMiddleware:
    def __init__(self, app, *, site: str | None = None):
        self.app = app
        self.site = site

    async def __call__(self, scope, receive, send):
        if scope['type'] not in {'http', 'websocket'}:
            return await self.app(scope, receive, send)
        state = scope.setdefault('state', {})
        host = next((value.decode('latin-1') for name, value in scope.get('headers', [])
                     if name == b'host'), '')
        site = self.site or site_from_host(host)
        recorded = None
        retry_after = 0.0

        async def record():
            nonlocal recorded, retry_after
            user_id = state.get('daily_activity_user_id')
            if user_id is None:
                return
            day = activity_day()
            identity = (day, user_id)
            if recorded == identity or time.monotonic() < retry_after:
                return
            if not visit_is_recorded(user_id, site, day):
                try:
                    await asyncio.to_thread(record_daily_visit, user_id, site)
                except Exception:
                    # Statistics must not break gameplay; retry on subsequent activity.
                    retry_after = time.monotonic() + 60
                    logger.exception('Could not record authenticated account activity')
                    return
            recorded = identity

        async def tracked_send(message):
            if (message['type'] == 'http.response.start' and message['status'] < 500
                    or message['type'] == 'websocket.send'):
                await record()
            await send(message)

        async def tracked_receive():
            message = await receive()
            if message['type'] == 'websocket.receive':
                await record()
            return message

        await self.app(scope, tracked_receive, tracked_send)
