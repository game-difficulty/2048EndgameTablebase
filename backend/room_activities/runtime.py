"""Bridge persisted content facts to room activities; no board rules here."""
import asyncio
import time
from . import predictions, lucky_bags


class RoomActivities:
    def __init__(self, hub):
        self.hub = hub
        self.lock = asyncio.Lock()
        self.state = dict(market=None)

    async def reconcile(self, batch):
        if not batch or batch.get('transition'):
            return
        room = self.hub.room.id
        await asyncio.to_thread(predictions.ensure_market, room, batch)
        await asyncio.to_thread(predictions.record_targets, room, batch['id'], batch.get('target_results', {}))
        if batch.get('closed') or batch['deadline'] <= time.time():
            await asyncio.to_thread(predictions.close, room, batch['id'])
        if batch['phase'] in ('settling','cooldown','void'):
            await asyncio.to_thread(predictions.settle, room, batch['id'], batch.get('winners',()), batch['phase']=='void')

    async def refresh(self):
        state = await asyncio.to_thread(predictions.listing, self.hub.room.id)
        # server_time changes do not cause a per-second broadcast.
        if state['market'] != self.state.get('market'):
            self.state = state
            self.hub.broadcast(dict(type='predictions', **state))

    async def drain(self):
        store = self.hub.store
        if not hasattr(store, 'pending_activities'):
            return
        for event in await asyncio.to_thread(store.pending_activities):
            await asyncio.to_thread(lucky_bags.create, 'batch:'+event['batch_id'], event['milestone'], room_id=self.hub.room.id)
            await asyncio.to_thread(store.activity_delivered, event['trigger_key'])
            await self.hub.refresh_lucky()
