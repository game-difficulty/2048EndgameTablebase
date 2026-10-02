"""Bridge persisted content facts to room activities; no board rules here."""
import asyncio
import time
from . import predictions, lucky_bags, competition_predictions


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
        if batch['phase'] != 'void':
            await asyncio.to_thread(predictions.record_targets, room, batch['id'], batch.get('target_results', {}))
        if batch.get('closed') or batch['deadline'] <= time.time():
            await asyncio.to_thread(predictions.close, room, batch['id'])
        if batch['phase'] in ('settling','cooldown','void'):
            await asyncio.to_thread(predictions.settle, room, batch['id'], batch.get('winners',()), batch['phase']=='void')

    async def refresh(self):
        if getattr(self.hub.room, 'content_kind', None) == 'competition-match':
            await asyncio.to_thread(competition_predictions.reconcile, self.hub.room.id, self.hub.content.projection)
            state = await asyncio.to_thread(competition_predictions.listing, self.hub.room.id)
            facts = self.hub.content.projection or {}
            state['available'] = bool(not facts.get('suspended') and (facts.get('prediction_window') or {}).get('open'))
            if state['markets'] != self.state.get('markets') or state['available'] != self.state.get('available'):
                self.state = state
                self.hub.broadcast(dict(type='predictions', **state))
            return
        state = await asyncio.to_thread(predictions.listing, self.hub.room.id)
        # server_time changes do not cause a per-second broadcast.
        if state['market'] != self.state.get('market'):
            self.state = state
            self.hub.broadcast(dict(type='predictions', **state))
        await self.flush_announcements()

    async def flush_announcements(self):
        room = self.hub.room.id
        events = await asyncio.to_thread(predictions.announcement_events, room, True)
        known = {item['id'] for item in self.hub.chat}
        for event in events:
            if event['id'] not in known:
                self.hub.chat.append(event)
                known.add(event['id'])
            self.hub.broadcast(event)
        if events:
            await asyncio.to_thread(predictions.announcements_delivered, room, [e['id'] for e in events])

    async def drain(self):
        store = self.hub.store
        if not hasattr(store, 'pending_activities'):
            return
        for event in await asyncio.to_thread(store.pending_activities):
            await asyncio.to_thread(lucky_bags.create, 'batch:'+event['batch_id'], event['milestone'], room_id=self.hub.room.id)
            await asyncio.to_thread(store.activity_delivered, event['trigger_key'])
            await self.hub.refresh_lucky()
