import { liveSupporterLevel, giftEffectDuration } from './supporterIdentity.js';
import { ceremonyGifts } from './giftArtwork.js';

export class GiftEvents {
  constructor() { this.active = []; this.queue = []; this.seen = new Set(); }
  receive(event, now = Date.now(), capacity = 2) {
    if (!event?.id || this.seen.has(event.id)) return;
    this.seen.add(event.id);
    if (this.seen.size > 500) this.seen.delete(this.seen.values().next().value);
    if (event.type === 'entrance' && liveSupporterLevel(event.actor) < 2) return;
    if (Math.abs(now - event.at * 1000) > 30000) return;
    this.tick(now, capacity);
    const existing = [...this.active, ...this.queue].find(item => item.key === (event.combo_id || event.id));
    if (existing) {
      existing.combo_count = Math.max(existing.combo_count, event.combo_count);
      if (existing.started != null) existing.until = Math.min(existing.started + 8000, Math.max(existing.until, now + 2500));
      return;
    }
    this.queue.push({ ...event, key: event.combo_id || event.id });
    this.queue.sort((a, b) => (b.tier ?? -1) - (a.tier ?? -1));
    this.queue = this.queue.slice(0, 20);
    this.tick(now, capacity);
  }
  tick(now = Date.now(), capacity = 2) {
    this.active = this.active.filter(item => item.until > now).slice(0, capacity);
    this.queue = this.queue.filter(item => now - item.at * 1000 < 30000);
    while (this.active.length < capacity && this.queue.length) {
      const hasCeremony = this.active.some(item => ceremonyGifts.has(item.gift_id));
      const index = this.queue.findIndex(item => !hasCeremony || !ceremonyGifts.has(item.gift_id));
      if (index < 0) break;
      const [item] = this.queue.splice(index, 1);
      const duration = giftEffectDuration(item);
      this.active.push({ ...item, started: now, until: now + duration });
    }
  }
  clear() { this.active = []; this.queue = []; }
}

export function mergeGiftHistory(history, incoming) {
  const result = new Map(history.map(item => [item.combo_id || item.id, item]));
  for (const item of incoming) {
    const key = item.combo_id || item.id, previous = result.get(key);
    if (!previous || (item.combo_count || 0) >= (previous.combo_count || 0)) result.set(key, item);
  }
  return [...result.values()].sort((a, b) => a.at - b.at).slice(-100);
}
