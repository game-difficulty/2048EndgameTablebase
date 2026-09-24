export const SPOTLIGHT_DURATION = 4400;

// Server marks the first threshold crossing; retries and later combo updates never replay it.
export class GiftSpotlights {
  constructor() { this.active = null; this.queue = []; this.seen = new Set(); }
  receive(event, now = Date.now()) {
    if (event?.type !== 'gift' || !event.id || Math.abs(now - event.at * 1000) > 30000) return;
    this.tick(now);
    const key = event.combo_id || event.id;
    const existing = [this.active, ...this.queue].find(item => item?.key === key);
    if (existing) {
      existing.combo_count = Math.max(existing.combo_count, event.combo_count);
      return;
    }
    if (!event.bulk_effect || this.seen.has(key)) return;
    this.seen.add(key);
    if (this.seen.size > 500) this.seen.delete(this.seen.values().next().value);
    if (this.queue.length < 6) this.queue.push({ ...event, key });
    this.tick(now);
  }
  tick(now = Date.now()) {
    if (this.active?.until <= now) this.active = null;
    this.queue = this.queue.filter(item => now - item.at * 1000 < 30000);
    if (!this.active && this.queue.length) this.active = { ...this.queue.shift(), until: now + SPOTLIGHT_DURATION };
  }
  clear() { this.active = null; this.queue = []; }
}
