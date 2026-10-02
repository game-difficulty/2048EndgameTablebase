// Network samples anchor the display, but late samples cannot stop or rewind it.
export class ServerClock {
  constructor({ now = () => performance.now(), wall = () => Date.now() } = {}) {
    this.localNow = now;
    this.anchorAt = now();
    this.anchor = wall();
    this.sample = -Infinity;
  }
  now() { return this.anchor + Math.max(0, this.localNow() - this.anchorAt); }
  observe(timestamp) {
    const sample = Date.parse(timestamp || '');
    if (!Number.isFinite(sample) || sample <= this.sample) return;
    const current = this.now();
    this.anchor = this.sample === -Infinity ? sample : Math.max(current, sample);
    this.anchorAt = this.localNow();
    this.sample = sample;
  }
}
