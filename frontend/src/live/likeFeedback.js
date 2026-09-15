export class LikeFeedback {
  constructor() {
    this.confirmed = 0;
    this.floor = 0;
    this.queued = 0;
    this.inFlight = 0;
    this.recent = [];
  }
  get count() { return Math.max(this.confirmed, this.floor) + this.queued; }
  get pending() { return this.queued > 0 || this.inFlight > 0; }
  update(count) { this.confirmed = Math.max(this.confirmed, Number(count) || 0); }
  begin(now = Date.now()) {
    this.recent = this.recent.filter(at => now - at < 60000);
    if (this.recent.length >= 20) return false;
    this.recent.push(now);
    this.queued++;
    return true;
  }
  takeBatch() {
    if (this.inFlight || !this.queued) return 0;
    this.inFlight = this.queued;
    this.floor = this.confirmed + this.inFlight;
    this.queued = 0;
    return this.inFlight;
  }
  finish(count) { this.update(count); this.floor = 0; this.inFlight = 0; }
  reject() { this.floor = 0; this.inFlight = 0; }
}
