// Ordered presentation only. Authoritative scores/clocks never wait for playback.
export class ProjectPlayback {
  constructor(present, { schedule = (fn, delay) => setTimeout(fn, delay), cancel = id => clearTimeout(id), onPending = () => {} } = {}) {
    Object.assign(this, { present, schedule, cancel, onPending });
    this.key = null;
    this.view = null;
    this.pending = new Map();
    this.timer = null;
  }
  reset(view, key) {
    this.close();
    this.key = key;
    this.view = view && { ...view, frames: [], payload: { ...view.payload, last_transition: { kind: 'restore' } } };
    this.present(this.view);
  }
  receive(next, key) {
    if (!next) { this.reset(null, key); return; }
    if (!this.view || key !== this.key || next.generation !== this.view.generation) {
      this.reset(next, key); return;
    }
    if (Number(next.sequence) < Number(this.view.sequence)) return;
    if (Number(next.sequence) === Number(this.view.sequence)) {
      // Server clock expiry/referee outcomes may update metadata without a move.
      this.view = { ...next, frames: [], payload: { ...next.payload, last_transition: this.view.payload.last_transition } };
      this.present(this.view);
      return;
    }
    const frames = next.frames?.length ? next.frames : [{ sequence: next.sequence, payload: next.payload }];
    for (const frame of frames) {
      if (frame.sequence > this.view.sequence && !this.pending.has(frame.sequence)) {
        this.pending.set(frame.sequence, { ...next, frames: [], sequence: frame.sequence, payload: frame.payload });
      }
    }
    // The producer explicitly advertises its retained history floor. A viewer
    // outside that window rejoins live instead of inventing missing movements.
    if (!this.pending.has(this.view.sequence + 1)
      && Number(next.frame_start ?? next.sequence) > this.view.sequence + 1) {
      this.reset(next, key); return;
    }
    this.start();
    this.onPending(this.pending.size > 0);
  }
  start() {
    if (this.timer != null || !this.pending.has(this.view.sequence + 1)) return;
    const frame = this.pending.get(this.view.sequence + 1);
    const elapsed = Number(frame.payload.elapsed_ms) - Number(this.view.payload.elapsed_ms);
    // Keep shared animation durations. Catch-up can interrupt an animation,
    // just like fast local input, but still presents every intervening state.
    const delay = this.pending.size > 20 ? 16 : this.pending.size > 8 ? 50
      : this.pending.size > 3 ? 100 : Math.max(100, Math.min(300, elapsed || 100));
    this.timer = this.schedule(() => {
      this.timer = null;
      this.pending.delete(frame.sequence);
      this.view = frame;
      this.present(frame);
      this.start();
      this.onPending(this.pending.size > 0);
    }, delay);
  }
  close() {
    if (this.timer != null) this.cancel(this.timer);
    this.timer = null;
    this.pending.clear();
    this.onPending(false);
  }
}
