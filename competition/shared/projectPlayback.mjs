// Ordered presentation only. Authoritative scores/clocks never wait for playback.
export class ProjectPlayback {
  constructor(present, { schedule = (fn, delay) => setTimeout(fn, delay), cancel = id => clearTimeout(id),
    now = () => performance.now(), onPending = () => {}, onGap = () => {}, onDiagnostic = () => {} } = {}) {
    Object.assign(this, { present, schedule, cancel, now, onPending, onGap, onDiagnostic });
    this.key = null;
    this.view = null;
    this.pending = new Map();
    this.timer = null;
    this.latest = null; this.waitingGap = false; this.anchor = null;
  }
  reset(view, key, reason = 'initial') {
    this.close();
    this.key = key;
    this.latest = view;
    this.view = view && { ...view, frames: [], payload: { ...view.payload, last_transition: { kind: 'restore' } } };
    this.present(this.view);
    this.onDiagnostic({ reason, played: view?.sequence });
  }
  receive(next, key) {
    if (!next) { this.reset(null, key); return; }
    if (!this.view || key !== this.key || next.generation !== this.view.generation) {
      this.reset(next, key); return;
    }
    if (Number(next.sequence) < Number(this.view.sequence)) return;
    if (!this.latest || Number(next.sequence) >= Number(this.latest.sequence)) this.latest = next;
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
    if (this.pending.size > 128) {
      this.reset(this.latest, this.key, 'playback_overflow'); return;
    }
    if (this.waitingGap && this.pending.has(this.view.sequence + 1)) {
      this.cancel(this.timer); this.timer = null; this.waitingGap = false;
    }
    this.start();
    this.onPending(this.pending.size > 0);
  }
  start() {
    if (this.timer != null || !this.view) return;
    const frame = this.pending.get(this.view.sequence + 1);
    if (!frame) {
      if (!this.latest || this.latest.sequence <= this.view.sequence) { this.anchor = null; return; }
      // Recheck at every dequeue: a gap may be hidden behind queued good steps.
      if (Number(this.latest.frame_start ?? this.latest.sequence) > this.view.sequence + 1) {
        this.reset(this.latest, this.key, 'history_expired'); return;
      }
      this.waitingGap = true;
      this.onGap({ after: this.view.sequence, latest: this.latest.sequence });
      this.timer = this.schedule(() => {
        this.timer = null; this.reset(this.latest, this.key, 'gap_timeout');
      }, 800);
      return;
    }
    const at = this.now(), elapsed = Number(frame.payload.elapsed_ms || 0);
    if (!this.anchor) this.anchor = { source: elapsed, wall: at + 60 };
    const due = this.anchor.wall + elapsed - this.anchor.source;
    // Never repeatedly shift the anchor while catching up: doing so compounds
    // lateness and can discard a completely intact queue on a slow connection.
    const delay = Math.max(16, Math.min(300, due - at));
    this.timer = this.schedule(() => {
      this.timer = null;
      if (this.now() - at > 5000) { this.reset(this.latest, this.key, 'browser_suspended'); return; }
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
    this.waitingGap = false; this.anchor = null;
    this.onPending(false);
  }
}
