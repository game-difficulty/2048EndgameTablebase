// Playback cadence depends only on queue depth, not layout or wire timestamps.
export const LIVE_PLAYBACK_BASE_INTERVAL = 40;
export const LIVE_PLAYBACK_MIN_INTERVAL = 4;
const targetInterval = depth => Math.max(LIVE_PLAYBACK_MIN_INTERVAL,
  LIVE_PLAYBACK_BASE_INTERVAL / (1 + Math.max(0, depth - 3) * 0.5));

export class LiveBoardPlayback {
  constructor(frames) {
    this.frames = [...frames];
    this.queues = [[], [], []];
    this.paintedAt = [-Infinity, -Infinity, -Infinity];
    this.intervals = [40, 40, 40];
  }

  sync(frames, lanes = [0, 1, 2]) {
    for (const lane of lanes) {
      this.frames[lane] = frames[lane];
      this.queues[lane] = [];
      this.paintedAt[lane] = -Infinity;
      this.intervals[lane] = LIVE_PLAYBACK_BASE_INTERVAL;
    }
    return [...this.frames];
  }

  enqueue(transitions) {
    for (const { lane, frame } of transitions) this.queues[lane].push(frame);
  }

  paint(now) {
    for (let lane = 0; lane < 3; lane++) {
      const queue = this.queues[lane];
      if (!queue.length || now - this.paintedAt[lane] < this.intervals[lane]) continue;
      // One step per callback, even after a stalled tab. Never skip queued moves.
      const target = targetInterval(queue.length);
      this.intervals[lane] += (target - this.intervals[lane]) * 0.25;
      this.frames[lane] = queue.shift();
      this.paintedAt[lane] = now;
    }
    return [...this.frames];
  }

  nextDelay(now) {
    return Math.max(LIVE_PLAYBACK_MIN_INTERVAL, Math.min(...this.queues.map((queue,lane) =>
      queue.length ? Math.max(0, this.paintedAt[lane] + this.intervals[lane] - now) : Infinity)));
  }

  get pending() { return this.queues.some(queue => queue.length > 0); }
}
