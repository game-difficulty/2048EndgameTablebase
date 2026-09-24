import { createSnapshotBoardFrame } from '../../components/boardFrame.js';

// Absorb network batching, not game time. Logical state is always decoded immediately.
export const LIVE_PLAYBACK_MAX_DELAY = 180;
export const LIVE_PLAYBACK_MAX_FRAMES = 12;
export const livePaintInterval = (layout, lane, mainLane) => layout === 'equal' ? 1000 / 30 : lane === mainLane ? 0 : 1000 / 15;

export class LiveBoardPlayback {
  constructor(frames) {
    this.frames = [...frames];
    this.queues = [[], [], []];
    this.paintedAt = [-Infinity, -Infinity, -Infinity];
  }

  sync(frames, lanes = [0, 1, 2]) {
    for (const lane of lanes) {
      this.frames[lane] = frames[lane];
      this.queues[lane] = [];
      this.paintedAt[lane] = -Infinity;
    }
    return [...this.frames];
  }

  enqueue(transitions, now) {
    for (const { lane, frame } of transitions) {
      const queue = this.queues[lane];
      queue.push({ frame, at: now });
      if (queue.length > LIVE_PLAYBACK_MAX_FRAMES) {
        this.queues[lane] = [{ frame: this.snapshot(frame), at: now }];
      }
    }
  }

  snapshot(frame) {
    return createSnapshotBoardFrame(`${frame.revision}:catchup`, frame.toBoard);
  }

  paint(now, layout, mainLane) {
    for (let lane = 0; lane < 3; lane++) {
      const queue = this.queues[lane];
      if (!queue.length || now - this.paintedAt[lane] < livePaintInterval(layout, lane, mainLane)) continue;
      if (now - queue[0].at > LIVE_PLAYBACK_MAX_DELAY) {
        this.frames[lane] = this.snapshot(queue[queue.length - 1].frame);
        this.queues[lane] = [];
      } else {
        this.frames[lane] = queue.shift().frame;
      }
      this.paintedAt[lane] = now;
    }
    return [...this.frames];
  }

  get pending() { return this.queues.some(queue => queue.length > 0); }
}
