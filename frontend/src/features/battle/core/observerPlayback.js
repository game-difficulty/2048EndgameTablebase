import { createSnapshotBoardFrame, createTransitionBoardFrame } from '../../../components/boardFrame.js';
import { battleActorRenderKey } from './battleActor.js';

// One timer per view handles correction expiry and local automatic playback.
// Repeated room snapshots reuse frames instead of restarting board animations.
export function createObserverPlayback({ resolve, canSee, publish, now = Date.now,
  schedule = setTimeout, cancel = clearTimeout }) {
  let room = null;
  let timer = null;
  let previous = new Map();
  const render = () => {
    if (timer != null) cancel(timer);
    timer = null;
    const time = now();
    const frames = {};
    const overlays = {};
    const next = new Map();
    let wake = Infinity;
    for (const result of room?.results || []) {
      if (!canSee(result)) continue;
      const view = resolve(result, time);
      if (!view) continue;
      const actor = battleActorRenderKey(result);
      const key = `${room.round?.round_id}:${actor}`;
      const revision = `${key}:${view.index}:${view.board.join(',')}:${view.overlay ? 'correction' : 'board'}`;
      const old = previous.get(key);
      let frame = old?.revision === revision ? old.frame : null;
      if (!frame) {
        const transition = old && view.index === old.index + 1 ? view.transition : null;
        frame = transition
          ? createTransitionBoardFrame(revision, transition, view.board)
          : createSnapshotBoardFrame(revision, view.board);
      }
      next.set(key, { revision, index: view.index, frame });
      frames[actor] = frame;
      if (view.overlay) overlays[actor] = view.overlay;
      if (view.nextAt > time) wake = Math.min(wake, view.nextAt);
    }
    previous = next;
    publish(frames, overlays);
    if (Number.isFinite(wake)) timer = schedule(render, Math.max(1, wake - time));
  };
  return {
    update(value) { room = value; render(); },
    clear() { room = null; render(); },
  };
}
