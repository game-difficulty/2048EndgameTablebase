import { move, VARIANTS } from "../../../frontend/src/human/engine.js";
let replay, snapshots;
function advance(state, event) {
  const [direction, index, value, delta] = event;
  const next = move(state.board, ...VARIANTS[replay.variant], direction);
  if (!next.changed || next.board[index] || ![2, 4].includes(value))
    throw Error("录像步数校验失败");
  next.board[index] = value;
  return {
    board: next.board,
    score: state.score + next.score,
    elapsed: state.elapsed + (delta === 0xffffffff ? 0 : delta),
  };
}
function seek(step) {
  const start = Math.floor(step / 256) * 256;
  let state = structuredClone(snapshots.get(start));
  for (let i = start; i < step; i++) state = advance(state, replay.moves[i]);
  self.postMessage({ type: "frame", step, ...state });
}
self.onmessage = ({ data }) => {
  try {
    if (data.type === "load") {
      replay = data.replay;
      snapshots = new Map();
      let state = { board: replay.initial, score: 0, elapsed: 0 };
      snapshots.set(0, state);
      replay.moves.forEach((event, index) => {
        state = advance(state, event);
        if ((index + 1) % 256 === 0) snapshots.set(index + 1, state);
      });
      self.postMessage({ type: "ready", total: replay.moves.length });
      seek(Math.min(replay.moves.length, data.step || 0));
    } else if (data.type === "seek")
      seek(Math.max(0, Math.min(replay.moves.length, Math.trunc(data.step))));
  } catch (e) {
    self.postMessage({ type: "error", message: e.message });
  }
};
