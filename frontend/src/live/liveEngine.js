import { simulateMove } from "../features/gamer/engine/classicMove.js";
import { createBoardFrame } from "../components/boardFrame.js";

export const DIRECTIONS = ["up", "right", "down", "left"];

export function applyLiveStep(run, packet) {
  if (!run || !(packet instanceof ArrayBuffer) || packet.byteLength !== 9)
    throw Error("snapshot_required");
  const bytes = new DataView(packet);
  const seq = bytes.getUint32(0, true),
    delta = bytes.getUint32(4, true),
    change = bytes.getUint8(8);
  if (seq !== run.seq + 1 || change > 127) throw Error("snapshot_required");
  const direction = DIRECTIONS[change & 3];
  const next = simulateMove(run.board, direction);
  const index = (change >> 2) & 15,
    value = change & 64 ? 4 : 2;
  if (next.board.every((v, i) => v === run.board[i]) || next.board[index])
    throw Error("snapshot_required");
  next.board[index] = value;
  const elapsed = run.elapsed_ms + delta;
  const nodes = { ...run.nodes };
  for (let tile = 512; tile <= Math.max(...next.board); tile *= 2)
    nodes[tile] ??= elapsed;
  return {
    run: {
      ...run,
      board: next.board,
      seq,
      score: run.score + next.scoreDelta,
      elapsed_ms: elapsed,
      nodes,
    },
    frame: createBoardFrame({
      revision: `${run.run_id}:${seq}`,
      kind: "move",
      fromBoard: run.board,
      toBoard: next.board,
      metadata: {
        direction,
        slide_distances: next.slideDistances,
        pop_positions: next.popPositions,
        appear_tile: { index, value },
      },
    }),
  };
}
