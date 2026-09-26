import { buildOptimisticMoveOnlyTransition, decodeBoard } from '../../../replay/engine/replayTransition.js';
import { correctionOverlayForResult } from '../../core/battleCorrection.js';

const decodeHex = (value) => {
  const text = String(value || '').trim().replace(/^0x/iu, '').toLowerCase();
  return /^[0-9a-f]{1,16}$/u.test(text) ? decodeBoard(BigInt(`0x${text}`)) : null;
};

export function observedFreeStepTransition(step, useVariant, fallbackBoardHex = '') {
  const fromBoard = decodeHex(step?.previous_board_hex || fallbackBoardHex);
  const toBoard = decodeHex(step?.board_hex);
  if (!fromBoard || !toBoard) return null;
  const moved = buildOptimisticMoveOnlyTransition(fromBoard, step?.executed_direction, useVariant);
  if (!moved) return null;
  const predicted = moved.board.slice();
  const metadata = { ...moved.metadata };
  const spawnIndex = Number(step.spawn_index);
  const spawnValue = Number(step.spawn_value);
  if (Number.isInteger(spawnIndex) && spawnIndex >= 0 && spawnIndex < 16 && [2, 4].includes(spawnValue)) {
    if (predicted[spawnIndex] !== 0) return null;
    predicted[spawnIndex] = spawnValue;
    metadata.appear_tile = { index: spawnIndex, value: spawnValue };
  }
  if (predicted.some((value, index) => value !== toBoard[index])) return null;
  return { fromBoard, toBoard, metadata };
}

export function observedFreeBoardView(result, time, useVariant) {
  const correction = correctionOverlayForResult(result, time);
  const board = decodeHex(correction?.previousBoardHex || result.mode_data?.board_hex);
  if (!board) return null;
  const step = result.mode_data?.last_step;
  const progress = Number(result.route_index || 0);
  const index = correction ? Math.max(0, progress - 1) : progress;
  const transition = !correction && step && Number(step.sequence) === Number(result.last_sequence)
    ? observedFreeStepTransition(step, useVariant) : null;
  return { board, index, transition, overlay: correction, nextAt: correction?.visibleUntil };
}
