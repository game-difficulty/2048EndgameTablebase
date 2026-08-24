import { restoreSuccessRate } from '../../../utils/successRate.js';
import { evaluationOfPerformance } from '../../replay/engine/replayAnalysis.js';

export function buildOptimisticTesterLastStep({
  board,
  results,
  dtype,
  direction,
  goodnessOfFit = 1,
} = {}) {
  const snapshotResults = results && typeof results === 'object' ? { ...results } : {};
  const bestMove = Object.entries(snapshotResults)
    .find(([, value]) => typeof value === 'number')?.[0] || null;
  const selectedRate = restoreSuccessRate(snapshotResults[direction], dtype);
  const bestRate = restoreSuccessRate(snapshotResults[bestMove], dtype);
  if (!bestMove || selectedRate == null || bestRate == null) return null;

  const ratio = Math.abs(bestRate - selectedRate) <= 3e-10
    ? 1
    : (bestRate > 0 ? selectedRate / bestRate : 1);
  return {
    board: Array.isArray(board) ? [...board] : [],
    board_lines: [],
    result_lines: [],
    results: snapshotResults,
    dtype: dtype || '?',
    message_lines: [],
    evaluation: evaluationOfPerformance(ratio),
    direction,
    best_move: bestMove,
    loss: 1 - ratio,
    goodness_of_fit: Number(goodnessOfFit ?? 1) * ratio,
  };
}
