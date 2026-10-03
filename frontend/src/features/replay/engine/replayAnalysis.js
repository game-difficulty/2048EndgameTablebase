import {
  replayChangeIsForced,
  replayStepGoodnessRatio,
} from './replayGoodness.js';
import { replayMarkerIndices } from './replayMarkers.js';
import { isPerfectResult } from '../../../utils/perfectTolerance.js';

export const PERFORMANCE_PERFECT_LABEL = 'Perfect!';
export const PERFORMANCE_EVALUATIONS = Object.freeze([
  { label: 'Excellent!', threshold: 0.999 },
  { label: 'Nice try!', threshold: 0.99 },
  { label: 'Not bad!', threshold: 0.975 },
  { label: 'Mistake!', threshold: 0.9 },
  { label: 'Blunder!', threshold: 0.75 },
  { label: 'Terrible!', threshold: -1 },
]);
export const PERFORMANCE_LABELS = Object.freeze([
  PERFORMANCE_PERFECT_LABEL,
  ...PERFORMANCE_EVALUATIONS.map((item) => item.label),
]);

export function evaluationOfPerformance(loss, selectedRate = loss, bestRate = 1, dtype = 'uint32') {
  const numericLoss = Number(loss);
  if (isPerfectResult(selectedRate, bestRate, dtype)) return PERFORMANCE_PERFECT_LABEL;
  return PERFORMANCE_EVALUATIONS.find((item) => numericLoss >= item.threshold)?.label
    || PERFORMANCE_EVALUATIONS[PERFORMANCE_EVALUATIONS.length - 1].label;
}

export function analyzeReplay(replay, markerThreshold = 1) {
  const count = Number(replay?.moveCount || 0);
  const losses = new Float64Array(count);
  const goodnessOfFit = new Float64Array(count);
  const combo = new Uint16Array(count);
  const forced = new Uint8Array(count);
  const counts = Object.fromEntries(PERFORMANCE_LABELS.map((label) => [label, 0]));

  let cumulative = 1;
  let comboCount = 0;
  let scoredMoves = 0;
  for (let index = 0; index < count; index += 1) {
    const offset = index * 4;
    let maximum = 0;
    for (let direction = 0; direction < 4; direction += 1) {
      maximum = Math.max(maximum, replay.rates[offset + direction] / 4e9);
    }
    const isForced = replayChangeIsForced(replay.changes[index]);
    forced[index] = isForced ? 1 : 0;
    const move = (replay.changes[index] >> 5) & 0b11;
    const player = replay.rates[offset + move] / 4e9;
    const stepLoss = isForced ? 1 : replayStepGoodnessRatio(player, maximum);
    losses[index] = stepLoss;
    cumulative *= stepLoss;
    goodnessOfFit[index] = cumulative;

    if (!isForced) {
      comboCount = isPerfectResult(player, maximum, 'uint32') ? comboCount + 1 : 0;
      counts[evaluationOfPerformance(stepLoss, player, maximum, 'uint32')] += 1;
      scoredMoves += 1;
    }
    combo[index] = comboCount;
  }

  const pointList = replayMarkerIndices(losses, markerThreshold);

  return {
    losses,
    goodnessOfFit,
    combo,
    forced,
    pointsRank: Int32Array.from(pointList),
    summary: {
      total_moves: scoredMoves,
      final_gof: count ? goodnessOfFit[count - 1] : 0,
      max_combo: count ? Math.max(...combo) : 0,
      counts,
    },
  };
}
