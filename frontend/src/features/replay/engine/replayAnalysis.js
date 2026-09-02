import {
  replayChangeIsForced,
  replayStepGoodnessRatio,
} from './replayGoodness.js';
import { replayMarkerIndices } from './replayMarkers.js';

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

export function evaluationOfPerformance(loss) {
  const numericLoss = Number(loss);
  if (numericLoss > 1 - 3e-10) return PERFORMANCE_PERFECT_LABEL;
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
      maximum = Math.max(maximum, replay.rates[offset + direction]);
    }
    const isForced = replayChangeIsForced(replay.changes[index]);
    forced[index] = isForced ? 1 : 0;
    const move = (replay.changes[index] >> 5) & 0b11;
    const player = replay.rates[offset + move];
    const stepLoss = isForced ? 1 : replayStepGoodnessRatio(player, maximum);
    losses[index] = stepLoss;
    cumulative *= stepLoss;
    goodnessOfFit[index] = cumulative;

    if (!isForced) {
      comboCount = stepLoss > 1 - 3e-10 ? comboCount + 1 : 0;
      counts[evaluationOfPerformance(stepLoss)] += 1;
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
