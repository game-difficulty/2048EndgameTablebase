import { isPerfectResult } from '../../../utils/perfectTolerance.js';

export const REPLAY_FORCED_FLAG = 0x80;

export function replayChangeIsForced(change) {
  return (Number(change) & REPLAY_FORCED_FLAG) !== 0;
}

export function replayStepGoodnessRatio(selectedRate, bestRate, dtype = 'uint32') {
  const selected = Number(selectedRate);
  const best = Number(bestRate);
  if (!Number.isFinite(selected) || !Number.isFinite(best)) return 1;
  if (isPerfectResult(selected, best, dtype)) return 1;
  return best > 0 ? selected / best : 1;
}
