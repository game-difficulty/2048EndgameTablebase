export const REPLAY_FORCED_FLAG = 0x80;

export function replayChangeIsForced(change) {
  return (Number(change) & REPLAY_FORCED_FLAG) !== 0;
}

export function replayStepGoodnessRatio(selectedRate, bestRate) {
  const selected = Number(selectedRate);
  const best = Number(bestRate);
  if (!Number.isFinite(selected) || !Number.isFinite(best)) return 1;
  if (Math.abs(best - selected) <= 3e-10) return 1;
  return best > 0 ? selected / best : 1;
}
