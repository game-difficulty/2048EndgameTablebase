// Only configured variants celebrate; the persisted flag belongs to the run.
export const VICTORY_TARGETS = Object.freeze({ '4x4': 2048 });

export function reachedVictory(previous, next, targets = VICTORY_TARGETS) {
  const target = targets[previous.variant];
  if (!target || previous.victoryShown || previous.reason) return 0;
  return previous.board.every(value => value < target) && next.board.some(value => value >= target) ? target : 0;
}
