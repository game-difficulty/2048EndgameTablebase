export const GAMER_SPAWN_POLICY = Object.freeze({
  RANKED: 'ranked',
  RANDOM: 'random',
  DIFFICULTY: 'difficulty',
});

export function resolveGamerSpawnPolicy({
  randomAfterUndo = false,
  rankedEligible = false,
  hasRankedRng = false,
} = {}) {
  if (randomAfterUndo) return GAMER_SPAWN_POLICY.RANDOM;
  if (rankedEligible && hasRankedRng) return GAMER_SPAWN_POLICY.RANKED;
  return GAMER_SPAWN_POLICY.DIFFICULTY;
}
