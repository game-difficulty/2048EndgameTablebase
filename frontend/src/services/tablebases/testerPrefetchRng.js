import { Xoshiro128StarStar, createRandomXoshiroState } from '../../utils/xoshiro128.js';

export const TESTER_PREFETCH_RNG_VERSION = 1;

export function createTesterPrefetchState() {
  return {
    state: createRandomXoshiroState(),
    turn: 0,
  };
}

export function createTesterSpawnRandomSource(prefetchState) {
  const rng = new Xoshiro128StarStar(prefetchState?.state);
  return {
    randomSource: () => rng.nextFloat(),
    nextState: () => ({
      state: rng.exportState(),
      turn: Math.max(0, Number(prefetchState?.turn) || 0) + 1,
    }),
  };
}

export function buildTesterPrefetchPayload(prefetchState, spawnRate4 = 0.1) {
  const rng = new Xoshiro128StarStar(prefetchState?.state);
  return {
    version: TESTER_PREFETCH_RNG_VERSION,
    state: rng.exportState(),
    turn: Math.max(0, Math.floor(Number(prefetchState?.turn) || 0)),
    spawn_rate_4: Math.max(0, Math.min(1, Number(spawnRate4) || 0)),
  };
}
