import { randomSpawnWithRng, Xoshiro128StarStar } from './seededRng.js';

export function createOrdinaryRng() {
  const words = new Uint32Array(4);
  globalThis.crypto.getRandomValues(words);
  if (!words.some(Boolean)) words[0] = 1;
  return new Xoshiro128StarStar([...words]);
}

export function copySpawnRng(rng) {
  return new Xoshiro128StarStar(rng.exportState());
}

function drawEvilBranch(rng, { difficulty = 0, randomOnly = false } = {}) {
  if (randomOnly) return false;
  const branch = rng.nextFloat();
  return difficulty >= 100 || (difficulty > 0 && branch < difficulty / 100);
}

export function needsEvilSpawn(rng, options) {
  return drawEvilBranch(copySpawnRng(rng), options);
}

// Ranked ordering is fixed: branch roll, then (only for random) position and value.
export function planGamerSpawn(board, rng, { difficulty = 0, spawnRate4 = 0.1, randomOnly = false } = {}) {
  const next = copySpawnRng(rng);
  const evil = drawEvilBranch(next, { difficulty, randomOnly });
  const spawn = evil ? null : randomSpawnWithRng(board, next, spawnRate4);
  return { evil, spawn, state: next.exportState() };
}
