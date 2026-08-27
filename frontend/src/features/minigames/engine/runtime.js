import { Xoshiro128StarStar, createRandomXoshiroState } from '../../../utils/xoshiro128.js';

const normalizeClock = (clock) => {
  if (clock && typeof clock.now === 'function') return clock;
  if (typeof clock === 'function') return { now: clock };
  return { now: () => Date.now() };
};

export function createMinigameRuntime({
  seedHex = '',
  rngState = null,
  clock = null,
  evilSpawn = null,
  onDeterminismFailure = null,
} = {}) {
  const rng = Array.isArray(rngState)
    ? new Xoshiro128StarStar(rngState)
    : seedHex
      ? Xoshiro128StarStar.fromSeedHex(seedHex)
      : new Xoshiro128StarStar(createRandomXoshiroState());
  const runtimeClock = normalizeClock(clock);

  return {
    seedHex: String(seedHex || '').trim().toLowerCase(),
    rng,
    clock: runtimeClock,
    evilSpawn: typeof evilSpawn === 'function' ? evilSpawn : null,
    onDeterminismFailure: typeof onDeterminismFailure === 'function' ? onDeterminismFailure : null,
    now() {
      return Math.max(0, Math.trunc(Number(runtimeClock.now()) || 0));
    },
    random() {
      return rng.nextFloat();
    },
    randomIndex(count) {
      return rng.chooseIndex(count);
    },
    markDeterminismFailure(code) {
      this.onDeterminismFailure?.(String(code || 'determinism_failure'));
    },
    exportSnapshot() {
      return {
        seedHex: this.seedHex,
        rngState: rng.exportState(),
      };
    },
  };
}

export function restoreMinigameRuntime(snapshot, options = {}) {
  const source = snapshot && typeof snapshot === 'object' ? snapshot : {};
  return createMinigameRuntime({
    ...options,
    seedHex: options.seedHex || source.seedHex || '',
    rngState: options.rngState || source.rngState || null,
    onDeterminismFailure: options.onDeterminismFailure || null,
  });
}

export function createVirtualMinigameClock(initialMs = 0) {
  let currentMs = Math.max(0, Math.trunc(Number(initialMs) || 0));
  return {
    now: () => currentMs,
    advance(deltaMs) {
      currentMs += Math.max(0, Math.trunc(Number(deltaMs) || 0));
      return currentMs;
    },
    set(value) {
      currentMs = Math.max(0, Math.trunc(Number(value) || 0));
      return currentMs;
    },
  };
}
