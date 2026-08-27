import assert from 'node:assert/strict';
import test from 'node:test';

import { createExclusiveRunLock } from '../src/services/concurrency/exclusiveRunLock.js';

test('only one minigame tab can hold the same ranked run', async () => {
  const heldNames = new Set();
  const originalLocks = globalThis.navigator?.locks;
  Object.defineProperty(globalThis.navigator, 'locks', {
    configurable: true,
    value: {
      request: async (name, _options, callback) => {
        if (heldNames.has(name)) return callback(null);
        heldNames.add(name);
        try {
          return await callback({ name });
        } finally {
          heldNames.delete(name);
        }
      },
    },
  });

  try {
    const first = createExclusiveRunLock({ namespace: 'minigame-ranked' });
    const second = createExclusiveRunLock({ namespace: 'minigame-ranked' });
    assert.equal(await first.acquire('run-1'), true);
    assert.equal(await second.acquire('run-1'), false);
    first.release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    assert.equal(await second.acquire('run-1'), true);
    second.release();
  } finally {
    Object.defineProperty(globalThis.navigator, 'locks', {
      configurable: true,
      value: originalLocks,
    });
  }
});

test('lock namespaces do not block unrelated ranked modules', async () => {
  const heldNames = new Set();
  const originalLocks = globalThis.navigator?.locks;
  Object.defineProperty(globalThis.navigator, 'locks', {
    configurable: true,
    value: {
      request: async (name, _options, callback) => {
        if (heldNames.has(name)) return callback(null);
        heldNames.add(name);
        try {
          return await callback({ name });
        } finally {
          heldNames.delete(name);
        }
      },
    },
  });

  try {
    const minigame = createExclusiveRunLock({ namespace: 'minigame-ranked' });
    const gamer = createExclusiveRunLock({ namespace: 'gamer-ranked' });
    assert.equal(await minigame.acquire('same-id'), true);
    assert.equal(await gamer.acquire('same-id'), true);
    minigame.release();
    gamer.release();
  } finally {
    Object.defineProperty(globalThis.navigator, 'locks', {
      configurable: true,
      value: originalLocks,
    });
  }
});
