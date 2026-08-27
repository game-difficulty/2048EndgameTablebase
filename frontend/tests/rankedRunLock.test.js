import assert from 'node:assert/strict';
import test from 'node:test';

import { createRankedRunLock } from '../src/features/gamer/services/rankedRunLock.js';

test('Web Locks allow only one tab to own a ranked run', async () => {
  let held = false;
  const originalLocks = globalThis.navigator?.locks;
  const locks = {
    request: async (_name, _options, callback) => {
      if (held) return callback(null);
      held = true;
      try {
        return await callback({ name: 'ranked-run' });
      } finally {
        held = false;
      }
    },
  };
  Object.defineProperty(globalThis.navigator, 'locks', {
    configurable: true,
    value: locks,
  });

  try {
    const first = createRankedRunLock();
    const second = createRankedRunLock();
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
