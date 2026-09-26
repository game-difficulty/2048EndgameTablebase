import test from 'node:test';
import assert from 'node:assert/strict';
import { saveReplaySource, saveReplayPosition, restoreReplaySession } from '../src/features/replay/services/replaySessionStore.js';

test('analysis stage identity, bytes and progress survive a page reload', () => {
  const values = new Map();
  globalThis.window = { sessionStorage: {
    setItem: (key, value) => values.set(key, value), getItem: key => values.get(key) ?? null,
  } };
  try {
    const bytes = new Uint8Array([1, 2, 3]).buffer;
    assert.equal(saveReplaySource(bytes, { analysisArtifactId: 'stage-a', filename: 'stage.rpl' }), true);
    saveReplayPosition(42);
    const restored = restoreReplaySession();
    assert.equal(restored.analysisArtifactId, 'stage-a');
    assert.equal(restored.step, 42);
    assert.deepEqual(new Uint8Array(restored.buffer), new Uint8Array(bytes));
    saveReplaySource(bytes, { filename: 'ordinary.rpl' });
    assert.equal(restoreReplaySession().analysisArtifactId, '');
  } finally { delete globalThis.window; }
});

test('unavailable session storage is reported rather than claiming replay persistence', () => {
  globalThis.window = { sessionStorage: { setItem: () => { throw new Error('quota'); } } };
  try { assert.equal(saveReplaySource(new ArrayBuffer(0)), false); }
  finally { delete globalThis.window; }
});
