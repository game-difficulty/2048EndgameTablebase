import assert from 'node:assert/strict';
import test from 'node:test';

await import('../public/verse-replay/replay-core.js');

const { snapshotToHex } = globalThis.ReplayCore;

test('verse board snapshots use one hexadecimal nibble per cell', () => {
  assert.equal(
    snapshotToHex(Uint8Array.from([
      0, 1, 2, 3,
      4, 5, 6, 7,
      8, 9, 10, 11,
      12, 13, 14, 15,
    ])),
    '0123456789abcdef',
  );
});

test('verse board snapshots encode 65536 and larger tiles as f', () => {
  assert.equal(
    snapshotToHex(Uint8Array.from([
      16, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 17,
    ])),
    'f00000000000000f',
  );
});
