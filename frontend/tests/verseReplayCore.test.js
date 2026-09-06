import assert from 'node:assert/strict';
import test from 'node:test';

await import('../public/verse-replay/replay-core.js');

const { decodeReplayBytes, decodeReplayText, snapshotToHex } = globalThis.ReplayCore;

function packedBoard(exponents) {
  return exponents.reduce(
    (packed, exponent, index) => (
      packed | (BigInt(exponent) << BigInt((15 - index) * 4))
    ),
    0n,
  );
}

function stateReplayBytes(records) {
  const bytes = new Uint8Array(records.length * 13);
  const view = new DataView(bytes.buffer);
  records.forEach((record, index) => {
    const offset = index * 13;
    const board = packedBoard(record.board);
    view.setUint32(offset, Number(board & 0xffffffffn), true);
    view.setUint32(offset + 4, Number((board >> 32n) & 0xffffffffn), true);
    view.setUint32(offset + 8, record.score, true);
    view.setUint8(offset + 12, record.move);
  });
  return bytes;
}

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

test('verse viewer decodes ranked 2048next records', () => {
  const replay = decodeReplayText(
    'REPLAY_v1RPL_B64_UlBMMUQAAhARg2QRAQAAAAEAAAACAAAAAwAAAASDAgRwb3cyg2UBAAt7g2YAhMwwCkc=',
  );
  assert.equal(replay.mode, 'ranked');
  assert.equal(replay.moveCount, 1);
  assert.equal(replay.scores[1], 8);
  assert.deepEqual(Array.from(replay.getBoardAt(1).slice(0, 4)), [8, 0, 2, 0].map((value) => (
    value ? Math.log2(value) : 0
  )));
});

test('verse viewer decodes 13-byte state VRS records', () => {
  const initial = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const afterLeft = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
  const afterRight = [1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
  const replay = decodeReplayBytes(stateReplayBytes([
    { board: initial, score: 0, move: 0 },
    { board: afterLeft, score: 4, move: 1 },
    { board: afterRight, score: 4, move: 2 },
  ]));

  assert.equal(replay.mode, 'state-vrs');
  assert.equal(replay.moveCount, 2);
  assert.deepEqual(replay.steps.map((step) => step.direction), ['left', 'right']);
  assert.deepEqual(
    replay.steps.map((step) => [step.spawnX, step.spawnY, step.spawnValue]),
    [[3, 3, 2], [0, 0, 2]],
  );
  assert.deepEqual(Array.from(replay.scores), [0, 4, 4]);
  assert.deepEqual(Array.from(replay.getBoardAt(2)), afterRight);
  assert.equal(replay.unknownTimings, 2);
  assert.equal(replay.playbackTimeMs, 200);
});

test('13-byte state VRS supports the packed 32k merge transition', () => {
  const initial = [15, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const afterLeft = [15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
  const replay = decodeReplayBytes(stateReplayBytes([
    { board: initial, score: 0, move: 0 },
    { board: afterLeft, score: 65536, move: 1 },
  ]));

  assert.equal(replay.steps[0].special32k, true);
  assert.equal(replay.steps[0].spawnValue, 2);
  assert.deepEqual(Array.from(replay.getBoardAt(1)), afterLeft);
  assert.equal(replay.scores[1], 65536);
});

test('13-byte state VRS rejects decreasing scores', () => {
  const board = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const moved = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
  assert.throws(
    () => decodeReplayBytes(stateReplayBytes([
      { board, score: 0, move: 0 },
      { board: moved, score: 4, move: 1 },
      { board: moved, score: 3, move: 1 },
    ])),
    /分数低于上一条/,
  );
});

test('13-byte state VRS rejects an invalid board transition', () => {
  const board = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const invalid = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
  assert.throws(
    () => decodeReplayBytes(stateReplayBytes([
      { board, score: 0, move: 0 },
      { board: invalid, score: 4, move: 1 },
    ])),
    /无法还原为合法移动和一次出数/,
  );
});

test('byte decoder keeps supporting textual Verse replay files', () => {
  const replay = decodeReplayBytes(new TextEncoder().encode('4x4-1_00000g'));
  assert.equal(replay.mode, '1');
  assert.equal(replay.moveCount, 0);
  assert.deepEqual(Array.from(replay.getBoardAt(0).slice(0, 4)), [1, 1, 0, 0]);
});

test('byte decoder rejects files larger than 500 KB', () => {
  assert.throws(
    () => decodeReplayBytes(new Uint8Array(500 * 1024 + 1)),
    /不能超过 500 KB/,
  );
});
