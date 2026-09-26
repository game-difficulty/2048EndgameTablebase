import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import { initialState, nextMove } from '../src/human/engine.js';

await import('../public/verse-replay/replay-core.js');

const { decodeReplayBytes, decodeReplayText, moveBoard, snapshotToHex } = globalThis.ReplayCore;

const humanSeed = '00000001000000020000000300000004';

function humanReplayBytes(variant, version) {
  let run = { ...initialState('viewer-test', variant, humanSeed), id: 'viewer-test', variant };
  const events = [];
  for (let index = 0; index < 40; index += 1) {
    const next = [3, 2, 1, 0].map((direction) => nextMove(run, direction, index * 137)).find(Boolean);
    if (!next) break;
    events.push(next.event);
    run = next.state;
  }
  const header = new TextEncoder().encode(JSON.stringify({
    version,
    rules_version: 1,
    run_id: run.id,
    variant,
    seed: humanSeed,
    reason: 'game_over',
    started_at: 0,
    timing: 'continuous-client-ms',
  }));
  const body = new Uint8Array(events.length * 5);
  const bodyView = new DataView(body.buffer);
  if (version === 1) {
    events.forEach(([code, delta], index) => {
      bodyView.setUint8(index * 5, code);
      bodyView.setUint32(index * 5 + 1, delta, true);
    });
  } else {
    events.forEach(([code, delta], index) => {
      body[index] = code;
      for (let plane = 0; plane < 4; plane += 1) {
        body[events.length * (plane + 1) + index] = (delta >>> (plane * 8)) & 255;
      }
    });
  }
  const bytes = new Uint8Array(8 + header.length + body.length);
  bytes.set(new TextEncoder().encode(`HPR${version}`));
  new DataView(bytes.buffer).setUint32(4, header.length, true);
  bytes.set(header, 8);
  bytes.set(body, 8 + header.length);
  return { bytes, run, events };
}

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

test('canonical reconstruction keeps a real 65536 tile and matching transition', () => {
  const board = Uint8Array.from([
    15, 0, 15, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  ]);
  const moved = moveBoard(board, 4, 3, 'left');

  assert.equal(moved.board[0], 16);
  assert.equal(moved.addedScore, 65536);
  assert.deepEqual(moved.transition.merges, [{ toIndex: 0, exponent: 16 }]);
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

test('RPL1 unknown timing sentinel stays unknown in the replay viewer', () => {
  const replay = decodeReplayText('REPLAY_v1RPL_B64_UlBMMUQAAgABP/////8PhCucGRo=');
  assert.equal(replay.moveCount, 1);
  assert.equal(replay.steps[0].deltaMs, null);
  assert.equal(replay.unknownTimings, 1);
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
  const afterRight = [1, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
  const replay = decodeReplayBytes(stateReplayBytes([
    { board: initial, score: 0, move: 0 },
    { board: afterLeft, score: 65536, move: 1 },
    { board: afterRight, score: 65536, move: 2 },
  ]));

  assert.equal(replay.steps[0].special32k, true);
  assert.equal(replay.steps[0].spawnValue, 2);
  assert.equal(replay.transitions[0].merges[0].exponent, 16);
  assert.deepEqual(
    Array.from(replay.getBoardAt(1)),
    [16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  );
  assert.deepEqual(
    Array.from(replay.getBoardAt(2)),
    [1, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  );
  assert.deepEqual(Array.from(replay.scores), [0, 65536, 65536]);
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

for (const version of [1, 2]) {
  test(`unified viewer decodes native HPR${version} 2x4 archives`, () => {
    const { bytes, run, events } = humanReplayBytes('2x4', version);
    const replay = decodeReplayBytes(bytes);
    assert.equal(replay.format, `hpr${version}`);
    assert.deepEqual([replay.height, replay.width], [2, 4]);
    assert.equal(replay.moveCount, events.length);
    assert.equal(replay.scores[events.length], run.score);
    assert.equal(replay.knownTimeMs, run.elapsed);
    assert.deepEqual(
      Array.from(replay.getBoardAt(events.length), (exponent) => exponent ? 2 ** exponent : 0),
      run.board,
    );
  });
}

test('Verse VRS fixtures use rows x columns for variant dimensions', () => {
  const fixtureDir = new URL('./fixtures/verse-replay/', import.meta.url);
  const fixtures = [
    ['Blueawa_3x4_2026-09-20_71356.vrs', 4, 3, 3244, 71356, '8192'],
    ['P-shiyi592_3x3_2026-09-20_11976.vrs', 3, 3, 691, 11976, '1024'],
    ['mmmcccc_4x4_2026-09-19_1285068.vrs', 4, 4, 41458, 1285068, '65536'],
    ['p56_4x4_2026-09-21_576348.vrs', 4, 4, 19976, 576348, '65536'],
    ['xzyszdj_2x4_2026-09-21_5228.vrs', 4, 2, 343, 5228, '512'],
  ];

  for (const [name, width, height, moveCount, score, finalMilestone] of fixtures) {
    const bytes = fs.readFileSync(new URL(name, fixtureDir));
    const replay = decodeReplayBytes(bytes);
    assert.equal(replay.width, width, name);
    assert.equal(replay.height, height, name);
    assert.equal(replay.moveCount, moveCount, name);
    assert.equal(replay.scores[moveCount], score, name);
    assert.equal(replay.getBoardAt(moveCount).length, width * height, name);
    assert.equal(replay.milestones.at(-1).key, finalMilestone, name);
  }
});

test('byte decoder rejects files larger than 2 MB', () => {
  assert.throws(
    () => decodeReplayBytes(new Uint8Array(2 * 1024 * 1024 + 1)),
    /不能超过 2 MB/,
  );
});
