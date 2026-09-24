import test from 'node:test';
import assert from 'node:assert/strict';
import { exportCurrentReplay } from '../src/human/replayExport.js';
import { initialState, nextMove, VARIANTS } from '../src/human/engine.js';
await import('../public/verse-replay/replay-core.js');
const { decodeReplayBytes, decodeReplayText } = globalThis.ReplayCore;
const seed = '00000001000000020000000300000004';

for (const variant of Object.keys(VARIANTS)) {
  test(`current ${variant} game round-trips as clipboard text and a binary file`, () => {
    let run = { ...initialState('export-fixture', variant, seed), id: 'export-fixture', variant, seed };
    const events = [];
    for (let i = 0; i < 100; i++) {
      const next = [3, 2, 1, 0].map(d => nextMove(run, d, i ? 137 * i : 0)).find(Boolean);
      if (!next) break;
      events.push(next.event); run = next.state;
    }
    const before = structuredClone(run), exported = exportCurrentReplay(run, events);
    for (const replay of [decodeReplayBytes(exported.binary), decodeReplayText(exported.text)]) {
      assert.deepEqual([replay.height, replay.width], VARIANTS[variant]);
      assert.equal(replay.moveCount, run.seq); assert.equal(replay.scores[run.seq], run.score);
      assert.equal(replay.knownTimeMs, run.elapsed); assert.equal(replay.unknownTimings, 0);
      assert.deepEqual(Array.from(replay.getBoardAt(run.seq), e => e ? 2 ** e : 0), run.board);
    }
    assert.deepEqual(run, before);
    assert.match(exported.filename, /\.vrs$/);
    // This header carries only the two initial tiles; it has no seed metadata extension.
    assert.equal(exported.binary[6], 2); assert.equal(exported.binary[9], events[0][0]);
  });
}
test('zero-move games and maximum millisecond deltas preserve exact state and timing', () => {
  const run = { ...initialState('empty', '4x4', seed), id: 'empty', variant: '4x4', seed };
  assert.equal(decodeReplayText(exportCurrentReplay(run, []).text).moveCount, 0);
  const next = [0, 1, 2, 3].map(d => nextMove(run, d, 0xffffffff)).find(Boolean);
  const result = decodeReplayBytes(exportCurrentReplay(next.state, [next.event]).binary);
  assert.equal(result.knownTimeMs, 0xffffffff);
});
test('snapshot sequence mismatch is rejected and the 200k-step encoding fits viewer limits', () => {
  const run = { ...initialState('limit', '4x4', seed), id: 'limit', variant: '4x4', seed };
  assert.throws(() => exportCurrentReplay({ ...run, seq: 1 }, []), /invalid_local_replay/);
  const encoded = exportCurrentReplay({ ...run, seq: 200000 }, Array.from({ length: 200000 }, () => [0, 0xffffffff]));
  assert.ok(encoded.binary.length < 2 * 1024 * 1024);
  assert.ok(encoded.text.length < 2 * 1024 * 1024);
});
