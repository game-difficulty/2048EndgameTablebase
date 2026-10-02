import test from 'node:test';
import assert from 'node:assert/strict';
import { createRenderer, nextTick, shallowRef } from 'vue';
import { useBoardAnimation } from '../src/components/useBoardAnimation.js';
import { createBoardFrame, createSnapshotBoardFrame } from '../src/components/boardFrame.js';

function mountAnimation(appearDelay) {
  const board = Array(16).fill(0);
  board[1] = 2;
  const frame = shallowRef(createSnapshotBoardFrame(0, board));
  let tiles;
  const renderer = createRenderer({
    createComment: () => ({}), insert() {}, remove() {}, parentNode() {}, nextSibling() {},
  });
  const app = renderer.createApp({
    setup() {
      tiles = useBoardAnimation({
        get frame() { return frame.value; }, animationDuration: 300,
        animationAppearDelay: appearDelay, isVariant: false,
      }, shallowRef({ visibleIndices: Array.from({ length: 16 }, (_, i) => i) }),
      shallowRef('4:4'), shallowRef({ offsetHeight: 400 })).activeTiles;
      return () => null;
    },
  });
  app.mount({});
  return { frame, tiles, app, board };
}

for (const [label, override, delay] of [['Play', 100, 100], ['main default', undefined, 125]]) {
  test(`${label} reveals spawn at its configured time and settles cleanly`, async t => {
    globalThis.document = { body: {} };
    t.after(() => { delete globalThis.document; });
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const { frame, tiles, app, board } = mountAnimation(override);
    t.after(() => app.unmount());
    const target = Array(16).fill(0);
    target[0] = target[15] = 2;
    frame.value = createBoardFrame({ revision: 1, kind: 'move', fromBoard: board, toBoard: target,
      metadata: { direction: 'left', slide_distances: [0, 1, ...Array(14).fill(0)],
        pop_positions: Array(16).fill(0), appear_tile: { index: 15, value: 2 } } });
    await nextTick();
    await nextTick();
    const spawn = () => tiles.value.find(tile => tile.isNew);
    assert.equal(spawn().isHidden, true);
    t.mock.timers.tick(delay - 1);
    assert.equal(spawn().isHidden, true);
    t.mock.timers.tick(1);
    assert.equal(spawn().isHidden, false);
    t.mock.timers.tick(300 - delay);
    assert.equal(tiles.value.length, 2);
    assert.ok(tiles.value.every(tile => !tile.isHidden && !tile.isNew));
  });
}

test('a snapshot interrupt cancels the pending spawn reveal', async t => {
  globalThis.document = { body: {} };
  t.after(() => { delete globalThis.document; });
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { frame, tiles, app, board } = mountAnimation(100);
  t.after(() => app.unmount());
  const target = [...board]; target[15] = 4;
  frame.value = createBoardFrame({ revision: 1, kind: 'move', fromBoard: board, toBoard: target,
    metadata: { appear_tile: { index: 15, value: 4 } } });
  await nextTick(); await nextTick();
  assert.ok(tiles.value.some(tile => tile.isHidden));
  const replacement = Array(16).fill(0); replacement[5] = 8;
  frame.value = createSnapshotBoardFrame(2, replacement);
  await nextTick();
  t.mock.timers.tick(500);
  assert.deepEqual(tiles.value.map(tile => [tile.row, tile.col, tile.value, tile.isHidden]), [[1, 1, 8, false]]);
});
