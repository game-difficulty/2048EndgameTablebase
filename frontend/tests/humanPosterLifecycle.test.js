import test from 'node:test';
import assert from 'node:assert/strict';
import { drawBestTenPoster, bestTenPreviewSize } from '../src/human/bestTenPoster.js';

test('preview covers CSS width and device pixels without exceeding its memory cap', () => {
  assert.deepEqual(bestTenPreviewSize(984, 1), { width: 984, height: 1661 });
  assert.deepEqual(bestTenPreviewSize(984, 2), { width: 1968, height: 3321 });
  assert.deepEqual(bestTenPreviewSize(390, 3), { width: 1170, height: 1974 });
  assert.equal(bestTenPreviewSize(984, 4).width, 2400);
});

test('a poster cancelled during font loading cannot reallocate its released canvas', async () => {
  const previous = globalThis.document;
  let ready, active = true;
  globalThis.document = { fonts: { ready: new Promise(resolve => { ready = resolve; }) } };
  try {
    const canvas = { width: 1, height: 1, getContext() { throw new Error('cancelled render touched canvas'); } };
    const render = drawBestTenPoster({ canvas, shouldRender: () => active });
    active = false;
    ready();
    assert.equal(await render, null);
    assert.equal(canvas.width, 1);
    assert.equal(canvas.height, 1);
  } finally {
    if (previous === undefined) delete globalThis.document;
    else globalThis.document = previous;
  }
});
