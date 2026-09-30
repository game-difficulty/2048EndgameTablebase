import test from 'node:test';
import assert from 'node:assert/strict';
import { drawBestTenPoster } from '../src/human/bestTenPoster.js';

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
