import test from 'node:test';
import assert from 'node:assert/strict';
import { roomIdFromPath } from '../src/live/roomRoute.js';
import { createGiftClient } from '../src/features/gifts/giftApi.js';

test('legacy paths and named rooms resolve strictly without fallback for unknown paths', () => {
  for (const path of ['/', '/live/', '/live/index.html']) assert.equal(roomIdFromPath(path), 'ai-classic');
  assert.equal(roomIdFromPath('/rooms/second-ai'), 'second-ai');
  assert.equal(roomIdFromPath('/live/rooms/second-ai/'), 'second-ai');
  for (const path of ['/rooms/', '/rooms/a/extra', '/rooms/../a', '/unrelated', '/rooms/%2f', '/rooms/UPPER']) assert.equal(roomIdFromPath(path), null);
});

test('simultaneous gift clients and uncertain retries remain bound to the original target', async () => {
  const original = globalThis.fetch, calls = [];
  let failed = false;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: options.body });
    if (url.endsWith('/send') && !failed) { failed = true; throw new Error('connection lost'); }
    return { ok: true, json: async () => ({ status: 'sent' }) };
  };
  try {
    const a = createGiftClient({ base: '/api/live/rooms/a/gifts', target: 'live:a' });
    const b = createGiftClient({ base: '/api/battles/b/gifts', target: 'battle:b:player-2' });
    const purchase = { request_id: 'same-purchase', quantity: 1 };
    await a.sendGift(purchase);
    await b.giftApi('catalog');
    await a.sendGift(purchase, true);
    assert.deepEqual(calls.map(c => c.url), [
      '/api/live/rooms/a/gifts/send', '/api/live/rooms/a/gifts/send',
      '/api/battles/b/gifts/catalog', '/api/live/rooms/a/gifts/orders/same-purchase',
    ]);
    assert.equal(calls[0].body, calls[1].body);
    assert.notEqual(a.pendingKey(1), b.pendingKey(1));
  } finally { globalThis.fetch = original; }
});
