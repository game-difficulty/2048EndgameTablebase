import test from 'node:test';
import assert from 'node:assert/strict';
import { openAsyncLink } from '../src/services/openAsyncLink.js';

test('reserves a detached tab before the asynchronous link request completes', async () => {
  const events = [];
  const tab = { opener: {}, closed: false, location: { replace: url => events.push(url) } };
  let finish;
  const pending = openAsyncLink(() => new Promise(resolve => { finish = resolve; }), {
    open: () => { events.push('open'); return tab; },
  });
  assert.deepEqual(events, ['open']);
  assert.equal(tab.opener, null);
  finish('https://2048tables.online/?tab=replay');
  await pending;
  assert.equal(events[1], 'https://2048tables.online/?tab=replay');
});

test('failed link request closes the reserved tab and propagates its error', async () => {
  let closed = false;
  await assert.rejects(openAsyncLink(async () => { throw new Error('expired'); }, {
    open: () => ({ closed: false, close: () => { closed = true; } }),
  }), /expired/);
  assert.equal(closed, true);
});

test('blocked popup falls back to navigation in the current tab', async () => {
  let destination;
  await openAsyncLink(async () => '/viewer', {
    open: () => null, location: { assign: url => { destination = url; } },
  });
  assert.equal(destination, '/viewer');
});

test('closing the reserved tab does not navigate the original page', async () => {
  await openAsyncLink(async () => '/viewer', {
    open: () => ({ closed: true }), location: { assign: () => assert.fail('unexpected navigation') },
  });
});
