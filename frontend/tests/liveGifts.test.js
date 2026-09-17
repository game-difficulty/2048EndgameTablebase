import test from 'node:test';
import assert from 'node:assert/strict';
import { GiftEvents, mergeGiftHistory } from '../src/live/giftEvents.js';
import { sendGift } from '../src/live/giftApi.js';
import { liveSupporterLevel, giftEffectDuration, entranceChat } from '../src/live/supporterIdentity.js';
import { giftAnimation, giftChat, mergeLiveChat, referenceArtwork, giftAsset } from '../src/live/giftArtwork.js';
import { existsSync, readFileSync } from 'node:fs';
import { giftPrice } from '../src/live/giftPrice.js';

test('bulk and custom gift totals match half-even token rounding', () => {
  for (const quantity of [1,10,100,1000]) assert.equal(giftPrice({base_units:2000,global_multiplier_units:333},quantity),666*quantity);
  assert.equal(giftPrice({base_units:1,global_multiplier_units:500},1),0);
  assert.equal(giftPrice({base_units:3,global_multiplier_units:500},1),2);
  for(const quantity of [0,1001,1.5,NaN])assert.equal(giftPrice({base_units:2000,global_multiplier_units:1000},quantity),null);
  assert.equal(giftPrice({base_units:65536000,global_multiplier_units:1000},1000),65536000000);
});

const gift = (id, other = {}) => ({ id, combo_id: id, type: 'gift', tier: 1, at: 1, combo_count: 1, ...other });

test('iii has inline motion for all senders without a ceremony', () => {
  for (const supporter_level of [0,1,2]) {
    assert.equal(giftAnimation(gift('iii-test', {gift_id:'iii', actor:{supporter_level}})), 'inline');
  }
  assert.equal(giftPrice({base_units:111000,global_multiplier_units:1000},1),111000);
});
test('only designated gifts animate, with supporter checks independent of rarity', () => {
  for (const gift_id of ['button','tea','whale','moai','chicken','serious','rip']) {
    assert.equal(giftAnimation(gift('a', { gift_id })), null);
    for (const supporter_level of [1,2]) assert.equal(giftAnimation(gift('a', { gift_id, actor: { supporter_level } })), 'supporter');
  }
  for (const gift_id of ['final','2048','crown','legend']) assert.equal(giftAnimation(gift('a', { gift_id })), 'ceremony');
  for (const gift_id of ['heart','flowers','two','four','dealer','666','bug','klbm','knowledge','meaning']) {
    assert.equal(giftAnimation(gift('a', { gift_id, actor: { supporter_level: 2 } })), null);
  }
});

test('RIP supporter animation ships as a small GIF without replacing the static gift icon', () => {
  const file = readFileSync(new URL('../public/live-gifts/rip-motion.gif', import.meta.url));
  assert.equal(file.subarray(0, 6).toString(), 'GIF89a');
  assert.equal(file.readUInt16LE(6), 240);
  assert.equal(file.readUInt16LE(8), 100);
  assert.ok(file.length < 50000);
  assert.equal(referenceArtwork.rip, undefined);
});
test('all reference posters are static; animated media is separate and bounded', () => {
  for (const asset of Object.values(referenceArtwork)) {
    const path = new URL(`../public${giftAsset(asset)}`, import.meta.url);
    assert.ok(existsSync(path));
    assert.equal(readFileSync(path).includes(Buffer.from('ANIM')), false);
  }
  for (const asset of ['button','tea','moai']) {
    const file = readFileSync(new URL(`../public${giftAsset(`${asset}-motion`)}`, import.meta.url));
    assert.ok(file.includes(Buffer.from('ANIM')));
    assert.ok(file.length < 70000);
  }
});
test('chat combos preserve DOM identity, place, count and recover without replays', () => {
  const first = gift('a', { gift_id: 'heart', actor: { name: 'A' } });
  const later = gift('b', { combo_id: 'a', combo_count: 6, at: 3, actor: first.actor });
  let chat = mergeLiveChat([], [first, { type: 'chat', id: 'text', text: 'Hello', at: 2 }]);
  chat = mergeLiveChat(chat, [later, first]);
  assert.equal(chat.length, 2);
  assert.equal(chat[0].id, 'gift:a');
  assert.equal(chat[0].combo_count, 6);
  assert.equal(chat[0].at, 1);
  assert.equal(chat[0].name, 'A');
  assert.deepEqual(mergeLiveChat(chat, [giftChat(later)]), chat);
  assert.equal(mergeLiveChat([], Array.from({length: 110}, (_, n) => gift(String(n)))).length, 100);
});
test('expensive ceremonies run one at a time while ordinary banners keep flowing', () => {
  const events = new GiftEvents();
  events.receive(gift('gold', { gift_id: 'crown', tier: 3 }), 1000);
  events.receive(gift('legend', { gift_id: 'legend', tier: 4 }), 1000);
  events.receive(gift('normal', { gift_id: 'heart', tier: 0 }), 1000);
  assert.deepEqual(events.active.map(item => item.id), ['gold','normal']);
  events.tick(6001);
  assert.deepEqual(events.active.map(item => item.id), ['legend']);
});
test('first-tier entrances are chat-only; gold entrances last five seconds', () => {
  const events = new GiftEvents();
  const first = { id: 'first', type: 'entrance', at: 1, actor: { name: 'First', supporter: true, supporter_level: 1 } };
  events.receive(first, 1000);
  assert.equal(events.active.length, 0);
  assert.equal(entranceChat(first).name, 'First');
  const gold = { ...first, id: 'gold', actor: { name: 'Gold', supporter_level: 2 } };
  events.receive(gold, 1000);
  assert.equal(events.active[0].until, 6000);
  assert.equal(liveSupporterLevel({ supporter: true }), 1);
  assert.equal(liveSupporterLevel({}), 0);
});
test('gold gifts retain their rarity and get an extra 1.5 seconds, not shortened by combos', () => {
  const events = new GiftEvents();
  const event = gift('gold', { tier: 4, actor: { supporter_level: 2 } });
  assert.equal(giftEffectDuration(event), 6500);
  events.receive(event, 1000);
  events.receive({ ...event, id: 'next', combo_count: 5 }, 1100);
  assert.equal(events.active[0].until, 7500);
  assert.equal(events.active[0].tier, 4);
});
test('combo updates do not restart entry or extend beyond eight seconds', () => {
  const events = new GiftEvents();
  events.receive(gift('a'), 1000);
  events.receive(gift('b', { combo_id: 'a', combo_count: 5 }), 3000);
  assert.equal(events.active.length, 1);
  assert.equal(events.active[0].started, 1000);
  assert.equal(events.active[0].combo_count, 5);
  events.receive(gift('b', { combo_id: 'a', combo_count: 7 }), 3500);
  assert.equal(events.active[0].combo_count, 5);
  for (let now = 4000; now < 9000; now += 1000) events.receive(gift(String(now), { combo_id: 'a', combo_count: now, at: now / 1000 }), now);
  assert.equal(events.active[0].until, 9000);
  events.tick(9000);
  assert.equal(events.active.length, 0);
});
test('bounded priority queue, stale events and mobile single lane', () => {
  const events = new GiftEvents();
  events.receive(gift('old'), 40000);
  assert.equal(events.active.length, 0);
  for (let i = 0; i < 40; i++) events.receive(gift(String(i), { tier: i === 35 ? 4 : 0 }), 1000, 1);
  assert.equal(events.active.length, 1);
  assert.equal(events.queue.length, 20);
  events.tick(4000, 1);
  assert.equal(events.active[0].id, '35');
});
test('history merges combos without replaying duplicate or regressing totals', () => {
  const result = mergeGiftHistory([gift('a', { combo_count: 5 })], [gift('b', { combo_id: 'a', combo_count: 2 })]);
  assert.equal(result.length, 1);
  assert.equal(result[0].combo_count, 5);
});
test('network retry preserves the exact purchase identity', async t => {
  const requests = [];
  t.mock.method(globalThis, 'fetch', async (path, options) => {
    requests.push(JSON.parse(options.body));
    if (requests.length === 1) throw new TypeError('network');
    return { ok: true, json: async () => ({ status: 'sent' }) };
  });
  const request = { request_id: 'same-id', quantity: 5 };
  assert.equal((await sendGift(request)).status, 'sent');
  assert.deepEqual(requests, [request, request]);
});
test('uncertain response checks committed order before reporting a failed retry', async t => {
  let calls = 0;
  t.mock.method(globalThis, 'fetch', async path => {
    calls++;
    if (calls === 1) throw new TypeError('network');
    if (calls === 2) return { ok: false, status: 429, json: async () => ({ detail: 'rate_limit' }) };
    assert.match(path, /orders\/same-id$/);
    return { ok: true, json: async () => ({ status: 'sent' }) };
  });
  assert.equal((await sendGift({ request_id: 'same-id' })).status, 'sent');
});
