import test from 'node:test';
import assert from 'node:assert/strict';
import { GiftSpotlights, SPOTLIGHT_DURATION } from '../src/live/giftSpotlights.js';

const event = (id, extra = {}) => ({id, combo_id: id, type: 'gift', gift_id: 'rip', bulk_effect: true,
  combo_count: 100, at: 1, actor: {supporter_level: 0}, ...extra});

test('only server-qualified events start a spotlight, including non-supporters and bulk 1000', () => {
  const events = new GiftSpotlights();
  events.receive(event('small', {bulk_effect: false, combo_count: 99}), 1000);
  assert.equal(events.active, null);
  events.receive(event('big', {combo_count: 1000}), 1000);
  assert.equal(events.active.key, 'big');
});

test('combo updates and retries do not restart or replay the spotlight', () => {
  const events = new GiftSpotlights();
  events.receive(event('first'), 1000);
  const until = events.active.until;
  events.receive(event('next', {combo_id: 'first', combo_count: 150, bulk_effect: false}), 1100);
  assert.equal(events.active.combo_count, 150);
  assert.equal(events.active.until, until);
  events.tick(until);
  events.receive(event('first'), until);
  assert.equal(events.active, null);
  events.receive(event('new-combo', {at: until/1000}), until);
  assert.equal(events.active.key, 'new-combo');
});

test('spotlights serialize and expire without unbounded queues or stale replay', () => {
  const events = new GiftSpotlights();
  events.receive(event('stale'), 40000);
  assert.equal(events.active, null);
  for (let i = 0; i < 20; i++) events.receive(event(`gift-${i}`), 1000);
  assert.equal(events.queue.length, 6);
  assert.equal(events.active.key, 'gift-0');
  events.tick(1000 + SPOTLIGHT_DURATION);
  assert.equal(events.active.key, 'gift-1');
  events.clear();
  events.receive(event('gift-1'), 6000);
  assert.equal(events.active, null);
  assert.equal(events.queue.length, 0);
});
