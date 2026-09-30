import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_FEATURED_GIFT_IDS, defaultGiftOrder, normalizeGiftOrder, moveGift } from '../src/features/gifts/giftOrder.js';

const catalog = [
  'heart', 'flowers', 'two', 'four', 'dealer', '666', 'button', 'whale', 'moai',
  'meaning', 'rip', 'tea', 'new-gift',
].map(id => ({ id }));

test('default order preserves the existing nine gift bar choices before the catalogue remainder', () => {
  const order = defaultGiftOrder(catalog);
  const featured = DEFAULT_FEATURED_GIFT_IDS.filter(id => catalog.some(item => item.id === id));
  assert.deepEqual(order.slice(0, featured.length), featured);
  assert.equal(new Set(order).size, catalog.length);
  assert.deepEqual(new Set(order), new Set(catalog.map(item => item.id)));
});

test('saved order removes stale duplicates and appends new catalogue gifts', () => {
  const order = normalizeGiftOrder(['dealer', 'dealer', 'retired', 'two'], catalog);
  assert.deepEqual(order.slice(0, 2), ['dealer', 'two']);
  assert.equal(order.at(-1), 'new-gift');
  assert.equal(new Set(order).size, catalog.length);
});

test('moving across position nine changes the derived gift bar without losing gifts', () => {
  const order = defaultGiftOrder(catalog);
  const moved = moveGift(order, 'dealer', 0);
  assert.equal(moved[0], 'dealer');
  assert.ok(!moved.slice(0, 9).includes(order[8]));
  assert.deepEqual(new Set(moved), new Set(order));
});
