import test from 'node:test';
import assert from 'node:assert/strict';

import { isProfileOwner } from '../src/human/profileOwnership.js';

test('profile ownership accepts an authoritative server result', () => {
  assert.equal(isProfileOwner({ player: { id: 73 }, is_owner: true }, { id: 99 }), true);
});

test('profile ownership compares serialized IDs without type sensitivity', () => {
  assert.equal(isProfileOwner({ player: { id: 73 } }, { id: '73' }), true);
  assert.equal(isProfileOwner({ player: { id: '73' } }, { id: 73 }), true);
  assert.equal(isProfileOwner({ player: { id: 73 } }, { id: 74 }), false);
  assert.equal(isProfileOwner({ player: { id: 73 } }, null), false);
});
