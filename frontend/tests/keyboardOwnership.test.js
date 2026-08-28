import assert from 'node:assert/strict';
import test from 'node:test';

import {
  KEYBOARD_OWNERS,
  keyboardInputAllowed,
  keyboardOwner,
  resetKeyboardOwnership,
  setKeyboardOwner,
  setSplitKeyboardMode,
} from '../src/app/keyboardOwnership.js';

test.afterEach(() => {
  resetKeyboardOwnership();
});

test('only the explicit owner receives input in split keyboard mode', () => {
  setSplitKeyboardMode(true);

  assert.equal(keyboardOwner.value, KEYBOARD_OWNERS.PRIMARY);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY), true);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.TRAINER), false);

  setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);

  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY), false);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.TRAINER), true);
});

test('normal single-page mode does not block the active page', () => {
  setKeyboardOwner(KEYBOARD_OWNERS.TRAINER);
  setSplitKeyboardMode(false);

  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.PRIMARY), true);
  assert.equal(keyboardInputAllowed(KEYBOARD_OWNERS.TRAINER), true);
});
