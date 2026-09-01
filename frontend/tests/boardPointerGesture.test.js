import assert from 'node:assert/strict';
import test from 'node:test';

import {
  boardSwipeDirection,
  boardSwipeThreshold,
} from '../src/components/boardPointerGesture.js';

test('scales the swipe threshold with the displayed board size', () => {
  assert.equal(boardSwipeThreshold(135), 10);
  assert.equal(boardSwipeThreshold(300), 13.5);
  assert.equal(boardSwipeThreshold(442), 19.89);
  assert.equal(boardSwipeThreshold(600), 24);
});

test('recognizes short mobile swipes without accepting tiny taps', () => {
  assert.equal(boardSwipeDirection(9, 1, 135), null);
  assert.equal(boardSwipeDirection(11, 2, 135), 'right');
  assert.equal(boardSwipeDirection(-2, -12, 135), 'up');
});

test('uses the dominant axis for diagonal gestures', () => {
  assert.equal(boardSwipeDirection(18, 12, 300), 'right');
  assert.equal(boardSwipeDirection(-8, 19, 300), 'down');
});
