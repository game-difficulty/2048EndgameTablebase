import test from 'node:test';
import assert from 'node:assert/strict';
import { createBoardSwipe } from '../src/projects/boardSwipe.js';

function setup() {
  const moves = [];
  let disabled = false;
  const target = { setPointerCapture() {}, hasPointerCapture() { return true; }, releasePointerCapture() {} };
  const event = (x, y, id = 1) => ({ clientX: x, clientY: y, pointerId: id, isPrimary: true,
    pointerType: 'touch', currentTarget: target, target, preventDefault() {}, stopPropagation() {} });
  return { moves, event, disable: () => { disabled = true; }, swipe: createBoardSwipe(() => disabled, direction => moves.push(direction)) };
}
test('board background, gaps and borders accept a gesture and move before release', () => {
  const { swipe, event, moves } = setup();
  swipe.down(event(0, 0)); // No tile target or cell index required.
  swipe.drag(event(17, 0)); assert.deepEqual(moves, []);
  swipe.drag(event(20, 0)); assert.deepEqual(moves, ['right']);
  swipe.drag(event(60, 0)); swipe.up(event(80, 0));
  assert.deepEqual(moves, ['right']);
  swipe.down(event(30, 30)); swipe.drag(event(30, 5));
  assert.deepEqual(moves, ['right', 'up']);
});
test('cancelled, secondary and disabled gestures never move', () => {
  const { swipe, event, moves, disable } = setup();
  swipe.down(event(0, 0)); swipe.drag(event(100, 0, 2));
  swipe.cancel(event(0, 0)); swipe.up(event(100, 0));
  swipe.down(event(0, 0)); disable(); swipe.drag(event(100, 0));
  assert.deepEqual(moves, []);
});
