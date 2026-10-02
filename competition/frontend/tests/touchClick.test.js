import test from 'node:test';
import assert from 'node:assert/strict';
import { bindTouchClick } from '../src/touchClick.js';

function setup() {
  const handlers = {};
  let clicks = 0;
  const element = {
    disabled: false,
    addEventListener(name, handler) { handlers[name] = handler; },
    removeEventListener(name) { delete handlers[name]; },
    getBoundingClientRect: () => ({ left: 0, top: 0, right: 100, bottom: 50 }),
    click() { clicks++; },
  };
  const cleanup = bindTouchClick(element);
  const event = (x = 20, y = 20) => ({ touches: [{ identifier: 1, clientX: x, clientY: y }],
    changedTouches: [{ identifier: 1, clientX: x, clientY: y }],
    prevented: false, preventDefault() { this.prevented = true; } });
  return { handlers, element, event, cleanup, clicks: () => clicks };
}

test('one touch tap activates the existing click handler and suppresses the compatibility click', () => {
  const { handlers, event, clicks, cleanup } = setup();
  handlers.touchstart(event());
  const end = event(23, 22);
  handlers.touchend(end);
  assert.equal(clicks(), 1);
  assert.equal(end.prevented, true);
  handlers.touchend(event());
  assert.equal(clicks(), 1);
  cleanup();
  assert.deepEqual(handlers, {});
});

test('scrolling away and returning, cancellation and multiple fingers are not taps', () => {
  const { handlers, event, clicks } = setup();
  handlers.touchstart(event()); handlers.touchmove(event(20, 40)); handlers.touchend(event());
  handlers.touchstart(event()); handlers.touchcancel(); handlers.touchend(event());
  const multi = event(); multi.touches.push({ identifier: 2, clientX: 25, clientY: 25 });
  handlers.touchstart(multi); handlers.touchend(event());
  assert.equal(clicks(), 0);
});

test('disabled buttons and releases outside the button do not activate undo', () => {
  const { handlers, element, event, clicks } = setup();
  element.disabled = true; handlers.touchstart(event()); handlers.touchend(event());
  element.disabled = false; handlers.touchstart(event()); element.disabled = true; handlers.touchend(event());
  element.disabled = false; handlers.touchstart(event(98, 20)); handlers.touchend(event(103, 20));
  assert.equal(clicks(), 0);
});
