import test from 'node:test';
import assert from 'node:assert/strict';
import { observeSurfaceBox } from '../src/live/roomSurfaceSize.js';

function surface({ observer = true } = {}) {
  let notify, watching = false, frame;
  const listeners = new Map(), style = { width: '710.5px', height: '420.25px' };
  const host = {
    getComputedStyle: () => style,
    addEventListener: (event, handler) => listeners.set(event, handler),
    removeEventListener: (event, handler) => { assert.equal(listeners.get(event), handler); listeners.delete(event); },
    requestAnimationFrame: handler => { frame = handler; return 1; },
    cancelAnimationFrame: () => { frame = null; },
  };
  const element = { ownerDocument: { defaultView: host }, getBoundingClientRect() { throw Error('Screen pixels would double-apply zoom'); } };
  if (observer) host.ResizeObserver = class {
    constructor(callback) { notify = callback; }
    observe(node) { assert.equal(node, element); watching = true; }
    disconnect() { watching = false; }
  };
  return { element, listeners, style, paint: () => frame?.(), changed: (width, height) => notify([{ contentRect: { width, height } }]), watching: () => watching };
}

test('fit uses unscaled layout pixels, updates for either constraint and releases the owning window', () => {
  const box = surface(), sides = [];
  const stop = observeSurfaceBox(box.element, (w, h) => sides.push(Math.min(w, h)));
  box.changed(710.5, 395.5);
  box.changed(210.75, 280);
  assert.deepEqual(sides, [420.25, 395.5, 210.75]);
  assert.equal(box.watching(), true);
  stop();
  assert.equal(box.watching(), false);
  assert.equal(box.listeners.size, 0);
});

test('a re-adopted surface measures and listens to its new document only', () => {
  const opener = surface(), pip = surface(), sizes = [];
  const stop = observeSurfaceBox(opener.element, (w, h) => sizes.push([w, h]));
  stop();
  // The real DOM node is unchanged; only its owner document changes.
  opener.element.ownerDocument = pip.element.ownerDocument;
  pip.element = opener.element;
  // Use the resize fallback here to exercise ownership without a DOM mock observer.
  pip.element.ownerDocument.defaultView.ResizeObserver = undefined;
  pip.style.width = '200px'; pip.style.height = '300px';
  const stopPip = observeSurfaceBox(pip.element, (w, h) => sizes.push([w, h]));
  pip.style.height = '150px'; pip.listeners.get('resize')(); pip.paint();
  assert.deepEqual(sizes, [[710.5, 420.25], [200, 300], [200, 150]]);
  assert.equal(opener.listeners.size, 0);
  stopPip(); assert.equal(pip.listeners.size, 0);
});

test('old engines without ResizeObserver still measure initially and on resize', () => {
  const box = surface({ observer: false }), sizes = [];
  const stop = observeSurfaceBox(box.element, (w, h) => sizes.push([w, h]));
  box.listeners.get('resize')();
  // A later resize listener changes the room scale/width before the next paint.
  box.style.width = '120px'; box.style.height = '160px'; box.paint();
  assert.deepEqual(sizes, [[710.5, 420.25], [120, 160]]);
  stop(); assert.equal(box.listeners.size, 0);
});
