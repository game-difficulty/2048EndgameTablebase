import assert from 'node:assert/strict';
import test from 'node:test';

import { clientPointToElementSpace } from '../src/features/minigames/model/pointerCoordinates.js';

function createElement({
  left = 0,
  top = 0,
  renderedWidth,
  renderedHeight,
  layoutWidth,
  layoutHeight,
}) {
  return {
    offsetWidth: layoutWidth,
    offsetHeight: layoutHeight,
    getBoundingClientRect: () => ({
      left,
      top,
      width: renderedWidth,
      height: renderedHeight,
    }),
  };
}

test('maps client coordinates into an element scaled down by the fixed layout', () => {
  const element = createElement({
    left: 80,
    top: 40,
    renderedWidth: 300,
    renderedHeight: 200,
    layoutWidth: 600,
    layoutHeight: 400,
  });

  assert.deepEqual(clientPointToElementSpace(element, 230, 140), { x: 300, y: 200 });
});

test('maps client coordinates independently for non-uniform display scaling', () => {
  const element = createElement({
    left: 10,
    top: 20,
    renderedWidth: 800,
    renderedHeight: 150,
    layoutWidth: 400,
    layoutHeight: 300,
  });

  assert.deepEqual(clientPointToElementSpace(element, 410, 95), { x: 200, y: 150 });
});

test('keeps coordinates unchanged when no transform is applied', () => {
  const element = createElement({
    left: 12,
    top: 18,
    renderedWidth: 500,
    renderedHeight: 500,
    layoutWidth: 500,
    layoutHeight: 500,
  });

  assert.deepEqual(clientPointToElementSpace(element, 137, 268), { x: 125, y: 250 });
});
