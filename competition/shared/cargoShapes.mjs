// Shared by the competition runtime and both player/spectator renderers.
export const CARGO_SHAPES = Object.freeze([
  Object.freeze({ key: 'square', name: '2×2', cells: [[0, 0], [0, 1], [1, 0], [1, 1]] }),
  Object.freeze({ key: 'horizontal', name: '1×2', cells: [[1, 0], [1, 1]] }),
  Object.freeze({ key: 'vertical-left', name: '2×1', cells: [[0, 0], [1, 0]] }),
  Object.freeze({ key: 'l2', name: 'L', cells: [[0, 1], [1, 0], [1, 1]] }),
  Object.freeze({ key: 'l3', name: 'L', cells: [[0, 0], [1, 0], [1, 1]] }),
  Object.freeze({ key: 'vertical-right', name: '2×1', cells: [[0, 1], [1, 1]] }),
]);

// Family first (25% each), then an equally weighted variant within the family.
export const CARGO_SHAPE_GROUPS = Object.freeze([
  Object.freeze([3, 4]), Object.freeze([1]), Object.freeze([2, 5]), Object.freeze([0]),
]);
