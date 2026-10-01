export function threeByThreeTileStyle(rows, cols, side) {
  if (rows !== 3 || cols !== 3 || !Number.isFinite(side) || side <= 0) return {};
  return {
    '--board-gap': `${side * 0.036 * 3 / (1 - 4 * 0.036)}px`,
    '--tile-label-small': `${side * 0.42}px`,
    '--tile-label-medium': `${side * 0.33}px`,
    '--tile-label-large': `${side * 0.28}px`,
    '--tile-corner-radius': `${side * 0.03}px`,
  };
}
