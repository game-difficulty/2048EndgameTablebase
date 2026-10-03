import { threeByThreeTileStyle } from '../components/threeByThreeTileStyle.js';

// Pixel sizes also work without container units and follow the actual tile,
// rather than the root font size or the device's screen width.
export function humanBoardTileStyle(rows, cols, side) {
  if (!Number.isFinite(side) || side <= 0) return {};
  if (rows === 3 && cols === 3) return threeByThreeTileStyle(rows, cols, side);
  return {
    '--tile-label-small': `${side * 0.42}px`,
    '--tile-label-medium': `${side * 0.33}px`,
    '--tile-label-large': `${side * 0.28}px`,
  };
}
