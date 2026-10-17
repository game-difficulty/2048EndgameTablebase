export function minigameBoardGeometry(width, rows, cols) {
  const density = Math.min(1, 5.5 / Math.max(rows, cols, 4));
  const gap = width * 0.025 * density;
  const height = width * rows / cols;
  return { width, height, gap,
    cellWidth: (width - gap * (cols + 1)) / cols,
    cellHeight: (height - gap * (rows + 1)) / rows };
}

export function minigameTilePosition(geometry, row, col) {
  return { width: `${geometry.cellWidth}px`, height: `${geometry.cellHeight}px`,
    transform: `translate(${col * (geometry.cellWidth + geometry.gap)}px, ${row * (geometry.cellHeight + geometry.gap)}px)` };
}

export function minigameSubgridCenter(geometry, row, col) {
  return { left: `${(col + 1) * (geometry.cellWidth + geometry.gap) - geometry.gap / 2}px`,
    top: `${(row + 1) * (geometry.cellHeight + geometry.gap) - geometry.gap / 2}px` };
}

export function minigameLabelSize(width, length, rem = 16) {
  const [minimum, ratio, maximum] = length >= 5 ? [0.8, 0.052, 1.55]
    : length === 4 ? [0.92, 0.058, 1.85] : length === 3 ? [1, 0.068, 2.15] : [1, 0.08, 2.5];
  return Math.max(minimum * rem, Math.min(width * ratio, maximum * rem));
}
