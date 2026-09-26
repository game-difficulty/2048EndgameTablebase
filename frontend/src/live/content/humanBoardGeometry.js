export function humanLiveBoardGeometry(rows, cols, width = 500, gap = 10) {
  const safeRows = Math.max(1, Number(rows) || 1);
  const safeCols = Math.max(1, Number(cols) || 1);
  const cell = (width - (safeCols + 1) * gap) / safeCols;
  return { width, height: cell * safeRows + (safeRows + 1) * gap, gap, cell };
}

// 38 px row, 6 px gap and 8 px list padding on both sides, matching Play.
export function humanLiveVisibleNodeCount(height) {
  return Math.max(0, Math.floor((Number(height) - 18 + 6) / 44));
}
