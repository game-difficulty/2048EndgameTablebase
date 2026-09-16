// Missing or unfamiliar metadata must not disable an otherwise usable table.
export function compileTableLayers(value) {
  if (value?.version !== 1 || !Number.isSafeInteger(value.nums_adjust)
    || !Array.isArray(value.ranges) || value.ranges.length > 4096) return null;
  let previous = -1;
  const ranges = [];
  for (const pair of value.ranges) {
    if (!Array.isArray(pair) || pair.length !== 2 || !pair.every(Number.isSafeInteger)
      || pair[0] < 0 || pair[0] > pair[1] || pair[1] > 0xffffffff || pair[0] <= previous) return null;
    ranges.push([...pair]);
    previous = pair[1];
  }
  return { numsAdjust: value.nums_adjust, ranges };
}

export function tableLayerAvailable(maskedBoard, coverage) {
  if (!coverage) return true;
  // The native reader receives the same 4-bit masked board and uses (sum + adjustment) / 2.
  const sum = maskedBoard.reduce((total, value) => total + Math.min(value, 32768), 0);
  const layer = (sum + coverage.numsAdjust) / 2;
  if (!Number.isSafeInteger(layer) || layer < 0) return true;
  let low = 0, high = coverage.ranges.length - 1;
  while (low <= high) {
    const middle = (low + high) >>> 1;
    const [start, end] = coverage.ranges[middle];
    if (layer < start) high = middle - 1;
    else if (layer > end) low = middle + 1;
    else return true;
  }
  return false;
}
