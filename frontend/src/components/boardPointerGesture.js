const DEFAULT_THRESHOLD_RATIO = 0.045;
const DEFAULT_THRESHOLD_MIN_PX = 10;
const DEFAULT_THRESHOLD_MAX_PX = 24;

export function boardSwipeThreshold(
  displaySize,
  {
    ratio = DEFAULT_THRESHOLD_RATIO,
    min = DEFAULT_THRESHOLD_MIN_PX,
    max = DEFAULT_THRESHOLD_MAX_PX,
  } = {},
) {
  const size = Number(displaySize);
  if (!Number.isFinite(size) || size <= 0) return max;
  return Math.min(max, Math.max(min, size * ratio));
}

export function boardSwipeDirection(dx, dy, displaySize) {
  const horizontal = Number(dx) || 0;
  const vertical = Number(dy) || 0;
  const absX = Math.abs(horizontal);
  const absY = Math.abs(vertical);
  if (Math.max(absX, absY) < boardSwipeThreshold(displaySize)) return null;
  if (absX >= absY) return horizontal >= 0 ? 'right' : 'left';
  return vertical >= 0 ? 'down' : 'up';
}
