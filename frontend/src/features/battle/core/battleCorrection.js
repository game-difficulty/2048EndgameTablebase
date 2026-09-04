const DIRECTIONS = new Set(['left', 'right', 'up', 'down']);

export function correctionOverlayForResult(result, now = Date.now()) {
  const correction = result?.mode_data?.correction;
  if (!correction || typeof correction !== 'object') return null;
  const selectedDirection = String(correction.selected_direction || '').toLowerCase();
  const standardDirection = String(correction.standard_direction || '').toLowerCase();
  if (!DIRECTIONS.has(selectedDirection) || !DIRECTIONS.has(standardDirection)) return null;
  const visibleUntil = Date.parse(String(correction.visible_until || ''));
  if (Number.isFinite(visibleUntil) && visibleUntil <= Number(now)) return null;
  return {
    selectedDirection,
    standardDirection,
    drop: Math.max(0, Math.min(1, Number(correction.goodness_drop) || 0)),
    previousBoardHex: String(correction.previous_board_hex || ''),
    previousRouteIndex: Number.isInteger(Number(correction.previous_route_index))
      ? Number(correction.previous_route_index)
      : null,
    visibleUntil: Number.isFinite(visibleUntil) ? visibleUntil : null,
  };
}
