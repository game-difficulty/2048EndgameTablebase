export const TRAINER_DOCK_PLACEMENTS = Object.freeze({
  NONE: 'none',
  RIGHT: 'right',
  BOTTOM: 'bottom',
});

export const TRAINER_DOCK_LAYOUT = Object.freeze({
  RIGHT_WIDTH_PX: 450,
  RIGHT_BOARD_WIDTH_PX: 324,
});

const VALID_DOCK_PLACEMENTS = new Set(Object.values(TRAINER_DOCK_PLACEMENTS));

export const normalizeTrainerDockPlacement = (placement) => (
  VALID_DOCK_PLACEMENTS.has(placement)
    ? placement
    : TRAINER_DOCK_PLACEMENTS.NONE
);

export const isTrainerDocked = (placement) => (
  normalizeTrainerDockPlacement(placement) !== TRAINER_DOCK_PLACEMENTS.NONE
);

export const resolveTrainerDockSurfaceHeight = ({
  placement,
  baseHeight,
  topBarHeight = 0,
} = {}) => {
  const normalizedBaseHeight = Number.isFinite(baseHeight) ? Math.max(0, baseHeight) : 0;
  if (normalizeTrainerDockPlacement(placement) !== TRAINER_DOCK_PLACEMENTS.BOTTOM) {
    return normalizedBaseHeight;
  }

  const normalizedTopBarHeight = Number.isFinite(topBarHeight)
    ? Math.min(normalizedBaseHeight, Math.max(0, topBarHeight))
    : 0;
  const contentPageHeight = normalizedBaseHeight - normalizedTopBarHeight;
  return normalizedTopBarHeight + contentPageHeight * 2;
};

export const resolveTrainerJumpDockPlacement = ({
  placement,
  dockAvailable,
  sourceIsHelp,
} = {}) => {
  if (!dockAvailable) return TRAINER_DOCK_PLACEMENTS.NONE;
  const normalized = normalizeTrainerDockPlacement(placement);
  if (isTrainerDocked(normalized)) return normalized;
  return sourceIsHelp
    ? TRAINER_DOCK_PLACEMENTS.RIGHT
    : TRAINER_DOCK_PLACEMENTS.NONE;
};
