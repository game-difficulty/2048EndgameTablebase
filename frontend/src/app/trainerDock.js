export const TRAINER_DOCK_PLACEMENTS = Object.freeze({
  NONE: 'none',
  RIGHT: 'right',
  BOTTOM: 'bottom',
});

export const TRAINER_DOCK_LAYOUT = Object.freeze({
  RIGHT_WIDTH_PX: 450,
  RIGHT_BOARD_WIDTH_PX: 324,
  MIN_VIEWPORT_WIDTH_PX: 900,
  MIN_VIEWPORT_HEIGHT_PX: 600,
});

const VALID_PLACEMENTS = new Set(Object.values(TRAINER_DOCK_PLACEMENTS));

export const normalizeTrainerDockPlacement = (placement) => (
  VALID_PLACEMENTS.has(placement) ? placement : TRAINER_DOCK_PLACEMENTS.NONE
);

export const isTrainerDocked = (placement) => (
  normalizeTrainerDockPlacement(placement) !== TRAINER_DOCK_PLACEMENTS.NONE
);

export const trainerDockAvailable = (width, height) => (
  Number(width) >= TRAINER_DOCK_LAYOUT.MIN_VIEWPORT_WIDTH_PX
  && Number(height) >= TRAINER_DOCK_LAYOUT.MIN_VIEWPORT_HEIGHT_PX
);

export const resolveTrainerJumpDockPlacement = ({ placement, dockAvailable, sourceIsHelp }) => {
  if (!dockAvailable) return TRAINER_DOCK_PLACEMENTS.NONE;
  const normalized = normalizeTrainerDockPlacement(placement);
  if (isTrainerDocked(normalized)) return normalized;
  return sourceIsHelp ? TRAINER_DOCK_PLACEMENTS.RIGHT : TRAINER_DOCK_PLACEMENTS.NONE;
};
