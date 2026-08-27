export const TRAINER_DOCK_PLACEMENTS = Object.freeze({
  NONE: 'none',
  RIGHT: 'right',
  BOTTOM: 'bottom',
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
