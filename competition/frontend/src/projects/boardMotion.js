// Keep the practice boards in step with frontend/src/components/BaseBoard.vue.
export const BOARD_ANIMATION_DURATION = 300;
export const BOARD_SLIDE_DURATION = BOARD_ANIMATION_DURATION / 3;
export const BOARD_POP_DURATION = BOARD_ANIMATION_DURATION * 2 / 3;
export const BOARD_MERGE_REVEAL_DELAY = BOARD_SLIDE_DURATION;
export const BOARD_SPAWN_REVEAL_DELAY = BOARD_ANIMATION_DURATION * 5 / 12;
// Quake: slide/merge, shift the cells, then pop the new spawn.
export const BOARD_QUAKE_SETTLE_DELAY = BOARD_SPAWN_REVEAL_DELAY + BOARD_POP_DURATION;
export const BOARD_QUAKE_DURATION = BOARD_QUAKE_SETTLE_DELAY + BOARD_POP_DURATION;
export const DICE_ROLL_DURATION = 700;
export const DICE_REVEAL_DURATION = 1400;

export function projectAnimationHoldMs(payload) {
  const transition = payload?.last_transition;
  return payload?.aftershock && transition?.quake && ['move', 'reshape'].includes(transition.kind)
    ? BOARD_QUAKE_DURATION : 0;
}
