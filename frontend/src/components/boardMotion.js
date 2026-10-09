// 2048Verse's Board CSS: 100ms slide, then 200ms pop/appear.
// Keep timers and CSS in step so hidden new tiles do not start animating early.
export const BOARD_ANIMATION_DURATION = 300;
export const BOARD_SLIDE_DURATION = 100;
export const BOARD_POP_DURATION = 200;
export const BOARD_MERGE_REVEAL_DELAY = 100;
export const BOARD_SPAWN_REVEAL_DELAY = 100;

export function boardAnimationTiming(duration = BOARD_ANIMATION_DURATION, appearDelay) {
  const total = Math.max(0, Number(duration) || 0);
  const scale = total / BOARD_ANIMATION_DURATION;
  const slide = BOARD_SLIDE_DURATION * scale;
  const pop = BOARD_POP_DURATION * scale;
  const spawn = appearDelay == null ? BOARD_SPAWN_REVEAL_DELAY * scale : Math.max(0, Number(appearDelay) || 0);
  return { slide, pop, merge: slide, spawn, settle: Math.max(total, spawn + pop) };
}

export function boardMotionStyle(duration = BOARD_ANIMATION_DURATION) {
  const { slide, pop } = boardAnimationTiming(duration);
  return { '--board-slide-duration': `${slide}ms`, '--board-pop-duration': `${pop}ms` };
}
