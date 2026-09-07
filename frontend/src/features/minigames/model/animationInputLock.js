export function animationInputLockMs(animation = {}) {
  if (animation.followUp) {
    const follow = animation.followUp;
    return follow.lockInput === false ? 0 : Number(follow.delayMs || 0) + Number(follow.durationMs || 0);
  }
  const effects = [...(animation.effects || []), ...(animation.pageEffects || [])]
    .filter((effect) => effect.lockInput !== false);
  if (!effects.length) return 0;
  return effects.reduce((duration, effect) => Math.max(duration,
    Number(effect.delayMs || 0) + Number(effect.durationMs || effect.animDurationMs || 430)), 430);
}
