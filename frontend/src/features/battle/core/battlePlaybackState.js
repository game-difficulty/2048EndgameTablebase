// Completed players may still have a certainty tail to animate; interrupted players do not.
export function isBattlePlaybackStopped(result) {
  return ['timed_out', 'disqualified', 'forfeited'].includes(result?.status);
}
