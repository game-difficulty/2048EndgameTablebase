// Natural endings always upload. Other endings upload only strictly above the
// threshold assigned to that game, including when revisiting older local saves.
export function needsReplayUpload(run) {
  return !!run && !run.guest && !!run.reason && !run.archived
    && (run.reason === 'game_over' || run.score > run.threshold);
}
