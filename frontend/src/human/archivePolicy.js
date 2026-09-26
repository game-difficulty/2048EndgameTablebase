// Every ended signed-in game is retained. Visibility is decided by the
// display threshold captured by the server when the run was created.
export function needsReplayUpload(run) {
  return !!run && !run.guest && !!run.reason && !run.archived;
}
