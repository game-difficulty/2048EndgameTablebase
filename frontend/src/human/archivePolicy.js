// Every ended signed-in game is retained. The server hides restarted archives
// by default and applies the display threshold captured when the run was created.
export function needsReplayUpload(run) {
  return !!run && !run.guest && !!run.reason && !run.archived;
}
