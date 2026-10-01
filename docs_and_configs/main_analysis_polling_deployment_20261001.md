# Main-site analysis polling and source-baseline recovery

Deployed 2026-10-01, approximately 17:41 (UTC+8).

## Cause and Recovery

- The Play-home/AI-card/bottom-leaderboard change was committed as `b026e95`
  and successfully deployed on September 29.
- The rollback entry saved before the October 1 replay deployment identifies
  build `5f194b1ff488-20260930112610851`. Its referenced
  `main-CJFwLRRh.js` contains `menu.ai` and `menu.descriptions.playSite`.
- However, `/opt/2048tables/app/frontend/src` still contained the older
  homepage, auxiliary entries, and announcement sources. Rebuilding those
  server files for `db6bba9` replaced the correct homepage with old content.
  The Git changes had not been reverted. This was a deployment-baseline error.
- Recovery uses the verified Git snapshot `5f194b1ff488a5081521de67eae121275c2a7d54`,
  the explicitly listed replay frontend files from `db6bba9`, and the
  analysis polling change from `53bb10f`. No dirty workspace changes are included.

## Analysis Fix

HTTP analysis jobs are handled by Play, while the main-site WebSocket points
to the main backend. Their upload/task directories differ. Subscribing an
HTTP-created job on that WebSocket could send `ANALYSIS_FAILED` for a missing
task and stop the correct HTTP poller. One reported-period task finished in
about four seconds; reopening the dialog fetched its saved results.

The dialog now uses HTTP as the sole status authority for uploaded jobs,
ignores competing WebSocket status messages, prevents overlapping/stale
responses, and times out hung requests after 15 seconds while retaining retries.

## Release

- Code commit: `53bb10f41934b5f6b24011f6eed25e8936f8cc7a`.
- Build ID: `53bb10f41934-20261001093835014`.
- Manifest: `/opt/2048tables/app/frontend/dist/main-source-manifest.json`.
- Rollback entry and verified source archive:
  `/opt/2048tables/backups/20261001-main-polling-53bb10f/`.
- Main-site sources were reconciled, excluding `src/live` and `src/human`.
  Those independently deployed applications remain unchanged. For a future
  combined build, use a verified Git snapshot and the recorded overlays,
  not arbitrary server source directories or the current dirty worktree.
- Only main-site static entry/assets/compatibility resources were published.
  Live/human entry hashes are recorded and unchanged. No backend restart,
  database migration, or database write was required.

## Verification and Retention

- Isolated production build and 13 targeted frontend tests passed.
- Browser regression: competing WebSocket failure no longer stops HTTP
  polling; a completed result appears automatically and polling then stops.
- Production browser confirmed Play's new-window link, AI-to-game navigation,
  bottom leaderboard navigation, contact entry, and the Play announcement.
- Production replay dark-mode/palette/mobile-width checks passed.
- All 54 referenced main-entry assets exist; entry gzip matches plain HTML.
  Main/Play/live/replay resources respond successfully, and Play health is OK.
- Removed two deployment-owned temporary artifacts: 11,367,637 bytes.
  Root filesystem remains at 52% used.
- Latest three main rollback snapshots and three Play releases retained.
  The older cross-application recovery record and mixed database snapshot were
  preserved for review; fresh verified database backups were not removed.
