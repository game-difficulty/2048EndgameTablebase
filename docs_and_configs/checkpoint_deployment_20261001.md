# Audited Checkpoint Deployment, 2026-10-01

## Release Identity

- Runtime/frontend source snapshot: `2a27eefd9874d44bd131a5147e0071f1635a851b`.
- Release: `20261001-checkpoint-2a27eef`.
- Main/live build: `2a27eefd9874-20261001100800069`.
- Main/live were built together from an isolated Git archive, not dirty WIP.
- Tournament frontend was built from the same archive.
- Main production `backend/app.py` was deliberately preserved: analysis and
  Play-owned administration remain routed to the independent Play service.
- Audited differing backend modules were overlaid; native binaries, nginx,
  tablebase configuration and local Worker startup parameters were not replaced.
- Input hashes, original entries and effective overlays are recorded in
  `/opt/2048tables/app/frontend/dist/main-source-manifest.json`.
- Verified source and rollback snapshot are under
  `/opt/2048tables/backups/20261001-checkpoint-2a27eef/`.

## Actual Differences From the Previous Production Versions

### Main Site

- Account preference sync recovery/error reporting is now included in the build.
- Managed test accounts have an identifying label and administrator password reset
  UI. The reset API was already deployed; resetting revokes target sessions.
- Seed-aware EvilGen handling and ranked replay protocol v2 are included in the
  game frontend; the matching seeded WASM was installed. The native verifier
  already supports seeded generation; the tested seed fixtures match exactly.
- Profile/theme API and account identity/admin support modules were reconciled.
  Saved theme schema creation has a verified pre-change auth backup.
- The Play homepage card, bottom leaderboard entry and 2048 AI card are preserved.
- Analysis history layout, HTTP polling recovery and replay appearance inheritance
  were already online and remain present, not new features of this release.

### Live Site

- Backend dynamic room generation replacement now reconnects retained viewers and
  avoids removing a replacement hub during stale-room maintenance.
- Chat moderation is active in the live routes: sensitive text is rejected;
  state/history responses hide disabled users' attributable messages.
- Fullscreen/focus, picture-in-picture, gift sorting, language persistence,
  competition previews and room capability UI were already in the deployed live
  bundle. The release rebuilds them alongside the main site; they are not newly
  introduced viewing workflows.
- Shared account UI/preferences benefit from the main/shared build updates.
- Lucky-bag amount changes in this checkpoint are explanatory comments/wording,
  not a new production prize algorithm or payout.

### Tournament Site

- Administrators can assign an event organizer, with dry-run account validation,
  permission checks, confirmation and audit recording.
- Event detail/enrollment/statistics expose the organizer and relevant controls.
- Previously deployed cargo weights `30/30/30/10`, pure-2 target-at-least behavior
  and bomb countdown range `12-32` remain in the frontend.
- Room gameplay rules, seeded native module, balances and existing matches remain
  unchanged. No active competition was present during restart.

### Not Deployed

- Play remains `20261001-replay-analysis-db6bba9`, PID `21149`; its entry and
  current symlink were verified unchanged.
- Local native/GPU/merge research and local Worker parameter changes were only
  checkpointed, not compiled, installed or restarted by this deployment.
- Concurrent uncommitted tournament touch-input changes were excluded.

## Verification and Retention

- Main/shared frontend: 612 tests passed; tournament frontend: 137 passed.
- Extended live/competition backend suite: 233 passed; account/preferences/
  rankings/history suite: 74 passed. One randomized ranking fixture failed on an
  earlier run, passed individually, then the complete 74-test suite passed.
- Three stale race fixtures were updated to a currently published race project;
  no gameplay rule was changed by this test adjustment.
- Browser checks: homepage entries, both main/live build IDs, live playback,
  fullscreen 1280x720, Escape recovery, mobile screenshot and event center.
- Live fullscreen check reported no JavaScript exceptions. Anonymous tournament
  session requests return expected 401 responses. Privileged production writes
  were not exercised.
- A first publication verification incorrectly expected tournament `/` to be
  HTML; it is a normal 302 to `/practice`. Automatic rollback succeeded, the
  probe was corrected and the release was reapplied successfully.
- Auth backup: 463736832 bytes, `quick_check=ok`; competition backup: 933888
  bytes, `quick_check=ok`. Runtime databases and existing backups were retained.
- Removed 2 reviewed old tournament frontend releases and 9 completed temporary
  artifacts: 206802827 bytes, about 197.2 MiB. Root disk usage: 53%.
- Tournament frontend retained: checkpoint, cargo-weights-r1, cargo-rigid-r1.
- Older tournament backend releases lack reliable deployment/success metadata;
  they were skipped rather than ordered by inherited copy timestamps. Mixed and
  Play recovery snapshots were also left intact.
- Final processes: main PID 68266 at `/opt/2048tables/app`; tournament PID 68268
  at its checkpoint release; Play PID 21149 at its unchanged release.
