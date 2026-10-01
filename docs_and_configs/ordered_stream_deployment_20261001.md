# Ordered tournament spectator streams — 2026-10-01

- Release: `20261001-ordered-stream-320662e`.
- Requested changes: `320662e` ordered frame transport/playback and `130c487`
  Huarong Dao square-piece easter egg. Includes pending `3681896` live-room expiry.
- Source baseline: `2a27eefd9874d44bd131a5147e0071f1635a851b`, isolated Git archive.
  Baseline main/live entry assets matched production bytes; archived frontend,
  competition and font sources matched Git. Tournament baseline sources matched;
  its rebuild uses the currently installed newer Vite toolchain.
- Preserved already deployed `819d700` prediction frontend overlay. Its backend
  was not replaced. Touchscreen undo WIP was excluded.
- Production hashes were rechecked immediately before publication. No gameplay
  was active at service stop; a new room was in pre-game preparation. Stop/recheck
  protects against a game starting during preparation of the release.
- Published tournament backend/frontend and shared main/live frontend; overlaid
  only three Live backend modules. Main dispatcher and Play service untouched.
- No database migration. Existing same-day verified database backups retained.
- Origin HTML and referenced assets verified byte-for-byte for all three hosts.
  Public tournament/live entries point to `index-71NYdhj_.js` and
  `live-BAIlwa6E.js`; public tournament health passes. Server-side public CDN
  probing returned HTTP 403, so origin verification used loopback host resolution
  and public checks were performed from the deployment workstation.
- Relevant pre-release checks: 72 transport/runtime/backend tests, both builds,
  and browser playback of all 30 moves before result transition passed.

## Runtime and rollback

- Cloud PID 120028: `/opt/2048tables/app`.
- Tournament PID 120029: tournament-app release above.
- Play PID 92732 still uses v100; its current frontend link remains v101.
- Backup, exact overlay, input manifest, rollback snapshot and verification report:
  `/opt/2048tables/backups/20261001-ordered-stream-320662e/`.
- Rollback: restore `previous.tar.gz` to app, restore tournament and tournament-app
  current links to `20261001-checkpoint-2a27eef`, restart Cloud and Tournament.
  No database rollback is needed. Keep Play unchanged.

## Retention

- Removed the reviewed unreferenced tournament `20261001-cargo-rigid-r1` and
  three completed deployment staging artifacts, reclaiming 61,623,222 bytes.
- Retained frontend releases: cargo-weights-r1, checkpoint-2a27eef, ordered-stream.
- Older backend releases and mixed backup artifacts have ambiguous provenance;
  preserved for manual review. Database backups were not deleted.
- Final disk used 51.6%; free 23,249,756,160 bytes.

Already open player and spectator pages must reload to use the new client.
