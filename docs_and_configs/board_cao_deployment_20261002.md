# 3x3 Board and Cao Lettering Release, 2026-10-02

- Requested source commits: `8df62ab` (3x3 typography, spacing and corners),
  `ef50dc6` (Cao lettering), and the already published UI commits `5dc915a`,
  `172f6d2`, `00feafb` retained together.
- Release: `20261002-board-cao-8df62ab`; main, live, tournament and Play frontends.
- No backend replacement, database migration, service restart or local AI Worker
  restart. Play's running backend remains at v102; its current frontend is new.

## Verified Sources

Main/live/tournament use the exact previously published rule-metrics source
snapshot, with byte-verified prior overlays, plus `ef50dc6` and `8df62ab`.
The previous production stream transport and recovery implementation is retained;
unpublished durable-stream/roster backend changes and uncommitted touchscreen WIP
are not imported by building the entire latest repository indiscriminately.

Play uses the independently reproduced v102 frontend, with only the common 3x3
board/style changes. Its terminal controls, hidden restarted archives, profile
preview board sum, authentication and preferences remain intact.

All active entry references, lazy-loaded chunks and shared static baseline files
were verified byte-for-byte on the server before publication. Actual source
archives, baseline/overlay manifests, mutable-file rollback snapshot, result and
retention accounting are in:

`/opt/2048tables/backups/20261002-board-cao-8df62ab/`

Play's baseline build script hardcoded revision `7187576`; the isolated target
was corrected to Git revision lookup. Its final entry/release metadata and helper
overlay are saved alongside `effective-manifest.json`; this correction changes
only build identity, not bundled asset contents or runtime behavior.

## Verification

- Main/live frontend: 619 tests passed. Tournament frontend: 148 passed.
- Play targeted geometry, gestures, profile ownership, appearance and label
  tests: 23 passed. Its isolated full-suite attempt also encountered missing
  backend fixtures in that frontend-only source tree; not claimed as a full pass.
- All frontend builds passed. Desktop/mobile 3x3 replay cells remain square;
  small-number font size is approximately half the tile side.
- Origin entries/referenced assets and three backend health endpoints passed.
  All backend PIDs/cwds remained unchanged.
- Main Play homepage link, bottom leaderboard entry and 2048 AI card preserved.

## Retention

Removed two reviewed, unreferenced old frontend releases (tournament checkpoint
and Play v100) and this deployment's completed staging/upload files.
Reclaimed 229,728,451 bytes, approximately 219.1 MiB; `df` disk usage: 54%.

Tournament frontend retained: ordered-stream, rule-metrics, board-cao.
Play retained: v101, v102, board-cao. v102 is additionally protected by the
running backend. Database backups, mixed-provenance legacy snapshots, active
backend releases, runtime data and tablebases remain untouched.
