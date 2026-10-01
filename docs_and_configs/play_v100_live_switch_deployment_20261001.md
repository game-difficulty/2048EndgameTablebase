# Play v100 and Live game switching — 2026-10-01

Published `8a69347` at https://play.2048tables.online/ as
`20261001-play-v100-live-switch-enter`.

## Scope and source verification

- `930253a`: retain producer/viewer connections across signed human game switches,
  fence actions until the matching ready response, serialize lease handoffs,
  refresh the room recovery window, and reset per-run milestone verification.
- `10b10f6` / `8a69347`: Enter and NumpadEnter restart an ended game directly,
  bypassing the restart confirmation. Practice redo and active-game controls
  retain their existing behavior.
- Frontend baseline: `0cde1abe9594` plus the previously published
  `f1dfbf6`, `a01f240`, `04925b0`, and `3ca467c` overlays. The preserved v99
  HTML and Human entry bundle matched production bytes before rebuilding.
  Only the live publisher module and terminal Enter handler were overlaid.
- Built in an isolated source directory. No tournament WIP or unrelated main-site
  preference changes were included. The Human manifest dependency closure and
  gzip copies were published while older assets remained available.
- Live backend: three reviewed modules under `/opt/2048tables/app/backend/live/`.
  Existing source matched the pre-change Git baseline after newline normalization.
- Play backend: only `backend/live/human_rooms.py` was overlaid on the previous
  release. Production dispatcher, other backend changes and runtime data remain
  inherited from that release.

Deployment input, file hashes, original Live modules, result and retention audit:
`/opt/2048tables/backups/20261001-live-switch-8a69347/`.

## Verification and startup recovery

Related checks: 22 Python tests and 11 frontend tests passed; the terminal Enter
handler was additionally exercised for both Enter keys, repeated keydown,
modal/input guards, practice redo and the existing R shortcut. Production build
passed. Origin homepage/profile/leaderboard responses and gzip contents match
the release. Public CDN HTML and Human entry JS match the local build bytes.
Public Play health and Live lobby respond successfully; both services are active.

The initial Play startup failed because the root-owned cloned release contained
a non-writable logger file. The deployment automatically rolled back. Ownership
of the new release was corrected to `ubuntu:ubuntu`, and the second deployment
passed all checks. Both services were restarted; this deployment itself briefly
interrupts existing connections.

Actual Play process cwd: the v100 release. Actual Cloud process cwd:
`/opt/2048tables/app`. No schema migration or user-data modification was needed;
the same-day verified database backups were retained.

## Retention and rollback

Retained Play releases: v100, v99, and `20261001-replay-analysis-db6bba9`.
The unreferenced v98 release and completed upload/extraction staging were removed
after checking process, configuration and symlink references. Reclaimed
141,207,163 bytes (approximately 134.7 MiB). Existing database backups were
preserved; nine older mixed-provenance backup files were skipped for review.
Final root filesystem use: 53%, 23,739,887,616 bytes available (about 22.1 GiB).

Rollback: restore the three original Live modules from the backup, restore the
Play current link to v99, ensure the Play logger is writable by the service user,
and restart both services. The replay-analysis release is also retained as the
last previously running backend baseline. No database rollback is required.
