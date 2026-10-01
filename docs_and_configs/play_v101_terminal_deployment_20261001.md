# Play v101 terminal controls — 2026-10-01

Published `09b3673` as `20261001-play-v101-terminal-controls`.
Ended-game Start new game and Enter share the direct restart path; the overlay
focuses that button, keeps its original two-second delay across archive receipts,
and no longer triggers automatic leaderboard/BEST requests on archive completion.

Built from the isolated, byte-verified v100 frontend source plus only the committed
HumanApp.vue and terminalOverlay.js patch. The 56,447-byte static archive included
the new Human entry, HTML, release metadata and gzip copies. Effective source
baseline, overlay and hashes are in the release's `v101-input.json` and
`ui-release.json`.

42 related tests and component compilation passed before release. Origin
homepage/profile/leaderboard/health, precompressed HTML and entry script passed.
Public CDN HTML and JS bytes match the build. Play and Cloud remain active.
No backend restart or database changes: Play PID 92732 still runs from v100.
Rollback is an atomic current-link switch to v100; no restart is necessary.

Retention checked process mappings, descriptors, service/nginx configuration and
symlinks. Removed the unreferenced replay-analysis release and completed upload;
retained v101, v100 and v99. Reclaimed 139,582,693 bytes (about 133.1 MiB).
Database backups were preserved; nine older mixed-provenance files were skipped.
Final filesystem use: 53%, with 23,728,762,880 bytes available (about 22.1 GiB).
