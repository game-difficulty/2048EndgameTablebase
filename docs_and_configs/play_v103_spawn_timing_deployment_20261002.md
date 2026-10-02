# Play spawn timing deployment — 2026-10-02

Published `4d3f6be` to `20261002-play-v103-spawn-timing`.
Play starts revealing new tiles after 100ms instead of 125ms, with a 200ms
appearance animation, matching Verse's published board CSS. All variants and
practice boards use the same timing. Main-site default timing is preserved.

Source baseline: independently reproduced v102 isolated frontend, with the
previously published Play-specific 3x3 overlays from `8df62ab`. BaseBoard remains
at the exact Play source archive baseline. All 119 published board-cao payload
files excluding build identity HTML/gzip and release.json matched SHA-256 before
applying the three committed animation source changes. Unpublished typography
changes and tournament WIP were excluded.

Build identity: `4d3f6be2cdf5-20261002033151328`. Overlay list and 120 payload file
hashes are recorded in `release-input-v103.json` and `ui-release.json`.
28 related tests and frontend builds passed. Origin homepage, profile,
leaderboard, all packaged assets and health checks passed. Public CDN homepage
and human entry match the build byte-for-byte.

Frontend-only deployment; no service restart, database migration or data change.
Play PID 170286 continues running from v102. Cloud PID 120028 remains running
from `/opt/2048tables/app`. Rollback: restore current to board-cao; no restart.

Retention checked nginx/systemd configuration, symlinks, process cwd/exe/fds/maps
and explicit pin markers. Removed unreferenced v101, reclaiming 139,820,594 bytes
(about 133.3 MiB). Retained v102, board-cao and v103. No verified backup files
older than seven days were found. Legacy mixed-provenance backups were preserved.
Completed payload staging was removed. Final disk use: 54%, available
23,224,516,608 bytes (about 21.6 GiB). Server accounting is saved in
`deployment-result-v103.json` in the new release.
