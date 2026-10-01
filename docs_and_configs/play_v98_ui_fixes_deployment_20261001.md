# Play v98 UI fixes — 2026-10-01

Published at https://play.2048tables.online/ with current pointing to
`/opt/2048tables/play/releases/20261001-play-v98-restart-poster`.

Frontend fixes: `f1dfbf6` (dark native datetime picker), `a01f240` (explicit
restart after server slot conflict, including variant isolation), and `04925b0`
(B10 preview resolution follows display width and device pixel ratio).

Built from the previous deployed frontend baseline `0cde1abe9594`, with only
the five affected Human frontend files overlaid. The separate build excluded
the unused Live entry and compatibility generator, and retained precompression.
Packaged the Human entry's recursive manifest dependency closure: 64 files,
535,093 bytes compressed. Build ID: `04925b073867-20261001055807880`.

No backend restart or database changes. The running Play backend remains on
v97. Existing static assets were retained for already-open browser sessions.
Rollback: atomically restore the Play current symlink to v97; no service restart
is required for this frontend rollback.

Checks: 25 session tests, two poster lifecycle/resolution tests, production
build, origin homepage/profile/health, 21 initial static assets, matching HTML
gzip, and public CDN entry and script byte comparison. Play and Cloud services
remain active.

Retention review retained v98, v97, v96 and v95. An initial cleanup incorrectly
ordered v95 and v96 by inherited directory mtimes; v96 was immediately recovered
from v95 and the unchanged v97 files, excluding the documented v97 reviewed
patches. Recovery evidence is in
`/opt/2048tables/backups/play-v98-retention-recovery.json`. No further release
deletions were made. Future ordering must use deployment manifests rather than
these inherited mtimes. Runtime user data was untouched.

Completed upload and deployment-script staging cleanup reclaimed approximately
0.51 MiB; no net release-directory space reclaim is claimed. Database backups
were preserved: none under `/var/lib/2048tables/backups` were older than seven
days, while 11 older files under `/opt/2048tables/backups` were retained because
their mixed backup provenance needs further review. Final filesystem usage:
51%, approximately 24 GiB available.

## v99: Enter confirms restart

Published `3ca467c` as `20261001-play-v99-restart-enter`. Cloned the then-current
`20261001-replay-analysis-db6bba9` release, preserving its backend changes. Its
frontend was still the byte-verified v98 bundle. Reused the isolated v98 source
baseline and applied only the restart-button focus/primary-style change.
Uploaded a 54,908-byte static patch with fresh HTML and JS gzip copies.
Origin and public CDN entry/script bytes match the build; both services are
active. No backend restart or database changes. Rollback target is the previous
replay-analysis release.

Retention used documented deployment chronology and checked current links,
service configuration and live process references. Removed the unreferenced
v97 release; retained v99, replay-analysis-db6bba9 and v98. Reclaimed 138,742,127
bytes (about 132.3 MiB), including upload staging. Database backups preserved.
Final filesystem use is 53%, with approximately 23 GiB available.
