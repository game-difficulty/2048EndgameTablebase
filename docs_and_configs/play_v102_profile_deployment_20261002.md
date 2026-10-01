# Play v102 profile updates — 2026-10-02

Published `ef5777c` and `7187576` as
`20261002-play-v102-hidden-restarts-board-sum`.

- New restarted games still retain uploaded replay archives, but default to
  hidden in owner/visitor history. Existing records are not rewritten. Explicit
  administrator approval can promote a hidden archive, respecting user deletion
  and the captured display threshold.
- Final-board preview shows the sum between its heading and variant. The common
  preview and computed getter cover 4x4, 3x4, 3x3 and 2x4, including other players'
  profiles. English text is included.

Frontend source: isolated byte-verified v101 baseline plus the committed profile,
CSS and translation patch. Backend source: exact production service/admin modules
plus the committed visibility patch. No tournament WIP was included. Baseline,
overlay list, build identity and SHA-256 files are recorded in the release's
`release-input-v102.json` and `ui-release.json`.

49 backend tests and the production frontend build passed. The actual computed
getter was exercised for all four variants. Origin homepage, two profile paths,
leaderboard, all packaged assets and gzip HTML passed. Public CDN HTML, Human
entry and PlayerProfile script match the build; public Play health and Live lobby
respond successfully. Both services are active.

Only Play restarted. Its process cwd is v102. There was no schema migration or
bulk user-data update. Rollback: restore current to v101 and restart Play.

Retention reviewed process descriptors/mappings, service/nginx configuration,
symlinks and pin markers. Removed unreferenced v99 and completed upload staging;
retained v102, v101 and v100. Reclaimed 140,183,982 bytes (about 133.7 MiB).
Database backups remain intact; nine older mixed-provenance files were skipped.
Final filesystem use: 54%, with 23,210,192,896 bytes available (about 21.6 GiB).
Deployment result and cleanup accounting: `deployment-result-v102.json` in v102.
