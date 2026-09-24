# Unified main-site and live-site builds

Build both sites from one committed frontend tree with `npm run build`.
The Vite multi-entry build produces main, live and optional play artifacts together;
publishing main/live does not enable the separate play site or its backend routes.

Before release:

1. Run `npm test` in `frontend` (includes the module-mocking flag).
2. Run backend and tablebase-worker tests against temporary databases.
3. Commit the effective source and tests. Do not stage runtime databases, secrets,
   compiled DLL backups, experiment output or browser logs.
4. Build once from that commit; do not rebuild each entry separately.
5. Verify `dist/index.html` and `dist/live/index.html` have the same `build-id`
   metadata and that `dist/release.json` identifies the source commit.
6. Upload new assets before publishing either entry. Keep old hashed assets for
   open tabs. Update uncompressed and gzip entries together and retain a rollback
   backup. Verify both public hosts, including gzip responses.

Source-only native changes do not authorize replacing running native modules.
Backend/API changes and enabling additional sites are separate release decisions.
