# Repository Instructions

## Production Deployment Retention

- Before any production deployment, read `docs_and_configs/deployment_retention.md`.
- After a successful deployment and health checks, perform the retention review and cleanup described there. Include reclaimed space and final disk usage in the deployment report.
- Keep the latest 3 releases per application, plus every release referenced by a current symlink, running process, or rollback pin. A running backend may still use an older release than the frontend's `current` link.
- Database backups expire after 7 days, but always preserve the latest 3 verified backups of each database, regardless of age. Never delete live database files or their WAL/SHM files.
- Use the SQLite online backup API (`tools/backup_sqlite.py`) rather than copying a live database file.
- Do not clean another application's directories, unknown artifacts, active staging directories, replay history, or tablebases as part of deployment cleanup.
- Do not schedule unattended deletion or broaden cleanup scope without explicit user authorization.

## Desktop Git Directives

Use forward slashes in Windows paths in Codex desktop git directive attributes.
