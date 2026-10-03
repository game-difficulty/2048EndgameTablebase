# Production Deployment Retention

Effective: 2026-09-30. Applies to future deployments of the 2048tables applications.
This is an agent execution policy, not an installed cron job or systemd timer.

## Build Resource Safety (2026-10-03 Incident)

Native compilation with only 2 parallel jobs exhausted the production host's
approximately 2 GiB RAM (no swap), making SSH and the cloud console unresponsive.
The user had to reboot. A low job count is not a sufficient safety measure.

- Do not compile native modules or build frontends on this production host.
  Build locally or on a separate builder. Native artifacts must match the target
  OS, architecture, Python ABI and runtime libraries; verify before switching.
- Check available memory, swap, disk and service health before deployment.
  Run remote preflight/maintenance in a named, bounded systemd unit. This recovery
  used `MemoryMax=700M` and `CPUQuota=50%`; these are ceilings, not a guarantee of
  spare capacity. Lower them or postpone if live services need more headroom.
- If memory pressure or SSH latency rises, stop only the deployment's own unit
  and verify recovery. Do not retry builds, add parallel jobs, change swap or
  reboot automatically. Failed checks must not trigger a release switch.
- Keep the current release intact until artifact checks and backups succeed;
  after switching, verify service health before cleaning deployment artifacts.

## Releases: Latest 3 Plus Protected Versions

For each application independently, retain its latest 3 successfully deployed
release directories. Also retain every version referenced by:

- A current symlink, nginx root, or systemd configuration.
- A running process's resolved working directory, executable, open files, or mappings.
- An explicit rollback pin or an incomplete deployment.

Protected versions are additional to the count limit. Do not terminate or restart
a service merely to make a release eligible for deletion. Determine ordering from
deployment timestamps/manifests, not lexical version ordering alone.

Known release root: `/opt/2048tables/play/releases`. Apply the same policy to
other 2048tables release roots only after identifying their active references.
Do not include `/opt/fund-pool-web` or `/data/app` in a 2048tables deployment cleanup.

## Database Backups: 7 Days, Minimum 3 Verified Copies

Retention is per source database across all backup directories, not per folder.
Keep backups for 7 days and always preserve the latest 3 verified, recoverable
backups of each database even if they are older than 7 days. Explicitly pinned
incident/recovery backups remain protected until the pin is removed.

Create a consistent backup before a database migration or destructive data change
using `tools/backup_sqlite.py` (SQLite online backup API and `PRAGMA quick_check`).
Do not copy a live `.sqlite3` file without its transactional state. For several
releases on the same day without database changes, reuse the day's verified
backup instead of creating another full copy. Database-changing releases still
require a fresh pre-change backup.

New backups go under `/var/lib/2048tables/backups/db/<database-name>/`, with a UTC
timestamp and release identifier in the name. Record source path, creation time,
release identifier, and verification result alongside the backup. Compress only
after validation; verify decompression and database integrity before replacing
the uncompressed copy.

Legacy backups exist under `/var/lib/2048tables/backups` and
`/opt/2048tables/backups`. Classify them by source database and verify retained
copies before removing expired copies. Unknown origin or unverifiable recovery
state means manual review, not automatic deletion. Treat a legacy database copy
and its associated WAL/SHM as a unit, separate from the live database.

Never delete runtime databases, live WAL/SHM files, replay archives, uploads, or
avatars under `/var/lib/2048tables` by directory-wide age rules.

## Deployment Snapshots and Temporary Files

- Non-database code/frontend deployment snapshots: retain the latest 3 successful
  rollback snapshots per application, plus explicitly protected snapshots.
- Mixed snapshots containing databases: apply database retention before deleting
  the enclosing directory; do not bypass it using the code snapshot count limit.
- Completed deployment archives/build directories in `/tmp` or staging roots:
  remove after successful verification, or after 24 hours when confirmed inactive.
- Never clear all of `/tmp`. Do not infer that an artifact is inactive from its
  name or age alone. Do not remove staging files belonging to another agent's
  in-progress deployment.
- Tablebase `.before-*` copies are outside this policy. Remove only after the
  replacement is validated and cleanup is explicitly authorized.

## Required Post-Deployment Procedure

1. Finish deployment, verify health endpoints and static assets, and confirm the
   actual process working directories. Do not prune after a failed deployment.
2. Inventory candidate paths, protected paths, retained backups and disk usage.
3. Resolve and validate every deletion target stays within its approved root.
   Do not follow symlinks into another root or use broad filesystem wildcards.
4. Check current configuration, symlinks and live process references. Preserve
   every referenced target; if classification is uncertain, skip and report it.
5. Delete only reviewed, eligible artifacts; keep runtime data and unrelated
   applications untouched. Recheck services and disk usage afterward.
6. Report deletion counts, reclaimed bytes, retained release identifiers and any
   skipped artifacts. Warn if disk use is at least 80%; at 90% prioritize a space
   review before a large build or upload, without bypassing safety checks.

## Cleanup Baseline (2026-09-30)

The approved cleanup removed 79 database/deployment backup entries, 155 older
deployment backup entries, 58 Play releases, and 11 explicitly selected temporary
artifacts. Disk usage fell from 91% to 49%, reclaiming approximately 19.4 GiB.

Play retained: `20260930-play-v93-start-time`,
`20260930-play-v94-immediate-swipe`, `20260930-play-v95-victory`.
At that time the backend still ran from v93 while `current` pointed to v95.
This is historical context, not a permanent pin: recheck references on every
deployment. Existing retained backups must not be reduced until at least 3 newer
verified backups exist for the same source database.
