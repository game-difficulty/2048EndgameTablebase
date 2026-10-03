# 64-bit Analysis History Reset (2026-10-03)

User authorization: remove old analysis results for 64-bit tables and commit
the dtype-based Perfect tolerance changes. Do not deploy those code changes.

## Scope and Result

The active main and Play manifests had three 64-bit tables: `3x3_1024`,
`3x3_sum-1790`, and `2x4_sum-894` (all float64). No other uint64/float64/
1-float64 table was configured. All previous analyses of these table keys
were invalidated, including older uint32-era results for `3x3_1024`.

The production cleanup committed at 2026-10-03 13:58:05 UTC:

- 70 history items and 68 now-empty history jobs removed.
- One mixed job retained its unrelated 32-bit item and corrected counters.
- 250 legacy transient analysis job records removed.
- 67 analysis artifact records removed; 64 existing derived replay files
  (703,550 bytes) moved out of the serving root into rollback quarantine.
- 44 Play summaries and 35 derived ranking results removed.
- 16 all-time best rows, 14 weekly best rows, and the affected catalog/week
  state entries removed. These counts overlap; they are not separate games.

Original games, original replay archives, uploaded inputs, account balances,
and other tables' analyses were retained. In-transaction hashes verified that
unrelated analysis rows and ranking rows were unchanged. The original Play
game count was unchanged at 169,407 during the transaction. Foreign-key checks
did not acquire new violations. Post-maintenance checks found zero target
rows in history, artifact, transient job, Play summary and ranking tables.

## Recovery and Runtime

Fresh SQLite online backups passed quick_check:

- `/var/lib/2048tables/backups/db/auth/20261003T135709Z-analysis64-reset.sqlite3`
- `/var/lib/2048tables/backups/db/human/20261003T135709Z-analysis64-reset.sqlite3`

Private operational scripts, removed-row inventory, file quarantine and the
receipt are under `/opt/2048tables/backups/20261003-analysis64-reset/`.
Do not put user identifiers or analysis content from those receipts in Git.
Backups follow `deployment_retention.md`; do not restore whole databases over
subsequent user activity without a reviewed recovery plan.

Main and Play services were briefly stopped to prevent analysis writers and
then started on the same existing releases, also clearing process-local jobs.
Their release paths did not change. Main live-lobby, Play health and competition
health endpoints returned HTTP 200. Maintenance used a bounded systemd unit
(400 MiB memory, 40% CPU); no compilation occurred. Final disk usage was 59%.

**No code/frontend deployment was performed.** Until the tolerance change is
deployed, new analyses still use the old policy. Re-audit and invalidate any
intervening old-policy results at the later deployment; do not assume this
cleanup makes subsequently generated results compatible with the new policy.
