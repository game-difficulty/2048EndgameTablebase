# 2026-10-03 3x3-1024 Table Replacement

## Published Change

- Source: `C:/2048_tables/cloud tables/33-1024`.
- Production: `/opt/2048tables/tablebases/3x3_1024`.
- EX zbook/zlut format; 572 consecutive layers, 0-571; 576 files,
  266,906,224 bytes including generation metadata.
- All layer headers declare Float64 and 8-byte values. Generation metadata
  declares an absolute deletion threshold of 0. The full value audit found
  10,986,746 of 11,615,327 values off the old uint32/4e9 grid, confirming that
  the contents are not merely the old quantized values stored in doubles.
- Every transferred file was verified against the local SHA-256 manifest.
  Existing main and Play native readers passed direct staged-table probes.
- Main and the active Play backend manifests now select float64. Both quota
  configurations advertise no pruning and layer range 0-571.
- Cloud and Play were briefly stopped and restarted after confirming no queued
  or active analysis work. No frontend, native binary, tournament, forum,
  database, or analysis scoring changes were deployed.

The runtime roots remain `/opt/2048tables/app` and
`/opt/2048tables/play/releases/20261003-cards-591e344`. Frontend/native hashes
were guarded before and after the switch. Main, Play and competition health
checks passed. Quota responses through the main, tables and Play nginx hosts
also passed. Direct public urllib probes from the server received HTTP 403;
the hostname checks therefore used nginx's loopback origin with TLS host
resolution, not the public CDN path.

## Rollback and Retention

Old table retained at
`/opt/2048tables/tablebases/3x3_1024.before-20261003-float64`.
Do not delete this table backup under generic release retention rules.

Receipts, original configs, deployment scripts and file manifest:
`/opt/2048tables/backups/20261003-table33-float64/`.
Local counterpart: `C:/Apps/2048endgameTablebase/deployments/20261003-table33-float64/`.

Removed only this deployment's verified upload archive: 1 file, 125,526,769
bytes reclaimed. Disk usage after cleanup: 58%, 21,501,325,312 bytes available.
Release and database backup inventories were recorded; existing releases,
recent database backups and unrelated artifacts were not deleted. Preserve
this table/config overlay in subsequent deployments, alongside the previously
published sum-1790/sum-894 replacement overlay.

## Perfect Tolerance Investigation (Not Implemented)

Historical investigation below. The subsequent user-approved implementation
uses dtype defaults, not table-specific overrides: 3e-10 for 32-bit formats,
1e-14 for 64-bit formats. See `perfect_tolerance.md`. That code change is
separate from this completed table deployment.

Current cloud analysis uses `3e-10`, not `3e-9`:

- `backend/analysis_core.py`, `Analyzer._analyze_one_step`: absolute difference
  `best_result - move_result <= 3e-10` counts as Perfect and increments combo.
- `engine_core/replay_utils.py`, `replay_step_goodness_ratio`: absolute
  differences <= 3e-10 become exactly 1, affecting cumulative goodness of fit.
- Python and frontend replay analysis additionally compare the relative ratio
  to `1 - 3e-10`; these are distinct from the absolute success-rate difference.
- The local desktop repository already centralizes a global tolerance in its
  performance evaluation configuration, but the cloud copy still hardcodes it.

Minimal proposed design: add a default tolerance and full-pattern overrides to
the existing performance evaluation config. Keep other tables at 3e-10; use
1e-12 for `3x3_1024` and `3x3_sum-1790`. Resolve once in Analyzer and pass that
same tolerance to the Perfect check and goodness-ratio helper. Keep the current
absolute-difference meaning; do not silently replace it with relative loss.
Record the applied policy/version in saved summaries. Existing summaries must
be reanalyzed to recover details discarded by the old tolerance.

Analysis posters consume float64 summary statistics directly, so the narrow
poster/analysis change does not require a replay-format migration. However,
`Analyzer.record_replay` stores each directional success rate as
`uint32(rate * 4e9)`, and the replay page reconstructs values from that format.
Its resolution is 2.5e-10: changing replay-page epsilon alone cannot recover
1e-12 precision. Consistent detailed replay statistics would require a
versioned float64 record or a high-precision sidecar, while retaining old-file
support. This is a separate follow-up, not part of the table replacement.
