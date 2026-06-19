# BC FamilyChain Solve Implementation Notes

This document records the current FamilyChain solve implementation. It replaces
the older pre-implementation plan. For shared recurrence, dtype semantics, and
the production runner, read
`docs_and_configs/exbc_exadbc_three_solve_chains_runtime_design_v1.md` first.

## 1. Current Status

FamilyChain solve is implemented as the low-memory BC backsolve route.

Primary code:

```text
native_core/include/BCFamilySolve.h
native_core/include/BCFamilySolvePlan.h
native_core/include/BCFutureFamilyWindow.h
native_core/include/BCFutureSuccessLookup.h
native_core/include/BCFamilySolveRunner.h
native_core/tests_src/test_bc_family_solve.cpp
native_core/tests_src/test_bc_family_solve_plan.cpp
native_core/tests_src/bench_bc_family_solve_full.cpp
```

Production dispatch is owned by `BCFamilySolveRunner`. The route can be forced,
but normal production uses `solve_route=auto` and selects resident, single, or
FamilyChain from the current/future row counts and available memory. The route
does not change the family modulus.

## 2. Scope And Output Contract

FamilyChain solve computes the same recurrence as resident and single-chunk
solve. It reads:

```text
current generated <prefix><ordinal>.bcpos
future2 exact     <prefix><ordinal+1>.bcpos + .bcsuc
future4 exact     <prefix><ordinal+2>.bcpos + .bcsuc
```

It writes the exact solved current layer:

```text
<prefix><ordinal>.bcpos
<prefix><ordinal>.bcsuc
```

The solved `.bcpos` is zero-compacted and can become a future layer. The solved
`.bcsuc` uses BC success format v2 with a per-cell value-offset table; readers
must use the offset table rather than assuming that success payloads are a
plain cid-prefix layout.

FamilyChain does not create generated-position entries and does not use
generation mutable structures:

```text
BCCellMutableBuilder
BCFamilyMutableStore
BCGenerationBlobIO
BCFamilyPositionWriter
```

## 3. Fixed Modulus Rule

BC calculation receives one caller-supplied family modulus. Generation and
solve routing must keep using that modulus for every layer and every route.

The runner does not attempt to infer or switch modulus mid-run. Reconnect and
resume read the recorded checkpoint and exact files but do not remap already
generated layers to a new modulus.

## 4. Source-Family Sweep

The implemented scheduler is source-family ordered, not target-family-major.
For a modulus/family count `F`:

```text
for fid = 0..F-1:
    row side:    cells (fid, x)       -> horizontal moves
    column side: cells (x, fid), x!=fid -> vertical moves
    diagonal:    cell  (fid, fid)     -> both horizontal and vertical moves
```

The order is fid ascending. This gives a controlled completion order for the
L-shaped cell boundary. Physical final-value writes may still be staged when a
cell completes before it can be flushed.

The removed interleave/block route is not a production path. Spawn4 and Spawn2
are separate full fid sweeps.

## 5. Future Fanout And Window

Future family coverage follows the same modulo family mapping rules as
double-block generation. For one exact axis coordinate, spawn fanout is at most
two. The third physical family appears only after modulo grouping. Therefore a
current fid pass normally needs two future families and occasionally three.

The production invariant is:

```text
future board-scale resident data <= 3 families
future resident data + current/source family cross <= 4 families occasionally
```

Family-scale metadata such as pass plans and family-cell matrices may be
resident. Full layer board data must not be resident in the FamilyChain route.

Future reuse is deterministic. The runner retains only cells that later fid
passes are guaranteed to need within the window; it does not keep arbitrary
adjacent families as a best-effort cache.

## 6. Two Spawn Sweeps

FamilyChain solve uses two phase sweeps:

```text
Spawn4 sweep:
    first direction of a non-diagonal cell writes compact partial4
    second direction reads partial4, merges max4, averages over empty slots
    writes weighted Spawn4 contribution into per-cell scratch4

Spawn2 sweep:
    first direction writes compact partial2
    second direction reads partial2 and scratch4
    merges max2, averages over empty slots, adds weighted Spawn2 contribution
    zero-compacts the cell and stages/writes final values
```

For each row:

```text
final = p4 * average(max4_per_empty_slot) +
        p2 * average(max2_per_empty_slot)
```

`spawn_rate4` is an external option. It is not hard-coded as 0.1/0.9 inside the
algorithm.

If a direction has no legal move/query for an empty slot, the stored value for
that slot is the dtype-specific `zero_value`. That value is a real dense slot,
not a hole.

## 7. Compact Partial Layout

Partial max is bucket-empty compact. It does not use a fixed 16-cell best array
as its primary layout.

For a current bucket:

```text
rows    = live board rows in bucket rank order
columns = empty cells from bucket_empty_mask in ctz order
lanes   = row_width
```

Physical value count:

```text
bucket_live_rows * popcount(bucket_empty_mask) * row_width
```

The hot layout stores only data needed for compact access. Redundant debug or
generic fields such as bucket key copies, bucket index copies, and
`empty_slot_ordinals[16]` are not on the production hot path.

## 8. Temporary Files

FamilyChain uses file-backed temporary values for:

```text
partial4
partial2
scratch4
family_final_values when final values cannot be flushed immediately
```

Temporary records use compact metadata:

```text
value_offset: uint64_t, UINT64_MAX means unwritten
value_count:  uint64_t
```

Temp IO supports direct-capable batched reads/writes. `compress_temp_files`
compresses retained temporary files in memory using the shared temporary-file
compression bridge; it does not decompress to a separate on-disk expanded file.
When `keep_temp_files=false`, temp files are removed at the end of the layer.

## 9. Future Lookup And Hot Path

The compute path reuses `BCFutureSuccessLookupView<StorageT>` and
`BCSolveEdgeKernel` logic. The trusted-axis collectors are the production fast
path:

```text
horizontal pass -> horizontal trusted collector
vertical pass   -> vertical trusted collector
diagonal pass   -> both trusted collectors
```

The planner is responsible for ensuring future-family coverage. In production
hot loops, bucket hit masks are not used as a defensive per-empty-slot filter.
Debug builds may assert coverage.

Detailed per-candidate counters are diagnostic-only. Production timing and CSV
stats should avoid query-level increments in the hot loop.

## 10. DType Policy

The six success dtype modes share the same FamilyChain control flow:

```text
UInt32
UInt64
Float32
Float64
OneMinusFloat32
OneMinusFloat64
```

The dtype dispatch chooses `StorageT` and dtype-derived zero/terminal values.
It must not choose a different algorithm route. Larger dtypes naturally perform
more IO.

One-minus float modes store `success - 1`; their zero and terminal values are
`-1` and `0` respectively. The max and weighted-average recurrence remains the
same because the transform is monotonic.

## 11. Final Output

Final `.bcpos/.bcsuc` writing uses the same exact output contract as resident
and single-chunk solve. `.bcsuc` v2 carries per-cell value offsets, so the
physical value payload order is described by the file metadata.

FamilyChain may stage completed final cell values in `family_final_values`
when immediate flush is not possible under the configured pending memory cap.
This staging is an implementation detail; readers see only the final exact
`.bcpos/.bcsuc` pair.

## 12. Tests And Benchmarks

Required validation:

```text
FamilyChain small fixtures equal ResidentSolve/SingleChunkSolve
all six dtypes use the same control flow and match typed expectations
single-layer forced resident/single/family routes produce equivalent exact rows
full-run benchmarks report compute, IO, temp, final-stage, and throughput stats
```

Useful performance fields are emitted by `BCFamilySolveRunner` and the
`bc_family_solve_full` thin CLI:

```text
solve_route
current_rows/live_rows/zero_pruned_rows
route_available_memory_bytes
route_resident_required_bytes
route_single_required_bytes
route_required_bytes
total_mrows_per_sec
solve_call_mrows_per_sec
open/partition/solve/compact/temp/final/archive/compress timings
```
