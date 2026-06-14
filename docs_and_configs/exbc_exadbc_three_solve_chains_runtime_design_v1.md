# BC Three Solve Chains Runtime Design

This document is the current solve-side handoff. It is aligned with the
generation implementation in `bc_resident_singlechunk_generation_impl_supplement.md`.

The current code state is intentionally split:

```text
Implemented and tested:
    BCBacksolve resident UInt32 baseline

Header-only scaffolding, not yet production:
    BCSolveEdgeKernel
    BCResidentSolve
    BCSingleChunkSolve
    BCFutureFamilyWindow
    BCPartialStore

Not implemented:
    FamilyChainSolve executor
    solve route dispatcher
    streaming/direct success writer
```

## 1. Solve Is Not Generation In Reverse

Solve reads existing position files and future success files, then writes the
current success file.

Forbidden in solve main paths:

```text
BCCellMutableBuilder
BCFamilyMutableStore
BCGenerationBlobIO
BCFamilyPositionWriter
whole-layer mutable key/rank hash builders
whole-layer position bitmap builders
```

Allowed objects:

```text
current position reader/scanner
future position + success lookup view/window
typed current success accumulator
current success writer
small route/pass plans
cell lists and work items
thread-local board/canonical/query buffers
```

## 2. Shared Recurrence

For each current board `b`:

```text
if terminal_success(b):
    value = terminal_value
else:
    sum = 0
    for each empty cell e:
        spawn2 = place rank 1 at e
        best2 = max success(move_d(spawn2)) over changing directions d

        spawn4 = place rank 2 at e
        best4 = max success(move_d(spawn4)) over changing directions d

        sum += (1 - p4) * best2 + p4 * best4

    value = empty_count > 0 ? sum / empty_count : zero_value
```

The reducer is per empty cell. Do not collapse a row before all relevant
future contributions have been considered.

## 3. Cell And Modulus Rules

Generation no longer has exact physical cells. Solve must follow the same
rules:

```text
physical cells are coord % modulus storage groups
bucket keys preserve raw board/quadrant sums
future candidate encoding always uses that future layer's own axis
FamilyId is local to one axis and one layer
```

Resident solve can scan the current layer in whatever physical cell layout it
was written with. Future lookup must encode candidates against the future
position axis and use that axis's own physical cells.

Single/family solve must support different current/future moduli by using
cell-list/window loaders and, when needed, bucket-key based remap. Do not assume
that a current physical cell id maps to the same future physical cell id.

## 4. Success DType Policy

BC success files support six dtype modes:

```text
UInt32
UInt64
Float32
Float64
OneMinusFloat32
OneMinusFloat64
```

The hot path should dispatch once to one physical `StorageT`:

```text
uint32_t
uint64_t
float
double
```

`OneMinusFloat32/64` use the same physical storage as `float/double`. The raw
value is `success - 1`, which is compatible with weighted averages and max
because the transform is monotonic and affine.

Do not switch dtype inside candidate lookup/reducer loops.

Current implementation caveat:

```text
BCBacksolve v1 supports UInt32 only.
BCResidentSolve/BCSingleChunkSolve templates are not yet tested production paths.
```

## 5. Implemented Baseline: BCBacksolve

Files:

```text
native_core/include/BCBacksolve.h
native_core/src/BCBacksolve.cpp
native_core/tests_src/test_bc_backsolve_resident.cpp
native_core/tests_src/bench_bc_backsolve_resident.cpp
```

Capabilities:

```text
resident current/future position readers
resident future success readers
UInt32 success dtype
dense row-aligned .bcsuc output
terminal success check
batch canonicalize
future2/future4 lookup
per-cell direct-entry future index
file output helper
basic stats
```

Current `BCFutureValueLayerView` builds a direct hash index per non-empty
future cell:

```text
bucket key -> rank_payload_offset, success_row_offset, bitmap_len
```

Lookup flow:

```text
canonical board
encode against future axis
find bucket entry in per-cell direct index
test/rank bucket bitmap using prefix256
read future success value
```

This is the resident correctness/performance baseline that the next solve
executor should preserve.

Known limitations:

```text
UInt32 only
success output is built in memory before optional file write
current cell values are stored in memory vectors
no route dispatcher
no direct success writer
no free9-256 solve benchmark yet
```

## 6. Shared Edge Kernel Scaffolding

File:

```text
native_core/include/BCSolveEdgeKernel.h
```

Intended role:

```text
terminal check
empty-cell ctz enumeration
spawn rank 1 / rank 2
move all or selected directions
optional target-family prefilter
batch canonicalize
encode to CellId + key + rank
collect prepared queries
reduce future lookup results
```

The header already has typed workspace and reducer templates, but it is not
compiled by an executable test today. The next implementation step should make
this kernel the single shared implementation for resident, single, and family
solve, then compare it against `BCBacksolve`.

Important adjustment from generation:

```text
family prefilter must use raw sums from bucket/board data and the target
future axis modulus. It must not compare FamilyId values across layers.
```

## 7. ResidentSolve Target Design

Header scaffold:

```text
native_core/include/BCResidentSolve.h
```

Target behavior:

```text
load current, future2, future4 position/success as resident views
build direct per-cell future indexes equivalent to BCBacksolve
scan every current cell
run shared BCSolveEdgeKernel
write dense current success
```

Required adjustment before using it:

```text
Current header uses future_position.cold_lookup() in the hot lookup lambda.
That is correct but not the performance path. Replace or wrap it with the
BCBacksolve direct-index view before benchmarking.
```

Acceptance:

```text
ResidentSolve == BCBacksolve byte-for-byte on synthetic tests
ResidentSolve == BCBacksolve on generated small free layers
UInt32 path reaches the same order of throughput as BCBacksolve
```

## 8. SingleChunkSolve Target Design

Header scaffold:

```text
native_core/include/BCSingleChunkSolve.h
```

Target memory contract:

```text
current position is loaded in cell chunks
future position/success can be resident one future at a time for v1
partial storage holds the not-yet-combined spawn4 contribution
final writer emits current cells in CellId order
```

The header currently has a two-phase design:

```text
phase 4:
    load future4 resident
    scan current chunks
    compute spawn4 partial contribution
    spool partial values by current cell

phase 2:
    load future2 resident
    rescan current chunks
    compute spawn2 contribution
    add stored spawn4 partial
    write final success cell
```

This matches the generation-side strict single idea: keep one future resident
layer plus a current chunk, not a whole resident current layer.

Required adjustment:

```text
Use direct future lookup views, not cold_lookup in the hot path.
Add executable tests that compare against ResidentSolve/BCBacksolve.
Add file-backed or direct-aware partial spool before large benchmarks.
```

## 9. FamilyChainSolve Target Design

FamilyChain solve should be target-family-major, not source-family-major.

For each target future family pass:

```text
load the needed future family view V(G)
scan relevant current cells/source families
generate prepared queries with target-family prefilter
apply partial updates for current cells
release future cells that leave the window
finalize current success cells only after both spawn phases are complete
```

The active future window is solve-specific. It is not the generation mutable
store. Use `BCFutureFamilyWindow` or its successor:

```text
native_core/include/BCFutureFamilyWindow.h
```

Partial success accumulation uses:

```text
native_core/include/BCPartialStore.h
```

Solve-side family window rules should mirror generation-side corrected modulus
rules:

```text
load physical cells using the future axis
if a requested logical family/cell cannot be represented exactly by old
physical cells, over-approximate reads but filter by bucket key/encoded query
before compute
do not keep over-approx data beyond the extraction/window stage
```

## 10. Success Writer Requirement

Family and large single solve need a streaming or direct-aware success writer.
The current memory writer is enough for `BCBacksolve` v1 but not for production
large solve.

Required writer behavior:

```text
begin layer from current .bcpos metadata
write typed cell block as raw dtype payload
mark empty cells
validate every cell exactly once
support CellId-order streaming first
later optionally support random-access cell writes
collect write bytes/seconds/backend stats
```

Initial implementation can require CellId-order writes. Family solve should
stage final cells until they can be flushed in CellId order.

## 11. Next Implementation Order

Recommended next-session order:

```text
1. Extract BCBacksolve future direct index into a typed reusable lookup view.
2. Wire BCSolveEdgeKernel into a tested ResidentSolve UInt32 path.
3. Prove ResidentSolve == BCBacksolve on synthetic and small generated layers.
4. Add streaming/direct success writer.
5. Implement SingleChunkSolve using the same edge kernel and direct lookup.
6. Only then start FamilyChainSolve target-family-major executor.
```

Do not start FamilyChainSolve before resident and single share the same tested
edge kernel; otherwise the project will have multiple subtly different solve
recurrences.

## 12. Solve File Map

Current implemented baseline:

```text
native_core/include/BCBacksolve.h
native_core/src/BCBacksolve.cpp
native_core/tests_src/test_bc_backsolve_resident.cpp
native_core/tests_src/bench_bc_backsolve_resident.cpp
```

Current scaffolding:

```text
native_core/include/BCSolveEdgeKernel.h
native_core/include/BCResidentSolve.h
native_core/include/BCSingleChunkSolve.h
native_core/include/BCFutureFamilyWindow.h
native_core/include/BCPartialStore.h
```

Shared IO/position dependencies:

```text
native_core/include/BCSuccessIO.h
native_core/include/BCPositionFile.h
native_core/include/BCPositionCellLoader.h
native_core/include/BCPositionScanner.h
native_core/include/BCLoadedCellScanner.h
native_core/include/BCFileIO.h
native_core/include/BCDirectFileIO.h
```
