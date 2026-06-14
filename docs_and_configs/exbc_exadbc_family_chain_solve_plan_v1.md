# BC FamilyChain Solve Plan

This document is the FamilyChain-specific solve plan. For shared solve
semantics, dtype policy, and current implementation status, read
`exbc_exadbc_three_solve_chains_runtime_design_v1.md` first.

## 1. Current Status

FamilyChain solve is not implemented yet.

Available building blocks:

```text
BCSolveEdgeKernel.h          header-only edge kernel scaffold
BCFutureFamilyWindow.h       active future cell window scaffold
BCPartialStore.h             typed current-cell partial block store
BCSingleChunkSolve.h         two-phase solve scaffold
BCBacksolve.cpp              implemented resident UInt32 baseline
```

There is no production FamilyChainSolve executor, no benchmark, and no
free9-256 solve validation yet.

## 2. Design Goal

FamilyChain solve exists to lower peak memory when resident/single solve cannot
fit. It must use the same recurrence and the same `.bcpos/.bcsuc` semantics as
resident solve.

It must not use generation-only structures:

```text
BCCellMutableBuilder
BCFamilyMutableStore
BCGenerationBlobIO
BCFamilyPositionWriter
```

It must not create target position entries. It only writes current success.

## 3. Scheduling Order

The main order is target-family-major:

```text
for phase in spawn4 then spawn2:
    for target future family G in solve order:
        load future view V(G)
        scan current/source cells that can query V(G)
        apply partial contribution to current rows
        release future cells that leave V(G)
```

Do not use source-family-major as the main order. It tends to reload future
views and makes IO amplification hard to bound.

The phase order mirrors single solve:

```text
spawn4 partial pass first
spawn2 finalize/add pass second
```

This allows a bounded partial representation and avoids needing two resident
future layers at once.

## 4. Future View V(G)

For a future axis with physical modulus `M`, the physical target family `G`
requires the row/column cross:

```text
row family G: cells (G, x), all x in [0, M)
col family G: cells (x, G), all x in [0, M), x != G
```

The helper equivalent is:

```text
bc_future_family_view_cells(axis, G)
```

The active window object should:

```text
load missing position/success cells
retain cells that stay in the next V(G)
release cells that leave the window
support typed success values
support direct-aware/coalesced reads
record active cell and IO stats
```

Existing scaffold:

```text
native_core/include/BCFutureFamilyWindow.h
```

## 5. Modulus And Remap Rules

Generation now uses modulo physical cells for every route. Family solve must
inherit that model.

Rules:

```text
candidate future board is encoded against the future layer's own axis
current physical cell ids are not comparable to future physical cell ids
FamilyId is local to one axis/layer
raw bucket keys preserve board sums and are the source of truth
```

If current/future files use different moduli, the loader may over-approximate
physical reads. Filtering must use encoded query or bucket-key semantics before
the data enters compute.

No over-approx loaded data should remain resident after extraction/window
construction.

## 6. Work Items

Preferred work item granularity:

```text
current cell
current bucket
current bitmap word range
```

The rank payload prefix256 table allows each bitmap word-range item to compute
its starting local success row without scanning from the bucket beginning.

For each source board in an item:

```text
run BCSolveEdgeKernel for one phase
target-family prefilter rejects candidates that cannot hit current G
prepared queries carry cid/key/rank/ref/lane
future window resolves queries to typed success values
partial store is updated for current row/ref/lane
```

## 7. Partial Store

Family solve cannot finalize a current row until all relevant target-family
passes for both spawn phases have contributed.

Use typed partial blocks:

```text
current CellId
local_success_row
lane
partial value
dirty/finalization state
```

Existing scaffold:

```text
native_core/include/BCPartialStore.h
```

The first production version may keep active partial cells in memory and spill
only when leaving the target-family window. Large-route production should add a
file-backed partial spool before full-size benchmarks.

## 8. Target-Family Prefilter

The prefilter is optional for correctness but important for performance.

It must answer:

```text
can this moved future board physically land in target family G of the future axis?
```

It should compute raw row/column coordinates from raw quadrant sums, then apply
the future axis modulo. Do not use current layer family ids.

Existing scaffold:

```text
bc_solve_physical_target_family_may_hit(...)
```

This function should be reviewed before FamilyChain solve production work. It
must use raw sums consistently with generation's `BCPositionCellLayout`.

## 9. Success Output

Family solve writes `.bcsuc`, not `.bcpos`.

Required first writer:

```text
CellId-order streaming success writer
typed block input
raw dtype output
empty-cell marking
finish-time validation
write stats
```

If target-family finalization order is not CellId order, the executor must
stage completed cells until they can be flushed in order. Random-access success
writer can be added later, but should not be required for v1.

## 10. Runtime Invariants

Production FamilyChain solve must assert:

```text
future active cell count is bounded by the configured V(G) window
no generation mutable builders are created
future success dtype matches the typed executor
future success metadata matches future position metadata
every current success cell is written exactly once
partial rows are finalized only after both spawn phases
```

It should record:

```text
future cells loaded/retained/released
future position/success read bytes and seconds
prepared query count
lookup miss count
prefilter skip count
partial active bytes peak
success write bytes and seconds
```

Avoid per-candidate stats in the default hot path; make detailed counters
diagnostic-only.

## 11. Correctness Plan

Small tests:

```text
BCBacksolve resident oracle still passes
ResidentSolve shared-edge output equals BCBacksolve
SingleChunkSolve output equals ResidentSolve for chunk sizes 1, 3, all
FamilyChainSolve output equals ResidentSolve on synthetic small layers
FamilyChainSolve handles different current/future moduli
target-family prefilter on/off produces identical output
```

Integration tests:

```text
generate a small free pattern with resident/single/family routes
solve one layer with ResidentSolve
solve same layer with FamilyChainSolve
compare .bcsuc byte-for-byte or row-by-row
```

Performance gate:

```text
free9 medium layer benchmark before free9-256
report recalc rows/s, IO GB/s, peak working set, active window peak
```

## 12. Recommended Next Step

Do not begin with the full FamilyChain executor.

First:

```text
extract BCBacksolve direct future index into a typed reusable view
wire BCSolveEdgeKernel into ResidentSolve
prove ResidentSolve == BCBacksolve
add success streaming writer
```

Then:

```text
make SingleChunkSolve use the same edge kernel and direct lookup
prove SingleChunkSolve == ResidentSolve
```

Only after that:

```text
implement FamilyChainSolve target-family-major scheduler
```
