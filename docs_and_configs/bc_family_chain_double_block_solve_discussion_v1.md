# BC FamilyChain Double-Block Solve Discussion Notes

This note records the agreed design context for BC double-block backsolve work.
It is a discussion snapshot plus current implementation notes, not a
replacement for the broader solve runtime design documents.

Related documents:

```text
docs_and_configs/exbc_exadbc_three_solve_chains_runtime_design_v1.md
docs_and_configs/exbc_exadbc_family_chain_solve_plan_v1.md
docs_and_configs/bc_resident_singlechunk_generation_impl_supplement.md
docs_and_configs/bc_compressed_result_format.md
```

Current implementation owner:

```text
native_core/include/BCFamilySolve.h
native_core/include/BCFamilySolvePlan.h
native_core/include/BCFamilySolveRunner.h
```

## 1. Scope And Current Production State

FamilyChain solve is implemented as the low-memory route under
`BCFamilySolveRunner`. It is selected by solve-side routing when resident and
single-chunk do not fit, or by a forced `solve_route=family`.

Current assumptions and invariants:

```text
current layer modulus == future2 layer modulus == future4 layer modulus
the modulus is supplied by the caller before BC calculation starts
generation and solve must keep using that modulus
layer-internal resume is not required
single-layer correctness is required
per-module performance should be reasonable from the first version
```

Modulus selection is deliberately left to the caller. The BC runtime may route a
layer through resident, single-chunk, or FamilyChain execution, but route
selection must not change the modulus. Generation-side auto routing still exists
for memory/performance choice; it uses the externally supplied modulus for every
route. Solve-side routing follows the same rule.

Solve-side auto route uses layer row counts:

```text
a = current generated layer row count
b = future n+2 exact/live row count after zero-success compaction
c = future n+4 exact/live row count after zero-success compaction

resident route fits when available physical memory >= 1 GiB + 5 * (a + b + c)
single route fits when available physical memory >= 1 GiB + 6 * max(b, c)
otherwise use FamilyChain double-block solve with the preset modulus
```

Layer-level checkpoint/resume is implemented in the runner. Layer-internal
resume is intentionally not supported.

Small struct layouts may still be refined, but the large execution flow is
settled: full Spawn4 fid sweep, then full Spawn2 fid sweep.

## 2. Shared Semantics

FamilyChain solve computes the same BC recurrence as resident and single-chunk
solve. It reads generated current `.bcpos` and solved future `.bcpos + .bcsuc`,
then writes solved compacted current `.bcpos + .bcsuc`.

It must not change success semantics, dtype semantics, or final exact
position/success file semantics. The current `.bcsuc` format is v2 and includes
a per-cell success value offset table; resident, single, FamilyChain, exact
reader, and compressed-result builder all use the same format.

Generation `.bcpos` stores boards after move and before spawn. Boards with no
empty tile are not valid stored current positions, because such boards are not
generated into these files.

Success boards are still processed through the same solve flow. They are not a
special case that avoids all partial/scratch bookkeeping.

## 3. Family Sweep And Direction Ownership

The sweep order is by increasing family id:

```text
for fid = 0..F-1:
    load current family cross for fid
```

The current family cross contains:

```text
row fid: cells (fid, x)
col fid: cells (x, fid), x != fid
```

Direction ownership follows how the cell enters the cross:

```text
(fid, x) row side     -> horizontal moves
(x, fid) column side  -> vertical moves
(fid, fid) diagonal   -> both horizontal and vertical moves
```

This matches the generation-side `BCFamilyGenerationScheduler` logic:

```text
source row cross cells are Horizontal
source column cross cells are Vertical
diagonal cell is Both
```

For a non-diagonal cell, the two directions are computed in two appearances:

```text
first appearance  -> compute one direction and write partial max
second appearance -> read partial max, compute the other direction, finalize
```

For a diagonal cell, both directions can be computed in one appearance and no
partial max file is needed for that cell.

The completed cells form the controlled L-shaped boundary of the top-left
rectangle as `fid` increases. Calculation order is fid order. Physical write
order does not have to be identical to cell-id order, as long as the final file
format remains logically compatible.

## 4. Future Family Fanout

Future family fanout should follow the generation-side double-block algorithm.
The useful references are:

```text
native_core/include/BCFamilyGenerationScheduler.h
native_core/include/BCFamilyPartitionPolicy.h
native_core/src/BCFamilyGeneration.cpp
```

Generation maps one source/current family to target/future families using:

```text
map_partition_source_family_to_target_families(...)
checked_partition_fanout3(...)
```

In current production, current and future layers are expected to use the fixed
caller-supplied modulus. The fanout is conceptually:

```text
current fid + spawn delta -> future target family ids
```

The future window normally needs two families and can occasionally need three.
The implementation should calculate this explicitly from the same family
mapping rules instead of relying on broad fallback checks in the hot path.

### Exact-coordinate proof

For one axis, let the two side sums be `(A, B)`. After side normalization:

```text
a = min(A, B)
b = max(A, B)
N = a + b
```

The exact family coordinate is `a`. If the spawned tile has axis delta `d`,
then the spawned tile can land on either side.

If it lands on the heavier side:

```text
(a, b) -> (a, b + d)
normalized family = a
```

If it lands on the lighter side:

```text
(a, b) -> (a + d, b)
normalized family = min(a + d, b)
```

Therefore exact-coordinate fanout is:

```text
{ a, min(a + d, b) }
```

It is strictly at most two. When the two candidates coincide, this is a repeated
root rather than a third family.

Horizontal moves preserve the top/bottom side sums, so the horizontal direction
uses row-family fanout. Vertical moves preserve the left/right side sums, so the
vertical direction uses column-family fanout.

Canonicalization may transform horizontal-looking boards into vertical-looking
canonical forms. That can change the final `cid`, but it does not invalidate
the fanout proof: the future window loads the whole row/column cross for each
fanout family, so a canonicalized result with that family on either axis is
still covered.

### Modulo fanout

With physical modulo `M`, one physical family `r` represents many exact
coordinates:

```text
a = r + kM
```

The exact candidates above map into the modulo set:

```text
{ r, r + d, N - r } mod M
```

The third candidate appears only because modulo merges many exact coordinates
and the light-side spawn can cross the normalization center. Duplicates are
common, so the practical fanout is one, two, or three families.

## 5. Two Spawn-Phase Sweeps

FamilyChain solve uses the same high-level phase order as single-chunk solve:

```text
Spawn4 sweep first
Spawn2 sweep second
```

Each phase sweeps `fid = 0..F-1`.

Spawn4:

```text
first direction of a non-diagonal cell:
    compute directional max4
    write compact partial max4

second direction of that cell:
    read compact partial max4
    compute second directional max4
    max4 = max(partial max4, second directional max4)
    avg4 = average(max4 over empty slots)
    scale avg4 by spawn4 probability
    write per-cell normal success scratch contribution
```

Spawn2:

```text
first direction of a non-diagonal cell:
    compute directional max2
    write compact partial max2

second direction of that cell:
    read compact partial max2
    compute second directional max2
    max2 = max(partial max2, second directional max2)
    avg2 = average(max2 over empty slots)
    scale avg2 by spawn2 probability
    read per-cell normal success scratch contribution from Spawn4
    add Spawn2 contribution
    zero-compact the cell
    stage/write the final compacted cell
```

The normal success scratch logic, timing of compaction, and use pattern should
match single-chunk solve as closely as possible. The scratch is dense over the
generated current cell rows. It is not compacted after Spawn4, because Spawn2
still needs alignment with the original current cell row order.

Equivalently, for each current row:

```text
Spawn4:
    success[row] = p4 * average(max4_per_empty_slot)

Spawn2:
    success[row] += p2 * average(max2_per_empty_slot)
```

Partial max stores only the per-empty-slot max for one spawn phase and one
already-computed direction. It never stores the average and never stores a
weighted contribution.

## 6. Per-Cell Normal Success Scratch

Normal success scratch is per cell, not a whole-layer resident array.

For a cell with `success_rows` generated current rows:

```text
value_count = success_rows * row_width
layout      = generated current cell row order
content     = weighted Spawn4 contribution
```

The scratch is released after the same cell is finalized in Spawn2.

This avoids keeping full-layer success values resident. It also avoids changing
the final `.bcsuc` format.

## 7. Compact Partial Max Format

Partial max is not a normal success array. It stores unweighted max values for
one spawn phase and one already-computed direction.

Partial max should be compact by using the bucket empty-mask invariant:

```text
one bucket key determines the board empty mask
all boards under that bucket have the same empty count
all boards under that bucket have the same empty cell positions
```

For each current cell, partial max can be represented as a logical matrix per
bucket:

```text
rows:    live board rows in bucket rank order
columns: empty slots from bucket_empty_mask, in ctz order
lanes:   row_width
```

Physical storage can be one contiguous 1D array. Offsets are computed from:

```text
bucket partial base
bucket-local live row ordinal
empty slot ordinal within bucket_empty_mask
lane
row_width
```

The value count for one bucket is:

```text
bucket_live_rows * popcount(bucket_empty_mask) * row_width
```

The partial value count is therefore based on actual empty count, not 16 fixed
cell slots.

After the first direction is computed, every partial max position for that cell
and phase is written. There should be no sparse holes and no need to rely on
zero-value fill inside the partial max payload.

If a direction has no legal move/query for a logical slot, the written value is
that direction's computed `zero_value`. This is still a real dense value, not a
missing entry.

The partial payload does not need to store:

```text
cell id
rank
empty cell index
empty count
direction
spawn rank
row offset
```

Those are determined by the outer scheduler, the current cell `.bcpos` bucket
layout, dtype, row width, and phase.

## 8. Final Output Staging

The physical order in which cells are calculated or staged does not have to
match cell-id order. The final logical `.bcpos + .bcsuc` output should remain
compatible with resident and single-chunk solve:

```text
.bcpos carries final compacted cell descriptors
.bcsuc v2 carries per-cell value offsets and typed success payload
success readers use the offset table plus final compacted success_rows
```

FamilyChain may stage finalized cells in `family_final_values.bcfstmp` when the
configured final pending memory cap does not allow immediate flush. That staging
file is not part of the public result format. It is accounted in
`final_stage_write/read_seconds` and related byte counters.

The `.bcsuc` v2 offset table is intentionally small, O(cell_count), and is not
O(layer board count).

## 9. Future Lookup Shape

`BCFutureFamilyWindow` is useful as a scaffold for family-window ownership and
load/retain/release accounting.

The production hot path should use the typed direct lookup view:

```text
load active future cells
build BCFutureSuccessLookupView<StorageT> with open_loaded(...)
batch lookup/reduce through the same direct lookup machinery used by
resident/single solve
```

This keeps the initial FamilyChain solve path closer to the optimized
resident/single implementation.

The current implementation uses trusted-axis collectors in production. Planner
coverage is treated as a correctness invariant; release hot loops do not use
bucket hit masks as per-empty-slot fallback filters.

## 10. Memory Window Invariant

For double-block solve, board-scale resident data must stay bounded by the
family window, not by layer board count.

At any point in the full flow:

```text
future resident board data <= 3 families
future resident board data + current/source family cross <= 4 families occasionally
```

This is the corrected bound. The earlier "around one hundred families" note was
an input mistake and should not be used as an implementation target.

Metadata may be resident when it is family-scale rather than board-scale:

```text
O(F) metadata is acceptable
O(F^2) metadata is acceptable
family-cell matrices and pass/keep plans are acceptable
```

What is not acceptable is `O(n)` resident board data, where `n` is the number of
positions/boards in the layer.

Future-family reuse must follow the fid execution order. For each pass, compute
which later passes are guaranteed to need the same future cells within the
three-family window; retain those cells after the pass and release the rest.
It should not blindly union adjacent passes and hope for cache hits.

Practical benchmark memory budget update:

```text
family modulus: 29
each cell is solved twice
average effective family volume: roughly layer / 15
one mid-layer family data volume: roughly 0.26 GB
current/future/partial-max live data may be budgeted as about 5 families
reasonable peak memory target for the tested mid-layer: <= 2 GB
```

BC double-block solve keeps Spawn4 and Spawn2 as separate full sweeps:
finish the Spawn4 fid sweep and persist its scratch/partial output, then start
the Spawn2 sweep and finalize cells. The previous bounded block interleave route
is deprecated and removed because it mixes the +4 and +2 phases and complicates
the production memory model.

Success dtype handling is also a production invariant. The six success dtypes
(`UInt32`, `UInt64`, `Float32`, `Float64`, `OneMinusFloat32`,
`OneMinusFloat64`) must run the same family solve control flow: same Spawn4
sweep, same Spawn2 sweep, same partial/scratch/finalize semantics, and same
cell ordering rules. The dtype dispatch may choose the `StorageT` instantiation
and typed zero/terminal values; 64-bit dtypes naturally move more bytes. It
must not select a different algorithm route for integral versus floating-point
success files.

## 11. Runtime And Statistics

Production execution is in-process through `formation_core.run_bc_family_build`.
The frontend no longer launches BC helper executables for production builds.
Progress is updated through `FormationProgress` after each generated layer and
after each archived/retired solved layer.

Solve stats now include route decision fields, route memory thresholds, total
and solve-call throughput, and detailed open/compute/temp/final/archive timing.
The solve-layer CSV ends with a `total` row.

The standalone `bc_family_solve_full` executable is a thin CLI wrapper around
the same runtime and emits the same solve stats shape.
