# BC Three Solve Chains Runtime Design

This document is the current solve-side handoff. It is aligned with the BC
position generation design in
`docs_and_configs/bc_resident_singlechunk_generation_impl_supplement.md` and
with the current resident/single-chunk solve implementation.

Current production state:

```text
Implemented and tested:
    typed BCFutureSuccessLookupView
    typed BCResidentSolve
    strict 1+x BCSingleChunkSolve full-run path
    direct-capable success/position/tmp4 IO
    ResidentSolve synthetic oracle tests for UInt32 and typed dtypes

Implemented as support or legacy entry points:
    in-memory single-chunk compacted build APIs
    file/file strict single solve API
    BCFutureFamilyWindow and BCPartialStore scaffolding

Not implemented as a production chain:
    FamilyChainSolve executor
    automatic solve route dispatcher
```

## 1. Solve Is Not Generation In Reverse

Solve reads existing `.bcpos` position files and future `.bcsuc` success files,
then writes solved `.bcpos + .bcsuc` files for the current layer.

Forbidden in solve production paths:

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
current position streaming reader / resident reader / loaded cell scanner
future position + typed success lookup view
typed current success accumulator for the active layer or chunk
row-slab plans and sub-cell work items
thread-local board/canonical/query buffers
streaming/direct output writers
```

Generation `.bcpos` files are kept as generated inputs. Solve output is written
to a separate solved directory with the same suffixes and file formats:

```text
<prefix><ordinal>.bcpos
<prefix><ordinal>.bcsuc
```

The solved `.bcpos` is a zero-compacted position file. Its layout remains the
same BC position layout as generation output, so it can be used directly as a
future layer by lower layers.

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

The reducer is per empty cell. Do not collapse a row before both spawn phases
and all legal directions have been considered.

Current hot paths use `BCSolveEdgeKernel` helpers for:

```text
terminal check
empty-cell ctz enumeration
spawn rank 1 / rank 2
move left/right/up/down according to direction mask
canonicalize and encode future queries
batch lookup/reduce into typed values
```

## 3. Cell And Modulus Rules

Generation no longer has exact physical cells. Solve follows the same storage
rules:

```text
physical cells are coord % modulus storage groups
bucket keys preserve raw board/quadrant sums
future candidate encoding always uses that future layer's own axis
FamilyId is local to one axis and one layer
```

Resident solve scans the current layer in its written physical layout. Future
lookup encodes candidates against the future position axis and uses that axis's
physical cells.

Single solve also keeps these rules. The current production full-run uses a
single generated/current modulus family for the tested free tables, but the
lookup and loader boundaries are axis-local and do not rely on current cell id
being equal to future cell id.

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

The solve code dispatches once to a physical `StorageT`:

```text
uint32_t
uint64_t
float
double
```

`OneMinusFloat32/64` use `float/double` storage. Their stored value is
`success - 1`, so zero success is `-1` and terminal success is `0`. The
transform is affine and monotonic, so max and weighted-average logic is shared
with normal float/double storage. The hot lookup/reducer loops do not branch on
one-minus mode; correctness depends on using the dtype-derived
`zero_value` and `terminal_value`.

Code must not hard-code zero as `0` or terminal as max inside generic typed
paths. Use:

```text
bc_success_zero_value_for_dtype<T>(dtype)
bc_success_terminal_value_for_dtype<T>(dtype)
BCResidentSolveOptions<T>::set_dtype(...)
```

`row_width >= 1` is supported in the typed APIs. Compact keeps a row when any
lane passes the keep predicate, preserving row alignment. Current production
full-run benches use UInt32 with `row_width = 1`.

## 5. Future Lookup

Typed future lookup is implemented in:

```text
native_core/include/BCFutureSuccessLookup.h
```

`BCFutureSuccessLookupView<StorageT>` builds a direct open-addressing hash
table per non-empty future cell. The current direct entry is intentionally
compact:

```text
uint64_t key
uint32_t rank_payload_offset
uint32_t success_row_offset
```

The entry is 16 bytes. Empty entries use `rank_payload_offset == UINT32_MAX`;
`key == 0` is not a special sentinel. Capacity is about 3.2x bucket count,
rounded to a power of two.

Lookup flow:

```text
canonical board
encode against future axis -> cid/key/rank/bitmap_len
hash key in the encoded future cell
prefetch direct entry
linear-probe until match or empty slot
compute bitmap offset from rank_payload_offset + bitmap_len
test bitmap bit first
only if bit is set, compute rank using prefix256 + local popcount
read success_row_offset + rank_before + lane
reduce max into the caller's best array
```

Whole-empty buckets are removed during compaction. For dead ranks inside a live
bucket, lookup first tests the bitmap bit and only ranks/popcounts when the bit
is set.

Scalar lookup is kept for tests and non-hot callers. Production resident and
single solve use batch lookup.

## 6. Resident Backsolve Route

Primary code:

```text
native_core/include/BCResidentSolve.h
native_core/tests_src/bench_bc_resident_solve_full.cpp
native_core/tests_src/bench_bc_resident_solve_layer.cpp
native_core/tests_src/test_bc_resident_solve.cpp
```

Resident route keeps three logical layers in memory:

```text
current generated position
future2 exact solved layer
future4 exact solved layer
```

Current raw success is computed into a typed value buffer, then zero-compacted
in place:

```text
read current generated .bcpos
raw = solve(current, future2_exact, future4_exact)
solved_layer = compact_zero_in_place(current, raw)
free raw offsets/buffers
write solved_layer as exact checkpoint
optionally prune old future4 for archive only when threshold > zero_value
future4 = move(future2)
future2 = move(solved_layer)
```

Important semantics:

```text
zero compact produces the exact layer that participates in later computation
threshold prune never produces a future layer for computation
threshold <= zero_value skips archive prune and archive rewrite
old future4 is the only layer eligible for threshold archive pruning
```

`bc_resident_compact_zero_in_place(...)` moves live values down inside the raw
success vector and then rebuilds compact position metadata. The non-in-place
`bc_resident_compact_layer(...)` remains available for tests and small
callers.

`bc_resident_prune_below_threshold_for_archive_in_place(...)` has the same
structural rebuild but uses `row[lane] > threshold` as its keep predicate. It
is not part of the exact future frontier.

The old `BCBacksolve` UInt32 oracle has been removed. Correctness coverage now
comes from the resident synthetic oracle, dtype roundtrip tests, resident layer
fixtures, and single-chunk comparisons against resident semantics.

## 7. Single-Chunk Backsolve Route

Primary production code:

```text
native_core/include/BCSingleChunkSolve.h
native_core/tests_src/bench_bc_single_chunk_solve_full.cpp
native_core/tests_src/bench_bc_single_chunk_solve_layer.cpp
native_core/tests_src/test_bc_single_chunk_solve.cpp
```

The production full-run path is:

```text
bc_single_chunk_solve_strict_1x_to_files_from_future4_frontier(...)
```

It is strict `1 + x`:

```text
1 = one resident future layer
x = one current row-slab chunk plus its typed value buffers
```

It never caches the whole current layer as decoded boards and does not keep
both future2 and future4 resident at the same time during the strict full-run
solve. The output current layer is streamed to files; it is not materialized as
the next future2 resident layer.

Full-run frontier order:

```text
initial:
    write terminal top layer
    write virtual empty top+1 layer
    future4_frontier = virtual empty layer

for ordinal from max-1 down to min:
    open current generated .bcpos as a streaming reader
    open solved ordinal+1 as future2 streaming readers
    pass 1 uses resident future4_frontier
    release future4_frontier
    load future2 into BCSingleChunkFrontierLayer
    pass 2 uses resident future2
    stream solved current ordinal to .bcpos + .bcsuc
    future4_frontier = move(future2)
```

This avoids rereading the layer that becomes `+4` in the next iteration. The
newly solved current layer becomes next iteration's `+2` by opening its just
written files, which keeps the strict memory boundary.

Resume support:

```text
--start-ordinal N requires solved N+1 and N+2 files
future4_frontier is loaded from solved N+2
the loop starts at N
```

## 8. Single-Chunk Current Chunking

Current production chunking is row-slab based:

```text
--current-chunk-rows N       default 128
--current-chunk-max-bytes B  default 512 MiB
```

`bc_single_chunk_next_row_slab_cids(...)` chooses all cells whose row family id
falls in the next row range, capped by the byte budget. This gives a much
larger unit than one cell while still preserving CellId order for output.

Each chunk is loaded with `BCPositionStreamingReader::load_cells(...)`.
Per-chunk planning builds:

```text
cell_offsets: row offset of each loaded cell inside the chunk value buffer
BCSingleChunkLoadedWorkItem: sub-cell bucket/bitmap-word work ranges
```

Large buckets are split into work items using the same granularity as resident
solve. Recalc then runs with OpenMP over these work items. Scanning/unpacking is
performed from `BCLoadedCell` data; there is no quadrant-board cache in the
production full-run path.

## 9. Single-Chunk Two-Pass Solve

The strict full-run route is two pass and uses no partial max store.

Pass 1:

```text
for each current row-slab chunk:
    load current cells
    allocate/reuse sum4_values for this chunk
    solve Spawn4 against resident future4
    scale by spawn4 probability
    write tmp4 chunk file
```

`tmp4` stores one `StorageT` weighted contribution per current row/lane:

```text
using BCSingleChunkSum4T<StorageT> = StorageT
```

For UInt32 this intentionally accepts one extra rounding step so tmp4 size is
the same as the final success payload size for that chunk.

Pass 2:

```text
release old future4
load future2 as a resident frontier
for each current row-slab chunk:
    reload current cells
    read tmp4 values
    allocate/reuse raw_values for this chunk
    solve Spawn2 against resident future2
    add scaled spawn2 contribution to tmp4 contribution
    compact the chunk by cell
    append compacted cell payloads and success values to final output files
    delete the tmp4 chunk file
```

The pass2 chunk is final after spawn2. There is no later merge operation.

## 10. Single-Chunk Output Streaming

Final output is handled by `BCSingleChunkFinalFileStreamer<StorageT>`.

At construction it:

```text
reserves the output .bcpos upper bound from the current generated position
reserves the output .bcsuc upper bound from current success rows
initializes descriptor metadata and CellId write tracking
starts direct-capable sequential stagers for bucket/rank/success payloads
```

For each compacted chunk it:

```text
accepts cells in CellId order
stores compacted descriptors in memory
appends bucket metadata and rank payload streams
appends typed success values
```

On finish it:

```text
validates every cell was written exactly once
writes the final position header/axis/descriptors
writes the success header and first aligned block
resizes files to logical size unless the caller keeps direct padding
returns logical position/success byte counts
```

The full-run bench defaults to keeping direct padding when `--direct-io` is
used:

```text
--keep-direct-padding    default
--trim-direct-padding    resize to logical size after each layer
```

## 11. Direct IO And Large Request Limits

Shared direct IO is in:

```text
native_core/include/BCDirectFileIO.h
native_core/include/BCFileIO.h
native_core/include/BCSuccessIO.h
native_core/include/BCPositionCellLoader.h
```

Single full-run direct IO options:

```text
--direct-io
--direct-queue-depth 16
```

These are wired into current/future position streaming readers, success
streaming readers, frontier loading, final position/success writers, and tmp4
direct read/write.

Windows direct IO requests are split before submission when a physical request
would exceed the Win32 DWORD byte count limit. Buffered writes also chunk large
requests. This avoids both silent truncation and "single request exceeds DWORD"
runtime failures on large BC files.

Direct readers may use buffered fallback when alignment/padding requirements
are not met. Success whole-payload reads use `BCSuccessOwnedValues<T>` so direct
payloads can be kept in aligned file-backed storage instead of forcing an extra
`std::vector<T>` copy.

## 12. Statistics And Production Benchmarks

Single full-run CSV is produced by:

```text
native_core/tests_src/bench_bc_single_chunk_solve_full.cpp
```

Layer CSV fields include:

```text
current_rows
live_rows
zero_pruned_rows
current_chunks
current_cells
current_work_items
tmp4_write_bytes/tmp4_read_bytes
current/future/backend read bytes and seconds
future2_index_seconds/future4_index_seconds
current_plan_seconds
future_release_seconds
recalc_seconds
compact_seconds
tmp4_write_seconds/tmp4_read_seconds
position_write_seconds/success_write_seconds
accounted_seconds/untracked_seconds
recalc_mrows_per_sec/total_mrows_per_sec
```

Summary CSV includes:

```text
total_rows
total_live_rows
total_zero_pruned_rows
total_position_bytes
total_success_bytes
max_layer_written_bytes
solve_wall_seconds
wall_mrows_per_sec
recalc_mrows_per_sec
peak_working_set_bytes
```

Current validation snapshot from the latest free10-256 single full-run:

```text
generated_position_dir = tmp/free10_256_resident_generated
solved_output_dir      = tmp/free10_256_single_chunk_solved
total_rows             = 92,740,544,730
solve_wall_seconds     = 1197.789 s
recalc_seconds         = 829.870 s
wall throughput        = 77.426 Mrows/s
recalc throughput      = 111.753 Mrows/s
peak working set       = 5,033,488,384 bytes
max layer written size = 2,764,431,468 bytes
input read seconds     = 93.196 s after current direct/padded read optimization
layer1 max success     = 3996176335 / 4000000000 = 0.99904408375
```

## 13. FamilyChainSolve Boundary

FamilyChain solve remains a future production route. Its intended direction is
target-family-major, not source-family-major.

For each target future family pass:

```text
load the needed future family view V(G)
scan relevant current cells/source families
generate prepared queries with target-family prefilter
apply partial updates for current cells
release future cells that leave the window
finalize current success cells only after both spawn phases are complete
```

Family solve is the route that will need `BCPartialStore` and disk partial max
semantics. Resident and current single-chunk solve do not use partial max
storage.

## 14. Solve File Map

Resident:

```text
native_core/include/BCResidentSolve.h
native_core/tests_src/test_bc_resident_solve.cpp
native_core/tests_src/bench_bc_resident_solve_full.cpp
native_core/tests_src/bench_bc_resident_solve_layer.cpp
```

Single-chunk:

```text
native_core/include/BCSingleChunkSolve.h
native_core/tests_src/test_bc_single_chunk_solve.cpp
native_core/tests_src/bench_bc_single_chunk_solve_full.cpp
native_core/tests_src/bench_bc_single_chunk_solve_layer.cpp
```

Shared solve support:

```text
native_core/include/BCSolveEdgeKernel.h
native_core/include/BCFutureSuccessLookup.h
native_core/include/BCFutureFamilyWindow.h
native_core/include/BCPartialStore.h
native_core/include/BCSuccessIO.h
native_core/include/BCPositionFile.h
native_core/include/BCPositionCellLoader.h
native_core/include/BCPositionScanner.h
native_core/include/BCLoadedCellScanner.h
native_core/include/BCFileIO.h
native_core/include/BCDirectFileIO.h
```
