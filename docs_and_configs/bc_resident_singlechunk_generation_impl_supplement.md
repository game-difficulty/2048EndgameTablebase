# BC Generation Runtime Design And Handoff

This document describes the generation implementation as of checkpoint
`6ed14a2`. It is the source of truth for the current resident, single, and
family generation routes.

## 1. Non-negotiable Semantics

- Every tile contributes its raw tile value to quadrant and layer sums.
- Physical cells are a storage partition only. A physical cell maps raw exact
  row/column coordinates through `coord % modulus`.
- Bucket keys still carry raw quadrant sums/ranks. The file format does not
  hide or change board semantics when the physical modulus changes.
- A `.bcpos` file is valid for all three generation routes. Resident, single,
  and family outputs are byte-layout compatible at the BC position level.
- `FamilyId` values are layer-local physical ids. Never compare family ids
  across layers or across different moduli.
- If an old source file uses a different physical modulus, the reader/remap
  layer may over-approximate old physical cell reads, but extraction must use
  bucket keys and keep only target logical cells before compute continues.

## 2. Position File Naming

Production benchmark output no longer puts the layer sum in the filename. The
name is:

```text
<pattern>_<target_tile>_<ordinal>.bcpos
```

Examples:

```text
free9_256_0.bcpos
free9_256_1.bcpos
free9_256_102.bcpos
```

`ordinal = (layer_sum - seed_sum) / 2`. The layer sum remains in the `.bcpos`
header.

## 3. Shared `.bcpos` Layout

All generation routes write the same position format:

```text
BC position header
axis coordinate table
cell descriptor table
bucket metadata stream
rank payload stream
```

Each descriptor records:

```text
bucket_count
success_rows
bucket_meta_offset
rank_payload_offset
rank_payload_bytes
flags
```

Each bucket entry stores:

```text
key
rank_payload_offset
success_row_offset
```

The rank payload is:

```text
prefix256[ceil(bitmap_len / 256)] as little-endian uint16
aligned bitmap words as little-endian uint64
```

`prefix256[i]` is the number of set bits before the `i`-th 256-bit bitmap
block. Lookup must use this prefix and must not rescan from bitmap word zero.

## 4. Physical Cell Layout

The shared layout helper is `BCPositionCellLayout`.

For modulo layout:

```text
raw row coord -> row coord % modulus
raw col coord -> col coord % modulus
physical cell id = row_mod * modulus + col_mod
```

The serialization axis has `family_count = modulus`. Bucket keys remain based
on raw board/quadrant content. This is why changing modulus only changes
physical grouping, not board identity.

Relevant files:

```text
native_core/include/BCPositionCellLayout.h
native_core/include/BCFamilyPartitionPolicy.h
native_core/include/BCFamilyPartitionAnalysis.h
native_core/include/BCPositionCellLoader.h
```

## 5. Route Dispatcher

The unified benchmark dispatcher is in:

```text
native_core/tests_src/bench_bc_family_generation.cpp
```

The route option is:

```text
--family-route auto|resident|single|family
```

`--family-route family --family-modulus N` remains the reproducible fixed
FamilyChain path. `--family-route-script <csv>` can force route/modulus and
available memory per layer for tests.

The CSV output includes:

```text
route
target_modulus
available_memory_bytes
route_estimated_peak_bytes
route_budget_bytes
```

Do not add duplicate source modulus CSV fields. The source file already carries
its axis and physical cell table.

## 6. Route Planner

Planner code:

```text
native_core/include/BCFamilyRoutePlanner.h
```

Budget:

```text
reserve = max(1 GiB, total_physical_memory * 3%)
budget  = max(available_physical_memory - reserve, 0)
```

Layer size inputs:

```text
L = max(source2_size, source4_size)
D = abs(source2_size - source4_size)
if source4 is missing, source4_size = source2_size
```

Peak estimates:

```text
resident_est = L * 3.5 + D * 2.5
single_est   = L * 1.2 + D
family_est   = 0.25 GiB + 12 * L / modulus
```

Supported family moduli are hard-coded primes from `13` through `293`. If no
prime fits the budget, choose `293` and continue.

Modulus sticky rule:

```text
budget = 0.2 GiB + k * L / previous_modulus
keep previous_modulus when 9 <= k <= 16
otherwise choose the smallest supported prime satisfying family_est <= budget
```

Route hysteresis:

```text
Resident is fastest, then single, then family.
Downgrade to a lower-memory route is immediate.
Upgrade to a faster route requires 2 consecutive fitting layers.
```

Forced route bypasses auto selection, except that route-script fields may still
override the target modulus and available-memory input.

## 7. Resident Route

Resident route means the generation step is allowed to keep the relevant layers
or mutable states in memory while still using the shared modulo physical cell
layout. It does not use FamilyChain scheduling.

Current dispatcher behavior:

```text
target layer S:
    current source = layer S - 2
    carry source   = layer S - 4, if not already carried

    current source may be a resident memory layer from the previous step
    otherwise it is materialized/remapped to the requested modulo layout

    carry_to_primary contains prior +4 contributions for S
    current +2 contributions are inserted into carry_to_primary
    final S is finalized/written
    current +4 contributions are generated as next carry for S + 2
```

The final written layer is complete. The carry mutable state is not written as
a standalone `.bcpos`.

Primary code:

```text
native_core/include/BCResidentGeneration.h
native_core/src/BCResidentGeneration.cpp
native_core/src/BCResidentGenerationInternal.h
native_core/src/BCResidentGenerationStreaming.cpp
```

Important entry points:

```text
generate_resident_mutable_carry_layer_to_file(...)
generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(...)
build_streaming_carry_to_primary(...)        // bench helper
```

## 8. Single Route

Single route is strict `1 + x` generation:

```text
1 = target/future mutable layer
x = current source cell chunk
```

It must not keep both `+2` and `+4` target mutable layers at the same time.
The route still uses modulo physical cells and the same `BCDynamicState`
target mutable backend as resident.

Current dispatcher behavior:

```text
target layer S:
    current source layer S - 2 is loaded in cell chunks
    carry_to_primary contains prior +4 contributions for S
    +2 current chunks are inserted into carry_to_primary
    S is finalized/written and primary mutable is released
    if S + 2 is still needed:
        rescan current chunks
        generate +4 contributions into next carry only
```

Only complete layers are written. A `+4`-only mutable carry is never finalized
as a production layer.

Primary code:

```text
native_core/include/BCSingleChunkGeneration.h
native_core/src/BCSingleChunkGeneration.cpp
native_core/src/BCResidentGeneration.cpp
native_core/src/BCResidentGenerationStreaming.cpp
```

Important entry point:

```text
generate_single_chunk_strict_layer_to_file(...)
```

## 9. Family Route

Family route is the two-blocked FamilyChain generation path. It is the memory
fallback and uses modulo family partitioning.

Current key properties:

```text
target family/cell builders are active only for the current family window
source families/cells are loaded by pass
physical family fanout may be up to 3
generation is allowed to keep up to 4 family views in the generation phase
```

Family route writes the same `.bcpos` format as resident and single. It does
not use resident/single carry states.

Primary code:

```text
native_core/include/BCFamilyGeneration.h
native_core/src/BCFamilyGeneration.cpp
native_core/include/BCCellMutableBuilder.h
native_core/include/BCFamilyGenerationScheduler.h
native_core/include/BCFamilyPartitionPolicy.h
native_core/include/BCFamilyPartitionAnalysis.h
```

Important benchmark path:

```text
generate_layer_to_file_for_route(... route = Family ...)
```

## 10. Remap / Modulus Changes

Changing physical modulus does not change board identity, but an old physical
cell can contain several new physical cells. Therefore remap cannot just reuse
old cell ids.

Current rule:

```text
read old physical cells as an over-approximation
scan bucket keys, not boards
compute the target logical physical cell from raw key sums
copy only matching bucket/rank payload into the output loaded cell
release non-target data before compute continues
```

This is used by generation route switching, modulus switching, checkpoint
resume, and later solve-side window loading.

## 11. Shared Hot Generation Backend

Resident and single use `BCDynamicState`:

```text
cell_array[slot]          atomic<uint32_t>, empty / pending / cid
key_array[slot]           uint64_t bucket key
bitmap_offset_array[slot] uint32_t arena offset
bitmap_arena[word]        atomic<uint64_t>
```

Insertion key:

```text
(physical cid, bucket key) -> bitmap bit for rank
```

The hash table is a manually managed flat open-addressing table with atomic
cell publication. It does not grow in place and does not delete. Capacity
failure marks overflow and the outer layer retries with larger reserves.

Recent hot-path rules:

```text
candidate stats that require per-candidate increments are disabled by default
hash grouping scans use relaxed loads after generation has ended
prepare initializes only cell_array empty sentinels
key_array and bitmap_offset_array are left uninitialized for empty slots
finalize groups hash slots by cell before cell-local sort/measure/write
parallel grouping is enabled only when cell_count >= 512
large per-cell key/value sorts use the native x86 SIMD sort adapter
```

## 12. Direct IO

Generation has direct-aware position IO. Current production defaults:

```text
Family route source/dump/reload/write paths use direct-capable IO where selected.
Resident/single file output can use direct writer through benchmark options.
No-QD paths still use direct sequential writes when direct output is requested.
```

The benchmark option names are currently:

```text
--target-output-io direct            // resident compute bench
--family-position-io direct-rank-first
--family-source-io direct|direct-auto
--family-blob direct
```

For large free9 runs, direct writer backend throughput has reached roughly
4.5 GB/s on the local SSD.

## 13. Current Validation Snapshot

Recent local runs after checkpoint:

```text
resident/non-block free9-256, m17:
    layers = 162
    total_seconds = 98.8559016
    total_throughput = 166.563837 M rows/s
    backend write = 4.527 GB/s

forced family free9-256, m17:
    layers = 162
    route = family for every layer
    total_seconds = 194.702517
    total_throughput = 84.5691089 M rows/s
    process peak working set = 191,348,736 bytes
    active_family_window_peak = 3
```

The forced family run confirms the FamilyChain path did not regress relative
to the previous 80-82 M rows/s range.

## 14. File Map For Generation

Core public interfaces:

```text
native_core/include/BCResidentGeneration.h
native_core/include/BCSingleChunkGeneration.h
native_core/include/BCFamilyGeneration.h
native_core/include/BCFamilyRoutePlanner.h
native_core/include/BCPositionCellLayout.h
```

Core implementations:

```text
native_core/src/BCResidentGeneration.cpp
native_core/src/BCResidentGenerationStreaming.cpp
native_core/src/BCSingleChunkGeneration.cpp
native_core/src/BCFamilyGeneration.cpp
native_core/src/BCPositionFile.cpp
```

Shared support:

```text
native_core/include/BCLut.h
native_core/include/BCKeyRank.h
native_core/include/BCCellBuilder.h
native_core/include/BCCellMutableBuilder.h
native_core/include/BCPositionFile.h
native_core/include/BCPositionCellLoader.h
native_core/include/BCPositionScanner.h
native_core/include/BCLoadedCellScanner.h
native_core/include/BCFileIO.h
native_core/include/BCDirectFileIO.h
native_core/include/BCSortUtils.h
```

Benchmarks/tests:

```text
native_core/tests_src/bench_bc_generation_compute.cpp
native_core/tests_src/bench_bc_family_generation.cpp
native_core/tests_src/bench_bc_resident_generation.cpp
native_core/tests_src/bench_bc_single_chunk_generation.cpp
native_core/tests_src/test_bc_resident_generation.cpp
native_core/tests_src/test_bc_family_generation_state.cpp
native_core/tests_src/test_bc_family_route_planner.cpp
native_core/tests_src/test_bc_family_partition_analysis.cpp
native_core/tests_src/test_bc_position_cell_loader.cpp
```
