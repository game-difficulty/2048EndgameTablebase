# EX Algorithm Context

This document is a handoff note for rebuilding context in a new model session. It records the current EX algorithm intent, file layout, runtime paths, performance decisions, and known caveats. The implementation is still experimental, so trust the current source tree over older notes if there is a conflict.

## One-Sentence Summary

EX is the low-memory tablebase algorithm that replaces classic `uint64[] boards + success[]` layer files with bitmap bucket layers, supports streaming enumeration and fast lookup, and keeps classic/AD paths separate.

## Naming History

- The feature was originally called `zmask`; many public bindings, CSV names, and helper functions still use `zmask`.
- The user-facing algorithm name should be `EX`.
- The current public native entrypoints are still:
  - `formation_core.run_pattern_build_zmask(...)`
  - `formation_core.run_pattern_solve_zmask(...)`
  - `formation_core.run_pattern_solve_zmask_single_layer(...)`
- In the current source tree these public EX entrypoints route to `EXPrefix36Runtime`, not to the older prefix40/zmask frozen implementation:
  - `native_core/src/BookGeneratorEX.cpp` calls `EXPrefix36Runtime::run_pattern_build(...)`.
  - `native_core/src/BookSolverEX.cpp` calls `EXPrefix36Runtime::run_pattern_solve(...)`.
- Older modules are still present and useful for reference or probes:
  - `ZMaskFrozen` in `EXFrozenLayer.*` / `EXFrozenIO.*`.
  - `Prefix40Baseline` in `EXPrefix40*`.
  - `EXPrefixOnly` in `EXPrefixOnly*`.

## Goals

- Avoid keeping full per-layer `boards[]` in memory.
- Avoid classic raw layer files for EX generation.
- Preserve sequential enumeration of all live states.
- Preserve fast high-frequency lookup.
- Preserve a stable dense index equivalent to classic `boards[]` position.
- Keep final `.zbook` structurally compact and suitable for later compression.
- Keep classic and AD behavior untouched.

## Main Source Map

- `native_core/src/BookGeneratorEX.cpp`
  Public EX build binding wrapper. Currently delegates to `EXPrefix36Runtime`.

- `native_core/src/BookSolverEX.cpp`
  Public EX solve binding wrapper. Contains older prefix40/prefix-only solve code but public wrapper currently delegates to `EXPrefix36Runtime`.

- `native_core/src/EXPrefix36Runtime.cpp`
  Current production EX runtime wrapper for `prefix36_suffix28`. Handles file IO, stats, build, solve, cold lookup, and compression.

- `native_core/src/EXPrefix36Core.inl`
  Large benchmark-derived core copied into production. Contains prefix36/suffix28 LUTs, layer structure, direct index, generation, lookup, compaction, and recalc kernels.

- `native_core/include/EXPrefix36Runtime.h`
  Current public native declarations for prefix36 runtime.

- `native_core/include/EXCompressedResult.h`, `native_core/src/EXCompressedResult.cpp`
  EX result compression and cold lookup support.

- `native_core/src/EXBuildPipeline.cpp`
  Older prefix40 pipeline. Still useful background for generation statistics and reserve-factor history.

- `native_core/include/EXFrozenLayer.h`, `native_core/src/EXFrozenLayer.cpp`
  Older zmask frozen layer format. Still used by LUT config building and older code paths.

- `native_core/include/EXFrozenIO.h`, `native_core/src/EXFrozenIO.cpp`
  Older zmask frozen IO and keep-bitset compact helper.

- `engine_core/BookBuilder.py`
  Python side chooses EX when `config["zmask_algo"] == True`.

- `frontend/src/features/settings/pages/SettingsPage.vue`
  Settings UI has an EX trigger via the historical `zmask_algo` config.

## Current Production Layout: `prefix36_suffix28`

The current EX runtime layout is named `prefix36_suffix28`.

### Key Split

- Board is split as:
  - `prefix36 = board >> 28`
  - `suffix28 = board & ((1 << 28) - 1)`
- Bucket key is:
  - `key64 = (prefix36 << 18) | remaining_sum18`
- `remaining_sum18` is the suffix tile sum required for the layer.

### Layer File

- Extension: `.zbook`
- Magic: `EXP36BK`
- Current layer version in source: `kLayerVersion = 2`
- Header includes:
  - `success_kind`
  - `dtype_mode`
  - `layer_sum`
  - `threshold_bits`
  - `large_rank_stride_words`
  - bucket counts and section sizes
  - `value_size`

### Persisted Sections

The prefix36 layer is still a compact bitmap + dense success format. Conceptually it contains:

- `bucket_keys[]`
- `bitmap_offsets[]`
- `success_offsets[]`
- `large_rank_offsets[]`
- `small_bitmap_bytes[]`
- `large_bitmap_words[]`
- `large_rank_bases[]`
- `success_values[]`

In memory, `BucketEntry` is an AoS entry used for faster lookup:

```cpp
struct BucketEntry {
    uint32_t prefix_low32;
    uint32_t bitmap_offset;
    uint32_t success_offset;
    uint32_t rank_offset_prefix_high;
};
```

Important properties:

- `BucketEntry` must stay 16 bytes.
- `rank_offset_prefix_high` packs prefix high 4 bits plus rank offset.
- Small buckets use a sentinel rank offset.
- Large buckets use `large_rank_bases[]` to avoid scanning every preceding word.
- `success_values[]` is dense over live bits and gives the stable dense index.

### LUT File

- Extension: `.zlut`
- Magic: `EXP36LT`
- Current LUT version in source: `kLutVersion = 1`
- The LUT includes:
  - tile-limit config
  - valid suffix masks
  - rank tables
  - packed rank pair table when applicable
  - packed metadata
  - `size_table`
  - `offset_table`
  - `unrank_array`
  - `high_base`

The suffix28 LUT is optimized by splitting suffix high nibble and low24:

- low24 rank tables are deduplicated by tile-limit signature.
- if exactly two variants exist, the two uint16 rank tables can be packed into one uint32 table.
- `high_base[high][group] + low_rank` gives the suffix28 rank.

### Generated Intermediate

- Extension: `.exgen`
- Used by generation and solve as an intermediate layer input.
- `existing_layer_input_path()` prefers final `.zbook`; otherwise falls back to `.exgen`.
- During solve, after a layer is solved and written as `.zbook`, the corresponding `.exgen` is removed.

## Older ZMask Frozen Format

The earlier zmask design used `prefix40 + suffix24 + zero_mask`.

Conceptual persisted sections were:

- `bucket_key64[]`
- `bitmap_offset32[]`
- `success_offset32[]`
- `small_bitmap_bytes[]`
- `large_bitmap_words[]`
- `success_values[]`

Bucket key layout:

- bits `63..24`: `prefix40`
- bits `23..18`: `zero_mask6`
- bits `17..0`: `remaining_sum18`

This format is still useful for context because many names and helper functions still reference `ZMaskFrozen`. However, current public EX build/solve uses `EXPrefix36Runtime`.

## Generation Workflow

### Current Prefix36 Production Flow

1. Build or load `.zlut`.
2. Build step 0 as a prefix36 layer and write `.exgen`.
3. For each forward layer:
   - generate spawn 2 and spawn 4 successors
   - move in 4 directions
   - skip stationary moves
   - apply pattern/canonical logic
   - insert into bitmap bucket layer
   - write `.exgen` for non-terminal layers
   - terminal layers are compacted and written as `.zbook`
4. Generation stats are written to:
   - `pathname + "zmask_generate_stats.csv"`

The CSV name still says `zmask`.

### Earlier Prefix40 Builder Work

The prefix40 builder work was migrated from `t/distribute1/benchmark_classic_generate_prefix40_omp.cpp`.

Important benchmark-derived behaviors:

- hash load upper target around `0.60`
- pending insert buffer of 128
- prefetch distance around 16
- OpenMP dynamic large scheduling
- thread-local chunk allocators
- hash-slot and bitmap-target prefetch
- no `occupied_slots[]` in the current preferred production mode, due memory cost concerns

Current prefix36 code has its own benchmark-derived implementation in `EXPrefix36Core.inl`.

## Solve Workflow

### Current Prefix36 Production Flow

1. Ensure forward artifacts exist.
2. Read future layers through `read_layer_input()`:
   - final `.zbook` if present
   - otherwise `.exgen`
3. Build load-time direct indexes for future layers.
4. For step `i` from `steps - 3` down to 0:
   - read current layer
   - recalculate success by enumerating current bitmap live states
   - use future layer lookup for spawn 2 and spawn 4 outcomes
   - compact zero-success states
   - write current `.zbook`
   - remove current `.exgen`
   - optionally compact/write future layer `i + 2` if `deletion_threshold > 0`
   - optionally compress `.zbook` into `.exzbook` if `compress == true`
5. Solve stats are written to:
   - `pathname + "zmask_solve_stats.csv"`

### Rolling Window

The solve path keeps a rolling window of future layers to avoid unnecessary rereads:

- `future1 = layer i + 1`
- `future2 = layer i + 2`
- after solving `i`, current becomes next window item

## Lookup Design

### Direct Index

The hot lookup path does not binary-search bucket arrays. It builds a load-time direct index:

- direct index is not persisted
- bucket key lookup is based primarily on prefix
- table size is chosen from bucket count and configured load factor
- current prefix36 runtime has both:
  - `DirectIndex`: stores bucket indices
  - `DirectEntryIndex`: stores full `BucketEntry` in the hash table to reduce dependent random loads

The direct hash path exists because lookup was the core solve bottleneck. It substantially improves over AdaptiveIndex/exact search style lookup.

### Dense Ordinal

After locating a bucket:

1. Use suffix rank from LUT.
2. Test bitmap bit.
3. Compute dense ordinal.
4. `success_index = success_offset + dense_ordinal`.

For small buckets:

- local popcount over preceding bytes/words.

For large buckets:

- use `large_rank_bases[]` every `large_rank_stride_words`.
- scan only within the stride block.

The stable dense index is the success array index. It replaces classic `boards[]` position semantics.

## Pattern And LUT Constraints

The EX LUT is constrained by global tile count limits and pattern restrictions.

Current intended rules:

- Before LUT generation, for non-free patterns, scan seed boards and compute per-tile maximum counts for tiles `>= target` and `< 32768`.
- Merge seed-derived count limits with global tile-count limits.
- `-1` unlimited was normalized conceptually to max count `6` to avoid max ambiguity.
- Valid suffixes must satisfy at least one original valid pattern mask in the suffix bits.
- Do not symmetry-expand valid pattern masks for this check. The current canonical insertion logic already ensures canonical boards satisfy an original pattern.
- For variant patterns, 32768 walls are fixed. OR all seed boards into `A`; suffix must satisfy the required wall bits from `A`.

Relevant code:

- `ZMaskFrozen::make_lut_tile_limit_config(...)`
- `ZMaskFrozen::build_zmask_luts(...)`
- prefix36 LUT construction in `EXPrefix36Core.inl`

## Success DTypes

The runtime recognizes six dtype names:

- `uint32`
- `uint64`
- `float32`
- `float64`
- `1-float32`
- `1-float64`

Native storage kind is one of four types:

- `uint32_t`
- `uint64_t`
- `float`
- `double`

The `1-float*` modes use floating storage but invert the zero/max interpretation:

- normal float zero/max: `0..1`
- one-minus mode zero/max: `-1..0`

For prefix36 runtime, in-memory success is generally normalized to uint32 fixed-scale for compute, with dtype conversion handled at file IO boundaries.

## Compaction

### Zero Compact

After recalculation, EX removes zero-success states:

- scan bitmap and dense success
- keep only `success > 0` or `success > threshold`
- remove empty buckets
- rebuild metadata and bitmap/success pools

This is reported in solve stats as `freeze_zero_compact_seconds` or `compact_seconds`.

### Future Threshold Compact

After solving layer `n`, if `deletion_threshold > 0`, layer `n + 2` may be compacted because later solve steps no longer need it in full.

The required persistence order for robust resume is:

1. solve and zero-compact layer `n`
2. write layer `n`
3. compact low-success layer `n + 2` if threshold is active
4. write layer `n + 2`

### Keep-Bitset Compact For Optimal Branch

Older zmask code includes:

- `ZMaskFrozen::compact_layer_by_keep_bitset(...)`

It uses a keep bitset indexed by stable dense success index. It rewrites the same frozen format and removes empty buckets. This is the intended design for optimal branch pruning, but current public prefix36 solve path does not yet route to that old implementation.

## `optimal_branch_only`

Intended semantics:

- After full solve, run a forward pass starting at step 21.
- For target layer `i`, read already-pruned/solved source layers `i - 2` and `i - 1`.
- From `i - 2`, spawn tile 4.
- From `i - 1`, spawn tile 2.
- For each source board and each empty cell, enumerate four moves.
- Mark only the first strictly best-success successor for each spawn cell.
- Use strict `>` tie behavior, matching classic.
- Compact target layer by keep bitset.

Current source-tree caveat:

- The old zmask keep-bitset implementation exists in `BookSolverEX.cpp` and `EXFrozenIO.cpp`.
- The current public EX solve wrapper delegates to `EXPrefix36Runtime::run_pattern_solve(...)`.
- In the older prefix-only solve function inside `BookSolverEX.cpp`, `optimal_branch_only` currently throws:
  - `"EX prefix-only solve does not yet support optimal_branch_only"`
- A future implementation should port the keep-bitset pruning to prefix36 `.zbook` format, not revive full `boards[]`.

## Compression

EX result compression exists separately from raw `.zbook`.

- Compressed extension: `.exzbook`
- Source:
  - `EXCompressedResult`
  - `compress_zbook_to_ex_result(...)`
  - `compress_prefix36_layer_view_to_ex_result(...)`
- Prefix36 solve can compress from memory after writing `.zbook`.
- If all compressed layers already exist, solve can return early.
- `compress_all_layer_results(...)` can compress all layers after solve and remove `.zbook` files.

Important distinction:

- `.zbook` is the compute/read-mostly layer format.
- `.exzbook` is a compressed result format for final storage/cold lookup.

## Statistics

### Generate CSV

Current file:

- `pathname + "zmask_generate_stats.csv"`

Current prefix36 header begins with:

```text
stage,step,layout,dtype_mode,canonical_batch_backend,direct_index_type,input_live,primary_live,secondary_live,...
```

Important fields:

- `input_live`
- `primary_live`
- `secondary_live`
- `bucket_count`
- `work_seconds`
- `finalize_seconds`
- `write_seconds`
- `compute_seconds`
- `compute_throughput_mbps`
- `total_seconds`
- `_total` row

The intended compute throughput excludes file IO but includes compute-side generation/finalization/compaction work.

### Solve CSV

Current file:

- `pathname + "zmask_solve_stats.csv"`

Current prefix36 header begins with:

```text
stage,step,layout,dtype_mode,canonical_batch_backend,direct_index_type,input_live,output_live,...
```

Important fields:

- `recalculate_seconds`
- `freeze_zero_compact_seconds`
- `future_index_seconds`
- `future_compact_seconds`
- `current_write_seconds`
- `future_write_seconds`
- `read_seconds`
- `compute_seconds`
- `active_throughput_mbps`
- `compute_throughput_mbps`
- `total_throughput_mbps`
- `metadata_bytes`
- `bitmap_density`

In prefix36 solve:

- `compute_seconds` includes recalc, compaction, direct-index build, and future compaction.
- `total_seconds` also includes read/write IO.

### Optimal Branch CSV

Older zmask keep-bitset implementation writes:

- `pathname + "zmask_optimal_branch_stats.csv"`

Fields:

```text
step,source0_live,source1_live,target_live_before,target_live_after,removed_count,read_seconds,index_seconds,mark_seconds,compact_seconds,write_seconds,total_seconds,throughput_mbps,time
```

This CSV is not yet guaranteed for the current prefix36 public path unless optimal branch pruning is ported there.

## Performance Notes Already Learned

- Direct hash lookup was one of the largest solve-stage wins.
- Direct index space is significant; storing only bucket indices is smaller, but storing compact `BucketEntry` can reduce dependent random loads.
- Prefix36 uses AoS `BucketEntry` to reduce random jumps in lookup.
- Sparse bitmap enumeration and board decode are cheap compared with lookup and move/canonical work.
- Decoding bitmap buckets into small per-thread board batches is better than full board materialization.
- `zero_cell_mask16()` iterates only empty cells rather than scanning all 16 cells.
- Batched lookup and prepared queries are important for latency hiding.
- Avoid extra O(n) board arrays unless explicitly benchmarking.
- Avoid per-thread `occupied_slots[]` in production unless memory cost is justified; no-occupied finalization was preferred due memory footprint.
- `std::atomic<uint32_t>[]` does not automatically pad like a struct with atomics, but atomic arrays still carry memory and contention considerations.
- Full free9 runs are expensive; use single-layer probes or small copied windows for testing.

## Testing Patterns

Useful backend/probe scripts:

- `backend/ex_test_build.py`
- `backend/ex_test_generate_single_step.py`
- `backend/ex_test_solve_single_layer.py`
- `backend/ex_compare_zbook.py`

Useful existing data directories from prior experiments:

- `tmp_ex_full_LL`
- `tmp_ex_full_predict`
- `tmp_ex_recalc_prod_verify`
- `tmp_ex_single_layer_compare`
- `tmp_ex_opt_smoke_ll2g`

For smoke tests:

- Prefer copying a small window of existing `.zbook` / `.exgen` / `.zlut` files.
- Avoid accidentally running full `free9-256`.
- A previous small optimal-branch smoke used LL2 layers 0..23 and `steps=24` to exercise target steps 21..23.

## Build Notes

On this Windows setup, native builds normally use:

```powershell
cmake --build C:\Apps\2048endgameTablebase\src\native_core\build-formation --config Release --target formation_core -j
```

Historical note: the user observed plain CMake configuration can hang unless run with elevated privileges. Prefer using the existing configured build directory when only recompiling.

## Current Caveats And To-Do Items

- Public EX binding names still say `zmask`; renaming user-facing/internal names fully to EX is incomplete.
- Old zmask/prefix40 code and new prefix36 code coexist. New sessions must identify which path a test is actually using.
- `optimal_branch_only` needs to be revalidated against the current prefix36 public path. The old keep-bitset compact design is still the right approach, but it should operate on prefix36 `.zbook`.
- Compression of final `.zbook` to `.exzbook` exists; compressed intermediate-file policy may still need UX clarification.
- CSV file names still say `zmask_*`.
- The experimental format is intentionally not backward-compatible. If magic/version mismatch occurs, rebuild.
- Do not change classic or AD behavior when working on EX.

## Mental Model

EX has three representations:

- `Carry/build state`: mutable hash/bitmap builder used during generation.
- `Generated input layer`: `.exgen`, compact bitmap layer used as solve input before final solve.
- `Result layer`: `.zbook`, compact bitmap + dense success layer used for solve output, final lookup, and compression.

The algorithmic invariant is that live boards are always represented as bitmap bits, and success values are always dense over live bits. Any operation that deletes states must rebuild metadata and drop empty buckets, while preserving the same `.zbook` structural contract.
