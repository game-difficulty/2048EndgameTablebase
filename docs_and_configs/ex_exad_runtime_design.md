# EX / EXAD Runtime Design Context

Last updated: 2026-06-06

This document is the current maintenance map for the EX and EXAD production
paths. It is meant to be a practical reference for future debugging and
optimization work. If this document conflicts with source code, trust the
source code.

Older files such as `ex_algorithm_context.md` and
`exad_prefix36_suffix28_generation_plan.md` remain useful historical context,
but they do not cover all current solve, compression, reader, chunked solve, and
resume behavior.

## 1. Scope and Naming

### User-facing algorithm choices

- `Classic`: legacy uint64 board arrays and success arrays.
- `AD`: advanced/masked algorithm for large 32768-style tablebases.
- `EX`: compact prefix36/suffix28 bitmap tablebase, classic semantics.
- `EXAD`: AD semantics plus EX-style compact prefix36/suffix28 storage.

### Historical names still present in code

The EX feature was originally named `zmask`. Several public bindings and CSV
files still use that name:

- `formation_core.run_pattern_build_zmask(...)`
- `formation_core.run_pattern_solve_zmask(...)`
- `formation_core.run_pattern_solve_zmask_single_layer(...)`
- `zmask_generate_stats.csv`
- `zmask_solve_stats.csv`

These public EX entrypoints currently route to `EXPrefix36Runtime`. They should
not be confused with older prefix40/zmask experiments.

### Main source map

Python routing:

- `engine_core/BookBuilder.py`
  - EX: `zmask_algo == true`, `advanced_algo == false`
  - EXAD: `zmask_algo == true`, `advanced_algo == true`
  - AD: `zmask_algo == false`, `advanced_algo == true`
- `engine_core/EXPhysicalPattern.py`
- `engine_core/BookReaderEX.py`
- `engine_core/BookReaderEXAD.py`

EX native:

- `native_core/src/BookGeneratorEX.cpp`
- `native_core/src/BookSolverEX.cpp`
- `native_core/src/EXPrefix36Runtime.cpp`
- `native_core/src/EXPrefix36Core.inl`
- `native_core/include/EXPrefix36Runtime.h`
- `native_core/include/EXCompressedResult.h`
- `native_core/src/EXCompressedResult.cpp`

EXAD native:

- `native_core/src/BookGeneratorEXAD.cpp`
- `native_core/src/BookSolverEXAD.cpp`
- `native_core/src/EXADBuilder.cpp`
- `native_core/include/EXADLayer.h`
- `native_core/include/EXADIO.h`
- `native_core/include/EXADSolvedLayer.h`
- `native_core/include/EXADCompressedResult.h`
- `native_core/src/EXADCompressedResult.cpp`

Shared runtime controls:

- `native_core/include/FormationRuntime.h`
- `native_core/src/ReaderRuntime.cpp`
- `native_core/src/FileIOUtils.cpp`
- `native_core/src/CompressionBridge.cpp`

## 2. Shared Concepts

### Physical pattern transform

EX and EXAD do not necessarily compute the logical pattern exactly as stored in
`patterns_config.json`. Before build/reader setup, Python calls
`resolve_ex_physical_pattern(...)`.

The resolver chooses an equivalent D4 board transform that makes the low 7
nibbles friendlier for EX suffix bitmaps. It transforms:

- `pattern_masks`
- `success_shifts`
- EXAD `fixed_32k_shifts`
- initial seed boards
- canonical mode for the supported min34 variant path

It also stores:

- `physical_transform`
- `inverse_physical_transform`
- `logical_pattern_signature`
- `physical_pattern_signature`

Build outputs, LUTs, solved files, and compressed files store the physical
metadata. Readers accept logical UI boards, transform lookup candidates into the
physical space, and inverse-transform sampled random states back to logical
space.

Classic and AD do not use this resolver.

Variant patterns:

- Python rejects AD/EXAD for variants.
- EX variants are allowed.
- For min34 variant patterns, the resolver may choose the `min34top` physical
  canonical mode for the vertical flip case.

### Board split

EX and EXAD both use the same board split:

```text
prefix36 = board >> 28
suffix28 = board & ((1 << 28) - 1)
```

The low 7 cells are the suffix. This is why fixed/restricted cells in those
positions hurt bitmap density and why the physical transform exists.

### EX bucket vs EXAD bucket terminology

EX:

- A bucket is keyed by `(prefix36, remaining_sum)`.
- The suffix LUT rank is determined by `remaining_sum`.

EXAD:

- A slot is one of the 48 AD semantic buckets.
- Inside each slot, an EXAD bucket is keyed by `(prefix36, semantic_suffix_sum)`.
- A row is one live board in a slot after bitmap lookup.
- `row_width[slot]` is the AD derive/matrix width for that slot.
- `success_values` is a global dense vector addressed by:

```text
value_index = slot_value_base[slot] + local_row * row_width[slot] + column
```

### Success dtype and deletion thresholds

`RunOptions.success_rate_dtype` supports:

- `uint32`
- `uint64`
- `float32`
- `float64`
- user-facing `1-float*` modes are represented through zero/max semantics in
  the runtime rather than by converting to real success first.

Deletion threshold controls are in `RuntimeControls`:

- `deletion_threshold`: absolute ratio against dtype scale.
- `relative_deletion_threshold`: ratio against the current layer maximum.
- `deletion_threshold_signal_path`: optional dynamic signal file. Each solve
  step refreshes it. A failed read keeps the previous threshold state.

For relative thresholding, the implementation first computes the layer max and
then keeps values above the effective threshold. For `1-float*`, this must stay
in native zero/max semantics and must not be converted to a real success value
before comparison.

### Progress model

Base build progress is `2 * steps`:

- generation: first half
- solve/recalculate: second half

EX with `optimal_branch_only` adds a third phase:

- generation
- solve
- optimal branch pruning

EXAD does not support `optimal_branch_only`.

## 3. EX Data Model

### Files

For pattern prefix `pathname` and `step`:

- LUT: `pathname + "_.zlut"`
- generated layer: `pathname + step + ".exgen"`
- generated temp archive: `.exgen.7z`
- solved layer: `pathname + step + ".zbook"`
- compressed solved layer: `pathname + step + ".exzbook"`
- solve stats: `zmask_solve_stats.csv`
- generation stats: `zmask_generate_stats.csv`
- optimal marker: `ex_optlayer`
- optimal completion marker: `ex_optimal_complete`

### `.zlut`

The EX LUT is a suffix28 LUT built from `ZMaskFrozen::TileLimitConfig`.

It contains:

- tile-limit config
- suffix size/rank tables
- low24 deduplicated rank tables
- packed rank-pair table when applicable
- offset/unrank/high-base tables
- physical pattern metadata

EX follows classic table semantics for target-tile handling. EXAD has a stricter
target tile count rule because AD treats reaching the target as immediate
success.

### `.exgen` and `.zbook`

Both are prefix36 layer files. The difference is lifecycle, not the conceptual
layout:

- `.exgen` is generated forward input.
- `.zbook` is solved output with success values populated and zero-compacted.

The persisted sections are conceptually:

- `bucket_keys[]`
- `bitmap_offsets[]`
- `success_offsets[]`
- `large_rank_offsets[]`
- `small_bitmap_bytes[]`
- `large_bitmap_words[]`
- `large_rank_bases[]`
- `success_values[]`

The in-memory direct bucket entry is kept compact. It currently packs lookup
fields into a 16-byte entry.

Large bitmap rank bases are runtime/format support for O(1)-ish ordinal
calculation. They are present in the raw `.zbook` format. Compressed EX stores
its own compressed block directories instead.

## 4. EX Generation

Public path:

```text
BookBuilder.py
  -> formation_core.run_pattern_build_zmask(...)
  -> BookGeneratorEX.cpp
  -> EXPrefix36Runtime::run_pattern_build(...)
```

Runtime flow:

1. Load or build `.zlut`.
2. Ensure generated layers exist through the requested target step.
3. Start from the highest local generation anchor:
   - step 0 can start from the seed layer.
   - step `c > 0` needs readable layers `c - 1` and `c`.
   - the carry for the next step is rebuilt from layer `c`.
4. Generate successors using the prefix36 dynamic builder.
5. Write generated layers as `.exgen` or `.exgen.7z`.
6. Do not overwrite a step that already has solved output.

Generation uses EX-specific bitmap insertion and finalization. It does not
materialize a classic `uint64[]` layer as the production representation.

### EX generation resume

Current generation resume is local-anchor based. It no longer requires all
generated files to exist.

Important helpers:

- `layer_input_exists(...)`
- `generated_layer_input_exists(...)`
- `solved_layer_exists(...)`
- `readable_layer_input_exists(...)`
- `find_generation_resume_step(...)`
- `ensure_ex_generated_through(...)`

`readable_layer_input_exists` accepts solved output or generated input. A
compressed solved `.exzbook` can be materialized back to `.zbook` when solving
needs it as a future layer.

## 5. EX Solve / Recalculate

Public path:

```text
BookBuilder.py
  -> formation_core.run_pattern_solve_zmask(...)
  -> BookSolverEX.cpp
  -> EXPrefix36Runtime::run_pattern_solve(...)
```

Solve scans from high step to low step:

```text
for step = steps - 3 down to 0:
    skip if solved step already exists
    ensure current, future1, future2 inputs exist
    promote generated futures to .zbook if needed
    load current/future1/future2
    recalculate current from future1/future2
    compact current
    write .zbook
    optionally compress no-longer-needed future
```

EX solve keeps two full future layers in memory, plus the current layer.

Core optimizations:

- direct-entry lookup index
- large bitmap rank bases
- batched lookup/reduce
- success prefetch
- per-thread workspaces
- zero compact using prefix36 bitmap layout
- compressed futures can be read by first materializing `.exzbook` to `.zbook`
  when the solve hot path needs raw layer access.

### EX solve resume

The solve phase is local-dependency driven:

- If a step is already solved (`.zbook` or `.exzbook`), it is skipped.
- If a current/future input is missing, `ensure_ex_generated_through(...)`
  attempts to fill only the needed gap.
- If compressed-only output is needed as a future, it is decompressed to raw
  `.zbook` for the solve hot path.

EX still has an all-compressed-complete fast return. With
`optimal_branch_only=true`, all compressed layers are not considered sufficient
unless `ex_optimal_complete` also exists.

## 6. EX Optimal-Branch Pruning

EX supports `optimal_branch_only`. AD/EXAD do not.

After normal solve, EX runs a third forward pruning pass starting at
`kOptimalBranchOnlyStartStep`:

1. For target layer `i`, read target `i` and source layers `i - 1` / `i - 2`.
2. Enumerate legal spawn/move successors from the source layers.
3. Canonicalize with the physical pattern's canonical mode.
4. Lookup target success through the EX direct lookup path.
5. Mark only strict-best successor dense indices in a keep bitset.
6. Compact target by keep bits.
7. Write/compress pruned target.
8. Update `ex_optlayer`.

When the pass finishes, `ex_optimal_complete` is written. If compression is
enabled, compression is delayed/ordered so files needed by the pruning window
remain readable.

## 7. EX Compression and Cold Reader

### `.exzbook`

EX compressed results are independent of EXAD compressed results.

The compressed format follows this broad layout:

- uncompressed header/directories
- independent LZMA bucket blocks
- independent LZMA success/value blocks

Block compression runs in parallel up to the CPU worker count. The compressed
file stores physical pattern metadata and must match the `.zlut`.

### EX cold lookup

Reader path:

```text
BookReaderEX.py
  -> formation_core.EXBookReader
  -> ReaderRuntime.cpp
  -> find_ex_value(...)
```

Lookup order:

1. raw `.zbook` via `EXPrefix36Runtime::lookup_zbook_cold(...)`
2. compressed `.exzbook` via `EXCompressedResult::lookup_cold(...)`

The raw cold lookup is single-point oriented:

- open `.zbook` and `.zlut`
- check physical metadata
- point-read suffix LUT data
- binary search bucket keys
- read target bitmap data
- compute ordinal
- read one success value

The compressed cold lookup uses the compressed block directory and decompresses
only the bucket/value block needed by that query.

Random sampling also supports raw and compressed EX files. Sampled physical
boards are inverse-transformed before returning to the UI.

## 8. EXAD Data Model

### Files

For pattern prefix `pathname` and `step`:

- LUT: `pathname + "_.exadlut"`
- generated layer: `pathname + step + ".exadtmp"`
- generated temp archive: `.exadtmp.7z`
- solved layer: `pathname + step + ".exadbook"`
- compressed solved layer: `pathname + step + ".exadzbook"`
- generation stats: `exad_generate_stats.csv`
- solve stats: `exad_solve_stats.csv`
- chunk scratch directory: `step.exadbook.chunks.tmp`
- chunk write marker: `step.exadbook.writing`

### `.exadlut`

The EXAD LUT is suffix28-based but uses AD semantic suffix sums. It is built
from the EX tile-limit config and then disables the target tile count, because
AD records target creation as success rather than storing target-tile boards.

The LUT stores:

- tile-limit config and signature
- suffix semantic sum/rank tables
- offset/unrank/high-base tables
- physical pattern metadata

Cold readers can read only the small LUT pieces needed for one suffix query.

### `.exadtmp`

Generated EXAD layers are grouped by 48 AD slots:

```text
Layer {
    BoardSet sets[48]
}

BoardSet {
    buckets[]
    small_bitmap_bytes[]
    large_bitmap_words[]
}
```

Inside a slot, each bucket is keyed by:

```text
pack_bucket_key(prefix36, semantic_suffix_sum)
```

### `.exadbook`

Solved EXAD layers keep the same 48-slot structure and add AD matrix/value
metadata:

- `slot_row_base[48]`
- `slot_value_base[48]`
- `row_width[48]`
- `success_values[]`

Runtime-only indexes are rebuilt after read/compact:

- per-slot direct-entry index
- per-slot large bitmap rank bases

The direct entry is currently 16 bytes. It is intentionally not persisted in
compressed EXAD output.

## 9. EXAD Generation

Public path:

```text
BookBuilder.py
  -> formation_core.run_pattern_build_exad(...)
  -> BookGeneratorEXAD.cpp
  -> ensure_exad_temp_through_cpp(...)
  -> run_pattern_solve_exad_cpp(...)
```

Generation keeps AD semantics:

- mask-new-tile logic
- derive/unmask behavior
- AD bucket/slot classification
- validate steps
- target success behavior

But it stores generated states in EXAD prefix36/suffix28 bitmap buckets instead
of AD's legacy matrix files.

The main carry generator is:

```text
EXAD::generate_two_layers_carry(...)
```

It creates:

- `arr1`: current `step` output
- `arr2`: carry for `step + 1`

Important generated stats fields:

- `input_live`
- `arr1_live`
- `arr2_live`
- `derive_candidate_count`
- `derived_output_count`
- `prepare_*`
- `loop_seconds`
- `insert_seconds`
- `finalize_seconds`
- `validate_removed_*`
- `retry_count`

### EXAD reserve prediction

Current EXAD generation uses conservative reserve factors:

- bucket default around `2.5`
- small bitmap default around `4.0`
- large bitmap default around `4.0`

Additional guards:

- early tiny layers get larger floors
- post-validate floor uses the pre-validate footprint for the next 3 layers
- a derived-output burst guard multiplies reserve prediction when recent
  `derived_output_count` changes from zero to positive and the layer sum is in
  the configured target-modulo window

The goal is retry-free generation with less virtual memory waste than older
large default factors.

### EXAD generation resume

`ensure_exad_temp_through_cpp(...)` is local-anchor based:

- if target temp already exists, return
- ensure layer0 temp exists
- scan down for the highest temp anchor where `step` and `step - 1` exist
- read `step - 1` as current
- rebuild carry from `step`
- continue only through the requested target
- do not overwrite steps that already have solved output

Generated temp can be raw `.exadtmp` or archive `.exadtmp.7z`.

## 10. EXAD Solve / Recalculate

EXAD solve has two modes:

- normal solve
- chunked solve

Both use the same AD semantic helpers and EXAD lookup/storage:

- `FormationAD::MaskerContext`
- `MatchCache`
- AD permutation/match tables
- EXAD direct lookup into future layers
- EXAD compact/write/compress

### Normal EXAD solve

Normal solve scans from high step to low step:

```text
for step = steps - 3 down to 0:
    skip if .exadbook/.exadzbook already exists
    ensure current .exadtmp exists
    read full current generated layer
    convert to SolvedLayer
    read full future1
    read full future2
    build direct indexes
    recalculate current
    optionally threshold future2
    zero-compact current
    write .exadbook
    optionally compress
    remove current .exadtmp/.exadtmp.7z
```

Normal solve keeps current, future1, and future2 in memory at the same time.

### EXAD recalculate optimizations

Currently applied:

- per-slot direct-entry index
- large bitmap rank bases rebuilt after read/compact
- batched future lookup
- prefetch for direct-index/rank/success reads
- vector-width aware query batching with a generic path for all widths
- `MatchCache` reuse across the solve run
- per-thread workspaces
- zero compact over EXAD bitmap rows

Avoided or removed:

- derived-output prefilter hash table
- deferred derive layer
- large fixed row caches for very wide `row_width`
- narrow benchmark-only special cases that did not pay for their complexity

### Deletion threshold during EXAD solve

The deletion threshold is applied to `future2` when it is no longer needed by
lower layers. For relative mode, EXAD scans the future layer to find the maximum
native success value and combines the relative threshold with the absolute one.

Stats record:

- requested absolute threshold
- requested relative threshold
- effective normalized threshold
- before/after future value counts
- retained ratios

## 11. EXAD Chunked Solve

Chunked solve is selected when:

- `RunOptions.chunked_solve == true`, or
- `EXAD_FORCE_CHUNKED_SOLVE` / `EXAD_FORCE_STANDARD_CHUNKED_SOLVE` is set

Test-only controls:

- `EXAD_TEST_AVAILABLE_MEMORY_MIB`
- `EXAD_TEST_CURRENT_CHUNK_ROWS`
- `EXAD_TEST_CURRENT_CHUNK_BUCKETS`
- `EXAD_TEST_STOP_AFTER_CHUNKED_FIRST_PASS_STEP`
- `EXAD_TEST_STOP_AFTER_CHUNKED_FINAL_CHUNKS_STEP`
- `EXAD_TEST_STOP_AFTER_CHUNKED_SOLVED_WRITE_STEP`

### Current chunking model

EXAD chunked solve reduces current-layer memory. It does not keep both future
layers in memory at the same time.

For each unsolved step:

1. Ensure future solved outputs for `step + 1` and `step + 2` exist.
2. Ensure current temp layer exists.
3. Remove stale chunk scratch for this step.
4. First pass:
   - load full `future1` (`step + 1`) and build direct indexes
   - stream current temp by AD slot through `EXAD::LayerSlotReader`
   - split a large slot into bucket-range chunks using a row budget
   - recalculate only spawn-2 contribution
   - write partial slot chunks
   - release `future1`
5. Second pass:
   - load full `future2` (`step + 2`) and build direct indexes
   - read partial chunks
   - recalculate spawn-4 contribution
   - zero-compact each chunk
   - write final slot chunks
   - release `future2`
6. Merge 48 slot chunks into one `.exadbook`.
7. Delete chunk scratch and current temp layer.

This means:

- current layer can be split inside large AD slots
- current slot chunks are persisted as scratch files
- `future1` and `future2` are never simultaneously resident
- each future layer is still fully materialized while its pass is running
- compressed future `.exadzbook` is materialized to a full solved layer for
  solve-time lookup

This is close to the AD chunked pattern in spirit: split the current layer, keep
future access straightforward and indexed, and tolerate extra IO to reduce peak
memory.

### Chunk row budget

The default row budget is derived from available memory:

```text
value_slots = max(1 << 28, available_memory_bytes * 0.9 / sizeof(T))
row_budget = max(1, value_slots / max(row_width, 1))
```

This protects extremely wide rows without allocating `rows * row_width` buffers
for huge ranges.

### Chunked resume behavior

Chunk scratch is not a durable resume format. If a run stops inside a chunked
step, the next run removes that step's scratch directory and recomputes the
current step.

Stable resume inputs are:

- solved futures: `.exadbook` or `.exadzbook`
- current temp: `.exadtmp` or `.exadtmp.7z`

Chunked and non-chunked EXAD solve can switch modes between runs, as long as the
needed temp and solved outputs exist or can be regenerated.

## 12. EXAD Compression and Cold Reader

### `.exadzbook`

EXAD compressed output is separate from EX compressed output.

Layout:

- uncompressed header
- uncompressed `SlotDir[48]`
- uncompressed bucket block directory
- uncompressed value block directory
- independent LZMA bucket blocks
- independent LZMA value blocks

Bucket blocks are grouped by slot and bucket order. Value blocks are global
contiguous `success_values` ranges, so very wide rows do not require reading or
decompressing the whole row for one cold lookup.

Compression can be called:

- from raw `.exadbook`
- directly from an in-memory solved layer

Block compression runs in parallel up to the CPU worker count.

### EXAD cold lookup

Reader path:

```text
BookReaderEXAD.py
  -> formation_core.EXADBookReader
  -> ReaderRuntime.cpp
  -> find_exad_value(...)
```

Lookup order:

1. raw `.exadbook` via `EXADCompressedResult::lookup_exadbook_cold(...)`
2. compressed `.exadzbook` via `EXADCompressedResult::lookup_exad_cold(...)`

Reader work:

1. Build the EXAD AD lookup target:
   - mask/derive/unmask semantics
   - target AD slot
   - target column
   - canonical physical board
2. Query `.exadlut` by point reader.
3. Construct `(prefix36, semantic_suffix_sum)` key.
4. Locate bucket in the slot.
5. Test suffix bitmap and compute local row ordinal.
6. Compute `value_index = slot_value_base + local_row * row_width + column`.
7. Read one raw value or one compressed value block.

Random sampling supports raw and compressed EXAD files and returns logical boards
after inverse physical transform.

## 13. Temporary File Compression

`compress_temp_files` is separate from final result compression.

EX:

- writes `.exgen.7z` instead of `.exgen`
- reads archives through a sequential archive reader
- promotes `.exgen.7z` to `.zbook` when solve needs the layer as solved input

EXAD:

- writes `.exadtmp.7z` instead of `.exadtmp`
- full read path can materialize it as a layer
- `LayerSlotReader` can stream it slot-by-slot for chunked solve

These archives are intermediate-layer storage only. Final compressed results
are controlled by `compress`, not by `compress_temp_files`.

## 14. Final Result Compression

`compress=true` controls `.exzbook` / `.exadzbook`.

EX:

- while solving, no-longer-needed raw `.zbook` files may be compressed and
  deleted
- after normal solve, `compress_all_layer_results(...)` finishes remaining raw
  outputs
- with `optimal_branch_only`, final compression is coordinated with the optimal
  pruning window and `ex_optimal_complete`

EXAD:

- raw `.exadbook` files can be compressed from memory immediately after write
  paths
- at the end, `compress_all_exad_solved_files(options, true)` compresses any
  remaining raw solved files and removes sources
- compressed-only futures are accepted by both normal and chunked solve, but
  solve-time lookup currently materializes them to full solved layers

## 15. Resume Matrix

### EX local dependencies

For solving step `s`, EX needs readable inputs for:

- `s`
- `s + 1`
- `s + 2`

Each input can be:

- `.zbook`
- `.exzbook`
- `.exgen`
- `.exgen.7z`

If a needed input is missing, EX calls `ensure_ex_generated_through(...)` for
only the needed target. It should not restart full generation just because the
complete generated prefix is absent.

### EXAD local dependencies

For solving step `s`, EXAD needs:

- current temp `s`: `.exadtmp` or `.exadtmp.7z`
- future `s + 1`: `.exadbook` or `.exadzbook`, unless terminal
- future `s + 2`: `.exadbook` or `.exadzbook`, unless terminal

If current temp is missing, EXAD calls `ensure_exad_temp_through_cpp(...)` for
that step. If a future solved layer is missing, the step cannot be solved until
higher steps are solved.

### Completion checks

Completed solved layers are skipped individually.

All-compressed fast return is allowed only when all required compressed final
outputs are present. For EX optimal pruning, `ex_optimal_complete` is also
required.

### File validation caveat

The product-level intent is local existence based resume. Some low-level readers
still validate magic/version/size/physical metadata when actually opening a
file. An empty placeholder may pass a high-level existence check but still fail
when the runtime tries to read it.

## 16. IO Model

Raw EX/EXAD binary read/write paths use the sequential IO helpers in
`FileIOUtils`.

Direct IO is controlled by:

- `RunOptions.direct_io`
- `direct_io_queue_depth`
- `direct_io_chunk_mib`

The Windows path probes direct IO capability on the target volume/path and can
fall back to buffered sequential IO when direct IO is not usable. Archive paths
use 7z streaming readers/writers instead of direct IO.

## 17. Stats Files

EX generation:

- `zmask_generate_stats.csv`
- includes per-layer generation/finalize/write data
- `_total` row should use weighted bitmap density where applicable

EX solve:

- `zmask_solve_stats.csv`
- includes solve, threshold, compact, read/write/compress, and optimal rows

EXAD generation:

- `exad_generate_stats.csv`
- includes detailed derive/generation/validate/reserve stats

EXAD solve:

- `exad_solve_stats.csv`
- `stage` is `solve`, `chunked_solve`, or `_total`
- includes:
  - current read/build
  - future read/index
  - recalculate
  - zero compact
  - future compact/write
  - compression
  - max success
  - deletion threshold fields
  - metadata/success bytes
  - bitmap density

CSV timestamps are expected to include sub-second precision.

## 18. Reader Integration

`ReaderRuntime.cpp` owns unified native reader behavior.

EX:

- Python creates `PatternSpec` with physical metadata.
- Native reader transforms logical move candidates into physical space.
- Raw `.zbook` is tried before `.exzbook`.

EXAD:

- Python creates `AdvancedPatternSpec` with physical metadata and AD fixed
  32768 shifts.
- Native reader applies AD mask/derive/column semantics before EXAD lookup.
- Raw `.exadbook` is tried before `.exadzbook`.

Trainer/tester/AI callers should use these unified readers, not custom file
parsing. Random initial state sampling also goes through the same raw/compressed
candidate search and inverse physical transform.

## 19. Current Caveats and Audit Points

- EX public names still say `zmask`.
- EX has no chunked solve mode. `BookBuilder.py` forces `chunked_solve=false`
  for EX.
- EXAD chunked solve chunks the current layer, including large slot interiors,
  but each future layer is still materialized whole while its pass is active.
- EXAD chunk scratch files are disposable. They are not intended as durable
  resume artifacts.
- EX raw `.zbook` cold lookup is single-point and avoids loading whole LUTs, but
  large raw bitmap ordinal calculation can still read part of one bitmap. The
  compressed EX reader uses block directories and can be more IO-shaped for cold
  lookups.
- EXAD solve currently uses `make_exad_solve_plan(...)` as a simple top-down
  scan. The local dependency behavior lives in the per-step checks and temp
  generation helper.
- Physical metadata mismatch means old EX/EXAD files should be rejected and
  recomputed. Current EX/EXAD formats are not intended to be backward compatible
  with earlier experimental files.

## 20. Quick Debug Checklist

When a build does not resume as expected:

1. Identify algorithm from config:
   - EX: `zmask_algo=true`, `advanced_algo=false`
   - EXAD: both true
2. Check physical config in `config.txt`.
3. Check LUT exists and matches physical metadata:
   - EX: `_.zlut`
   - EXAD: `_.exadlut`
4. Check current step dependencies:
   - EX: current/future can be solved, compressed, generated, or archive.
   - EXAD: current must be temp/archive; futures must be solved/compressed.
5. Check whether compressed-only futures are being materialized.
6. Check stale chunk scratch:
   - EXAD removes `*.exadbook.chunks.tmp` and `*.exadbook.writing` for solved or
     restarted chunked steps.
7. Check stats row stage:
   - EXAD chunked rows must show `chunked_solve`.
8. Check dynamic deletion threshold signal file if compact ratios look wrong.

