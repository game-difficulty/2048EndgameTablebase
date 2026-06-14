# EXBC / EXADBC FamilyChain Solve 开发计划 v1

本文是 `exbc_exadbc_dense_family_cell_latest_plan_v7.md` 的 solve 阶段修订指南。

v7 中的 Dense Family-Cell Matrix、BC position file、key/rank 格式、Resident / SingleChunk / Family generation 设计仍然有效；但 FamilyChain solve 的对象边界、调度顺序、success dtype 处理、partial 语义需要按本文修正。

核心结论：

- solve 不是 generation 的反向版。
- solve 不创建 position entry，不插入 key/rank bitmap，不写 rank payload。
- solve 只读取 current position、future position、future success，并写 current success。
- generation-only 对象不能进入 solve 主链路。
- FamilyChain solve 的性能主顺序应为 target-family-major。
- solve success 数据必须支持 6 个 dtype mode / 4 个物理类型；hot path 通过外层 typed dispatch 选择物理类型，不能写死为 `uint32`。

## 1. 非目标与对象边界

以下对象是 generation-only，不应出现在 solve 主链路：

```text
BCCellMutableBuilder
BCFamilyMutableStore
BCGenerationBlobIO
BCFamilyPositionWriter
```

原因：

```text
generation:
    source position -> spawn/move/canonicalize/encode -> insert target position
    target position 尚不存在
    需要 mutable builder、dump/reload、boundary finalize、position writer

solve:
    current position 已存在
    future position 已存在
    future success 已存在
    只计算 current success values
    不创建 position
    不插入 key/rank
    不写 bucket/rank payload
```

solve 主链路的核心对象只有：

```text
Current position reader/scanner
Future position + success lookup/cache
Current success accumulator / partial store
Current success writer
```

## 2. 与现有代码的对齐点

现有代码中需要直接对齐的事实：

- `BCSuccessIO.h` 当前已有 `BCSF` header 和 `dtype` 字段。实现必须支持 dtype-aware payload size、typed read/write 和 streaming load；`uint32` 只能作为默认兼容路径。
- `EXADSolvedLayer.h` 中 `DTypeMode` 编号为：
  - `UInt32 = 1`
  - `UInt64 = 2`
  - `Float32 = 3`
  - `Float64 = 4`
  - `OneMinusFloat32 = 5`
  - `OneMinusFloat64 = 6`
- EX prefix36 runtime 也使用同样的 6 个 dtype mode。
- 6 个 dtype mode 映射到 4 个 storage kind：
  - `uint32_t`
  - `uint64_t`
  - `float`
  - `double`
- EX/EXAD 文件格式公开 6 个 dtype mode。BC solve 不能把 success hot path 写死为 `uint32`，否则后续 `uint64`、`float32`、`float64`、`1-float*` 都会返工。

BC solve 的正确做法是外层 dtype dispatch 一次：

```text
run_solve_typed<Mode, StorageT>(...)
    StorageT in {uint32_t, uint64_t, float, double}
    active success blocks use StorageT
    reducer/update hot path uses StorageT
    no dtype switch inside candidate loop
```

## 3. Success DType 设计

新增或整理 `BCSuccessTypes`，用于 success dtype、layout、typed physical block 和 raw file block view。

### 3.1 DTypeMode

BC success header 的 `dtype` 必须对齐 EX/EXAD：

```cpp
enum class BCSuccessDTypeMode : uint32_t {
    UInt32 = 1,
    UInt64 = 2,
    Float32 = 3,
    Float64 = 4,
    OneMinusFloat32 = 5,
    OneMinusFloat64 = 6,
};
```

不要发明新的编号。`kBCSuccessDTypeUint32 = 1` 可以保留为兼容别名。

### 3.2 Storage kind

`DTypeMode` 与 storage kind 分开：

```text
UInt32           -> uint32_t, value_size 4
UInt64           -> uint64_t, value_size 8
Float32          -> float raw bits, value_size 4
Float64          -> double raw bits, value_size 8
OneMinusFloat32  -> float raw bits, value_size 4
OneMinusFloat64  -> double raw bits, value_size 8
```

### 3.3 Typed success 语义

solve 内部不统一转换为 `uint32`。每个 solve layer 根据 dtype mode 做一次 typed dispatch：

```text
UInt32           -> StorageT = uint32_t
UInt64           -> StorageT = uint64_t
Float32          -> StorageT = float
Float64          -> StorageT = double
OneMinusFloat32  -> StorageT = float
OneMinusFloat64  -> StorageT = double
```

各 dtype 的 zero / full-scale 语义：

```text
UInt32:
    zero = 0
    full = 4000000000

UInt64:
    zero = 0
    full = 1600000000000000000

Float32:
    zero = 0.0f
    full = 1.0f

Float64:
    zero = 0.0
    full = 1.0

OneMinusFloat32:
    zero = -1.0f
    full = 0.0f

OneMinusFloat64:
    zero = -1.0
    full = 0.0
```

`1-float*` 的 raw value 是 `success - 1`。因为 expectation 的权重和为 1，直接对 raw value 做 average 与先还原 success 再减 1 等价；max reducer 也保持单调。

### 3.4 Hot path 限制

禁止在 candidate update 内做：

```text
switch(dtype)
virtual call
std::variant visit
float/double reducer
```

允许在外层做一次 dtype dispatch：

```text
read future success block:
    raw dtype -> typed StorageT block

solve kernel:
    StorageT only

write current success block:
    typed StorageT block -> raw dtype
```

## 4. Success IO 模块要求

`BCSuccessIO` 需要从 `uint32` 专用扩展为 dtype-aware。

### 4.1 Header 校验

校验项：

```text
magic
format_version
header_bytes
dtype in {1..6}
row_width > 0
family_count
descriptor_count
payload_offset
position metadata fingerprint
payload_bytes
reserved fields
```

### 4.2 Payload bytes

payload byte count 必须使用 dtype value size：

```text
payload_bytes = total_success_rows * row_width * dtype_value_size(dtype)
```

不能再固定乘 `sizeof(uint32_t)`。

### 4.3 Block view

需要两类 block view：

```text
raw block view:
    指向文件 dtype payload bytes
    用于 file roundtrip / streaming reader / writer

typed block view:
    contiguous StorageT values
    dtype dispatch 后用于 solve hot path
```

建议接口：

```text
BCSuccessLayout
    dtype_mode
    row_width
    row_stride_bytes
    total_rows
    value_size
    payload_bytes

BCSuccessTypedBlock<T>
    rows
    row_width
    vector<T> values

BCRawSuccessBlockView
    dtype_mode
    rows
    row_width
    bytes

decode_raw_success_block_to_typed<T>(...)
encode_typed_success_block_to_raw<T>(...)
```

### 4.4 Streaming writer

当前 BC 有 memory writer 和 streaming reader，但 Family / SingleChunk solve 需要 streaming writer：

```text
BCSuccessStreamingWriter
    open path
    write header placeholder or final header
    write typed block as raw dtype at cell offset
    mark empty cell
    finish / validate all cells written
```

如果实现时不想做 out-of-order random writes，可先要求 cell 按 `CellId` 升序写入；Family solve 的 finalize 顺序若不满足升序，则需要 random-access writer 或 staged cell-order flush。

## 5. Solve 基本语义

对每个 current board：

```text
if terminal success:
    output = max_scale
else:
    enumerate empty cells
    for each empty cell:
        spawn 2:
            move 4 directions
            unchanged move skipped
            canonicalize
            encode future cell/key/rank
            lookup future success
            best2[empty_cell] = max(best2[empty_cell], future_success)

        spawn 4:
            same
            best4[empty_cell] = max(best4[empty_cell], future_success)

    output = average_over_empty_cells(
        (1 - spawn_rate4) * best2[empty_cell]
        + spawn_rate4 * best4[empty_cell]
    )
```

这个语义必须保持，不得把 candidate lookup 的 max 与 spawn expectation 混在一起提前简化。

## 6. 三条 Solve 链路

### 6.1 ResidentSolve

适用：

```text
current position resident
future +2 position/success resident
future +4 position/success resident
output success resident or streaming writer
```

特点：

```text
无 partial
无 Family window
无 dump/reload
作为 correctness oracle
```

流程：

```text
load current position
load future +2 position/success
load future +4 position/success
scan all current cells
run fixed-success solve kernel
write current success file
```

### 6.2 SingleChunkSolve

适用：

```text
current position 不能全驻留
future position/success 可以 resident 或 cache
current 按 cell chunk streaming
```

特点：

```text
current chunk 内同时完成 +2/+4
chunk result block 驻留
处理完直接写 current success file
不需要 Family partial
```

流程：

```text
for current cell chunk:
    load current position cells
    allocate chunk typed result blocks
    process +2 and +4 futures
    write typed result as output dtype
    write success cells
    release chunk
```

### 6.3 FamilyChainSolve

适用：

```text
current position 不能整层驻留
future position/success 不能整层驻留
必须控制 family-level memory window
```

FamilyChain solve 不使用 generation mutable store。它使用：

```text
future family view window
current cell scanner
cell-local fixed-success partial
success writer
```

## 7. FamilyChainSolve 调度

FamilyChain solve 的主顺序是 target-family-major：

```text
for delta in {+4, +2}:
    for future target family G in coord order:
        load future V(G) once
        process all current source cells whose candidates can hit G
        update current cell-local partial
        release future V(G)
```

其中：

```text
V(G) = row G union col G
```

不要使用 source-family-major 作为 solve 主顺序：

```text
for current source family F:
    for G in T_delta(F):
        load future V(G)
```

该顺序会重复读取 future view，导致 solve IO 放大。

### 7.1 Candidate owner event

为了避免同一 candidate 在 row/col 两个 future family pass 中重复处理，需要定义 owner event。

建议规则：

```text
horizontal move:
    owner = future row family
    只在 pass G == encoded_future.row_family 时处理

vertical move:
    owner = future col family
    只在 pass G == encoded_future.col_family 时处理
```

这样每个 candidate 只属于一个 `(delta, direction-axis, future family)` event。

current diagonal cell 不特殊重复处理；它只是同时可能产生 horizontal owner event 和 vertical owner event。

### 7.2 Reverse source families

对给定 `delta` 和 future family coord `G`，需要找到可能映射到 `G` 的 current source family coord：

```text
G in T_delta(F, target_axis_total)
```

必须使用 `FamilyCoord` 做跨层比较，不能用 layer-local `FamilyId`。

`T_delta` 仍使用 v7 修正后的公式：

```text
T_delta(a, N) = unique({ a, min(a + d, N - a) })
```

其中 `a` 是 source family coord，`N` 是 target layer 的 normalized total coord。

## 8. 3-Family 内存窗口

FamilyChain solve 的窗口对象与 generation 不同：

```text
generation:
    source position
    target mutable builder

solve:
    future position/success V(G)
    current position source view
    current partial/result blocks
```

设计目标：

```text
任意时刻 family-level 活跃窗口 <= 3
    1 个 future target family G
    <= 2 个 reverse current source families
```

附加约束：

```text
不能创建 whole-layer hash
不能创建 whole-layer bitmap
不能 whole-layer success resident
允许 cell-local typed partial
允许 thread-local buffers
允许小型 pass plan / cid list / stats
```

## 9. Partial / Result Accumulator 语义

这是 FamilyChain solve 最容易出错的部分。

### 9.1 不要过早把 partial 定义成最终 row value

EX solve 对每个 board 的真实 reducer 是：

```text
best2[empty_cell] = max over directions/future hits
best4[empty_cell] = max over directions/future hits
final = average over empty cells of weighted best2/best4
```

如果 target-family-major 把同一个 current row 的 candidate 分散到多个 future family pass，那么中间 partial 可能必须保存 per-empty best，而不是只保存最终 row success。

因此 Family partial schema 需要先证明以下二者之一：

```text
方案 A:
    某 current cell/row 的所有 relevant candidate 能在一个 resident accumulator 生命周期内完成
    则 partial 可保存 rows * row_width 的 collapsed contribution

方案 B:
    candidate 会跨 future family pass 分散
    则 partial 必须保存 per-empty best arrays
    例如 rows * row_width * max_empty_slots 的 typed best block
```

在没有完成该证明前，禁止把 Family partial 固定为：

```text
success_rows * row_width * dtype_size
```

这对最终 success block 是正确的，但对跨 pass partial 未必正确。

### 9.2 +4 / +2 phase

推荐分两阶段：

```text
+4 phase:
    target-family-major
    accumulate best4 per current row/empty/lane
    phase end collapse to result_accum_4 if row/cell complete

+2 phase:
    target-family-major
    accumulate best2 per current row/empty/lane
    combine with result_accum_4
    write final current success block
```

`result_accum_4` 可以是 collapsed row contribution：

```text
spawn_rate4 * average_empty(best4)
```

但只有在 `best4` 对该 row 的所有 relevant future family pass 已完成后才能 collapse。

## 10. Edge Kernel

`BCSolveEdgeKernel` 负责从 current board 产生 future lookup 和 typed result update。

它不持有文件，不持有 position builder，不知道 Family generation blob。

输入：

```text
current board
current cell id
current local row
lane
spawn tile
direction mask
future axis
future lookup callback
result update callback
```

热路径必须复用 EX / BC generation 已有优化：

```text
bc_zero_cell_mask16 + ctz 枚举 empty
BoardMover / VBoardMover
CanonicalBatch::canonicalize_inplace
BC quadrant hot encode
physical_target_family_may_hit / family prefilter
PreparedQuery batch
future lookup batch
```

不要写单 candidate 的 slow canonicalize / slow encode 版本作为生产路径。

## 11. FutureFamilyWindow

FamilyChain solve 不需要传统 LRU/whole-layer cache。这里的对象应是 `BCFutureFamilyWindow`，负责 future family view 的加载、保留和释放。

它绑定：

```text
future position reader
future success reader
future axis
dtype mode
typed StorageT
```

提供：

```text
load_future_view(G) = V(G)
retain cells still needed by the next family pass
release cells outside the active window
lookup_typed(cid, key, rank, lane) -> StorageT
batch_lookup_typed(prepared_queries) -> StorageT values / max reducer
stats
```

加载 future success 时，应读入 active view 需要的 typed physical block。不要把所有 dtype 强制 decode 成 `uint32`；`uint32`、`uint64`、`float`、`double` 都应按外层 typed dispatch 进入 hot path。

当前 BC loaded cell lookup 可作为 correctness baseline；性能版应补 cell-local direct index 或 prepared batch lookup，避免 bucket binary search 成为瓶颈。

## 12. PartialStore

`BCPartialStore` 负责 current cell-local typed partial/result block。

它不是 generation blob，不保存 mutable key/bitmap builder，也不做 append-only dump。

基本要求：

```text
cell-local
typed StorageT hot block
memory-only mode for tests
file-backed mode for FamilyChain
prepare_cells / load_cell / update / release_except / finalize_cell
stats for loads, writes, bytes, seconds
```

实现前必须先明确 partial schema：

```text
collapsed row contribution:
    rows * row_width
    只适用于该 row 的 per-empty best 已经完整时

per-empty best block:
    rows * row_width * max_empty_slots
    适用于 target-family-major 将同一 row 的 candidates 分散到多个 future family pass 的情况
```

不要为了节省空间提前 collapse。只有当 owner event 和 pass lifecycle 能证明某 row 的相关 future passes 已全部完成，才能把 per-empty best 折叠为最终 row contribution。

FamilyChain 中 `release_except` 必须配合 active window invariant：

```text
离开窗口的 clean block 可直接释放
离开窗口的 dirty block 必须写 partial/result tmp
不得常驻全层 partial
不得构建全层 hash/bitmap 辅助结构
```

## 13. Success Writer

solve 输出是 current success file，不是 position file。

writer 要支持：

```text
write typed StorageT block as configured dtype
empty cell mark
cell offset validation
streaming output
finish validation
stats
```

如果 Family finalize 顺序不是 `CellId` 升序，writer 需要支持 random-access cell writes，或 solve executor 需要 staging 到可按 `CellId` 升序 flush 的小窗口。

## 14. BC Rank Checkpoint

BC position rank payload must keep a cell-local checkpoint table for each bucket bitmap.

Current format:

```text
bucket.rank_payload_offset:
    prefix256[ceil(bitmap_len / 256)] as little-endian uint16
aligned bitmap offset:
    bitmap words as little-endian uint64
```

The stored prefix value is cumulative popcount before the 256-bit segment, not only the segment-local count.
This is intentionally stronger than "count inside this segment": lookup of a rank or a word-range start becomes:

```text
block = bit_index / 256
rank_before = prefix256[block] + popcount(bitmap words within the same 256-bit block before bit_index)
```

So each query needs at most 4 bitmap-word popcounts after reading one uint16 checkpoint.
For FamilyChain solve word-range work items:

```text
bucket_seen_at_word_begin =
    rank_before_word_index(prefix256, bitmap_words, bitmap_len, word_begin)

local_success_row =
    bucket.success_row_offset + bucket_seen_at_word_begin
```

Then scanning the selected word range only increments `bucket_seen` for set bits inside that range.
This prevents every split work item from rescanning a large bucket from word 0.

Constraints:

```text
checkpoint scope is one bucket inside one cell
no whole-layer rank table
no whole-layer bitmap auxiliary structure
RankPrefix remains uint16 because kBCMaxBucketBitmapLen <= 46656
payload overhead is ceil(bitmap_len / 256) * 2 bytes per non-empty bucket
```

Implementation hooks:

```text
kBCRankPrefixBits = 256
rank_prefix_type = uint16
build_prefix256 / append_prefix256_le when finalizing cells
rank_before_bit_index / rank_before_word_index for native bitmap words
rank_before_bit_index_le_bytes / rank_before_word_index_le_bytes for loaded position payload
BCLoadedCellScanner::for_each_bucket_word_range uses this for range-local row starts
```

## 15. Work Items and Parallelism

并行粒度需要在设计阶段固定，避免后续 Family 大表下出现负载不均。

推荐 work item：

```text
current cell id
loaded current cell index
bucket index
bitmap word begin
bitmap word end
direction / owner event
delta
future target family G
```

构建要求：

```text
每个 bitmap word 在同一 owner event 下只扫描一次
word range 尽量均衡，避免大 bucket 造成长尾
work item 不持有全层状态
thread-local workspace 保存 board/canonical/encoded/query buffers
```

线程安全要求：

```text
同一个 current row 的 update 必须由单 writer 负责，或使用明确的 row-local reduction 合并
不要让多个线程无保护地写同一 partial row/lane/empty slot
flush_family_canonical_buffer 必须批量 canonicalize -> encode -> lookup/update
```

## 16. Runtime Invariants

FamilyChain solve 必须在 runtime 收集并断言窗口。

至少统计：

```text
active_family_window_current
active_family_window_future
active_family_window_partial
active_family_window_total
active_family_window_max
active_current_family_window_max
active_future_family_window_max
active_partial_family_window_max
```

硬约束：

```text
active_family_window_total <= 3
```

如实现需要临时超过 3 family，必须在文档和 options 中显式标记为 debug / fallback，不允许成为 production FamilyChain solve 默认路径。

## 17. Stats

需要统计到 thread-level、family-level、layer-level。

建议字段：

```text
stage
step
chain
dtype_mode
row_width
current_cells_loaded
future_cells_loaded
future_cell_cache_hits
future_cell_cache_misses
current_boards_scanned
empty_slots_scanned
move_candidates
unchanged_moves
prefilter_rejects
canonicalized_candidates
encoded_candidates
future_lookup_count
future_lookup_misses
partial_loads
partial_writes
partial_bytes_read
partial_bytes_written
success_bytes_written
read_seconds
compute_seconds
lookup_seconds
partial_io_seconds
write_seconds
total_seconds
active_family_window_max
active_future_family_window_max
active_current_family_window_max
active_partial_family_window_max
work_items
bitmap_words_scanned
partial_rows_updated
partial_release_count
partial_finalize_count
```

Free9-256 EX sample stats can be used as performance reference:

```text
C:\2048_tables\free9\free9_256_zmask_solve_stats.csv
```

该样本为 `uint32`、Direct IO、AVX512 canonical batch。BC stats 不必字段完全相同，但需要能解释 compute / read / write / lookup / partial IO 的成本。

## 18. 已对齐设计与仍需落地项

### 已对齐

```text
generation-only 对象不进入 solve 主链路
solve 核心对象限定为 current reader / future family window / accumulator-partial / success writer
DTypeMode 1..6 对齐 EX/EXAD
hot path 按 dtype mode 外层 dispatch 到 StorageT
candidate update hot loop 不做 dtype switch
FamilyChain solve 使用 target-family-major
edge kernel 复用 canonical batch、bitmask/ctz、hot quadrant encode、prefilter
```

### 仍需落地

```text
PartialStore schema proof and implementation
FutureFamilyWindow memory/streaming batch load and retain/release
cell-local direct or batch lookup path
cell + bucket + bitmap word range work item builder
runtime active-window assertions
success streaming writer or random-access writer
thread-level and family-level stats
```

## 19. 开发顺序

### Module 1: Success dtype / typed block / raw IO

目标：

```text
BCSuccessDTypeMode
dtype_value_size
BCSuccessLayout
BCSuccessTypedBlock<T>
raw block typed encode/decode
BCSuccessIO dtype-aware payload validation
memory writer/reader dtype-aware roundtrip
streaming reader dtype-aware load
```

非目标：

```text
不实现 solve edge kernel
不实现 FutureFamilyWindow
不实现 PartialStore
不实现 ResidentSolve
不实现 FamilyChainSolve
不改 position file 格式
```

测试：

```text
6 dtype mode roundtrip
uint32 legacy tests continue passing
payload_bytes uses dtype size
typed raw roundtrip matches EX/EXAD dtype mode layout
row/lane bounds
streaming reader at least uint32 + one 8-byte dtype
```

### Module 2: FutureFamilyWindow

目标：

```text
support memory and streaming readers
load future position cells
load future success cells
retain cells still needed by the next family pass
release cells outside active window
load success as typed StorageT block
lookup_typed
batch lookup API
efficient future success row location
cell-local direct/batch lookup extension point
stats
```

### Module 3: PartialStore

目标：

```text
cell-local typed partial block
memory-only and file-backed modes
in-place max/reducer over StorageT
release_except
finalize_cell
partial schema validation
no whole-layer resident partial
stats
```

### Module 4: SolveEdgeKernel

目标：

```text
current board -> spawn/move/canonicalize/encode
future lookup callback
per-empty best reducer
typed result update callback
batch canonicalize
direction-specific prefilter
cell/bucket/bitmap-word work item support
thread-local workspace flush
```

### Module 5: ResidentSolve

目标：

```text
correctness oracle
small table high-performance path
typed success accumulator
success writer output
memory future position/success
stats
```

### Module 6: SingleChunkSolve

目标：

```text
current cell chunk streaming
future family window reuse
direct chunk result write
no Family partial
success writer output
chunk stats
```

### Module 7: FamilyChainSolve skeleton

目标：

```text
target-family-major pass builder
per-family pass executor skeleton
candidate owner event
reverse source family mapping
V(G) cell set builder
3-family window validation
partial schema proof
current cell partial lifecycle
future family window integration points
boundary finalize logic skeleton
```

### Module 8: FamilyChainSolve executor

目标：

```text
+4 target-family-major pass
+2 target-family-major pass
future family window for V(G)
future family window retain/release
current source cell streaming
partial/result store
success writer finalize
runtime active-window assertions
stats
```

### Cross-cutting: Success streaming writer

`BCSuccessStreamingWriter` 是 solve 输出能力，不应被忘掉。可以在 Module 1 之后单独实现，也可以在 ResidentSolve / SingleChunkSolve 前作为输出子模块实现。

目标：

```text
file-backed current success output
typed block -> raw dtype write
cell write validation
ordered or random-access policy explicit
finish validation
write stats
```

## 20. 验收顺序

### Correctness

```text
BCSuccessIO 6 dtype roundtrip
ResidentSolve equals synthetic oracle
SingleChunkSolve equals ResidentSolve for chunk size 1 / 3 / all
FamilySolvePlan owner event no duplicate candidate
Family work item builder scans each bitmap word once per owner event
FamilyChainSolve equals ResidentSolve on small layers
```

### Memory

```text
ResidentSolve no unexpected whole-layer extra structures
SingleChunkSolve current chunk bound enforced
FamilyChainSolve active family window <= 3
runtime active-window assertion enabled in tests
no whole-layer hash / bitmap / success resident in FamilyChain
```

### IO

```text
future V(G) loaded once per target-family pass
partial load/write count explainable
success writer writes each cell exactly once
dirty partial blocks are dumped only when leaving window
no generation blob usage
no position writer usage
```

### Performance

```text
edge kernel uses batch canonicalize
future success blocks decoded once per load
lookup stats visible
cell/bucket/bitmap word work item balance measured
Direct IO path available for large runs
free9-256 mini/perf run compared against EX stats
```

## 21. Implementation Guardrails

必须遵守：

```text
Do not use BCCellMutableBuilder in solve.
Do not use BCFamilyMutableStore in solve.
Do not use BCGenerationBlobIO in solve.
Do not use BCFamilyPositionWriter in solve.
Do not create whole-layer hash maps in FamilyChain solve.
Do not create whole-layer bitmaps in FamilyChain solve.
Do not make dtype switch inside candidate update hot loop.
Do not finalize row success before all per-empty best contributions are complete.
Do not compare FamilyId across layers.
Do not skip runtime active-window accounting in FamilyChain solve.
Do not allow multiple threads to race on the same partial row/lane/empty slot.
```

允许：

```text
small pass plans
cell id lists
cell/bucket/bitmap word work items
thread-local board/canonical/encoded buffers
cell-local typed partial blocks
future cell cache for V(G)
streaming position/success readers
streaming or random-access success writer
```

## 22. 下一步建议

下一步只实现 Module 1。

推荐任务边界：

```text
开始 EXBC solve 的 Module 1：
Success dtype / typed success block / raw IO / dtype-aware BCSuccessIO。

不要实现 ResidentSolve。
不要实现 SingleChunkSolve。
不要实现 FamilyChainSolve。
不要实现 board edge kernel。
不要使用 generation mutable builder / generation blob / position writer。

目标：
    对齐 EX/EXAD DTypeMode 1..6；
    BC success file payload size 支持 dtype value size；
    raw dtype block 与 typed physical block 可双向读写；
    旧 uint32 测试继续通过；
    为后续 solve hot path 提供 typed StorageT block。
```
