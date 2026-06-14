# EXBC / EXADBC Three Solve Chains Runtime Design v1

本文档梳理 BC 系列回算阶段的三条链路：

- `ResidentSolve`
- `SingleChunkSolve`
- `FamilyChainSolve`

目标是把模块组成、计算逻辑、细节优化、IO 顺序统一下来。本文不替代
`exbc_exadbc_family_chain_solve_plan_v1.md`，而是作为三条 solve 链路的横向运行设计。

## 1. Scope

适用对象：

```text
BC position file: current / future layer 的 position 集合
BC success file: future lookup 输入、current solve 输出
BC dense family-cell matrix
Resident / SingleChunk / FamilyChain 三条 solve executor
```

非目标对象：

```text
BCCellMutableBuilder
BCFamilyMutableStore
BCGenerationBlobIO
BCFamilyPositionWriter
generation-phase mutable key/rank builder
```

solve 不创建 position entry，不插入 key/rank bitmap，不写 rank payload。solve 只读取 position，读取 future success，写 current success。

## 2. Shared Semantics

### 2.1 Layer Dependency

对 current layer `L` 中的 board 回算，需要查询：

```text
spawn +2 -> future layer L + delta2
spawn +4 -> future layer L + delta4
```

在 classic / EX 语义里，这对应 EX runtime 中的 `future1` 和 `future2`。在 BC family-cell 语义里，delta 对应目标 family fanout：

```text
T_delta(a, N) = unique({a, min(a + delta, N - a)})
```

实现时禁止跨 layer 直接比较 `FamilyId`。跨层必须使用 `FamilyCoord` 或 route planner 输出的目标 family。

### 2.2 Success DType

BC success 支持 6 个 dtype mode：

```text
1 uint32
2 uint64
3 float32
4 float64
5 1-float32
6 1-float64
```

物理 hot path 只应有 4 类 `StorageT`：

```text
uint32_t
uint64_t
float
double
```

`1-float32` / `1-float64` 是文件语义和数值解释模式，物理存储仍是 `float` / `double`。不要在 candidate hot loop 内做 dtype switch；executor 外层按 dtype dispatch 到模板实现。

### 2.3 Per-Board Recurrence

对 current board `b`：

```text
if terminal_success(b):
    success(b) = max_success
else:
    success_sum = 0
    empty_count = 0
    for empty cell e:
        spawn2 = place tile +2 at e
        best2 = max_d success(move_d(spawn2))

        spawn4 = place tile +4 at e
        best4 = max_d success(move_d(spawn4))

        success_sum += (1 - p4) * best2 + p4 * best4
        empty_count += 1

    success(b) = empty_count > 0 ? success_sum / empty_count : zero
```

`best2` 和 `best4` 是 per-empty-cell reducer。不能提前把某个 row collapse 成最终 success，除非该 row 的所有 relevant future passes 都已经完成。

### 2.4 Cell / Bucket / Rank

BC position lookup 的基本单位：

```text
CellId
bucket key
BucketRank
local_success_row
success lane / row_width column
```

`local_success_row` 的计算依赖 bucket bitmap 内 rank 前的 set-bit 个数：

```text
local_success_row = bucket.success_row_offset + rank_before(bitmap, rank)
```

BC 的单 bucket bitmap 平均长度大于 EX，因此 rank 查询必须使用 cell-local `prefix256` checkpoint，不能从 bucket 起点线性 popcount。

## 3. Shared Modules

### 3.1 Success IO / DType Policy

职责：

```text
validate success file dtype and row_width
read cell success block as typed physical StorageT
write current success block as configured dtype
provide raw block path for streaming IO
collect read/write bytes and seconds
```

要求：

```text
uint32 old API remains compatibility path
typed APIs are primary solve path
future success block loaded once -> reused as typed block
no repeated per-lookup decode
no dtype switch inside candidate loop
```

### 3.2 Position Reader

三条链路都读取 current/future position，但 resident 范围不同：

```text
ResidentSolve:
    current/future position can be resident

SingleChunkSolve:
    current position is chunked
    future position/success uses cache/window

FamilyChainSolve:
    current/future position are family view windows
    active family window <= 3
```

需要的基础能力：

```text
load cell list
coalesce cell bucket/rank extents
return BCLoadedCellView
support buffered/direct backend
stats: requested_extents, coalesced_extents, bytes, backend ops
```

### 3.3 Rank Checkpoint

BC position rank payload 格式：

```text
bucket.rank_payload_offset:
    prefix256[ceil(bitmap_len / 256)] as little-endian uint16
aligned bitmap offset:
    bitmap words as little-endian uint64
```

`prefix256[i]` 是第 `i` 个 256-bit segment 开始前的累计 set-bit 个数。

rank 查询：

```text
block = rank / 256
rank_before =
    prefix256[block]
  + popcount(words inside same 256-bit block before rank)
```

word-range work item 起点：

```text
bucket_seen = rank_before_word_index(prefix256, bitmap_words, bitmap_len, word_begin)
local_success_row = bucket.success_row_offset + bucket_seen
```

这使 `cell + bucket + bitmap word range` work item 不需要从 word 0 重扫。该 checkpoint 是 cell-local position payload，不是 whole-layer bitmap/hash 辅助结构。

### 3.4 SolveEdgeKernel

职责：

```text
scan source board
terminal success check
empty-cell ctz enumeration
spawn +2/+4
move all directions
skip unchanged move
direction/family prefilter
batch canonicalize
encode to CellId + key + rank + lane/ref
future lookup
update best2/best4 or partial store
```

推荐数据流：

```text
source bitmap word scan
    -> board buffer
    -> canonical_buffer2 / canonical_buffer4
    -> prepared_query2 / prepared_query4
    -> batch future lookup
    -> reducer update
```

EX 可迁移优化：

```text
thread-local workspace
ctz scan empty mask and bitmap words
batch canonicalize_inplace
prepared query separates encode from lookup
lookup returns row index first, then reads success values
prefetch success values after row-index batch lookup
optional sort by success row for cache locality
```

BC 需要调整：

```text
EX prefix36 key46 -> BC CellId + key + rank
EX whole-layer direct index -> BC cell-local direct index / active-view index
EX fixed uint32 solve value -> BC typed StorageT policy
```

### 3.5 Future Lookup

baseline API：

```text
lookup_typed(CellId cid, uint64_t key, BucketRank rank, uint32_t lane) -> StorageT
```

hot API：

```text
prepare queries:
    cid
    key
    rank
    lane / column
    ref = board_slot * 16 + empty_cell

batch lookup:
    queries -> found flags + row indices or values
    reduce max into best[ref]
```

性能要求：

```text
future position lookup should not binary-search bucket for every candidate in hot path
future success value should not decode per lookup
cell-local direct index is allowed inside active view
whole-layer index is not allowed in FamilyChain
```

### 3.6 Current Accumulator

Resident / SingleChunk 可以直接写 current result block：

```text
result[cell][row][lane]
```

FamilyChain 需要 partial store：

```text
per-empty best2/best4 while future family passes are incomplete
collapse to final row only after all owner events for the row are done
file-backed or memory-backed cell-local block
release_except active family window
```

### 3.7 Success Writer

职责：

```text
write current success file
write typed StorageT blocks in CellId order or explicit random-access policy
validate row_width and cell row count
mark empty cell
finish layer header/descriptors
stats: cell writes, bytes, seconds
```

solve 不写 position writer。

### 3.8 Stats

至少收集：

```text
stage, step, chain
dtype_mode, row_width
current_cells_loaded
future_cells_loaded
source_boards_scanned
bitmap_words_scanned
empty_slots_scanned
move_candidates
unchanged_moves
prefilter_rejects
canonicalized_candidates
encoded_queries
future_lookup_count
future_lookup_misses
partial_loads, partial_writes
success_bytes_read, success_bytes_written
position_bytes_read
read_seconds
scan_seconds
spawn_move_seconds
canonical_seconds
encode_seconds
lookup_seconds
partial_io_seconds
write_seconds
total_seconds
```

FamilyChain 额外收集：

```text
active_family_window_current
active_family_window_future
active_family_window_partial
active_family_window_total
active_family_window_max
retain_count
release_count
```

## 4. ResidentSolve

### 4.1 Module Composition

```text
BCResidentSolveExecutor
BCResidentPositionView
BCResidentFutureLayer<StorageT>
BCFuturePreparedLookup<StorageT>
BCSolveEdgeKernel<StorageT>
BCResidentResultAccumulator<StorageT>
BCSuccessLayerWriter
BCSolveStatsCollector
```

ResidentSolve 是 correctness oracle 和高性能 baseline。它允许 current/future position 与 future success 常驻，但仍不应创建无意义的 whole-layer hash/bitmap 复制。

### 4.2 IO Order

推荐顺序：

```text
1. open current position L
2. open future position/success L + delta2
3. open future position/success L + delta4
4. load or map resident future position cells
5. load or map resident future success blocks
6. build future lookup acceleration
7. allocate current result blocks
8. scan current position
9. write current success
10. flush stats
```

如果 current position 也 resident：

```text
read all current position once
parallel scan cells/buckets/word ranges
write success once
```

如果 current position 使用 streaming reader，也可以按 cell batches 读入，但这已经接近 SingleChunk。

### 4.3 Compute Logic

```text
for each current cell:
    get buckets + rank payload
    for each bucket:
        split bitmap words into work items if needed
        scan set bits
        unrank key/rank -> board
        edge kernel produces +2/+4 prepared queries
        future lookup reduces best2/best4
        finalize board success
        write result[cell][local_success_row]
```

由于 ResidentSolve 当前 result 全部常驻，row update 可以直接写 `result[cell][row]`。每个 source row 只应由一个 work item owner 写 final result，避免 atomic 写 success。

### 4.4 Optimizations

直接采用：

```text
thread-local RecalcWorkspace
canonical_buffer2 and canonical_buffer4 separated
prepared query vector separated by delta
lookup row-index batch then success prefetch
terminal success short-circuit
ctz bitmap scan
prefix256 for local_success_row
cell-local direct index for future lookup
```

可选优化：

```text
sort successful future row indices before reading success
interleave +2/+4 lookup if memory latency dominates
precompute bucket decoder for hot bucket
reuse board buffer and query vector capacity across work items
```

避免：

```text
per-candidate dtype switch
per-candidate heap allocation
binary-search buckets for every candidate when direct index is available
decoding success from raw bytes for every lookup
```

### 4.5 Correctness Role

ResidentSolve 应作为：

```text
small layer oracle
SingleChunkSolve 对拍目标
FamilyChainSolve 对拍目标
dtype behavior oracle
partial collapse correctness oracle
```

## 5. SingleChunkSolve

### 5.1 Module Composition

```text
BCSingleChunkSolveExecutor
BCCurrentChunkPlanner
BCPositionStreamingReader current_reader
BCFutureCellWindow<StorageT> or chunk future view
BCFuturePreparedLookup<StorageT>
BCSolveEdgeKernel<StorageT>
BCChunkResultAccumulator<StorageT>
BCSuccessStreamingWriter
BCSolveStatsCollector
```

SingleChunkSolve 目标是限制 current layer 常驻内存，同时尽量复用 ResidentSolve 的 hot kernel。

### 5.2 Chunk Planning

chunk 粒度可以按：

```text
cell count
rank_payload bytes
success row bytes
estimated bitmap words
```

推荐以 memory budget 为主：

```text
current position chunk bytes
+ current result chunk bytes
+ future active view/cache bytes
+ thread workspace bytes
<= budget
```

chunk 内部仍按 `cell + bucket + bitmap word range` 拆并行 work item。

### 5.3 IO Order

推荐顺序：

```text
1. open current position streaming reader
2. open future position/success readers
3. open current success writer
4. build chunk plan over current cells
5. for each current chunk:
       load current cells
       determine future cells needed by this chunk
       load/retain future cells and typed success blocks
       build/update cell-local future lookup acceleration
       parallel scan current chunk
       write solved success blocks for chunk cells
       release current chunk
       retain future cells still useful for next chunk
6. finish success writer
7. flush stats
```

重要点：

```text
current chunk position read is sequential/coalesced
future reads should be batched by cell list
success writes should be in CellId order when possible
```

### 5.4 Compute Logic

SingleChunk 的 row 生命周期：

```text
load current cell
scan its rows
compute final success completely inside chunk
write output success for that cell
release current cell
```

因此它不需要 FamilyChain 的 cross-family partial store。只要 current row 的 +2/+4 future lookup 都在本 chunk processing 中完成，就可以立即写 result block。

### 5.5 Optimizations

直接复用 ResidentSolve：

```text
same SolveEdgeKernel
same prepared query / batch lookup path
same prefix256 rank-before
same thread-local workspace
```

SingleChunk 特有：

```text
chunk future cell need-list dedup
future retain/release across adjacent chunks
coalesced position and success reads
preload next chunk future view when IO backend supports overlap
avoid reload of future success block if same cell appears in next chunk
```

需要防止：

```text
chunk too small -> excessive future reload and writer overhead
chunk too large -> memory budget violation
future cache degenerates to whole-layer resident
random small success reads instead of batched cell block reads
```

### 5.6 Correctness Role

SingleChunkSolve 应验证：

```text
chunk_size = all cells equals ResidentSolve
chunk_size = 1 equals ResidentSolve
random chunk partition equals ResidentSolve
future retain/release does not affect result
```

## 6. FamilyChainSolve

### 6.1 Module Composition

```text
BCFamilyChainSolveExecutor
BCFamilySolvePlanner
BCFamilyCurrentWindow
BCFutureFamilyWindow<StorageT>
BCPartialStore<StorageT>
BCSolveEdgeKernel<StorageT>
BCFamilyWorkItemBuilder
BCSuccessStreamingWriter
BCActiveWindowAsserter
BCSolveStatsCollector
```

FamilyChainSolve 的核心约束：

```text
active family window <= 3
no whole-layer hash map
no whole-layer bitmap
no whole-layer success resident
future view is V(G), not LRU cache
partial/result is cell-local and windowed
```

### 6.2 Target-Family-Major Scheduling

FamilyChain solve 应按 target family major 顺序调度：

```text
for target family G in solve order:
    load future view V(G)
    retain overlap from previous family if still needed
    prepare current cells whose owner event targets G
    scan work items
    update current partial/result
    finalize boundary cells whose all owner events are complete
    write finalized success cells
    release cells outside active family window
```

`V(G)` 是 family cross：

```text
row G: (G, x)
col G: (x, G), x != G
```

这不是传统 cache。它只负责：

```text
load V(G)
retain cells still needed by next G
release cells not in active window
```

### 6.3 IO Order

推荐单 family pass 的 IO：

```text
1. determine target family G
2. compute needed future cells V(G)
3. future window:
       retain overlap
       batch load missing future position cells
       batch load missing future success cells
       build cell-local lookup acceleration for loaded cells
4. current window:
       load source/current cells needed by owner events for G
5. partial store:
       load dirty partial blocks for affected current cells
6. parallel scan current work items
7. flush thread workspaces
8. boundary finalize:
       finalize cells whose owner events are complete
       write typed success blocks
9. partial store release_except
10. future/current release_except
11. update active-window stats and assertions
```

IO 原则：

```text
future V(G) batch load once per target-family pass
success block loaded as typed physical block
partial dirty block only dumps when leaving window
finalized success cell writes immediately
no generation blob
no position writer
```

### 6.4 Work Items

FamilyChain work item 建议字段：

```text
current loaded cell index
current CellId
bucket index
bitmap word begin
bitmap word end
owner event / target family G
delta: +2 or +4
direction mask
```

构建要求：

```text
same owner event 下每个 bitmap word 只扫描一次
large bucket split by word range
small buckets can be grouped
word range size target should balance CPU without creating too many tasks
```

range scan 必须使用 `prefix256`：

```text
row_at_word_begin =
    bucket.success_row_offset
  + rank_before_word_index(prefix256, bitmap_words, bitmap_len, word_begin)
```

这样 split bucket 的每个 work item 都能直接得到自己的起始 success row。

### 6.5 Compute Logic

```text
for each work item:
    load bucket decoder
    compute row_at_word_begin by prefix256
    for each set bit in word range:
        rank -> quadrant words -> board
        local_success_row = row_at_word_begin + set_bits_seen_in_range
        terminal check if applicable
        spawn/move/canonicalize/encode
        prefilter by target family G
        future lookup in V(G)
        update partial row/lane/empty slot
```

`+4 phase` 和 `+2 phase` 分开：

```text
+4 phase:
    generate/update partial
    do not finalize

+2 phase:
    generate/update partial
    advance L-shaped boundary
    finalize completed cells
```

如果某 row 的 relevant future owner events 还没全部完成，不能 collapse 成 final success。

### 6.6 PartialStore

FamilyChain 的 partial store 是 solve 的核心对象。

候选 schema：

```text
per-empty best:
    row_count * row_width * max_empty_slots * per-delta best

collapsed row:
    row_count * row_width
```

使用规则：

```text
per-empty best is required while owner events are incomplete
collapse only when all future passes affecting the row are complete
dirty blocks leaving active window must be dumped
clean blocks leaving active window can be released
finalized blocks are written to success writer and no longer reloaded
```

线程安全：

```text
same partial row/lane/empty slot must have single writer
or use explicit row-local reduction then merge
no unprotected multi-thread writes to same partial slot
```

### 6.7 Active Window

Runtime invariant：

```text
active_family_window_total <= 3
```

统计分解：

```text
current window families
future window families
partial window families
total unique active families
max observed
```

建议每个 family pass 结束时 assert：

```text
current cells resident only in allowed families
future cells resident only in allowed families
partial cells resident/dumped/finalized state is consistent
dirty outside window count == 0 after release_except
```

### 6.8 Optimizations

直接迁移 EX/EXAD：

```text
thread-local workspace
batch canonicalize
prepared query
lookup row-index batch
success prefetch
ctz bitmap scan
terminal success short-circuit
```

BC 特有：

```text
prefix256 for word-range local row start
family prefilter before future lookup
physical_target_family_may_hit before expensive encode/lookup when possible
future V(G) retain overlap instead of LRU
cell-local direct index for active future cells
partial store release_except
```

难以应用：

```text
EX whole-layer future rolling cache
EX whole-layer direct index
EX optimal-branch keep bitmap
EX compact_layer rewriting position file
```

## 7. Cross-Chain Comparison

| Item | ResidentSolve | SingleChunkSolve | FamilyChainSolve |
|---|---|---|---|
| Current position | resident | chunked | family window |
| Future position/success | resident | chunk/window retained | V(G) family window |
| Current result | resident typed blocks | chunk typed blocks | partial store + finalize |
| Memory goal | fastest baseline | bounded current memory | <= 3 active families |
| IO style | few large reads/writes | chunked batched IO | family view + partial IO |
| Parallelism | cells/buckets/ranges | chunk cells/buckets/ranges | family pass work items |
| Main risk | memory | reload overhead | lifecycle correctness |
| Correctness role | oracle | resident equivalence | final production path |

## 8. Shared Hot Path Checklist

All chains should share the same edge kernel where possible:

```text
source rank payload scan uses prefix256 when local row is needed
bitmap scan uses ctz
empty mask scan uses ctz
spawn +2/+4 buffers separated
unchanged moves skipped
family/pattern prefilter before lookup
canonicalize in batches
encode in batches
lookup in batches
success dtype is template StorageT
future success block decoded/loaded once per cell block
stats are thread-local then reduced
```

Do not regress into:

```text
per-candidate allocation
per-candidate dtype switch
per-candidate future success decode
whole-layer hash/bitmap in FamilyChain
source-major FamilyChain future reload storm
unbounded future cache
unprotected partial row write races
```

## 9. Recommended Implementation Order

The production path should still be incremental:

```text
1. Success dtype / typed block / streaming writer
2. FutureFamilyWindow and future typed block loading
3. prefix256 word-range scanner APIs
4. SolveEdgeKernel typed baseline
5. ResidentSolve oracle
6. SingleChunkSolve with chunked current
7. PartialStore schema and tests
8. FamilyChainSolve skeleton
9. FamilyChainSolve executor with active-window assertions
10. performance counters and free9-256 comparison
```

Current status notes:

```text
BCSuccessIO dtype-aware path exists
BCFutureFamilyWindow exists as first version
prefix256 payload exists and word-range helper exists
Resident/SingleChunk/Family solve executors still need implementation
PartialStore still needs schema proof and implementation
```

## 10. Validation Matrix

Correctness:

```text
typed success IO roundtrip for all 6 dtype modes
prefix256 rank_before against naive popcount
loaded-cell word-range scanner equals full scanner
ResidentSolve equals synthetic oracle
SingleChunkSolve equals ResidentSolve for chunk size 1 / random / all
FamilyChainSolve equals ResidentSolve on small layers
FamilyChain +2/+4 phase order correctness
```

Memory:

```text
ResidentSolve reports expected resident bytes
SingleChunkSolve respects chunk budget
FamilyChainSolve active family window <= 3
no whole-layer hash/bitmap/success resident in FamilyChain
```

IO:

```text
future cell load count explainable
future success block not decoded repeatedly
partial dirty dump only when leaving window
success writer writes each cell exactly once
position writer is not used in solve
generation blob is not used in solve
```

Performance:

```text
scan/encode/lookup throughput measured
lookup miss rate measured
prefilter reject rate measured
prefix256 avoids range-start rescans
FamilyChain hot path within target multiple of non-blocked pass
free9-256 compared against EX solve stats
```
