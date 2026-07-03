# EXBC / EXADBC 当前设计：Raw-Sum Modulo Family、Streaming Position Writer 与 Direct IO

本文定义当前实现基线。核心目标是使用同一套 BC position layer 逻辑格式支撑三条生成链路，并让大层生成在有限内存内完成：

```text
Dense family-cell matrix
+ cell-local key -> rank bitmap
+ compact dense success rows
+ source-family sweep
+ mutable cell dump/reload
+ direct IO friendly .bcpos output
```

设计约束：

```text
所有 layer sum、family sum、quadrant sum 都使用 raw tile sum。
bucket key 继续使用 compact sum_id + empty_mask，不直接存 raw sum。
FamilyCoord 表达 raw half-sum / family_unit 后的 sum 语义。
FamilyId 表达当前 layer/partition 内的 dense storage id。
生成阶段允许 modulo partition 导致的 target fanout3。
三条生成链路输出同一套 .bcpos 逻辑格式。
```

---

## 1. Board、Tile Sum 与 Quadrant LUT

### 1.1 Board 布局

4x4 board 使用 `uint64_t` 表示，每个 tile 占 4 bit exponent。

```text
A4 B4 | C4 D4
A3 B3 | C3 D3
------+------
A2 B2 | C2 D2
A1 B1 | C1 D1
```

四个 2x2 quadrant：

```text
NW = A4 B4 A3 B3
NE = C4 D4 C3 D3
SW = A2 B2 A1 B1
SE = C2 D2 C1 D1
```

### 1.2 Raw Sum 语义

当前语义下，tile 的 sum 始终按真实 tile value 累加：

```text
tile 0  -> 0
tile r  -> 2^r, r in [1, 15]
```

`rank15` 不具有特殊 sum 语义。只要 tile rank 合法，它的 sum 就参与 quadrant sum、half sum、layer sum 和 family coord 计算。

每个 `.bcpos` layer 必须满足：

```text
board_raw_sum(board) == header.layer_sum
```

数据转换工具、抽样校验和全量一致性检查都应以 raw sum 为准。

### 1.3 Legal Tile Alphabet

`BCLut` 由 target/table 的 legal tile alphabet 构建。非法 tile 会让对应 16-bit quadrant word 的 `BCWordDesc.valid=false`。

通用 encode 和测试路径需要检查 `valid`；Family generation 热路径的输入来自已验证的 source layer、spawn、move、canonicalize，因此使用 trusted helper 跳过重复合法性检查。

### 1.4 BCWordDesc

每个 16-bit quadrant word 的 descriptor 当前包含：

```cpp
struct BCWordDesc {
    uint32_t sum;
    uint16_t sum_id;
    uint16_t packed_sum_mask;
    uint16_t group_count;
    uint8_t empty_mask;
    BucketRank rank;
    bool valid;
};
```

字段含义：

```text
sum              raw quadrant sum，用于 family half-sum 计算
sum_id           compact LUT sum group id，用于 bucket key
packed_sum_mask  (sum_id << 4) | empty_mask，用于 key 拼装
group_count      同 sum_id+empty_mask 下的 legal word 数
empty_mask       quadrant 内空格 mask
rank             word 在 sum_id+empty_mask group 内的 rank
valid            word 是否由 legal alphabet 构成
```

`sum` 和 `packed_sum_mask` 同时保留是当前热路径设计：family 计算直接读 raw sum，key 生成直接读 packed sum/mask，避免额外查表和重复转换。

---

## 2. Bucket Key、Rank 与 Rank Payload

### 2.1 Bucket Key

当前 key mode：

```text
NW exact + NE/SW/SE packed_sum_mask
```

64-bit key 布局：

```text
[63:48] exact 16-bit NW word
[47:32] NE packed_sum_mask
[31:16] SW packed_sum_mask
[15: 0] SE packed_sum_mask
```

每个 `packed_sum_mask`：

```text
[15:4] compact sum_id
[ 3:0] empty_mask
```

`sum_id` 只用于 compact bucket grouping，不等价于 raw sum。raw sum 由 `BCWordDesc.sum` 提供。

### 2.2 Rank 公式

对非 exact quadrant：

```text
NE, SW, SE
```

分别有：

```text
rankNE, countNE
rankSW, countSW
rankSE, countSE
```

bucket 内 rank：

```text
rank = (rankNE * countSW + rankSW) * countSE + rankSE
```

bitmap length：

```text
bitmap_len = countNE * countSW * countSE
```

固定 `sum_id+empty_mask` 的 2x2 word 最多 36 种，因此：

```text
bitmap_len <= 36^3 = 46656
```

所以：

```cpp
using BucketRank = uint16_t;
using BucketBitmapLen = uint16_t;
```

中间乘法使用 `uint32_t`，结果在写入 `uint16_t` 前必须满足范围约束。

### 2.3 BCBucketEntry

bucket metadata 中每个 entry 固定 16 bytes：

```cpp
struct BCBucketEntry {
    uint64_t key;
    uint32_t rank_payload_offset;
    uint32_t success_row_offset;
};
```

含义：

```text
key                  bucket key
rank_payload_offset  cell-local rank payload byte offset
success_row_offset   cell-local dense success row base
```

bucket 在 cell 内按 `key` 升序且唯一。

### 2.4 Rank Payload

每个 bucket 在 cell-local rank payload 中有一段：

```text
[prefix256 uint16 array]
[padding to 8-byte alignment]
[bitmap uint64 little-endian words]
```

定义：

```text
prefix_block_bits = 256
prefix_count = ceil(bitmap_len / 256)
bitmap_word_count = ceil(bitmap_len / 64)
```

`prefix[block]` 保存该 256-bit block 之前已经置位的 bit 数。lookup 命中时：

```text
local_success_row =
    bucket.success_row_offset + popcount(bitmap bits before rank)
```

---

## 3. FamilyCoord、Partition Family 与 Cell

### 3.1 Exact FamilyCoord

一个 layer 的 raw sum 为 `S`。board 的 row/col half sum：

```text
row_top    = sum(NW) + sum(NE)
row_bottom = sum(SW) + sum(SE)
col_left   = sum(NW) + sum(SW)
col_right  = sum(NE) + sum(SE)
```

exact side-normalized family raw sum：

```text
row_family_raw = min(row_top, row_bottom)
col_family_raw = min(col_left, col_right)
```

FamilyCoord：

```text
FamilyCoord = family_raw / family_unit
```

标准 2048 raw-sum 语义下：

```text
family_unit = 2
spawn2_delta_coord = 1
spawn4_delta_coord = 2
```

### 3.2 FamilyCoord 与 FamilyId

二者不能混用：

```text
FamilyCoord:
    raw half-sum / family_unit 后的 sum coordinate。
    可跨 layer 比较，参与 source->target 映射和 wavefront boundary。

FamilyId:
    当前 layer 当前 partition axis 内的 dense storage id。
    用于 cell_id = row_id * family_count + col_id。
```

### 3.3 Exact Axis 与 Modulo Partition Axis

实现支持两种 partition policy：

```cpp
BCFamilyPartitionPolicy::exact()
BCFamilyPartitionPolicy::modulo(M)
```

exact policy 中，axis 由该 layer 所有可达 exact FamilyCoord 构成。

modulo policy 中，storage axis 固定为：

```text
FamilyId = coord % M
family_count = M
axis coords = 0..M-1
```

当前 benchmark 默认：

```text
M = 29
```

每个 layer 会构建 `BCFamilyPartitionLayerMap`：

```cpp
struct BCFamilyPartitionLayerMap {
    LayerSum layer_sum;
    uint16_t family_unit;
    FamilyCoord total_coord;
    BCFamilyPartitionPolicy policy;
    vector<vector<FamilyCoord>> exact_coords_by_family;
    vector<uint8_t> active_family;
    vector<FamilyId> coord_to_family_lut;
};
```

热路径中：

```text
FamilyCoord -> FamilyId
```

通过 `coord_to_family_lut[coord]` 完成。取模已经在构建 LUT 时展开，不在 encode 热路径执行 `%`。

### 3.4 Cell Matrix

每个 layer 的 position 数据按 dense family-cell matrix 组织：

```text
cell(row_id, col_id)
cid = row_id * family_count + col_id
```

一个 cell 包含所有映射到该 row partition family 和 col partition family 的 boards：

```text
cell -> sorted buckets -> bucket rank bitmap -> dense local success rows
```

---

## 4. Source Family 到 Target Families

### 4.1 Exact Coord 映射

对 source layer：

```text
N = source_total_coord
a = source exact FamilyCoord
b = N - a
a <= b
```

spawn delta：

```text
d = 1 for spawn2
d = 2 for spawn4
```

一个 exact source coord 可能触达的 exact target coords：

```text
{ a, min(a + d, b) }
```

然后把 exact target coord 映射到 target partition family：

```text
target_id = target_partition.coord_to_family_lut[target_coord]
```

### 4.2 Modulo Fanout

modulo partition 中，一个 source `FamilyId` 可以包含多个 exact coords：

```text
exact_coords_by_family[source_id]
```

因此 source partition family 的 target family 集合是：

```text
dedup sort union over exact source coords:
    target(coord)
```

集合大小当前允许为：

```text
1..3
```

使用 `FamilyIdList3` 表示。若映射结果超过 3，generation backend 报错。

### 4.3 Active Family Window

单个 FamilyChain pass 的 active target set 为：

```text
Gset = mapped target families, size <= 3
NeedCells = dedup union_{g in Gset} (row g union col g)
```

因此 active family window 峰值为：

```text
1 source family + up to 3 target families = 4 families
```

生成阶段按该窗口 reload target mutable cells、处理 source cells、插入 candidates，然后 release 不需要保留的 cells。

---

## 5. 三条生成链路

三条生成链路输出同一套 `.bcpos` 逻辑格式。链路差异只影响计算和中间状态管理。

| 链路 | 调度单位 | target mutable | source/future 驻留 | 输出 |
|---|---|---|---|---|
| ResidentChain | cell / cell range | 全驻留 | 全驻留 | `.bcpos` |
| SingleChunkChain | current cell chunk | chunk 驻留 | 每次需要的 source/future | `.bcpos` |
| FamilyChain | source partition family pass | active window 驻留，其他 dump/reload | streaming source cells | `.bcpos` |

当前大层生成主线是 FamilyChain。

---

## 6. FamilyChain Generation

### 6.1 输入与阶段

生成一个 target layer 时输入：

```text
source4: layer_sum - 4, spawn tile rank 2, delta coord 2
source2: layer_sum - 2, spawn tile rank 1, delta coord 1
```

流程：

```text
begin target mutable store
begin streaming position writer

if source4 exists:
    run +4 phase
    no final boundary finalize

run +2 phase
    boundary finalize/write cells

finish position writer
delete temporary blob/spool
```

`+4` phase 只产生 target mutable state。`+2` phase 在 source-family sweep 中推进 boundary，并把完成的 cells finalize 到 position writer。

### 6.2 Pass Cache

每个 phase 开始时构建 source-family pass cache。每个 cached pass 包含：

```text
has_source_rows
source pass descriptor
source cells
source cids
target_need_cells
boundary_cells
keep_cells
```

`has_source_rows` 通过 source family activity scan 得到。空 source family 仍会推进 finalize/release 边界。

`target_need_cells` 是 `Gset` 对应 family crosses 的去重 cell 集合。

`boundary_cells` 由 last-producer boundary 计算。+2 phase 中，当一个 cell 最后一次可能被写入的 source pass 完成后，该 cell 可以 finalize。

`keep_cells` 来自后续 pass 的 target_need_cells 前瞻集合，用于减少 dump/reload。

### 6.3 Source Cell Loading

一个 source family pass 读取：

```text
row source cells: cell(f, x)
col source cells: cell(x, f)
```

source cells 按 cid 排序后由 streaming reader 批量加载。加载得到 `BCLoadedCell`，其中包括：

```text
bucket entry view
rank payload view
source direction mask
```

### 6.4 Source Range Work

每个 loaded source cell 被切成 `FamilySourceRangeWork`。切分单位是 bucket bitmap words：

```text
source_bitmap_words_per_work_item = 64
```

大 bucket 会按 word range 切分，小 bucket 会合并成一个 work item。这样可以让并行调度以 bitmap word 扫描量而不是 bucket 数量近似负载均衡。

### 6.5 Board 生成流程

每个 source rank bit 命中后：

```text
decode source board from bucket key + rank
spawn tile
move left/right/up/down according to source direction mask
discard no-op moves
optional terminal success filter
canonicalize
encode target family candidate
enqueue into active cell buffer
```

terminal success filter 由 `keep_only_success_generated_boards` 控制，只用于 terminal generation mode。普通 layer generation 不启用该过滤。

### 6.6 Canonical Batch

worker 先把 move 后的 board 放入 canonical buffer：

```text
canonical_batch_size = 8192
canonical_symm_mode = Full
```

buffer flush 时批量 canonicalize，再逐个 encode。这样把 canonicalize 的固定成本和后续 encode/insert 分开，减少小批量调用开销。

---

## 7. Family Encode 热路径

### 7.1 从 Canonical Board 到 Candidate

encode 输入是 canonicalized `uint64_t board`。流程：

```text
unpack board -> NW/NE/SW/SE 16-bit words
load BCWordDesc for each quadrant
raw sums -> row/col FamilyCoord
partition LUT -> row/col FamilyId
active family window -> active_index
packed_sum_mask -> 64-bit bucket key
quadrant ranks/group counts -> BucketRank
```

family sum 计算使用：

```text
nw.sum + ne.sum
nw.sum + sw.sum
target_layer_sum - half_sum
```

key 生成使用：

```text
NW exact word
NE/SW/SE packed_sum_mask
```

### 7.2 Trusted Coord LUT

当 `family_unit == 2` 且 active window 已建立时，热路径使用 trusted helper：

```text
coord_to_family_id_trusted(coord)
```

该 helper 是一次 LUT 读取。coord 有效性由 source layer、spawn/move、canonicalize 和 target layer sum 约束保证。

### 7.3 Active Index

`active_index` 不存入 candidate。它只用于选择当前 worker 的 active cell buffer。

对 `Gset = [g0, g1, g2]`，active buffers 表示：

```text
V(g0) = row g0 union col g0
V(g1) excluding cells already covered by V(g0)
V(g2) excluding cells already covered by V(g0), V(g1)
```

`active_index -> cid` 在 worker 准备 active window 时固定。enqueue 时用 `active_index` 找到 buffer，flush 时直接使用该 buffer 的 `cid`。

### 7.4 Candidate 结构

Family encode 输出的 candidate 只保留：

```cpp
struct FamilyEncodedCandidate {
    uint64_t key;
    BucketRank rank;
};
```

不保存：

```text
valid
cid
active_index
bitmap_len
```

设计原因：

```text
valid 由返回的 active_index 是否有效表达。
cid 属于 active cell buffer。
active_index 是 enqueue 路由信息，不属于 candidate payload。
bitmap_len 只有创建 bucket 或扩 bitmap 时才需要，可从 trusted key 现场计算。
```

---

## 8. Per-Thread Active Cell Buffer 与 Mutable Insert

### 8.1 Buffer 布局

每个 worker 对当前 active window 建立固定容量 buffer：

```text
one buffer per active cell
capacity = pending_insert_buffer_size
default pending_insert_buffer_size = 512
```

buffer 中连续存储：

```cpp
struct BCCellTrustedKeyRankInsert {
    uint64_t key;
    BucketRank rank;
};
```

buffer 首次触达时加入 `touched_buffer_indices`。buffer 满或 worker pass 结束时 flush。

### 8.2 Builder Bind

enqueue 时若 buffer 未绑定 builder：

```text
target_store.get_or_create(cid)
```

该调用只允许 active target window 内的 cell。若 cell 是 Empty，则创建 `BCCellMutableBuilder`；若 cell 是 Dumped，则必须已在 prepare 阶段 reload；若 cell 已 Finalized，则报错。

### 8.3 Batch Resolve/Apply

flush 时调用：

```text
BCCellMutableBuilder::resolve_and_apply_trusted_key_rank_batch
```

处理顺序：

```text
ensure hash/bitmap batch capacity
resolve key -> home slot / bucket / bitmap word
必要时 grow hash table 或 bitmap arena
prefetch upcoming hash slots and bitmap words
apply atomic bitmap OR
mark builder dirty when new rank appears
```

当前路径支持动态 grow。hash/bitmap 容量不足时会 stop-world grow 后重试本批。

### 8.4 Prefetch

batch resolve/apply 使用固定 lookahead：

```text
kBCMutableBatchPrefetchDistance = 16
```

预取目标包括 hash slot 和即将访问的 bitmap word。该参数是当前实测后保留的生成热路径常量。

---

## 9. Mutable Store 与 Generation Blob

### 9.1 Cell 状态

target mutable store 为每个 cell 常驻 header：

```cpp
enum class BCMutableCellState : uint8_t {
    Empty,
    Resident,
    Dumped,
    Finalized,
};

struct BCMutableCellHeader {
    BCMutableCellState state;
    BCDumpRef dump_ref;
    uint32_t dump_generation;
};
```

每个 cell 还具有：

```text
atomic builder pointer
lifecycle mutex
active/keep/resident marks
optional resident BCCellMutableBuilder
```

### 9.2 Prepare Target Window

pass 开始时：

```text
prepare_target_window_for_cached_cells(Gset, target_need_cells)
```

动作：

```text
mark active cells
reload dumped active cells from blob
keep resident active cells
do not create empty builders until first insert
```

### 9.3 Release

pass 结束后：

```text
release_except(keep_cells)
```

动作：

```text
keep cells remain resident
dirty resident cells outside keep set are serialized to blob
clean resident cells outside keep set release builder and keep dump ref
empty cells remain empty
finalized cells remain finalized
```

### 9.4 Generation Blob

FamilyChain generation 使用单个 append-only blob 保存 mutable builder dump。

blob 存储的是可恢复的 builder 状态，不是 final position payload，也不是 C++ 对象 raw memory。

blob IO 当前支持 buffered 和 direct backend。benchmark 默认：

```text
family_blob = direct
direct_queue_depth = 8
```

layer 完成后临时 blob 删除。

---

## 10. Boundary Finalize 与 Streaming Position Writer

### 10.1 Boundary Finalize

FamilyChain 在 +2 phase 中按 source-family pass 推进 finalize boundary。一个 cell 一旦不会再收到任何 candidate，就可以 finalize。

finalize 过程：

```text
if cell Dumped:
    reload builder
if cell Empty:
    write empty descriptor
if cell Resident:
    finalize builder into sorted buckets + rank payload
    stream cell payload into position writer
    release builder
mark cell Finalized
```

### 10.2 Streaming Position Writer

`BCFamilyPositionWriter` 为 boundary finalize 设计。每个 cell finalize 一次，writer 立即接收 payload。

writer 常驻：

```text
axis
cell descriptors
written flags
small staging buffers
metadata scratch
rank/bucket copy buffer
```

writer 不常驻所有 finalized payload。

### 10.3 Buffered Layout

buffered writer 逻辑布局：

```text
header
axis coord table
cell descriptor table
bucket meta stream
rank payload stream
```

bucket stream 直接写 final file，rank payload 先写 spool，finish 时 copy 到 final file。

### 10.4 Direct Rank-First Layout

direct no-buffering backend 要求 aligned IO，不允许破坏未写入的同一 aligned block。当前 direct output 使用 rank-first layout：

```text
metadata prefix, padded to 4096
rank payload stream
padding to 4096
bucket meta stream
optional trailing padding < 4096
```

header 中仍记录：

```text
bucket_meta_offset / bucket_meta_bytes
rank_payload_offset / rank_payload_bytes
```

reader 按 header offset 读取，不依赖物理段顺序。

direct-rank-first 的 `.bcpos` 是同一套 `.bcpos v2` 逻辑格式。

---

## 11. `.bcpos` 逻辑格式

### 11.1 Header

header 固定 112 bytes：

```cpp
struct BCPositionHeader {
    uint32_t magic;
    uint32_t format_version;
    uint32_t header_bytes;
    uint32_t key_mode;
    uint32_t rank_prefix_bits;
    uint32_t rank_prefix_type;
    uint32_t rank_payload_align;
    uint32_t family_unit;
    uint32_t axis_base_coord;
    uint32_t family_count;
    uint64_t layer_sum;
    uint64_t descriptor_count;
    uint64_t descriptor_table_offset;
    uint64_t descriptor_table_bytes;
    uint64_t bucket_meta_offset;
    uint64_t bucket_meta_bytes;
    uint64_t rank_payload_offset;
    uint64_t rank_payload_bytes;
    uint64_t axis_coord_table_bytes;
};
```

约束：

```text
magic = BCPF
format_version = 2
key_mode = Q4_NW_EXACT_NE_SW_SE_SUM_MASK_PREFIX256
rank_prefix_bits = 256
rank_prefix_type = uint16
rank_payload_align = 8
```

### 11.2 Axis Coord Table

紧跟 header，包含 `family_count` 个 `uint32` coord。

exact policy 中 coord 是 exact FamilyCoord。modulo policy 中 coord 是 partition storage coord：

```text
0..M-1
```

### 11.3 Cell Descriptor

每个 cell descriptor 固定 40 bytes：

```cpp
struct BCPositionCellDescriptor {
    uint32_t bucket_count;
    uint32_t success_rows;
    uint64_t bucket_meta_offset;
    uint64_t rank_payload_offset;
    uint64_t rank_payload_bytes;
    uint32_t reserved0;
    uint32_t flags_or_padding;
};
```

descriptor offset 是相对 stream 的 cell-local offset：

```text
bucket file range =
    header.bucket_meta_offset + desc.bucket_meta_offset
    length = desc.bucket_count * 16

rank payload file range =
    header.rank_payload_offset + desc.rank_payload_offset
    length = desc.rank_payload_bytes
```

空 cell：

```text
flags_or_padding has kBCPositionCellFlagEmpty
bucket_count = 0
success_rows = 0
rank_payload_bytes = 0
```

### 11.4 Validation

reader 必须校验：

```text
header fields match current key/prefix mode
axis table size matches family_count
descriptor table range is valid
bucket and rank ranges are inside file logical size
bucket/rank streams do not overlap
descriptor cell ranges stay within their streams
bucket keys are sorted and unique inside a cell
rank payload offset/size is valid for each bucket bitmap_len
```

direct output may have physical trailing padding. Reader allows:

```text
0 <= physical_size - logical_data_end < 4096
```

---

## 12. IO Policy

### 12.1 Defaults

Family generation benchmark defaults:

```text
family_blob = direct
family_position_io = direct-rank-first
family_source_io = direct-auto
direct_queue_depth = 8
```

### 12.2 Source Reader

`direct-auto`:

```text
if source physical size is padded enough for direct IO:
    use BCDirectFileReader with logical_size
else:
    use buffered reader
```

`direct` requires a padded source file and throws if direct IO cannot be opened safely.

### 12.3 Blob IO

Generation blob uses direct qd=8 by default. It records backend read/write bytes, op counts, and backend seconds.

### 12.4 Position Output IO

Final `.bcpos` output uses direct rank-first by default. The writer keeps physical padding instead of trimming the file, so later direct reads can use the file without rewriting.

### 12.5 Stats

Benchmark stats distinguish:

```text
wall-stage seconds:
    source_load_seconds
    parallel_seconds
    dump_seconds
    reload_seconds
    finalize_seconds
    write_seconds

backend IO seconds:
    blob_backend_read_seconds
    blob_backend_write_seconds
    target_backend_read_seconds
    target_backend_write_seconds
```

Throughput should be computed from backend seconds when judging device IO capability. Wall-stage seconds include CPU work, scheduling, serialization, coalescing, flush, and reader/writer bookkeeping.

Main generation stats intentionally avoid per-board, per-rank, per-candidate, and per-hash-probe counters. Those counters require writes inside the hottest loops and are not part of the default benchmark output. The retained top-level counters are:

```text
source families/cells loaded
source/blob/target IO bytes, ops, and backend seconds
target cell lifecycle counts
builder grow counts
active/source/workspace peak sizes
stage wall-clock seconds
```

Detailed memory attribution is emitted only when memory checkpoints are enabled.

---

## 13. Memory Accounting

Memory checkpoints record process memory and BC-owned components:

```text
active_builder_bytes
thread_workspace_bytes
source_loaded_payload_bytes
source_loaded_allocated_bytes
source_reader_metadata_bytes
range_work_bytes
pass_cache_bytes
store_static_metadata_bytes
store_allocated_bytes
position_writer_bytes
finalized_payload_bytes
external_staging_bytes
released_builder_bytes_total
last_release_batch_builder_bytes
```

`residual_bytes` is:

```text
process working set - accounted bytes
```

`baseline_adjusted_residual_bytes` subtracts the process baseline. Windows heap retention can keep released pages in the process working set, so `released_builder_bytes_total` is tracked separately from live BC objects.

---

## 14. Layer Compatibility Rules

A `.bcpos` file is compatible with the current Family generation path only if:

```text
header.format_version == 2
header.key_mode matches current BC key mode
header.rank_prefix_bits == 256
header.rank_prefix_type == uint16
header.rank_payload_align == 8
all boards represented by the file have raw sum == header.layer_sum
the file physical size is either exact logical size or has direct-IO padding < 4096
```

Converted or generated layer sets must be internally consistent:

```text
source2.header.layer_sum + 2 == target.layer_sum
source4.header.layer_sum + 4 == target.layer_sum
family_unit and partition policy agree across the run
legal tile alphabet agrees with the target rank
```

For converted high-rank layers, conversion validation should assert:

```text
decoded board raw sum equals target raw layer sum
rank15 tile count and expected delta are consistent with the conversion contract
row count and payload hash match the target reference when available
```

---

## 15. Current Implementation Entry Points

Primary generation API:

```cpp
BCFamilyGenerationStats generate_family_position_layer_v1(
    const BCLut& lut,
    const BCFamilyTable& target_axis,
    const BCFamilyStreamingGenerationSource* source4,
    const BCFamilyStreamingGenerationSource& source2,
    BCFamilyMutableStore& target_store,
    BCFamilyPositionWriter& position_writer,
    const BCFamilyGenerationOptions& options
);
```

Benchmark driver:

```text
native_core/tests_src/bench_bc_family_generation.cpp
```

Important options:

```text
--family-modulus
--family-blob buffered|direct
--family-position-io buffered|direct-rank-first
--family-source-io buffered|direct|direct-auto
--target-direct-queue-depth
--family-memory-checkpoints
--single-layer-sum
--source2-file
--source4-file
```

Core implementation files:

```text
native_core/include/BCLut.h
native_core/include/BCKeyRank.h
native_core/include/BCFamilyPartitionPolicy.h
native_core/include/BCFamilyGenerationScheduler.h
native_core/include/BCFamilyMutableStore.h
native_core/include/BCCellMutableBuilder.h
native_core/include/BCFamilyPositionWriter.h
native_core/include/BCPositionFile.h
native_core/include/BCPositionCellLoader.h
native_core/src/BCFamilyGeneration.cpp
```
