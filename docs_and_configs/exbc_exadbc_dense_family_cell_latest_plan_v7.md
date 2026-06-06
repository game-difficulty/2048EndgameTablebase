# EXBC / EXADBC 最新设计计划 v7：Dense Family-Cell Matrix、1-Exact Q4 Key、三链路与 Source-Family 顺序 IO

本文档记录当前敲定的 EXBC / EXADBC 设计。v7 在 v6 基础上去除了生产路径中的 `CellUseSchedule`，修正了 near-center 时 `T_delta` 的 side-normalization 公式，并进一步明确 FamilyCoord / FamilyId、generation blob、cell-granular future cache、cell-local partial 与文件 IO 顺序。

本文只描述最新计划，不保留旧 key mode、旧 target-family 外层、旧 per-cell dump 文件、pending-producer 生命周期、CellUseSchedule 生产结构、rectangle planner、source-major 主路径、离线 OR merge 等已废弃内容。

核心目标：

```text
小表 / 中表：保持简单高速路径；
大表：使用 family 顺序双分块，避免 future success 与 target mutable cells 全量驻留；
三条链路最终输出同一套 position file 与 success file 格式。
```

---

## 1. 总体架构与三条链路

### 1.1 目标问题

经典算法按 layer 存储全部局面，并在回算时将 future success matrix 常驻内存。随着 table 放大，future success matrix、生成阶段 target mutable cells、以及 FamilyChain solve 的中间 partial 都可能无法全量驻留。

本设计使用：

```text
Dense family-cell matrix
+ cell-local key -> bitmap
+ compact dense success rows
```

并提供三条计算链路：

```text
ResidentChain      完全不分块，所有必要数据内存驻留
SingleChunkChain   单分块，+2/+4 分开，current layer 按 cell chunk 分块
FamilyChain        双分块，按 source family 顺序进行 family sweep
```

### 1.2 链路选择原则

链路按简单到复杂选择：

```text
若完全驻留可行：使用 ResidentChain
否则若单分块可行：使用 SingleChunkChain
否则：使用 FamilyChain
```

链路选择是 per layer / per operation 的。同一 layer 的 generation 与 solve 可以选择不同链路。

### 1.3 三条链路差异

| 项目 | ResidentChain | SingleChunkChain | FamilyChain |
|---|---|---|---|
| 调度单位 | cell / cell range | current cell chunk | source family `f`，按 FamilyCoord 升序 |
| +2/+4 | 可一起或分开 | 分开 | 分开，固定 +4 phase 后 +2 phase |
| current layer | 全驻留 | current 分块 | 按 source family 的 row/col stripe 扫描 |
| future success | 全驻留 | 单个 delta 的 future success 驻留 | finalized future cells 按 NeedFutureCells 加载与缓存 |
| target mutable | 全驻留 | 全驻留 | MutableCell headers 常驻，CellBuilder resident/dumped/reloaded |
| 四方向 | 同 board 内同时算 | 同 board 内同时算 | row event / col event 分散到 family pass |
| partial max | 不需要 | 不需要 | solve 需要 cell-local wavefront partial |
| generation dump | 不需要 | 不需要 | 单 append-only generation blob |
| finalize 顺序 | 通常 cid 顺序 | 通常 cid / chunk 顺序 | +2 phase 的 family wavefront L 边界 |
| 最终文件格式 | 相同 | 相同 | 相同 |

非双分块链路没有 future fanout 问题，因此调度单位是 current cell 或 current cell chunk，四方向可在同一 board 局部同时计算，不需要 partial max。

双分块链路为了避免 future view 和 target mutable cells 全量驻留，必须按 source family 顺序推进；因此 solve 需要 partial，generation 需要 mutable dump / reload / retain。

---

## 2. 4x4 Board、四象限与 2x2 Coarse Board 视角

### 2.1 Board 布局

4x4 board 用 `uint64_t` 表示，每个 tile 4 bit exponent。

坐标：

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

### 2.2 2x2 Coarse Board

四象限划分本质上把 4x4 board 投影为一个 2x2 coarse board：

```text
NW  NE
SW  SE
```

family 是这个 coarse 2x2 board 的 row/col half-sum 标准化；cell 是 coarse board 的 row-family × col-family 格子。

这个视角解释了 FamilyChain 的核心形状：

```text
一次 source family 调度会触达一个或两个 target family 的“十字”：
    V(g) = row g ∪ col g

+2 phase 中每个 source family boundary 推进后，能 finalize 的 target cells 是
左上已完成子正方形的新 L 型边界：
    row boundary, columns <= boundary
    col boundary, rows    <  boundary

solve 中每个 source family boundary 推进后，能 finalize 当前 delta contribution 的 current cells
也具有同样的 L 型边界。
```

### 2.3 为什么 FamilyChain solve 需要 partial

如果所有 future success 在内存中，同一 board 的四个方向可同时查得：

```text
best = max(left, right, up, down)
```

此时不需要 partial。

FamilyChain 中，horizontal event 与 vertical event 被 source-family 顺序调度拆开：

```text
cell(r,c):
    horizontal event 在 source row-family r 处理
    vertical event   在 source col-family c 处理
```

若 `r != c`，两者发生在不同 pass。因此需要 cell-local partial 保存跨 pass 的中间 max：

```text
partial[board, empty, lane] = max(已处理方向的 success)
```

当 `max(r,c)` 对应的第二个 event 完成后，该 cell 的 partial 才能 finalize 成该 delta 的 contribution。

---

## 3. 1-Exact + 3 Sum&Mask Q4 Bucket Key

### 3.1 主 Key Mode

当前主 key mode：

```text
NW exact + NE/SW/SE sum_id+empty_mask
```

64-bit key 布局：

```text
[63:48] exact(NW)
[47:32] NE sum_id + NE empty_mask
[31:16] SW sum_id + SW empty_mask
[15: 0] SE sum_id + SE empty_mask
```

每个 `sum_id + empty_mask` 占 16 bit：

```text
[15:4] sum_id
[ 3:0] empty_mask
```

`sum_id` 是 compact LUT id，不是 raw sum。若某 target 的 `sum_id` 不能放入 12 bit，该 key mode 对该 target 不适用，应在 LUT 构建阶段报错。

默认 exact quadrant 是 canonical board 的 NW quadrant。该选择来自统计验证。key mode 对 canonical output board 通用：

```text
final_board = canonicalize(board, target_canonical_mode)
encode(final_board)
```

canonical mode 可以是 full-D4，也可以是更窄标准化。key mode 只要求有确定的 final representative board。

### 3.2 Rank 公式

对三个非 exact quadrant：

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

固定 `sum_id + empty_mask` 的 2x2 word 最多 36 种可能，因此：

```text
countNE <= 36
countSW <= 36
countSE <= 36
bitmap_len <= 36^3 = 46,656 < 65,536
```

所以：

```cpp
using BucketRank = uint16_t;
using BucketBitmapLen = uint16_t;
```

中间乘法使用 `uint32_t`，最终断言后 cast 到 `uint16_t`。

### 3.3 BCBucketEntry

bucket entry 保持 16B：

```cpp
struct BCBucketEntry {
    uint64_t key;

    // cell-local byte offset into rank payload stream
    uint32_t rank_payload_offset;

    // cell-local dense success row offset
    uint32_t success_row_offset;
};
```

不再使用旧的 `bitmap_bit_offset`。`rank_payload_offset` 是 cell-local byte offset，指向该 bucket 的 rank payload 起点。

### 3.4 Rank Payload 布局

每个 bucket 的 rank payload：

```text
[prefix uint16 array][padding to 8-byte alignment][bitmap uint64 words]
```

定义：

```text
prefix_block_bits = 256
prefix_count = ceil(bitmap_len / 256)
bitmap_word_count = ceil(bitmap_len / 64)
```

每个 prefix 存该 256-bit block 之前已置位的 bit 数：

```text
prefix[block] = popcount(bitmap bits before block)
```

prefix 类型：

```cpp
using RankPrefix = uint16_t;
```

因为 bucket 内最多 46,656 bit，prefix 值也小于 65,536。

prefix 体积约为 bitmap 体积的 1/16：

```text
每 256 bit bitmap = 32 bytes；对应 prefix = 2 bytes；
2 / 32 = 1 / 16。
```

### 3.5 Rank-to-Row Lookup

lookup 时：

```cpp
block = rank >> 8;   // rank / 256
off   = rank & 255;  // rank % 256
```

先 test bit：

```cpp
exists = bitmap_test(bitmap, rank);
if (!exists) miss;
```

若存在：

```text
count_before =
    prefix[block]
  + popcount(bitmap words within this 256-bit block before rank)

local_success_row =
    bucket.success_row_offset + count_before
```

每次最多 popcount 4 个 `uint64_t` word。

### 3.6 Cell Descriptor

主 cell descriptor 使用 rank payload，而不是旧 bitmap stream 命名：

```cpp
struct FinalCellDescriptorLogical {
    uint32_t bucket_count;
    uint32_t success_rows;

    uint64_t bucket_meta_offset;
    uint64_t rank_payload_offset;
    uint64_t rank_payload_bytes;

    uint32_t reserved0; // 第一版必须为 0
};
```

on-disk descriptor 必须使用固定 serialized layout，不依赖 C++ struct padding。推荐序列化顺序：

```text
u32 bucket_count
u32 success_rows
u64 bucket_meta_offset
u64 rank_payload_offset
u64 rank_payload_bytes
u32 reserved0
u32 descriptor_flags_or_padding
```

其中：

```text
bucket_meta_offset / rank_payload_offset 是 position file 中的 global uint64 offset；
rank_payload_bytes 是该 cell 的 rank payload 总字节数；
success_rows 是该 cell 的 dense success row 数；
reserved0 第一版固定写 0，不表示全局 success offset。
```

success file 的 cell offset 由 `success_rows` 做 prefix sum，或由 `BCSuccessIO` 构造独立 uint64 offset table。

### 3.7 Generation 中不存 Prefix

mutable CellBuilder 内只需要：

```text
key -> bitmap
```

不需要 rank prefix。prefix 只在 cell finalize 时生成，供回算 lookup 使用。

---

## 4. BCLUT 设计

### 4.1 目标

BCLUT 让 hot path 的 board encode / key/rank / bitmap length 计算以 table lookup 为主，无堆分配、低分支。

### 4.2 必要 LUT

```text
sum4_id[raw_sum] -> compact sum id
sum4_value[sum_id] -> raw_sum
word_desc[65536]
count4[sum_count][16]
offset4[sum_count][16]
sum4_words[]
legal_tile_alphabet
move LUT
canonical transform helpers
```

`word_desc[65536]` 推荐为 32-bit packed descriptor：

```text
bits 0..15   rank_in_sum_empty
bits 16..19  empty_mask
bits 20..31  sum_id
```

非法 4-cell word 的 descriptor 标记 invalid。

### 4.3 Legal Tile Alphabet 与 Sum 语义

LUT 必须由 target/table 的合法 tile alphabet 构造，不能默认 `{0..15}` 全合法。

实现中需要区分：

```text
rank_sum_value(tile)
family_sum_value(tile)
layer_sum_value(tile)
```

若三者相同，可共用。若 table 有 sentinel / cap 语义，必须通过 target abstraction 明确给出。

---

## 5. FamilyCoord、FamilyId、Cell 与 Dense Matrix

### 5.1 Half Sum

```text
row_top    = sum(NW) + sum(NE)
row_bottom = sum(SW) + sum(SE)
col_left   = sum(NW) + sum(SW)
col_right  = sum(NE) + sum(SE)
```

一个 layer 中：

```text
row_top + row_bottom = S
col_left + col_right = S
```

### 5.2 Side-Normalized Family

family 由较小 half sum 表示：

```text
family_raw = min(half_sum, S - half_sum)
```

定义 family coordinate：

```text
FamilyCoord = family_raw / family_unit
```

标准 2048 value-sum 语义下通常：

```text
family_unit = 2
spawn2_delta_coord = 1
spawn4_delta_coord = 2
```

### 5.3 FamilyCoord 与 FamilyId

本设计同时使用两种语义，不能混用：

```text
FamilyCoord / SumCoord：
    normalized half-sum / family_unit。
    它表达 sum 语义，跨 layer 可比较，参与 T_delta 归一化计算、调度顺序和 wavefront boundary。

FamilyId / AxisIndex：
    某个 layer 内 dense matrix 的连续下标，范围 0..F-1。
    它表达 storage 语义，用于 cell_id = row_id * F + col_id。
```

每个 layer 保存自己的 `FamilyTable`：

```cpp
struct FamilyTable {
    uint16_t family_unit;
    uint16_t axis_base_coord;  // FamilyId 0 对应的 FamilyCoord
    uint16_t family_count;     // F

    FamilyCoord id_to_coord(FamilyId id) const;
    FamilyId coord_to_id(FamilyCoord coord) const;
    bool contains_coord(FamilyCoord coord) const;
};
```

当前版本要求 theory axis 在 FamilyCoord 上连续。构建期必须 assert：

```text
id_to_coord[i+1] == id_to_coord[i] + 1；
coord_to_id(axis_base_coord + i) == i；
边界越过 target small-side 上界可合法跳过；
中间缺口默认报错。
```

不同 layer 的 `axis_base_coord` / `family_count` 可能不同。跨 layer 不能把 source layer 的 `FamilyId` 直接当成 target layer 的 `FamilyId`。必须通过：

```text
source FamilyId -> source FamilyCoord -> target FamilyCoord -> target FamilyId
```

### 5.4 Board 所属 Family

```text
R = family_id(row_top, S)
C = family_id(col_left, S)
```

board 属于：

```text
row_family R
col_family C
```

### 5.5 Cell

cell 是有序 pair：

```text
cell(R, C)
```

含义：

```text
所有 row_family == R 且 col_family == C 的 buckets / rank payload / success rows
```

### 5.6 Dense Matrix

```cpp
using FamilyId = uint16_t;  // layer-local dense axis index
using CellId   = uint32_t;

inline CellId cell_id(FamilyId r, FamilyId c, uint32_t F) {
    return uint32_t(r) * F + uint32_t(c);
}
```

`cell_id` 永远是 **layer-local** 的：

```text
generation:
    source cells 使用 source layer 的 FamilyTable / F / cid；
    target mutable cells 使用 target layer 的 FamilyTable / F / cid。

solve:
    current cells / partial 使用 current layer 的 FamilyTable / F / cid；
    future loaded cells 使用 future layer 的 FamilyTable / F / cid。
```

因此不同 layer 的 `cid` 不能互相复用。跨 layer 只通过 `FamilyCoord` 和各自 layer 的 `coord_to_id` 转换。

### 5.7 Theory Axis

生成 target layer 前，基于目标 layer sum `S` 构造理论 family axis：

```text
axis = {
    family_id(h, S)
    | h in possible_8tile_sums
    | S - h in possible_8tile_sums
}
```

`possible_8tile_sums` 由 8 个 tile 的合法取值 DP / enumeration 得到。

真实统计显示 theory axis 对 actual families 冗余很小，dense matrix 冗余可接受：

```text
layer          boards       actual families  actual cells  theory axis  matrix cells  matrix/cells  density
free10_512_70  571,484,170  191              25,808        191          36,481        1.414x        70.74%
free10_512_75  632,552,570  204              27,985        204          41,616        1.487x        67.25%
free10_512_79  687,251,413  209              30,255        210          44,100        1.458x        68.61%
free10_512_80  691,807,863  211              30,926        211          44,521        1.440x        69.46%
free10_512_81  692,436,071  211              29,769        213          45,369        1.524x        65.62%
free10_512_82  718,205,735  213              32,463        213          45,369        1.398x        71.55%
```

---

## 6. Family View、Fanout 与 Source-Family Sweep

### 6.1 Family View

一个 family `G` 的 view 是 dense matrix 中一行加一列：

```text
V(G) = row G ∪ col G
```

即：

```text
V(G) = { cell(G, x) | x in axis }
     ∪ { cell(x, G) | x in axis }
```

对角 cell `cell(G,G)` 只出现一次。

### 6.2 Canonical 语义

key mode 只编码 canonicalize 后的 final representative board：

```text
final_board = canonicalize(board, target_canonical_mode)
encode(final_board)
```

full-D4 影响的是 family view 是否必须 row/col union，以及 canonical 后 NW exact 的统计效果；不影响 key mode correctness。

### 6.3 正确的 Source Family 到 Target Families 公式

这是调度正确性的核心。不能简单写成 `{coord, coord + d}`，因为 near-center 时需要重新做 side-normalization。

设 source layer total coord 为：

```text
N = source_layer_total_sum / family_unit
```

source family coord 为：

```text
a = source_coord
b = N - a
```

由于 family 是 side-normalized：

```text
a <= b
```

spawn delta 为 `d`：

```text
spawn2: d = 1
spawn4: d = 2
```

两种 spawn side：

```text
spawn 到 large / opposite side:
    (a, b + d) -> normalized target coord = a

spawn 到 current small side:
    (a + d, b) -> normalized target coord = min(a + d, b)
```

因此正确公式：

```text
T_delta(a, N) = unique({ a, min(a + d, N - a) })
```

示例：

```text
N = 11, a = 5, b = 6, spawn4 d = 2
small side spawn: (5+2, 6) = (7,6)
normalized target coord = 6
T_4(5,11) = {5,6}
```

如果错误写成 `{5,7}` 并因 7 超过 target small-side 上界而跳过，就会漏掉 target coord 6。

实现必须统一调用：

```cpp
SmallVector<FamilyId, 2> map_source_family_to_target_families(
    const FamilyTable& source_table,
    const FamilyTable& target_table,
    FamilyId source_id,
    SpawnDeltaCoord d
);
```

伪代码：

```cpp
FamilyCoord a = source_table.id_to_coord(source_id);
FamilyCoord N = source_table.total_coord();
FamilyCoord b = N - a;

FamilyCoord target0 = a;
FamilyCoord target1 = std::min<FamilyCoord>(a + d, b);

add_if_valid(target_table.coord_to_id(target0));
add_if_valid(target_table.coord_to_id(target1));
dedup_targets();
```

边界规则：

```text
先计算 target1 = min(a+d, b)，再检查 target axis 是否包含该 coord；
如果 target coord 在理论上应存在但 coord_to_id invalid，则报错；
不要用 “a+d 超过 target small-side 上界就跳过” 作为归一化逻辑。
```

### 6.4 Fanout 仍然不超过 2

修正后的：

```text
T_delta(a,N) = {a, min(a+d, N-a)}
```

仍然最多两个 target family。中心附近可能去重为一个。

并且：

```text
min(a+d, N-a) >= a
```

所以 target family 不会小于 source family。这保证了 +2 phase 的 wavefront finalize 逻辑仍然成立。

### 6.5 横移 / 竖移守恒量

横移 left/right 不改变 row half-sum，因此由 source row-family 控制。竖移 up/down 不改变 col half-sum，因此由 source col-family 控制。

处理 source family `f` 时：

```text
row source cells cell(f,x) -> left/right
col source cells cell(x,f) -> up/down
```

生成 / 回算中需要的 target future/mutable cells 是：

```text
NeedCells(f, delta) = union_{g in T_delta(f)} V(g)
```

其中 `T_delta` 必须使用 Section 6.3 的 side-normalized 公式。

### 6.6 调度变量

FamilyChain 外层变量是 source family `f`，按 FamilyCoord 升序：

```text
+4 phase:
    for source family f ascending

+2 phase:
    for source family f ascending
```

不要把 FamilyChain 外层写成 target family `G`。每个 pass 由：

```text
pass = (delta, source_family f)
```

确定。

### 6.7 NeedCells 必须去重

`NeedCells(pass)` 是一个去重后的 CellId 集合。这个去重逻辑是通用工具，不只用于 generation，也用于 solve future cell cache、mutable reload/dump scan、future-cell load、partial/finalize 边界枚举等路径。

实现推荐在 `BCDenseCellMatrix` 中提供：

```cpp
class BCDenseCellMatrix {
public:
    // Collect union_{g in family_ids} V(g), where V(g)=row g ∪ col g.
    // Output is deduplicated and layer-local.
    void collect_family_crosses(span<FamilyId> family_ids,
                                std::vector<CellId>& out);
};
```

内部可使用 dense epoch array：

```cpp
std::vector<uint32_t> mark(F * F, 0);
uint32_t epoch = 1;
std::vector<CellId> need_cells;

void add_need_cell(CellId cid) {
    if (mark[cid] == epoch) return;
    mark[cid] = epoch;
    need_cells.push_back(cid);
}
```

同一 pass 中，任意 cid 的 reload / retain / dump / finalize / future load 最多执行一次。

---

## 7. 文件格式：Position File、Success File 与 Header

### 7.1 Position File

```text
Header
FamilyTable
DenseCellDescriptor[F * F]
BucketMetaStream
RankPayloadStream
```

position file 不存 success data。

### 7.2 Success File

success data 写入独立 success file。bucket entry 的 `success_row_offset` 是 cell-local row offset。

第一版 position descriptor 中不保存全局 success offset，`reserved0` 必须为 0。success writer 用 cell descriptor 中的 `success_rows` 做 prefix sum 或自定义布局，构造 `uint64_t` cell offset table。

```text
success_file_offset[cid] = prefix_sum(success_rows before cid) * row_width * dtype_bytes
```

如果后续需要其他 success file layout，应通过 header/version 增加明确字段，不复用 `reserved0` 的模糊语义。

### 7.3 Header 必须记录的字段

文件 header 至少包含：

```cpp
struct BCFileHeader {
    uint32_t magic;
    uint16_t format_version;

    uint8_t  key_mode;              // Q4_NW_EXACT_NE_SW_SE_SUM_MASK_PREFIX256
    uint16_t rank_prefix_bits;      // 256
    uint8_t  rank_prefix_type;      // uint16
    uint8_t  rank_payload_align;    // 8

    uint32_t layer_sum;
    uint16_t family_unit;           // e.g. 2
    uint16_t axis_base_coord;       // FamilyCoord of FamilyId 0
    uint16_t family_count;

    uint8_t  canonical_mode;
    uint8_t  success_dtype;
    uint16_t row_width;
};
```

读取时必须校验 header，不允许用旧 key mode 解释新文件。`axis_base_coord` 与 `family_count` 定义本 layer 的 dense matrix axis，跨 layer 调度必须通过 FamilyCoord 换算。

### 7.4 Finalize 与 Offset 分配

主口径为 batch finalize：

```text
1. cell-local finalize 生成 bucket metadata / rank payload，并计算输出大小；
2. writer 对本批待写 cell 做 offset assignment；
3. pwrite 到 position file；
4. 回填 descriptor[cid]。
```

offset assignment 推荐对待写 cell 做 prefix sum。pwrite 可顺序也可并行；瓶颈通常是 IO 带宽，实现可根据平台决定。

本文中“append-order”指逻辑上 cell 按完成顺序进入 writer batch，不表示每个 finalize worker 直接 fetch_add 并写文件。第一版推荐先 measure size，再分配 offset，再写，以保证可复现和易测试。

---

## 8. ResidentChain：完全不分块

### 8.1 Generation

target mutable 全驻留，不使用 dump。

```text
load source layer for +4
load source layer for +2
initialize target MutableCell[F*F]

for +4 source boards:
    spawn4
    move left/right/up/down
    canonicalize
    encode -> cid,key,rank
    insert target CellBuilder

for +2 source boards:
    spawn2
    move left/right/up/down
    canonicalize
    encode -> cid,key,rank
    insert target CellBuilder

finalize all nonempty cells
write position file
```

调度单位：cell / cell range。四方向同 board 内同时处理。

### 8.2 Solve

future success 全驻留。

```text
for current board:
    for empty:
        spawn4 -> compute 4 directions -> best4
        spawn2 -> compute 4 directions -> best2
    result = 0.1 * avg(best4) + 0.9 * avg(best2)

write success file
```

不需要 partial。

---

## 9. SingleChunkChain：单分块

### 9.1 Generation

target mutable 全驻留，current source 按 cell chunk 读。

```text
initialize target MutableCell[F*F]

for delta in {spawn4, spawn2}:
    for source current cell chunk:
        read chunk
        generate 4 directions
        canonicalize
        encode
        insert target CellBuilder

finalize all nonempty cells
write position file
```

无 dump，无 partial。

### 9.2 Solve

每个 delta 的 future success 可驻留，current layer 分 chunk。

```text
for current chunk:
    result buffer = 0

    load / ensure future +4 resident
    compute spawn4 four-direction best
    write 0.1 contribution to result buffer or result_accum

    load / ensure future +2 resident
    compute spawn2 four-direction best
    add 0.9 contribution
    write final success chunk
```

无 partial。`result_accum` 只是 delta 间累计，不是 direction partial。

---

## 10. FamilyChain Generation：双分块生成

### 10.1 Phase 顺序

固定顺序：

```text
+4 phase: source family f 从小到大
+2 phase: source family f 从小到大
```

+4 phase 永不 final finalize。+2 phase 是最后 contributor，负责所有 final finalize。

### 10.2 Pass 定义

```text
pass = (delta, source_family f)
Gset = T_delta(f)
NeedCells(pass) = dedup union_{g in Gset} V(g)
```

其中 `T_delta` 必须使用 Section 6.3 的公式，不允许写死 `{f, f+d}`。

### 10.3 不使用 CellUseSchedule

生产实现不维护 per-cell `CellUseSchedule`，也不使用 `MaxCellUses` 小数组。该结构已被以下公式化逻辑取代：

```text
当前 pass 是否需要 cell：
    cid in NeedCells(pass)

cell 是否应 final finalize：
    +2 phase 的 boundary sweep：prev_boundary < max(row_coord, col_coord) <= current_boundary

cell 是否应保留 resident：
    根据未来若干 pass 的 T_delta 与 V(g) 即时判断 next use 是否接近
```

也就是说，generation 生命周期由：

```text
phase(+4/+2)
source family coordinate f
NeedCells(delta,f)
+2 finalize boundary
retain window / memory policy
```

共同驱动。没有额外的 per-cell use schedule 数据结构。

### 10.4 单 Append-Only Generation Blob

FamilyChain generation 使用单个 append-only blob：

```text
mutable_generation.blob
```

每个 cell header 常驻。builder 指针必须具备并发保护，因为同一个 pass 内多个 worker 可能同时第一次向同一 target cell 插入 candidate。

```cpp
enum class CellState : uint8_t {
    Empty,
    Resident,
    Dumped,
    Finalized
};

struct MutableCell {
    CellState state;

    // Resident 时有效。首次创建 / reload / release 需要 CAS 或 cell-local lock 保护。
    std::atomic<CellBuilder*> builder;

    uint16_t dump_generation;
    uint64_t dump_offset;
    uint64_t dump_bytes;
    uint32_t dump_version;

    bool dirty_since_load;
};
```

说明：

```text
dump_generation 表示该 cell 最新 dump record 的代际 / version；
dump_offset / dump_bytes 指向 mutable_generation.blob 中的 latest dump record；
同一个 cell 可以在 blob 中有多个旧 record，但 metadata 只指向最新 record；
旧 record 不原地修改，layer 完成后删除整个 blob。
```

状态转移表：

| 当前状态 | 事件 | 动作 | 新状态 |
|---|---|---|---|
| Empty | candidate insert | 线程安全创建 CellBuilder，插入 key/rank，`dirty=true` | Resident |
| Dumped | pass 需要该 cell | 读取 latest dump record，恢复 builder，`dirty=false` | Resident |
| Resident | candidate insert | 插入 key/rank，`dirty=true` | Resident |
| Resident | release 且 `dirty=false` | 释放 builder，保留旧 dump pointer | Dumped |
| Resident | release 且 `dirty=true` | append 新 dump record，更新 dump pointer，释放 builder | Dumped |
| Resident/Dumped/Empty | final boundary | 若 Dumped 先 reload；若 Empty 写 empty descriptor；否则 final emit | Finalized |

`get_or_create_builder(cid)` 必须线程安全，可用 atomic CAS 或 cell-local lock 实现。

### 10.5 CellBuilder Dump Record

dump 是 CellBuilder 的可恢复序列化状态，不是 final position format，也不是 raw memcpy C++ 对象。

推荐 header：

```cpp
struct CellBuilderDumpHeader {
    uint32_t magic;          // e.g. "BCDG"
    uint16_t version;
    uint16_t key_mode;

    uint32_t cid;
    uint16_t dump_generation;
    uint16_t flags;

    uint32_t hash_slots;
    uint32_t used_keys;

    uint64_t payload_bytes;
    uint64_t checksum;
};
```

payload 至少包含：

```text
hash control array
key entries
mutable bucket metadata
bitmap arena block table
bitmap arena raw bytes
aux arena data if needed
```

checksum 覆盖 header 中除 checksum 外的固定字段以及 payload。reload 时必须校验：

```text
magic / version / key_mode / cid / dump_generation / payload_bytes / checksum
```

dump 中不能保存裸指针。若 CellBuilder 内有 arena block pointer / bitmap pointer / hash entry pointer，dump 中应保存 block id / offset / length，reload 后修复运行时指针。

dump 时必须 freeze builder，不允许并发插入与扩容同时发生。dump 只在 pass barrier 后发生。

### 10.6 Pass IO 顺序

对每个 pass `(delta, source_family f)`：

#### Step 1：计算 NeedCells

```text
Gset = T_delta(f)
NeedCells = dedup union_{g in Gset} V(g)
```

`NeedCells` 使用 target layer 的 FamilyTable / F / cid。

#### Step 2：Reload

```text
ReloadCells = { cid in NeedCells | state == Dumped }
```

按 `dump_offset` 排序，批量读 `mutable_generation.blob`，并恢复 CellBuilder。已经 Resident 的 cell 直接复用，无 IO；Empty cell 不提前创建，只有实际 candidate 插入时才创建 builder。

#### Step 3：读取 source cells

读取 source position file 中：

```text
row source cells: cell(f,x) -> left/right
col source cells: cell(x,f) -> up/down
```

这里的 source `cell(f,x)` / `cell(x,f)` 使用 source layer 的 FamilyTable / F / cid。按 source file offset 排序 coalesce read。

#### Step 4：生成并插入

```text
spawn delta
move
compute physical target family
if physical target family not in Gset:
    skip

canonicalize
encode -> target cid,key,rank
get_or_create CellBuilder // thread-safe
insert key/rank
dirty_since_load = true
```

#### Step 5：Finalize Boundary

只在 +2 phase 做 final finalize。+4 phase 只生成 mutable state，不 final finalize。

令 `prev_boundary` 是上一次已完成的 +2 finalize boundary，当前完成的 source family coordinate 为 `f_coord`。处理完当前 source family 后，推进：

```text
prev_boundary < max(row_coord, col_coord) <= f_coord
```

对应的 target cells 全部 final finalize。若 family axis 连续且每步推进 1，新增边界为一个 L 型：

```text
FinalizeCells(f) =
    { cell(f, x) | x <= f }
  ∪ { cell(x, f) | x <  f }
```

若 +2 phase 实际 source family 有跳跃，则对每个缺失 boundary 补扫：

```text
for b in (prev_boundary+1)..f_coord:
    finalize cells with max(row_coord,col_coord)==b
```

+2 phase 结束后，必须将 `prev_boundary` 推进到 `target_axis_max_coord`，确保 +4-only cell 或没有 +2 source work 的高 boundary cell 也被 final finalize。

如果被 finalize 的 cell 是 Dumped，先 reload；如果是 Empty，写 empty descriptor；如果是 Resident，直接 final emit。

#### Step 6：Retain / Dump

对未 finalized 的 Resident mutable cells：

```text
if cell will be needed by near-future passes and memory allows:
    keep resident
else:
    if dirty_since_load or no valid dump:
        append dump record to mutable_generation.blob
        update dump pointer
    release builder
    state = Dumped
```

near-future 判断不依赖 CellUseSchedule，而是根据未来若干 pass 的 `T_delta(future_f)` 计算：

```text
cell(r,c) will be needed iff r in Gset_future or c in Gset_future
```

如果内存压力仍然较高，优先 dump：

```text
next use 最远、builder 最大的 cells。
```

### 10.7 +4 到 +2 的过渡

+4 phase 结束时，不 final finalize。进入 +2 phase 前，根据 +2 初始 `NeedCells` 和内存水位保留即将使用的 resident cells，其余按 generation blob dump。

---

## 11. FamilyChain Solve：双分块回算

### 11.1 Phase 顺序

```text
+4 solve pass
+2 solve pass
```

+4 写 `result_accum.tmp`。+2 读取 `result_accum.tmp`，写 final success file。

### 11.2 Future Cell Cache

future cache 是 cell-granular，不是 view-granular。`V(g)` 只是一个 CellId set，不拥有 LoadedCell。

每个 pass：

```text
Gset = T_delta(f)
NeedFutureCells = dedup union_{g in Gset} V(g)
```

对每个 future cell：

```text
if LoadedFutureCell[cid] already exists:
    reuse
else:
    load finalized cell from future position/success files
```

硬不变量：

```text
同一个 finalized future cell 在内存中最多加载一份。
```

释放粒度也是 cell，不是 view。对 future cell 预计算或即时维护 `last_use_pass[cid]`。pass 结束后：

```text
if current_pass >= last_use_pass[cid]:
    release this LoadedFutureCell
else:
    keep if memory allows
```

内存不足时可以提前 evict read-only future cell；后续 reload 即可。提前 evict 是性能退化，不影响 correctness。

### 11.3 Partial Cell Lifecycle

对 current cell `cell(r,c)`，在一个 delta 内：

```text
first_use = min(r,c)
last_use  = max(r,c)
```

当 `f = first_use`，写入第一半方向的 max。当 `f = last_use`，合并另一半方向并 finalize 该 delta contribution。

对角 cell `cell(f,f)` 的 row/col event 同 pass 完成，通常不需要落盘 partial。

### 11.4 Partial Store

partial 是 cell-local dense block：

```text
partial block for current cell
size = sum_empty_slots(cell) * row_width * sizeof(PartialT)
```

partial state：

```cpp
struct PartialCellState {
    PartialState state;     // None / Resident / Dumped / Finalized
    void* resident_ptr;
    uint64_t dump_offset;
    uint32_t dump_bytes;
    bool dirty_since_load;  // 可选，但保留以统一语义
};
```

partial 临时文件：

```text
partial.tmp
```

+4 与 +2 复用同一个 partial file。delta 切换时清空 / 重用。

第一版约束：每个 current cell 每个 delta 的 partial block 最多 dump 一次、reload 一次。因为一个 cell 的 partial 只有：

```text
first_use = min(row_family, col_family)
last_use  = max(row_family, col_family)
```

两个方向事件。若 `first_use == last_use`，即对角 cell，通常不写 partial.tmp。

### 11.5 Result Accumulator

`+4` pass finalizes current cell 后，将：

```text
0.1 * avg4
```

写入：

```text
result_accum.tmp
```

`+2` pass finalizes same current cell 后，读取 result_accum，计算：

```text
final = result_accum + 0.9 * avg2
```

写 final success file。

result_accum 使用 current-cell descriptor：

```cpp
struct ResultAccumCellDesc {
    uint64_t offset;
    uint32_t rows;
    uint32_t bytes;
};
```

注意 result_accum block 与 partial block 大小不同：

```text
result_accum_bytes(cell) = success_rows(cell) * row_width * sizeof(AccumT)
partial_bytes(cell)      = sum_empty_slots(cell) * row_width * sizeof(PartialT)
```

`ResultAccumCellDesc` 应由 current layer finalized position metadata / success_rows 计算，不能复用 PartialCellState 的 offset。

### 11.6 Row / Col 更新互斥

FamilyChain solve 中同一个 current cell 的 partial block 同一时刻只能有一个 writer。

实现推荐每个 source family pass 分两阶段：

```text
Stage A:
    并行计算所有 row source cells，执行 left/right。

barrier

Stage B:
    并行计算所有 col source cells，执行 up/down。

barrier
```

这样即使是对角 cell `cell(f,f)`，row 和 col 更新也不会并发。

对角 cell `cell(f,f)` 的横向和竖向都在同一个 pass 内完成；可以使用 pass-local buffer 直接计算该 delta contribution，不写 partial.tmp。若为了实现统一而进入 partial path，也必须保证 row/col 两阶段串行。

### 11.7 Partial Retain / Dump

处理完 source family `f` 后，新产生的 incomplete partial cells：

```text
cell(f,x), x > f
cell(x,f), x > f
```

这些 cell 的 next use 是 `x`。

若：

```text
x <= f + retain_window
```

且内存允许，保留 resident，不写盘。否则 dump 到 `partial.tmp`。

处理 source family `f` 时，需要 finalize 的 cells：

```text
cell(f,x), x < f
cell(x,f), x < f
```

若 partial resident，直接复用。若 dumped，按 dump offset 批量 read。

### 11.8 +4 Solve IO 顺序

对 source family `f`：

1. 计算 `Gset = T_4(f)`，使用 Section 6.3 的公式。
2. 确保 `NeedFutureCells` loaded / reused。
3. 读取 current source cells：
   ```text
   cell(f,x) -> left/right
   cell(x,f) -> up/down
   ```
4. 对 candidate lookup future success，更新 partial。
5. finalize `max(r,c)==f` 的 current cells：
   ```text
   read/reuse partial
   average over empty slots
   write 0.1 contribution to result_accum.tmp
   release partial
   ```
6. 按 next_use 策略 dump/retain incomplete partial。
7. release last-use future cells。

### 11.9 +2 Solve IO 顺序

对 source family `f`：

1. 计算 `Gset = T_2(f)`，使用 Section 6.3 的公式。
2. 确保 `NeedFutureCells` loaded / reused。
3. 读取 current source cells。
4. lookup future success，更新 partial。
5. finalize `max(r,c)==f` 的 current cells：
   ```text
   read/reuse partial
   average over empty slots
   read result_accum.tmp
   final = result_accum + 0.9 * avg2
   write final success file
   release partial
   ```
6. dump/retain incomplete partial。
7. release last-use future cells。

---

## 12. CellBuilder、空间预测与扩容

### 12.1 CellBuilder

CellBuilder 内部沿用 EX 已有并发 hashmap + bitmap 结构：

```text
key -> bitmap
bitmap word atomic OR
arena allocation
buffer / prefetch 降低访存延迟
```

新 board 插入：

```text
encode -> cid,key,rank
get_or_create CellBuilder
key lookup / create
bitmap word atomic OR
```

`get_or_create CellBuilder` 必须线程安全。因为同一个 pass 内多个 worker 可能同时第一次向同一 Empty cell 插入 candidate。推荐实现：

```cpp
CellBuilder* get_or_create_builder(MutableCell& cell, CellId cid) {
    CellBuilder* p = cell.builder.load(std::memory_order_acquire);
    if (p) return p;

    CellBuilder* q = allocate_builder_with_predicted_capacity(cid);
    CellBuilder* expected = nullptr;
    if (cell.builder.compare_exchange_strong(expected, q,
            std::memory_order_release,
            std::memory_order_acquire)) {
        cell.state = CellState::Resident;
        return q;
    }

    destroy_unused_builder(q);
    return expected;
}
```

如果实现中 reload / release 与 insert 可能交错，还需要 cell-local lock 或 state CAS。第一版推荐在 pass barrier 外进行 reload / dump / release，在 worker 阶段只允许 Resident builder 被插入；这样可将并发保护限制在首次创建与 builder 内部插入。

允许小 buffer 批量插入，但不做全量 append-first，不做离线 OR merge。

### 12.2 初始化容量预测

首次创建 target cell `(r,c)` 的 builder 时，使用 current layer 邻近 cell finalized 体积估计。

邻域：

```text
4-neighbor 或 3x3 neighborhood
```

估计：

```text
estimate = max(valid_neighbors)
initial_capacity = 2 * estimate
```

无有效邻居时使用默认小容量。

### 12.3 自动扩容

```text
hashmap load factor 超阈值 -> cell-local lock + rehash
bitmap arena 不足 -> 追加 arena block
```

扩容采用简单锁 + 数据搬运。扩容正确性优先，不要求无锁化。

---

## 13. Chain Planner

### 13.1 Planner 输入

```text
table config
layer id
source layer stats
target theory axis
available memory
available disk
success dtype
EXAD lane width
historical cell stats
previous layer actual stats
```

### 13.2 选择规则

```text
if ResidentChain estimated_peak <= budget:
    choose ResidentChain
else if SingleChunkChain estimated_peak <= budget:
    choose SingleChunkChain
else:
    choose FamilyChain
```

第一版可以允许手动指定链路，并记录实际 stats。自动预测后续根据真实层数据优化。

### 13.3 Runtime 回退

若实际超过 hard limit：

```text
ResidentChain -> restart layer with SingleChunkChain or FamilyChain
SingleChunkChain -> shrink current chunk; if target mutable still too large, restart with FamilyChain
FamilyChain -> dump more mutable cells, shrink retain window, lower parallelism
```

第一版 recovery 策略：

```text
若中断，删除该 layer 的 temp files 与未完成 output，整层重算。
```

---

## 14. Implementation Modules

### 14.1 BCFamilyTable

```text
construct theory axis
FamilyCoord / FamilyId ordering
axis_base_coord / family_count
id_to_coord / coord_to_id
T_delta(a,N) = {a, min(a+d, N-a)}
possible_8tile_sums
```

职责要求：

```text
1. 保证 layer 内 FamilyId 是 dense matrix 下标；
2. 保证 FamilyCoord 用于跨 layer调度和 +2/+4 算术；
3. 构建期校验 axis 连续性；
4. 提供 source FamilyId -> source coord -> target FamilyId 的转换接口；
5. 禁止在 source/target axis_base 不一致时直接把 source FamilyId 当 target FamilyId；
6. 禁止调用者私自写 {f,f+d}，必须统一调用 BCFamilyTable 的 T_delta 映射。
```

### 14.2 BCDenseCellMatrix

```text
cid(r,c)
FinalCellDescriptor[F*F]
MutableCell[F*F]
iterate V(g) = row g ∪ col g
iterate L boundary max(r,c)==f
NeedCells dedup by epoch array
```

### 14.3 BCLUT

```text
1-exact Q4 key/rank LUT
sum_id / empty_mask LUT
word_desc
count4
move LUT
canonical transform helpers
```

### 14.4 BCEncoder

```text
canonicalize board
board -> row_family / col_family / cid / key / uint16 rank
physical target family prefilter
```

### 14.5 BCCellBuilder

```text
EX-style hashmap + bitmap
concurrent insert
atomic bitmap OR
auto expand
generation dump / restore
finalize emit with prefix256 rank payload
```

### 14.6 BCPositionWriter

```text
bucket metadata write
rank payload write
descriptor table write / rewrite
position file checksum
```

### 14.7 BCSuccessIO

```text
success file read/write
cell-local row offset mapping
cell success offset prefix table
dtype encode/decode
```

### 14.8 BCFutureCellCache

```text
load finalized future cells
collect descriptors from NeedFutureCells
sort by file offset
coalesce reads
cache loaded cid -> cell span
release by cell-level last_use / memory policy
```

### 14.9 BCGenerationBlobIO

```text
single mutable_generation.blob append-only writer
batch read by offset
dump record checksum/version
CellBuilder restore
```

### 14.10 BCPartialStore

```text
cell-local partial blocks
partial.tmp append/read
resident partial cache
retain/dump by next_use
```

### 14.11 BCResultAccum

```text
result_accum.tmp
ResultAccumCellDesc
write +4 contribution
read during +2 finalize
```

### 14.12 ResidentGenerator / ResidentSolver

```text
cell scheduling
all data resident
no partial
no dump
```

### 14.13 SingleChunkGenerator / SingleChunkSolver

```text
current cell chunk scheduling
+2/+4 split
no partial
no dump
```

### 14.14 FamilyGenerator / FamilySolver

```text
source-family ordered sweep
generation blob dump/reload
future cell cache
cell-local partial
wavefront L-boundary finalize
```

---

## 15. Correctness Tests

### 15.1 Q4 / LUT / Rank Payload

```text
legal word maps to valid sum_id / empty_mask / rank
fixed sum+mask count <= 36
bitmap_len <= 46656
rank fits uint16
rank encode/decode roundtrip
prefix256 lookup equals naive popcount
rank_payload_offset / success_row_offset monotonic and valid
```

### 15.2 Family Axis / Dense Matrix

```text
theory axis contains all actual families
FamilyId ordering is coordinate-aligned and continuous where required
axis_base cross-layer mapping correct
T_delta near-center correctness: {a, min(a+d, N-a)}
T_delta never emits target < source
target coord maps to valid target FamilyId or known invalid boundary
cid(r,c) unique
actual boards encode into valid cid
V(g) == row g ∪ col g
L-boundary max(r,c)==f enumerates correct cells
NeedCells dedup correctness
```

### 15.3 Generation Lifecycle

```text
NeedCells constructed from T_delta and V(g), not from CellUseSchedule
reload occurs before use
+4 phase never final finalizes
+2 boundary finalizes all cells with prev_f < max(r,c) <= f
+2 final sweep reaches target_axis_max
state transitions of mutable_generation.blob are correct
dirty resident cell dump preserves data
non-dirty reload/release keeps old dump pointer
```

### 15.4 Generation Equivalence

```text
Resident / SingleChunk / Family generation produce identical position sets on small layers
CellBuilder dump/restore preserves all key/rank bits
no offline OR merge needed
finalized cell receives no later inserts
```

### 15.5 Solve Partial

```text
partial first_use=min(r,c), last_use=max(r,c)
resident partial reuse avoids unnecessary dump/read
disk partial reload equals resident partial
row stage and col stage do not concurrently update same partial block
Family solve result equals Resident solve on small layers
result_accum mapping correct
```

### 15.6 Future Cell Cache

```text
NeedFutureCells dedup correctness
same future cid loaded at most once
cell-granular release by last_use correct
lookup through loaded future cell equals cold point lookup
```

### 15.7 File Format

```text
position file reopen / descriptor roundtrip
header rejects incompatible key_mode / prefix bits
rank payload lookup equals full scan
success file row mapping correct
offset assignment descriptor valid
```

---

## 16. Performance Tests

### 16.1 Key Mode

```text
key_count
rank_payload_bytes
prefix_bytes
metadata_bytes
insert throughput
lookup throughput
rank lookup ns
LLC miss / lookup
finalize time
```

### 16.2 Matrix / View

```text
family_count
matrix_cells
nonempty_cells
matrix density
view cell count
NeedCells size
NeedCells dedup cost
future cell cache hit rate
coalesced range count
```

### 16.3 Generation

```text
candidate_count
prefilter_skip_count
canonicalize_count
insert_count
duplicate_bit_count
new_cell_count
new_bucket_count
hash_probe_count
hashmap_expand_count
arena_expand_count
generation_blob_read_bytes
generation_blob_write_bytes
resident_cell_reuse_count
dump_avoided_by_next_use
obsolete_dump_bytes_estimate
```

### 16.4 Solve

```text
future_cell_load_count_per_delta
future_cell_cache_hit
cell_read_multiplicity
future_actual_read_bytes
lookup_count
bucket_lookup_steps
bitmap_word_load_count
success_row_load_count
partial_update_count
partial_dump_bytes
partial_reload_bytes
partial_reuse_hit
result_accum_read_bytes
result_accum_write_bytes
```

### 16.5 Finalize / IO

```text
finalize_cell_count
finalize_seconds
bucket_sort_seconds
rank_payload_build_seconds
position_write_bytes
success_write_bytes
read/write bandwidth
offset_assignment_seconds
```

### 16.6 Planner

```text
estimated_peak_memory
actual_peak_memory
estimated_temp_disk
actual_temp_disk
chosen_chain
fallback_count
```

---

## 17. Non-Negotiable Invariants

1. FamilyCoord 表示 normalized half-sum / family_unit；FamilyId 表示 layer-local dense matrix 下标。二者不能混用。
2. 跨 layer 调度必须通过 `source FamilyId -> source FamilyCoord -> target FamilyCoord -> target FamilyId` 转换。
3. `T_delta(a,N)` 的唯一正确公式是 `{a, min(a+d, N-a)}`，去重后得到最多两个 target family。
4. 禁止在生产逻辑中私自写 `{f,f+d}` 作为实现公式；所有 NeedCells / future cache / retain / verifier 必须调用统一 T_delta。
5. 构建期必须 assert family axis 连续；边界越界可跳过，中间缺口默认报错。
6. `cell_id = row_family_id * F + col_family_id`，其中 F 和 FamilyId 都是当前 layer-local。
7. source/current layer 的 cid 与 target/future layer 的 cid 不可互用。
8. Dense matrix 与 cell headers 常驻。
9. 主 key mode 是 `NW exact + NE/SW/SE sum_id+empty_mask + prefix256`。
10. bucket entry 是 `key + rank_payload_offset + success_row_offset`，保持 16B。
11. position file 与 success file 格式对三条链路一致。
12. 非双分块链路调度单位是 cell / current cell chunk。
13. 非双分块链路四方向同 board 内同时计算，不使用 partial。
14. FamilyChain 外层是 source family `f` 升序，不是 target family `G` 外层。
15. FamilyChain pass 的 target cell set 是 `T_delta(f)` 对应的去重 `NeedCells`。
16. 生产路径不使用 CellUseSchedule / MaxCellUses；retain、dump、finalize 由公式化 NeedCells、future window、+2 boundary 驱动。
17. Future cache 是 cell-granular；同一个 future cell 在内存中最多加载一份。
18. FamilyChain generation 使用单 append-only generation blob。
19. +4 phase 不 final finalize；+2 phase 通过 `prev_boundary < max(r,c) <= current_boundary` 的 L 型边界推进 finalize，并在 phase 结束推进到 target_axis_max。
20. Generation 中 release Resident builder 时，若未 dirty 可保留旧 dump pointer，不重复写 dump。
21. FamilyChain solve 使用 cell-local partial store、result_accum 和 cell-granular future cache。
22. `partial_bytes(cell)` 与 `result_accum_bytes(cell)` 不同，必须分别计算和寻址。
23. 新 board 生成后直接进入 live CellBuilder 插入路径。
24. `get_or_create_builder` 必须有 atomic CAS 或 cell-local lock 保护。
25. 不存在全量 append-first / 离线 OR merge。
26. 回算 / 生成 hot path 必须先 physical family prefilter，再 canonicalize / encode。
27. row source 只执行横移，col source 只执行竖移；lookup / insert 逻辑一致。
28. 同一 current cell 的 partial block 同一时刻只能有一个 writer。
29. finalize 输出 compact final payload，descriptor 是唯一权威入口。
30. 第一版不支持 layer 中途 resume；中断后删除该 layer 临时文件并整层重算。
