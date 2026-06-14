# BC Resident / SingleChunk Generation Implementation Supplement

本文是 `exbc_exadbc_dense_family_cell_latest_plan_v7.md` 的实现补充，记录当前已经落地的 BC 非分块生成链路和单分块生成链路。它不替代总设计文档；总设计文档仍描述最终 EXBC / EXADBC dense family-cell 体系、三条链路和未来 FamilyChain 设计。本文只解释当前代码中已经实现并经过 free9-256 generation 验收的部分，以及主设计文档中没有展开的工程接口、性能路径和 caveat。

当前范围：

- ResidentChain / non-block generation。
- SingleChunkChain generation，严格 `1+x` layer residency 路径。
- Position memory/file format、streaming source loader、target file output。
- generation hot mutable backend `BCDynamicState`。
- 不包含 solve、FamilyChain、cell dump/reload、partial/result_accum、generation blob。

## 1. 当前模块地图

| 模块 | 主要文件 | 说明 |
|---|---|---|
| family / cell 基础 | `BCTypes.h`, `BCFamilyTable.h`, `BCCellMatrix.h` | layer-local dense axis、cell id、NeedCells 基础工具。 |
| key/rank/LUT | `BCLut.h`, `BCKeyRank.h` | `NW exact + NE/SW/SE sum_id+empty_mask` key mode、mixed-radix rank、prefix256。 |
| finalized cell | `BCCellBuilder.h` | correctness builder、`BCBucketEntry`、rank payload、finalized lookup。 |
| position file | `BCPositionFile.h/.cpp` | memory position layer、file writer、streaming metadata reader。 |
| source cell loader | `BCPositionCellLoader.h` | descriptor 常驻，按 cell/cell batch 读取 bucket/rank payload。 |
| scanner | `BCPositionScanner.h`, `BCLoadedCellScanner.h` | memory reader / loaded cell 共用扫描逻辑，支持 board callback。 |
| board glue | `BCBoardCodec.h`, `BCBoardOps.h` | quadrant pack/unpack、empty cell、spawn、canonical board encode。 |
| success IO | `BCSuccessIO.h` | memory success file 与 streaming success reader，当前只支持 `uint32_t` 测试 dtype。 |
| file IO | `BCFileIO.h`, `BCDirectFileIO.h` | buffered positioned IO、direct IO prototype、`read_many/write_many` stats。 |
| Resident generation | `BCResidentGeneration.h`, `BCResidentGeneration.cpp`, `BCResidentGenerationStreaming.cpp`, `BCResidentGenerationInternal.h` | non-block / streaming source generation、dynamic mutable backend、file output finalize。 |
| SingleChunk generation | `BCSingleChunkGeneration.h/.cpp` | SingleChunk wrappers，包含 strict `1+x` generation step。 |
| benchmark | `bench_bc_resident_generation.cpp`, `bench_bc_single_chunk_generation.cpp` | free9 generation benchmark、EX CSV 对照、stats CSV。 |

## 2. 共同生成热路径

Resident 和 SingleChunk 当前共享同一套 board-level generation hot path：

```text
source position entries
  -> scanner.for_each_board / loaded_cell scanner
  -> zero_cell_mask16(board), ctz enumerate empty cells
  -> spawn tile rank 1(+2) or 2(+4)
  -> BoardMover::move_all_dir(spawned)
  -> skip moved == spawned
  -> CanonicalBatch::canonicalize_inplace(batch)
  -> unpack_board_to_quadrants
  -> encode_canonical_quadrants_position_hot
  -> BCDynamicState insert: hash(cid,key) + atomic bitmap OR
  -> dynamic finalize
  -> BC position file
```

关键点：

- move/canonicalize 不在 BCBoardOps 里重写，直接接项目已有 `BoardMover::move_all_dir` 和 `CanonicalBatch::canonicalize_inplace`。
- scanner 使用 board callback，不构造 `vector<BCScannedPositionEntry>` 作为热路径中间结果。
- empty cell 枚举使用 bitmask + ctz，不走 checked `spawn_tile()`。
- canonicalize 使用 batch buffer，默认 `canonical_batch_size = 8192`。
- encoded candidate 先进入每线程 pending buffer，默认 `pending_insert_buffer_size = 1024`，再批量 hash resolve + bitmap OR。
- `family_tile_sum_values` 支持 free9 这类语义 sum，其中 tile `15` 可以按配置计为 0。

## 3. Target mutable backend: BCDynamicState

当前 ResidentChain 和 SingleChunkChain 的 target mutable 使用 `BCDynamicState`。这是 whole-layer mutable backend，只适用于 Resident / SingleChunk，不用于未来 FamilyChain。

逻辑键：

```text
(cid, bucket_key) -> bitmap
```

内部结构：

```text
cell_array[slot]          atomic<uint32_t>, empty / pending / cid
key_array[slot]           uint64_t bucket key
bitmap_offset_array[slot] uint32_t arena offset
bitmap_arena[word]        atomic<uint64_t>
```

插入流程：

1. 用 `(cid,key)` hash 到 home slot。
2. 线性探测。
3. 空 slot CAS 到 `kPendingCell`。
4. 为该 bucket 从 bitmap arena 分配 word range。
5. 清零 bitmap words。
6. 写 `key_array` 和 `bitmap_offset_array`。
7. 发布 `cell_array=cid`。
8. 对 rank 对应 bit 做 atomic OR。

去重语义：

- 同一 `(cid,key,rank)` 重复生成只设置同一 bit。
- `duplicate_candidates` 由 atomic OR 发现重复后统计。
- `output_success_rows` 是所有 live bits 数。

容量与 retry：

- `make_bc_dynamic_state(cell_count, bucket_estimate, bitmap_word_estimate)` 根据估计构造 hash table 与 bitmap arena。
- hash capacity 使用 power-of-two capacity 和 load guard。
- overflow 会设置 `overflowed`，外层按 reserve factor retry。
- 当前生产默认 hash load threshold 已提升到 0.75；实际 layer load 仍受 source estimate、reserve guard、power-of-two rounding 影响。

## 4. Dynamic finalize / file output

动态 finalize 不再把所有 `FinalizedCellPayload` 全层常驻后再统一写文件。file-output 路径使用 streaming finalize：

1. 扫 dynamic hash slots，按 `cid` 统计 bucket_count。
2. prefix-sum 得到每个 cell 的 slot-ref 区间。
3. 再扫 hash slots，把 `{key, slot, packed(bitmap_len, live_count)}` 放到 cell-local range。
4. 对每个 cell 内部按 key 排序。
5. 并行 measure cell payload：
   - 统计 live bit count。
   - 缓存 `bitmap_len` 和 `live_count` 到 slot ref。
   - 计算 bucket metadata bytes / rank payload bytes。
6. 生成 header 和 descriptor table。
7. bucket metadata stream 使用 bounded batch 并行序列化。
8. rank payload stream 使用 bounded batch 并行序列化。
9. 通过 `BCDynamicSequentialWriteStager` 顺序写入 `BCWritableFile`。

只要求 cell 内 key 升序，因为 finalized lookup 在单 cell bucket entries 内二分；不同 cell 的 key 没有全局排序语义。当前已经避免全局 bucket sort。

file-output finalize 会返回：

- `target_position_file_logical_bytes`
- `output_success_rows`
- `dynamic_bucket_slots_used`
- `dynamic_bitmap_words_used`
- `target_position_write_*` stats

因此 strict SingleChunk 的 mutable generation 阶段可以关闭中间 live-bit 预统计，避免同一 bitmap 在 mutable result 和 finalize 中重复扫描。

## 5. Position file 与 streaming source

Position file 仍是总设计文档规定的统一格式：

```text
header
descriptor table
bucket metadata stream
rank payload stream
```

单 cell descriptor 记录：

```text
bucket_count
success_rows
bucket_meta_offset
rank_payload_offset
rank_payload_bytes
flags
```

`BCPositionStreamingReader` 打开时只常驻：

- header
- family axis
- descriptor table

按需加载 cell：

```cpp
BCLoadedCell load_cell(CellId cid, BCCellLoadStats* stats = nullptr) const;
std::vector<BCLoadedCell> load_cells(const std::vector<CellId>& cids,
                                     BCCellLoadStats* stats = nullptr) const;
```

`load_cells` 保留调用方传入的 `cids` 顺序。它不要求 `cids` 全局有序；连续 chunk 只是 SingleChunk planner 的访问模式，不是数据结构语义。读取时会对当前请求顺序中的相邻 extent 做 coalescing，并通过 `BCReadableFile::read_many` 执行批量 IO。

loaded cell 扫描：

```cpp
BCLoadedCellScanner(lut, cell.view()).for_each_board(fn);
```

这与 memory `BCPositionCellScanner` 共用底层 scanning helper。

## 6. BCFileIO 当前约定

统一抽象：

```cpp
class BCWritableFile {
    virtual void write_at(uint64_t offset, const void* data, uint64_t bytes) = 0;
    virtual void write_many(const std::vector<BCFileWriteRequest>&, BCFileIOStats*) = 0;
    virtual void resize(uint64_t bytes) = 0;
    virtual void prepare_full_overwrite(uint64_t bytes);
    virtual void flush() = 0;
};

class BCReadableFile {
    virtual void read_at(uint64_t offset, void* data, uint64_t bytes) const = 0;
    virtual void read_many(const std::vector<BCFileReadRequest>&, BCFileIOStats*) const = 0;
    virtual uint64_t size() const = 0;
};
```

Buffered backend：

- 使用 ordinary OS-buffered file IO。
- `write_many/read_many` 对连续 requests 只 seek 一次。
- `prepare_full_overwrite()` 用于新文件完整顺序写，避免 buffered writer 预先 close/reopen resize。

Direct backend：

- 已有 Windows direct IO prototype。
- 当前 direct writer 注释语义是：不做 read-modify-write preservation，只适合 new-file stream writes / planned full writes。
- direct 后端当前不是默认 generation output backend。

线程安全约定：

- `BCReadableFile` backend 默认不保证线程安全，除非具体 backend 明确说明。
- 当前 streaming source load 在单 IO stage 调用；并行发生在 loaded cells 的 board scan / generation 计算阶段。

## 7. ResidentChain / non-block generation

Resident generation 的语义是 source / target 都在内存或可直接访问，不强制 current layer chunking。

主要接口：

```cpp
BCResidentGenerationResult generate_resident_position_layer(
    const BCLut& lut,
    const BCFamilyTable& target_axis,
    const std::vector<BCResidentGenerationSource>& sources,
    const BCResidentGenerationOptions& options = {});

BCResidentGenerationResult generate_resident_position_layer_to_file(
    const BCLut& lut,
    const BCFamilyTable& target_axis,
    const std::vector<BCResidentGenerationSource>& sources,
    BCWritableFile& output_file,
    const BCResidentGenerationOptions& options = {});
```

streaming source 兼容接口：

```cpp
BCResidentGenerationResult generate_resident_position_layer_from_streaming_source(
    const BCLut& lut,
    const BCFamilyTable& target_axis,
    const std::vector<BCResidentStreamingGenerationSource>& sources,
    const BCResidentGenerationOptions& options = {});

BCResidentGenerationResult generate_resident_position_layer_from_streaming_source_to_file(...);
```

Pair / mutable carry 接口：

```cpp
BCResidentGenerationPairResult generate_resident_position_layer_pair_with_mutable_carry_to_file(...);
BCResidentGenerationPairResult generate_resident_position_layer_pair_from_streaming_source_with_mutable_carry_to_file(...);
```

Pair path 可以在一次 current scan 中同时生成 primary(+2) 与 secondary(+4)，generation wall time 是共享的。因此 `primary.generation_seconds` 和 `secondary.generation_seconds` 不应相加；pair-level 有：

```cpp
current_boards_scanned
shared_generation_seconds
total_pair_compute_seconds
```

注意：pair path 是 Resident-like 性能原型或兼容路径，不是 strict SingleChunk 的 `1+x` 内存约束路径；它可能同时持有 primary 和 secondary mutable states。

## 8. Strict SingleChunk generation

SingleChunk 的生产语义：

```text
source current layer: 按 cell chunk streaming load
target mutable: whole-layer BCDynamicState
output: position file
memory: 1 + x layers
```

其中：

- `1` 是 future/target mutable layer。
- `x` 是 current source layer 的 loaded cell chunk。
- 不允许同时持有 primary(+2) 和 secondary(+4) 两个 whole-layer mutable states。

当前 strict step：

```cpp
BCSingleChunkGenerationStepResult generate_single_chunk_position_layer_strict_to_file(
    const BCLut& lut,
    const BCFamilyTable& primary_axis,
    const BCPositionStreamingReader& current,
    uint32_t current_cell_chunk_size,
    std::unique_ptr<BCResidentGenerationMutableLayer> carry_to_primary,
    const BCFamilyTable* secondary_axis,
    BCWritableFile& primary_output_file,
    const BCResidentGenerationOptions& options = {});
```

严格执行顺序：

```text
1. consume current chunks with +2
   target = carry_to_primary if present, otherwise new primary mutable
   finalize/write primary position file
   release primary mutable

2. if secondary exists:
   rescan current chunks with +4
   build next_carry mutable only
   do not finalize
   do not write +4-only file
```

下一层：

```text
previous next_carry -> carry_to_primary
current layer +2 inserts into that mutable
finalize/write complete layer
```

这解决了之前容易混淆的点：`+4` 分支结果不应单独 finalize/write 成 `secondary` 文件；它以 mutable carry 形式直接参与下一层的 `+2` generation。只有完整 layer 才写 position file。

SingleChunk wrapper：

```cpp
BCResidentGenerationResult generate_single_chunk_position_layer(...);
BCResidentGenerationResult generate_single_chunk_position_layer_to_file(...);
```

这些接口保留为普通 SingleChunk / compatibility path；严格 `1+x` 路径应使用 `generate_single_chunk_position_layer_strict_to_file`。

## 9. Benchmark route and stats

benchmark target：

```text
bc_single_chunk_generation_bench
```

free9-256 strict file-output 路径：

```text
seed layer -> bc_layer_16.bcpos
for S = 18..340 step 2:
    current = bc_layer_(S-2).bcpos
    carry_to_primary = previous +4 mutable, if any
    +2 current chunks -> complete bc_layer_S.bcpos
    +4 current chunks -> next_carry, unless terminal
```

输出文件名：

```text
bc_layer_<sum>.bcpos
```

不再使用 `primary_` / `secondary_` / `buffered_` 前缀。

CSV 口径：

- `compute_seconds = generation_seconds + finalize_seconds`
- `total_seconds = compute_seconds + write_seconds + small wrapper overhead`
- `compute_throughput_mbps = throughput_live / compute_seconds / 1e6`
- `total_throughput_mbps = throughput_live / total_seconds / 1e6`
- `throughput_live` 对齐 EX generation 口径：有 output rows 时使用 output live，否则使用 source live。
- 生成性能统一看 EX/live 口径的 compute/total throughput。

最新一次完整 strict SingleChunk free9-256 file-output 验收：

```text
CSV:
  C:\2048_tables\free9\bc_singlechunk_strict_generation_20260607_152746.csv

Output dir:
  C:\2048_tables\free9\bc_singlechunk_strict_generation_20260607_152746

Generated files:
  163 .bcpos files
  8,754,682,168 bytes total

Layer correctness:
  162 / 162 generated layers ex_match=1

Total:
  generation_seconds          121.217532
  finalize_seconds              5.737336
  write_seconds                 1.740855
  compute_seconds             126.954868
  total_seconds               128.707274
  compute_throughput_mbps     129.698203
  total_throughput_mbps       127.932306
```

## 10. 当前已经应用的 EX 对齐优化

当前 BC generation 已经对齐或近似对齐 EX generation 的主要优化：

- OpenMP parallel over loaded source cells / buckets。
- `BoardMover::move_all_dir` 一次生成四方向。
- bitmask + ctz empty-cell enumeration。
- `CanonicalBatch::canonicalize_inplace` batch canonicalize。
- per-thread canonical buffer。
- per-thread pending insert buffer。
- dynamic hash insert 单查路径，不做 `contains()+insert()` 双查。
- atomic bitmap OR 去重。
- source file streaming load with coalesced extents。
- dynamic finalize 使用 per-cell grouping + cell-local key sort。
- rank payload / bucket metadata bounded parallel serialization。
- output file streaming write，不构造整层 serialized bytes vector。

与 EX 仍不同的点：

- strict SingleChunk 为满足 `1+x` 内存约束，对非 terminal current layer 做两次 scan：先 +2，再 +4。
- 当前 target mutable 是 whole-layer `BCDynamicState`，FamilyChain 后续不能复用这个 backend。
- SingleChunk 暂无 generation blob，因此不会把第一次 scan 的中间结果保存给第二个 delta phase。
- 当前 output 默认 buffered backend；direct backend 已有 prototype，但不是生产默认。

## 11. Caveats and TODO

已冻结或当前可依赖：

- Position file 格式：header + descriptor + bucket stream + rank payload stream。
- `BCBucketEntry` / `BCRankPayloadView` / `BCBucketEntryView` / finalized lookup。
- `BCPositionStreamingReader::load_cells` 与 `BCLoadedCellScanner`。
- Resident / SingleChunk target mutable backend `BCDynamicState`。
- Strict SingleChunk 不写 `+4-only` 文件，`+4` 以 mutable carry 进入下一层。

后续仍需实现：

- Route planner：按 layer / memory budget 选择 Resident / SingleChunk / FamilyChain，并决定 chunk size。
- FamilyChain mutable store：cell-local builder + dump/reload + finalize boundary，不能直接使用 `BCDynamicState`。
- generation blob 或其他机制，若要减少 strict SingleChunk 的 double scan，需要单独设计，不能破坏 `1+x` 约束。
- Solve chain。
- Success file production writer/loader 接主链路。
- Direct IO production policy：Windows high-QD overlapped direct read/write、extent sorting/coalescing、logical/physical size 管理。
- Streaming position reader 的多线程 reader 策略：当前 backend 默认不保证 shared-reader 并发安全。

## 12. 维护建议

1. 不要把 `BCDynamicState` 扩展到 FamilyChain。它是 Resident / SingleChunk whole-layer mutable backend。
2. strict SingleChunk 的 `+4` carry 不应 finalize/write；只有完整 layer 写 `bc_layer_<sum>.bcpos`。
3. 如果优化没有稳定收益，应回退。已经验证过无效或不适合保留的方向包括：
   - 在 insert hot path 上用 atomic 维护 bucket/bitmap stats，统计扫描减少但 hot insert 变慢。
   - 过大的 sequential write staging buffer，减少 write ops 但 buffered write seconds 变差。
4. 单分块性能验收应强制 current layer chunking，例如 5-8 chunks，而不是退化为全层 source resident。
5. CSV 对 EX 校验应继续使用 `free9_256_zmask_generate_stats.csv`，至少检查每层 `input_live` 和 `primary_live`。
