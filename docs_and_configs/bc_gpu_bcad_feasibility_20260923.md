# BC GPU 与 BCAD 可行性考察

日期：2026-09-23。范围：当前工作区源码，目标为 12/16/24 GiB 级消费独显。
本机查询结果：RTX 4080 SUPER，16376 MiB 显存。本文保留最初设计评估时的结论与边界。
后续已实现独立 CUDA 实验后端；小定式结果见 [首轮实验](../experiments/bc_gpu/RESULTS.md)，
free10-512 的 GPU 生成、compact、FamilyChain 与容量测试见 [大表实验](../experiments/bc_gpu/FREE10_RESULTS.md)。
下文“尚未实现／未测量”描述的是最初设计评估阶段；BCAD 仍未实现。

## 1. 结论

1. **纯 BC 可以先上 GPU，不必等待 BCAD。** 保留 family/cell/bitmap 和流式文件框架，把一个显存窗口内的 decode、spawn、move、canonicalize、lookup、reduce、compact 放在 GPU。
2. **BCAD 的冲突存在一种有数学依据的解法：半盘和先模 64，再进行两侧归一化。** 大砖隐藏、恢复与排列不改变这个分区。对单个出砖阶段，一个源 family 只需两个目标 family 的行列十字窗口。
3. **新分区不等于已经解决所有 BCAD 工程问题。** 每层只有 16 或 17 个不同 family，分区细度有限；AD 宽向量、特殊转移及列映射仍须移植。其内存能否小于 12 GiB 尚无真实数据验证。
4. 第一阶段应以纯 BC 单层回算为基线。若 AD 行宽和映射复用率有利，再实现 residue-family BCAD。不要以峰值浮点算力估算加速比。

## 2. 当前实现及需要保留的语义

### EX / EXAD / BC 的不同作用

| 路线 | 位置表示 | 计算共享 | 内存控制 |
|---|---|---|---|
| EX | prefix36 + suffix28 rank bitmap | 每个具体棋盘 | 主要是整层未来数据 |
| EXAD | 48 个 AD slot，各自 prefix36 + suffix28 bitmap | 一个代表棋盘对应一条胜率向量，排列映射关联列 | 支持单未来层及当前 chunk |
| BC | 2×2 quadrant key/rank bitmap，family-cell matrix | 当前生产主线是标量胜率 | resident / single / family 三路 |

BC key = NW exact + NE/SW/SE 的 `(raw sum_id, empty_mask)`；rank 是后三个 quadrant 的混合进制编号。
每 quadrant 同 sum+empty-mask 最多 36 个 word，故 bucket bitmap 最大 46656 bit，rank 可用 uint16。
这保证同一 bucket 中所有棋盘空位相同，也适合 GPU 按 bucket 批量枚举。

BC 的小内存主要来自依赖分解：横向移动保持上下半盘和，纵向移动保持左右半盘和；D4 canonicalize 可能交换轴，所以加载目标 family 的整条行与列。
这不是一般 hash 分片能替代的性质。

现有 family 回算将 Spawn4 / Spawn2 分开，非对角 cell 的两个方向访问之间保存逐空位 partial max：

```text
best[board, empty, spawn] = max(horizontal_best, vertical_best)
value[board] = average_empty(p2 * best2 + p4 * best4)
```

必须先合并方向 max，再对空位求平均。不同 AD 列也要分别完成这个过程。
临时结果只能在完成全部依赖后删除零行；阈值裁剪用于归档，不能用于 exact future frontier。

核心依据：

- `native_core/include/BCKeyRank.h`、`BCBoardCodec.h`、`BCLut.h`：位置编码和枚举。
- `native_core/include/BCFamilyPartitionPolicy.h`、`BCFamilySolvePlan.h`：当前先 min 再 modulo 的分区及调度。
- `native_core/include/BCSolveEdgeKernel.h`：spawn/move、查询准备、long double 归约。
- `native_core/include/BCFamilySolve.h`、`BCSingleChunkSolve.h`：partial/scratch 与流式回算。
- `native_core/src/BCFamilyGeneration.cpp`、`native_core/include/BCCellMutableBuilder.h`：生成、动态 hash 扩容、原子 bitmap OR。
- `native_core/src/BookSolverEXAD.cpp`、`BookSolverAD.cpp`、`BoardMaskerAD.cpp`、`BoardMoverAD.cpp`：AD 的实际语义。
- `native_core/include/EXADSolvedLayer.h`：slot 行宽及排列数量。

**源码与文档差异：** 当前 `BCFutureSuccessLookup.h` 的 16-byte DirectEntry 已存 `bitmap_offset`，热路径另建 uint32 `word_rank_bases`；不能只按旧文档的 prefix256 查找成本估算。
hash capacity 是 `next_pow2(ceil(3.2 * bucket_count))`，每个非空 cell 至少 4 项。索引可能比原始 bucket 元数据大很多。
`row_width` 参数的存在也不代表 BC 已支持 AD：当前 BC 查询主要以指定 lane 工作，缺少 AD slot 分派与列置换语义。

## 3. 为什么直接拼接 BC 与 AD 会失败

AD 默认将 rank >= 6 的砖（64 及以上）表示为 F；F 在棋盘字中是 rank15，但可能代表不同真实大砖。
`BoardMoverAD::merge_line` 会把 32+32 等产生的 >=64 砖写为 F，F 本身不按普通相等砖合并。
实际大砖移动、重复大砖、生成新大砖等由 derive / mask-new-tile / permutation 分支处理。

因此必须区别：

1. 原始真实棋盘的总和，用于回算层和 AD 大砖组合推导。
2. 把 F 当 32768 后得到的代表棋盘 raw sum，用于一种无损位置编码。
3. family 所用的不变量。

它们不能继续共用一个 `layer_sum`。

一个代表棋盘的不同 AD 列可能把 64、128 放在不同半盘。真实半盘和变化，当前 `min(half1,half2)/2 % M` 随之变化。
例如上下半盘 `(64+2,128+4)` 与 `(128+2,64+4)`，当前公式即使用 M=32 也分别得到 1 和 2。
两者的 masked skeleton 相同。因此不能把一整条 AD 向量直接分配到当前 raw-sum family。

## 4. 可行的新分区：先取余，再对两侧归一化

设一层真实总和为 S，以半值为单位：

```text
T = (S / 2) mod 32
r = (上半盘和 / 2) mod 32
F_T(r) = min(r, (T-r) mod 32)

row_family_label = F_T(top_half_residue)
col_family_label = F_T(left_half_residue)
```

这里所有输入 residue 都在 0..31。每层再将合法 label 映射为 dense FamilyId，不能把跨层 label 和 storage id 混用。

### 4.1 不变量证明

所有被隐藏的大砖都是 64 的倍数，F 的编码值 32768 也是 64 的倍数。
所以在任何半盘中：

```text
真实大砖 -> F
F -> 真实大砖
大砖在原有大砖位置之间排列
```

均不改变半盘和 mod 64。所有 AD lanes 共享 row/col family。
32+32 -> F 的编码值变化为 32768-64，也是 64 的倍数。
真实大砖的合并与再掩码同样不破坏方向半盘 residue；终局分支无需查询未来。
前提是维持当前“隐藏阈值 64”的语义，不能无审计地推广到任意阈值或其他规则。

上下翻转把 r 换成 T-r，F_T 不变；转置交换 row/col family。因此仍能使用行列十字窗口覆盖所有 D4 canonicalize 结果。

### 4.2 fanout 只有两个

出 2 或 4 时，令 d=1 或 2，目标层 T'=(T+d) mod32。
对源 family label f，依赖的目标 labels 为：

```text
{ F_T'(f), F_T'(f+d) }
```

原因是出砖只发生在某一侧；方向移动不改变该方向半盘 residue。
无论 f 代表原来的哪一侧，两侧互补及 F 的归一化都会落入上述集合。
两个 labels 去重后加载对应目标行列十字；AD slot 则取真实转移能触达的 slot 集合。
这是 family 维度的 fanout2，不能误读成只有两个目标 cell 或两个 AD slot。

### 4.3 代价与限制

- T 偶数时有 17 个 family，T 奇数时有 16 个。两 family 的十字并集，在 F×F 的矩阵中有 `4F-4` 个 cell，约占 22%～23%；这只是 cell 数比例，实际字节数可能严重不均。
- 这个模数受隐藏阈值约束，不能任意改成 37 或 61 来进一步压缩窗口。仅将旧 BC 配置设为 M=32 也不成立，必须改变归一化顺序。
- 层间 family 轴会变化，现有 stride sweep / last-producer boundary 必须根据新依赖图重建。
- 如果一个依赖窗口仍超过显存，缩小 current batch 无法解决。需要进一步分 future shard，并逐 shard 处理有界查询，或保留 CPU 路线；反复全扫 current 的代价必须计入。

### 4.4 分区和 bucket 压缩应分开

**只对 family 使用 residue，不要将 BC bucket 的 sum_id 直接替换为模 64 的和。**
后者会把许多不同大砖编码混进同一个 quadrant group，破坏 group<=36 和 uint16 rank 的前提。
本次全字母表穷举得到：raw sum+empty-mask 最大 group=36，residue+empty-mask 最大 group=10734。

建议保留现有 raw-sum quadrant LUT 和 key/rank，将存储组织改为：

```text
logical layer(original S)
  -> AD slot
    -> residue-family cell
      -> existing raw-sum bucket key
        -> bitmap live row
          -> AD success vector
```

同一逻辑层内可存在不同代表棋盘 raw sum；这并不影响每个 bucket 的无损编码。
但 `.bcpos` 当前要求 `decoded raw_sum == header.layer_sum`，热 encoder 还使用 `layer_sum-half_sum`，均须改版，不能把新数据伪装成现有 `.bcpos v2`。
需要显式区分 logical S、family policy、AD slot/row_width 和 codec signature，并独立验证原始 EXAD 的特殊分支。

## 5. GPU 后端：先纯 BC，再 AD

### 5.1 纯 BC 回算推荐结构

```text
CPU: 读取、规划 cell 窗口、异步 I/O、检查点和归档
  -> 有界 pinned staging
GPU: 当前压缩位置 + 完整可覆盖的未来查询窗口
  -> bitmap 枚举/解码
  -> spawn / move / canonicalize / encode
  -> device-local lookup
  -> 逐空位 max + 加权归约
  -> partial 或 compact 后的 position/success
CPU: 有界接收、顺序写盘
```

第一版让一个线程处理一个具体棋盘作为易校验基线；同 bucket 的任务有相同空位，减少不一致循环。
同时比较每空位/方向使用小组线程的实现；不能预先断言一个 warp 一个标量棋盘最快。
bitmap 稀疏时先在有界 batch 中压紧 live rank；不生成整层 uint64 boards 或整层转移图。

工作调度按 live rows、empty count、AD width 估算，而不是均分 bucket 数。大 bucket 分片，小 bucket 合并。
行移动表约有 65536 项；quadrant 表也有 65536 项。它们适合设备只读全局内存/缓存，不应假定整个 LUT 放入 constant memory。
保留移位、掩码、popcount 的 device 实现，不直接移植 CPU BMI2/AVX/prefetch 指令。

### 5.2 查表与工作集

先复用当前 direct hash 算法建立正确性基线，但在显存里使用平坦数组和 offsets，不复制 C++ 对象、vector、CPU 指针或锁。
未来窗口内的 hash、bitmap、success 均须在 device；不要为每条查询经 PCIe 回 CPU 取值。
复用相邻 family pass 的 future cells 和索引，增量加载新 cell；双缓冲只复制 staging，不默认复制整套 future。

应比较三种 rank 支持：

1. 原样 uint32 word-rank-base：单次局部 popcount，索引占用较大。
2. uint16 bucket-local word-rank-base：BC 单 bucket live rows <=46656，数值范围足够；全局地址与 row offset 仍须足够宽。
3. 只保留文件中的 prefix256：最多附加三个完整 word popcount，换取更低显存开销。

hash 可比较原负载率与更紧凑的静态表；提高负载率会增加 probe，不能只按内存节省选择。
先测 fused kernel。若随机查询成本高，再尝试有界 query queue 按 target cell/hash 分组；排序成本与额外读写必须包括在比较中。

### 5.3 AD 向量何时有利

普通分支中，一个代表棋盘完成一次移动、canonicalize、bucket lookup，之后读取 `future_row[permutation[lane]]`，这部分可跨很多 lane 摊薄。
`BookSolverEXAD.cpp::update_osr_exad_ranked` 正是这种结构。

- 宽度小：一个 warp 内放多个代表棋盘。
- 宽度接近 32：一个 warp 处理一行，广播公共 query/row base，各 lane 做 max 和平均。
- 宽度大：按 lane tile 分给多个 warp；不要让一个线程保存整条向量，也不要每个 lane 重复 hash lookup。

AD 向量访问是 permutation gather，不天然等于完美合并访存。小排列可以比较按目标列连续加载后 warp shuffle 回源列；大排列需要块内 staging 或显式映射，不能套用跨 warp shuffle。
公共列映射可缓存或以紧凑描述表达；避免把所有 edge×width 的映射物化到全局数组。
mask-new-tile、重复砖、三颗 64、终局等分支应分类成有界任务队列，分别调用经过校验的 kernel；不能为了并行简化这些语义。
GPU 并不保证 derive 很宽就更快：value traffic 和 partial 空间也按 width 增长。

### 5.4 生成阶段另行处理

当前生成 builder 的 stop-world grow、共享 hash 与小批 atomic OR 是 CPU 方案，不适合逐字搬到 GPU。
建议 GPU 产生有界 `(slot, cid, key, rank)` candidates，排序去重，合并为局部 bitmap，再与 active target 合并。
可保留 CPU boundary writer 与 blob 生命周期，先测 GPU 去重后再传回是否已足够；后续再比较预分配 GPU hash。
必须预留 radix sort 双缓冲和临时空间，并支持容量不足时分批/重放，不能丢 candidate。
不缓存整层 edges。生成与回算的临时对象、依赖和瓶颈不同，分别做性能预算。

## 6. 12/16/24 GiB 的预算方式

不能从 CPU 峰值 RSS 直接推导 VRAM。工作区还有 file cache、CPU heap、设备索引构建暂存、pinned staging 等不同成本。

```text
VRAM_required = future_position + future_index + future_values
              + current_batch + partial/scratch_batch
              + compact_or_sort_workspace + transfer_staging + reserve

future_index(current implementation)
  ~= sum_cell[16 * next_pow2(max(4, ceil(3.2 * buckets)))
              + 4 * ceil(rank_payload_bytes / 8)]

partial_bytes = sizeof(T) * sum_bucket(live_rows * empty_count * AD_width)
```

这些是 live allocation 预算；建立新索引时旧副本和输入元数据是否仍在也要统计。
现有 CPU 的 2 GiB `value_batch_max_bytes` 只限制数字 payload，并不是完整内存上限，不能直接作为 GPU 显存控制器。

| 标称显存 | 初始算法预算建议 | 路线选择 |
|---|---|---|
| 12 GiB | 约 9～10 GiB，按实时空闲显存下调 | family；单未来层实际能放下时选 single |
| 16 GiB | 约 12～13 GiB | 同上，增大缓存和 current batch |
| 24 GiB | 约 19～20 GiB | 更多层可用 single/resident，降低窗口搬运 |

以上只是初始 reserve 策略，不是对 free12 或 BCAD fit 的承诺。路由须在 load 前按 descriptor 实际估算；压不下时先压工作区，再改变 future 分片策略。
12 GiB 若要同时驻留四个相近 family，每个 family 的 position+index+values 就必须显著小于 2 GiB，才有空间给 current 和 partial；实际共享交叉 cell 可减小这一上界。
更大显存的潜在收益往往来自减少重复搬运，不只是容纳更大的线程 batch。

## 7. 性能与数值边界

- 2048 这里主要是整数位运算、查表、gather、max/average，Tensor Core 不直接匹配。
- 先统计查找边数、有效命中、probe 分布、缓存命中、每条边的实际显存 transaction 字节、传输和磁盘字节。
- `GPU boards/s <= effective_memory_bandwidth / bytes_per_board` 只给带宽上界；依赖加载延迟、指令量和 occupancy 可能先饱和。
- 以有效边数 q、每边实际显存流量 b 为例，100 Mboards/s、q=40、b=64 bytes 意味着约 256 GB/s；这是示例需求，不是 q/b 的实测或性能预测。
- 端到端时间要包含读取、索引、传输、计算、partial I/O、compact、写盘及请求范围内的压缩。重叠的理想下界取各阶段最大值，未重叠时按和累计。
- 例如只有 70% 原耗时可加速，即使 kernel 快 10 倍，理想总加速也仅 `1/(0.3+0.7/10)=2.70`，额外 PCIe 成本还会降低它。

UInt32 建议最先支持，但不能直接用 float32 累加大定点胜率。整数 sum 用 uint64 进行；最终概率缩放另行规定舍入。
现有 CPU 用 long double，single/family 分阶段截断又可能不同于 resident 一次截断。GPU double、FMA 或精确 1/10 有理数计算不自动与当前 double p4 转 long double 的路径位级一致。
先固定 CPU oracle 路线与数值契约，验证边界值及多层误差。若需要位级一致，要实现兼容的算术/舍入；不能默许若干 ULP 差异改变零裁剪。
Float64/UInt64/one-minus 模式分别验证，零值与终值不可硬编码为 0/1。

## 8. 已执行验证及其边界

新增独立探针 `tools/probe_bcad_residue_partition.cpp`，链接仓库真实 `BoardMoverAD.cpp`，不依赖 GPU。

```powershell
g++ -std=c++17 -O2 -I native_core/include tools/probe_bcad_residue_partition.cpp native_core/src/BoardMoverAD.cpp -o tmp/probe_bcad_residue_partition.exe
$env:PATH = 'C:/Apps/mingw64/bin;' + $env:PATH
./tmp/probe_bcad_residue_partition.exe
```

2026-09-23 输出：

```text
row_move_checks=131072
large_tile_replacement_checks=163840
fanout_algebra_checks=8192
random_board_direction_symmetry_checks=1280000
family_count_range=16..17
raw_sum_empty_mask_max_group=36
residue_empty_mask_max_group=10734
old_partition_hidden_permutation_counterexample=1,2
PASS (partition probe only; no complete AD solver or GPU validation)
```

行检查覆盖 65536 种输入和两个方向；随机棋盘使用固定 seed，覆盖两种 spawn、四方向、八个 D4 变换。
这是分区数学性质和实际 AD mover 的验证。尚未跑完整 EXAD generation/derive/query 轨迹、未验证新的文件格式、没有 BCAD 端到端结果、没有 GPU benchmark。
本机 PATH 未发现 nvcc。本次未安装工具链、未修改生产算法，也未运行昂贵的 free12 全量计算。

## 9. 推荐实施顺序与通过标准

1. **纯 BC 单层 CUDA 回算。** 先用已有数据做两个 future 均驻留的标量基线，逐步接 single、family；对照当前 CPU 的棋盘集合、胜率和裁剪，包含传输后仍明显加速才继续。
2. **真实容量画像。** 对 free12 代表层记录每个窗口 position/index/value 字节、current/partial 峰值、重用率、吞吐。用实际 12 GiB 限额验证，不能只在 24 GiB 卡上推断。
3. **AD 新分区 tracer。** 在现有 EXAD 的所有查询出口记录原 source、slot、目标、列映射，检查 residue window 覆盖；统计每 family/slot 的实际字节和 width 分布。
4. **小规模 CPU BCAD oracle。** 新 codec/container 与 residue scheduler 对照 EXAD，覆盖无新大砖、新 64、重复大砖、三 64、终局、D4、空 cell 和层间 family 轴变化。
5. **GPU AD 快分支，然后特殊分支。** 比较标量展开 vs 共享代表棋盘，统计每秒实际 AD values，避免用 skeleton rows/s 误报加速。
6. **生成 GPU 化。** 比较 sort/unique 与 bounded hash，验证棋盘集合完全相同、不会因为 batch 边界丢失状态，再做全程流水线。

优先决策：纯 BC GPU 是较低语义风险的工程路线；residue-family BCAD 是有证明和局部实验支持的新设计路线。二者可以共用 GPU lookup/window/bitmap 基础设施，但应独立验收。

## 10. 外部依据

CUDA 的设备驻留、合并访存与有界 pinned/异步传输原则参见 [NVIDIA CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。它支持本文的实现方向，不构成本项目加速比证据。
Windows 等平台的 Unified Memory 能力取决于设备属性，不能假定显存超额后自动分页仍有稳定性能，参见 [NVIDIA Unified and System Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)。首版使用显式窗口和传输。
批量排序可用 CUB，临时空间应查询真实 API 需求，参见 [NVIDIA DeviceRadixSort 示例](https://github.com/NVIDIA/cccl/blob/main/cub/examples/device/example_device_radix_sort_custom.cu)。
