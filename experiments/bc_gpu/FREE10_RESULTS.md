# free10 → 512：GPU 生成、压紧与 FamilyChain

本报告对应 `C:/2048_tables/free10-512` 的既有 EX 表。已完成 292 个位置层的 GPU 生成和 290 层 GPU 链式回算，累计回算 **239,073,978,447** 个局面；没有重算一套 CPU free10 表。
原始证据见 `results/free10_summary.json` 和 `results/free10_*.json`。

## 实测结果与重要异常

| 阶段 | 旧 CPU EX 记录 | GPU 各步骤合计 | 比值 |
|---|---:|---:|---:|
| 生成 | 1816.789 s | 390.469 s | 4.65× |
| 回算 | 2388.294 s | 1252.427 s | 1.91× |
| 合计 | 4205.083 s（70.08 min） | 1642.896 s（27.38 min） | 2.56× |

所有生成层的局面数、两个终端层和每个回算输入层数量都与旧 CSV 一致。GPU 查表／归约 kernel 合计 165.434 s，占回算 wall time 约 13.2%；主要剩余耗时在加载、缓存搬运、压紧和 I/O。
这些是分阶段开发、续算的步骤合计，并非最终版本在独占环境下一次冷启动的统一基准；具体差异见后文。

**旧第 0 层存在显著数值异常，不能写成“前六层全部与旧表一致”。**

- 第 1–5 层各抽样 1,024 个，共 5,120 个；旧表保留的 5,113 个样本最大差异为 8 个 UInt32 单位，即胜率差 `2e-9`。其余 7 个均符合旧表 0.1 的归档删除阈值。
- 第 0 层抽样 1,024 个，其中 1,017 个与旧文件差异远超舍入误差。直接读取旧文件的 value section 与原生 EX reader 完全一致，排除了此次 cold reader 读值偏移的解释。
- 旧第 0 层有 17,925 行，却只有 100 个不同胜率。其文件时间仍为 2026-08-08；本实验没有改写旧表。
- 对相同的 1,024 个初始样本，用独立 Python 移动／D4 和旧第 1、2 层的查询值检查一步 Bellman 递推。GPU 最大误差仍只有 8 个整数单位；旧第 0 层有 1,017 个不满足递推，最大差异约 0.3094。
- 被旧归档删除的未来值按 `[0, 0.1]` 处理；在这些样本中，另一个已知方向的 max 都消除了该不确定性，1,024 个递推检查没有使用未知值替代成零。

这支持 GPU 初始层结果，并说明旧第 0 层不能作为可靠数值 oracle。产生旧层错误的历史代码原因尚未定位；没有修改生产 EX 或旧表来掩盖差异。
例如 `0x000f0021f2ff03ff`：旧层 0 值为 `2212750408`，由旧未来层递推得到 `3450379711`，GPU 为 `3450379707`。
`compare_samples.py` 会有意对这批旧层 0 数据返回失败；`diagnose_initial.py` 提供独立检查，汇总报告同时保留直接比较失败与诊断结果。

另对第 241 层抽样 4,096 个，旧表中保留 103 个，最大差异为 3 个整数单位；其余均满足旧归档阈值。
248 个层的非零数量与旧 EX 存在差异，单层最大 52,360 行；不能以行数相等代替数值验证，也不能把极小值处的截断差异隐藏掉。

## 真实大窗口容量和路线测试

保留第 166 层及其两个 GPU future 的不可变快照。当前层 **1,249,797,355** 个局面，以下五种路线的完整压紧 position/value 文件 SHA256 一致：
`c4ca2f1ed1311e82c01e78f291bffb45047d08b65d66f997a2b000c753fb2f3d`。

| 设备内存池硬上限 | 路线 | 一次整层 wall | 整卡 NVML 使用量峰值 |
|---|---|---:|---:|
| 10 GiB | 双 future 合并 | 10.31 s | 13700 MiB |
| 8 GiB | 双 future 合并 | 18.26 s | 11467 MiB |
| 8 GiB | 分出砖 partial/scratch | 29.43 s | 11743 MiB |
| 8 GiB | 双 future 合并 + 异步写盘 | 13.74 s | 11455 MiB |
| 10 GiB | 完整 future 驻留 + 分批 current + 异步写盘 | 5.27 s | 未采集 |

这是容量和精确等价性验证，wall 每种只测一次，存在顺序、文件缓存和其他任务竞争，不能把表中差值直接认定为稳定加速比。
整卡峰值按 200 ms 周期取样，包含桌面和驱动，并非逐 allocation 的绝对峰值；设备池上限本身是硬约束。
8 GiB split 路径实际写入并读回约 3.88 GB 磁盘临时值，结果仍完全相同，验证了缓存溢出的真实 scratch/partial 链路。

最后一行由 `resident_stream.py` 实测，两个完整 future 约 6.92 GiB，设备活动数组峰值约 7.58 GiB；不展开整层棋盘，也不分配整层 current values。
其计算 kernel 为 0.443 s，省去了 family partial。这说明 **能放下两个完整 future 时应选择驻留路线；超预算时才使用 FamilyChain**。
本次全量 290 层有意走 FamilyChain 来验证核心链路，没有用这一个驻留层的测量外推或替换全量回算总时间。

全量回算记录的设备活动数组峰值约 9.26 GiB。后 209 层启用了 10 GiB 池硬上限，覆盖最大 future 工作集区间；早期步骤只有活动分配预算，不能倒称全程都已限制池保留块。
全量进程 host working-set 峰值约 9.47 GiB，Windows private/pagefile commit 峰值约 21.69 GiB。后者必须保留在内存结论里，不能笼统声称“总进程占用低于 16GB”。

## 并行度与索引测量

第 166 层最大对角 cell 含 9,067,060 个局面，全部至少有一个空位，平均 1.266 个空位；该层小砖和低于 512，没有目标终局快返回。
对 64/128/256 threads/block 和 hash 容量因子 2.0/3.2 分别先连续预热至少 0.75 秒，再测 9 次中位数；单次 launch 预热曾出现明显低频／驻留瞬态，未用于最终参数结论。

| kernel | hash 因子 | 64 threads | 128 threads | 256 threads | future 已分配字节量 |
|---|---:|---:|---:|---:|---:|
| resident cells | 2.0 | 1.891 ms | 1.896 ms | 1.904 ms | 2.922 GiB |
| family fused，双方向 | 2.0 | 1.892 ms | 1.881 ms | 1.858 ms | 2.922 GiB |
| resident cells | 3.2 | 1.855 ms | 1.858 ms | 1.864 ms | 3.057 GiB |
| family fused，双方向 | 3.2 | 1.855 ms | 1.853 ms | 1.820 ms | 3.057 GiB |

所有配置逐值相同。当前样本中块大小只带来小幅差异；较大的 hash 略快，但多占显存。
继续采用 128 threads 和因子 2.0 作为内存优先的基线，没有据这个单 cell 宣布全局最优。
表里的 future 字节量是加载窗口大小，并非该 cell 实际触达的全部字节；重复 kernel 的缓存局部性优于全层扫描，不能用此吞吐推算整表时间。

当前最有价值的下一步是生产级加载／写盘重叠、缓存预算分配和低层 resident 路由，而不是只增加线程数。

## 比较口径

- GPU 为本机 RTX 4080 SUPER，实际显存约 16 GiB。没有把型号误写成 12 GiB 版本。
- 旧表配置：EX、Full D4、p4=0.1、UInt32、归档删除阈值 0.1、非 optimal-branch。
- 使用与生产代码相同的 free10 初始集合，共 17,925 个局面。LUT 与初始集合由小型 C++ setup 导出；之后没有再计算一套 free10 CPU 表。
- 按旧表的有限层数边界运行：总计 292 个位置层，step 0..291；step 290、291 仅保存成功终局。不是把目标 512 一直展开到所有可能小砖总和。
- GPU forward 的 source success check 与旧 EX 一致：`step > 247`。所有生成层的数量逐层对照旧 CSV，两个终端层也单独核对。
- 计算与归档阈值分开。GPU future 仅删除精确零值，抽样遇到旧表已按 0.1 删除的局面时检查阈值界限，不把“旧表缺失”解释成胜率零。
- 旧 EX 用逐空位 long-double 加权累加；实验 BC 先累计各出砖类别的整数和，再按 CPU80 契约归约。不同顺序可能产生微小截断差异，也可能改变极小胜率处的零裁剪数量。

## 已实现的链路

### 正向生成

GPU 按 bitmap word 批量解码当前压缩位置，执行出砖、移动、D4 canonicalize、BC key/rank 编码，再向 GPU bucket hash 和 bitmap 插入。
同一目标的重复状态用 atomic OR 合并。hash 或 bitmap arena 容量不足时扩容并重放整批；已经插入的数据保留，不丢 candidate。

两个目标层以压缩形式滚动保留。冻结时在 GPU 按 cell/key 排序并重排 bitmap，然后写盘。
不保存整层 uint64 棋盘，也不保存整层 edges。当前生成器仍属于双目标层压缩驻留方案；尚未实现生成阶段的完整 family 边界调度。

### GPU position compact

每个 bitmap word 根据对应 dense value 的非零谓词生成新 bitmap；GPU prefix sum 重建行号，删除空 bucket，并压紧 values。
输出继续保持 cell/key/rank 顺序。验证同时检查位置集合、顺序、值，而不只检查保留行数。

### FamilyChain

每个 source family 只加载能覆盖查询的未来行列十字窗口。非对角 cell 的两个方向访问分开；第一次保存每个空位的 max，第二次先合并方向 max，再做概率归约。
每次查询都检查目标 cell 是否属于已加载窗口，缺失依赖会终止计算。

提供两种实现：

1. `family_split`：先 Spawn4，使用逐空位 partial 和逐局面 scratch；再 Spawn2 并合并。scratch 保存 UInt64 的未加权 sum4，以便最终结果遵守 resident CPU80 契约。
2. `family_fused`：两种出砖的 future 窗口同时容纳时，每次方向访问一起处理两种出砖；只保存两组逐空位 partial，省去整层 Spawn4 scratch，并减少当前位置扫描。

这不是把方向平均提前计算。两条路径都执行 `max(horizontal, vertical)` 后才对空位平均。
实验的 split scratch 数值契约有意避免生产 FamilyChain 中间阶段的额外截断，因此不声称复刻了生产 split 链路每一处舍入。

### 访存和缓存

- future 每 cell 使用平坦设备数组和指针描述表，避免每条查询通过 PCIe 访问 CPU。
- bucket 内 word-rank-base 用 UInt16；BC 单 bucket 的 rank 范围支持这一压缩。
- 静态 hash 起始容量因子由旧实验的 3.2 改为 2.0，另保留对照测量；实际容量仍按 2 的幂取整。
- family 窗口间增量保留 cell；层间保留仍会作为 future 的设备缓存。
- partial/scratch 优先用有预算的 GPU 缓存，溢出到有预算的 host 缓存，再溢出到临时文件。
- 新产生的精确 future 使用有界 host cell 缓存，避免刚写盘又立刻从盘重读。
- 若当前位置本身能放进预留空间，只上传一次，之后按 cell 使用视图。
- 可选两 cell 队列的异步写盘：有限 host staging 与下一 cell 的 GPU 计算重叠，完成全部写入后才发布层 manifest。

## 内存预算与不能外推的结论

常规大表实验采用 10 GiB CuPy 内存池硬上限；它同时限制活动分配和池中保留的块，而不只是统计 live buffer。
CUDA context、驱动和桌面显示占用不属于此内存池。需要分别记录设备池、`nvidia-smi` 整卡使用量、host working set 和 Windows private commit。
Windows private commit 不能与实际驻留 RAM 混为一谈，也不能用设备数组之和冒充整个进程占用。

更小显存的验证使用 8 GiB 内存池预算，为 context、驱动和桌面保留余量；这是在 16 GiB 卡上施加预算，不等于已经在一张实际 12 GiB 卡上实测。
24 GiB 卡可以增加 future 缓存，潜在收益主要在减少重载和 partial 搬运；没有本次实测的加速倍数。

free10 第 166 层的两个完整 future，按新设备布局估算合计约 6.92 GiB；最大的双 future family 窗口约 2.92 GiB，单 future 窗口约 1.49 GiB。
这是真实数据的窗口缩减，不能直接当成 free12 的比例。

当前容器与算子针对标量 UInt32、Full D4、freeN 语义；尚未接生产 `.bcpos/.bcsuc` I/O、BCAD、其他 dtype 或完整恢复协议。
实验单个 `Position` 的 dense row offset 为 UInt32；达到 2^32 行时明确拒绝，需进一步按 cell 分开读取。生成阶段的全局 mutable target 也没有 free12 任意规模的保证。
如果一个最小 future 窗口自身超过设备预算，当前代码会报错；未来还需查询分片方案，不能靠缩小 current batch 解决。

## 验证与计时说明

小定式使用此前已审计的 CPU fixture 文件，覆盖 GPU index、decode、position compact、fused partial 和 split scratch。
既测试全部临时结果落盘，也测试 GPU、host 和磁盘混合缓存。大表只使用 GPU 链式输出作为未来值。
大表正确性按固定种子在前六层各抽样 1,024 个局面，通过原生 EX cold reader 读取旧值；另对较高层做诊断抽样。

生成和回算报告包含解码、索引、传输、临时结果处理和位置/值写盘的 wall time，单列 kernel time。
全量过程分阶段改进并续算，汇总是已完成步骤耗时之和，不是最终版本从冷启动连续运行一次的严格统一基准。
JIT、一次性 setup、调试重跑、诊断抽样、等待用户任务和原有小实验不计入生产步骤合计；不能将 kernel 吞吐写成端到端收益。

实验使用普通缓冲文件 I/O，旧 EX 配置使用 direct I/O。当前机器还有其他定式任务运行，CPU、RAM、磁盘竞争会影响 wall time；这些数据不是空闲独占机器上的硬件排行。
未生成全部长期压缩归档，计算后只保留滚动 exact future、前六层结果与选定剖析快照；不能据此宣布完整产品导出耗时。

## 复现入口

```powershell
$gpuPython = './experiments/bc_gpu/.venv/Scripts/python.exe'
$env:PATH = 'C:/Apps/mingw64/bin;' + $env:PATH
& ./experiments/bc_gpu/build/native_fixture.exe --setup-free experiments/bc_gpu/data/free10_512_gpu 10 9
& $gpuPython experiments/bc_gpu/stream.py experiments/bc_gpu/data/free10_512_gpu
& $gpuPython experiments/bc_gpu/family.py experiments/bc_gpu/data/free10_512_gpu --init-terminal
& C:/Anaconda/python.exe experiments/bc_gpu/sample_reference.py experiments/bc_gpu/data/free10_512_gpu
& C:/Anaconda/python.exe experiments/bc_gpu/compare_samples.py experiments/bc_gpu/data/free10_512_gpu
& C:/Anaconda/python.exe experiments/bc_gpu/diagnose_initial.py experiments/bc_gpu/data/free10_512_gpu --count 1024
& C:/Anaconda/python.exe experiments/bc_gpu/collect_free10.py experiments/bc_gpu/data/free10_512_gpu
```

对当前这份旧数据，`compare_samples.py` 的层 0 失败是应当保留的检查结果；后两条命令独立诊断并汇总，绝不改写旧值。
容量/参数测试使用 `snapshot.py` 在滚动删除前保留第 166 层及未来层，再运行 `profile_family.py SNAPSHOT --step 166`。
完整 future 驻留对照使用 `resident_stream.py SNAPSHOT --step 166 --async-writes`。
GPU、host、磁盘三层临时缓存和异步写盘的小定式回归命令：

```powershell
& $gpuPython experiments/bc_gpu/validate_stream.py experiments/bc_gpu/data/free8_32_p010 --sums 52 --gpu-temp-gib 0.01 --temp-gib 0.01 --host-cache-gib 0.05 --cache-current --async-writes
```

`stream.py --resume N` 从已保存的第 N 层续做生成，会从前一位置层重建 Spawn4 carry。
`family.py --start N` 要求 N+1/N+2 的 GPU exact future 已存在。中断后的本实验临时文件可用 `--clean-temp` 清理；它只删除固定命名的实验临时文件。
外部旧表目录始终只读。
