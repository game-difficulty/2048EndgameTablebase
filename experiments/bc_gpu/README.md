# BC GPU 实验路径

这是独立实验，不接入生产入口，不修改原有 BC/EX/EXAD 算法。
当前完成的是 **标量 UInt32 BC 的真实 CUDA 回算正确性和吞吐基线**。
数据由本目录的 C++ 程序自行正向生成，再调用仓库生产 BC resident 回算作为 CPU oracle。
详见 [实测结果](RESULTS.md) 和 `results/` 中的原始 JSON/CSV。

## 文件

| 文件 | 作用 |
|---|---|
| `native_fixture.cpp` | 正向生成、生产 BC CPU 回算、导出 `.bcpos` 和设备实验布局、CPU 性能测试与边界参考 |
| `kernels.cu` | 实际 CUDA：BC bitmap 解码、移动、D4 canonicalize、hash/bitmap/rank 查找、UInt32 归约 |
| `run.py` | GPU 从高层到低层回算，逐层 GPU 输出成为后续 future，逐局面／逐值验证 |
| `recalculate.py` | 对相同生成位置用另一出砖概率重新执行 CPU 回算；只硬链接不可变位置数据 |
| `compact.py` | 使用生产 BC compactor 生成零裁剪后的未来层 |
| `benchmark.py` | CPU 1/16/32 线程，GPU 64/128/256 threads/block，驻留及准备好索引后的重新加载计时 |
| `capacity.py` | 复制独立未来表，测量超过 L2 后的缓存容量压力；明确属于合成实验 |
| `validate_primitives.py` | 移动／D4 与 CPU80 舍入的独立参考验证 |
| `collect_results.py` | 审计全部层覆盖、CPU value SHA256、零差异及基线结果，汇总小型证据文件 |
| `setup.ps1` | 实验目录内的 venv 和 C++ 编译 |

`.venv/`、`build/`、`data/` 被本目录 `.gitignore` 排除。可复现源代码和小型结果放在 Git 可见路径。

## 已运行的硬件与软件

- Windows，RTX 4080 SUPER，约 16 GiB 显存，64 MiB L2。
- Ryzen 9 9950X，16 核／32 逻辑处理器。
- Python 3.12，CuPy 14.2.0，CUDA component wheels 12.9。
- MinGW g++：`-std=c++17 -O3 -march=native -fopenmp`。
- CUDA 通过 CuPy RawModule / NVRTC 编译，`--std=c++17 --fmad=false`。
- 无须安装全局 CUDA Toolkit 或 nvcc，已有 NVIDIA 驱动即可使用这个实验环境。

## 数据语义

1. free8、free9 使用项目 `patterns_config.json` 中相同的种子、Full D4、无额外 pattern mask；目标分别降低到 32 与 16，以完整生成可管理的真实可达图。
2. F 仍是普通 BC 的 32768 不合并砖，未使用 AD masking 语义。
3. 从种子逐层枚举所有空位、出 2／4、四个有效移动，canonicalize 后去重。到达目标即终止扩展。
4. 非终局每个可用格的砖值至多 target/2，故生成到 `free_cells * target/2 + 4`，再加两个显式空 future。没有截断 horizon，也没有随机未来胜率。
5. 分区为当前 BC 的 modulo，free8 用 7，free9 用 11。位置写为真实 `.bcpos v2`；实验额外导出平坦设备 hash、bitmap、word-rank-base、任务表。
6. CPU 使用 `bc_resident_solve_raw_values<uint32_t>`，终值 4,000,000,000。GPU 所有未来胜率来自前一步 GPU 输出，不将 CPU 值注入回算。
7. sparse 模式使用 CPU 生产 compactor 生成位置。GPU 自己计算 keep 谓词及压紧 value 序列，并核对压紧 board 集合、顺序与数值；还没有 GPU position compactor。

实验 layer 目录按“非 F 砖的总和”命名。free8-32 的目录 `52` 是 ordinal 19，真实 raw layer sum 为 `8*32768+52=262196`。

## 复现

以下命令在仓库根目录 PowerShell 执行。首次生成需要数 GiB 磁盘空间；不要与性能测试并发运行其他生成或 GPU 工作。

```powershell
./experiments/bc_gpu/setup.ps1
$env:PATH = 'C:/Apps/mingw64/bin;' + $env:PATH
$gpuPython = './experiments/bc_gpu/.venv/Scripts/python.exe'
$fixtureExe = './experiments/bc_gpu/build/native_fixture.exe'

# 自行生成两种非平凡定式，以及一个小型 smoke 定式。
& $fixtureExe experiments/bc_gpu/data/free6_8_p025 6 3 16 0.25
& $fixtureExe experiments/bc_gpu/data/free8_32_p025 8 5 16 0.25
& $fixtureExe experiments/bc_gpu/data/free9_16_p010 9 4 16 0.1 11

# 用同一组自行生成的位置，独立回算默认 10% 概率。
# 此命令要求目标目录尚不存在；已有数据可以直接执行后面的验证命令。
& $gpuPython experiments/bc_gpu/recalculate.py experiments/bc_gpu/data/free8_32_p025 experiments/bc_gpu/data/free8_32_p010

& $gpuPython experiments/bc_gpu/run.py experiments/bc_gpu/data/free6_8_p025 --repeats 3
& $gpuPython experiments/bc_gpu/run.py experiments/bc_gpu/data/free8_32_p025
& $gpuPython experiments/bc_gpu/run.py experiments/bc_gpu/data/free8_32_p010
& $gpuPython experiments/bc_gpu/run.py experiments/bc_gpu/data/free9_16_p010

# 真实零裁剪未来层的查表／回传验证。
& $gpuPython experiments/bc_gpu/compact.py experiments/bc_gpu/data/free8_32_p010
& $gpuPython experiments/bc_gpu/run.py experiments/bc_gpu/data/free8_32_p010 --compact-futures

# 独立移动与数值边界验证。
& $fixtureExe --probes experiments/bc_gpu/data/probes
& $gpuPython experiments/bc_gpu/validate_primitives.py

# CPU/GPU 对照；分别运行，不并发。
& $gpuPython experiments/bc_gpu/benchmark.py experiments/bc_gpu/data/free8_32_p010
& $gpuPython experiments/bc_gpu/benchmark.py experiments/bc_gpu/data/free8_32_p010 --compact-futures
& $gpuPython experiments/bc_gpu/benchmark.py experiments/bc_gpu/data/free9_16_p010 --sums 32 38 42 46
& $gpuPython experiments/bc_gpu/capacity.py experiments/bc_gpu/data/free8_32_p010
& $gpuPython experiments/bc_gpu/collect_results.py
```

命令中的 MinGW DLL 路径应与本机编译器一致。任何验证差异使 Python 返回非零退出码。

## 数值契约

直接用 GPU double 对 10% 概率加权，不能位级复现本机 CPU 的 long double，实测会产生大量 1～若干整数单位差异。
`reduce_cpu80` 用宽整数实现正数乘法和加法的 64 位有效尾数 round-to-nearest-even，并补上除法最终舍入影响，再向 UInt32 截断。

当前契约限制：

- 当前 CPU oracle 的 `LDBL_MANT_DIG=64`；不是 MSVC 的 53 位 long double。
- UInt32 胜率与本项目 4e9 终值，每个棋盘最多 16 空位。
- p4 在 [0,1] 内，其 double 精确有理数的分母指数不超过 59；0.1、0.25、0.01 均覆盖。
- 未声称支持其他 dtype、AD 向量、one-minus 或所有极小概率。

`--rounding fp64` 保留为失败对照，不能用于声称默认概率正确。

## 计时口径

- CPU：生产 resident raw-solve API 的重复 wall time，包含解码、工作计划／缓冲准备、查表和归约；未来层已加载并建索引。
- GPU 驻留：预分配缓冲、未来表已上传，测 decode + solve；同时记录 CUDA Event 和 launch/synchronize wall time。预热不计入，中位数取 9 次。
- GPU 重新加载：当前任务表、两个未来表的设备布局文件读取、H2D、解码、回算、D2H。索引已预构建，OS 文件缓存是热的；不是物理 SSD 冷读，也不包含索引生成、最终写盘或压缩。
- 数据准备、CPU 索引构建、正确性比对、JIT、初始化不会混入 kernel 吞吐。相应阶段不能因此被解释为免费。
- `peak_live_device_bytes` 是核心 buffer 大小的估算，不是包含 CUDA context、pool reserve、sparse 重建暂存的整进程峰值。

## 首轮实验的边界及后续进展

首轮实验验证了纯 BC 的 GPU 回算可行性，当时尚未实现 GPU 正向生成、GPU position compact、FamilyChain 的逐方向 partial/scratch 流水线、BCAD 或多 GPU。
真实测试数据的核心工作集只有百 MiB 量级，不能据此宣布 free12 可在 12 GiB 内完成。
容量实验虽把设备未来表扩大到约 5.94 GiB，仍是复制表的压力测试。

下一步应接入现有 FamilyChain 调度，按真实 `.bcpos` descriptor 规划 12 GiB 工作集，增量保留 future cells；同步完成设备索引构建与 position compact，再测有磁盘和 PCIe 开销的完整层。现有独立实验和 CPU oracle 可继续作为回归基线。

后续已新增 `stream.py / stream.cu` 的 GPU 正向生成和 position compact，以及
`family.py` 的 split partial/scratch、双 future 合并计算、有界多级缓存和可选异步写盘。
`validate_stream.py` 验证这些链路；`sample_reference.py / compare_samples.py` 读取既有 EX 表核对；
`snapshot.py / profile_family.py` 保存真实大层并测试预算、路线与线程块配置。
`resident_stream.py` 在两个完整 future 能放下时，对照完整 future 驻留、current 分批的路线。
`diagnose_initial.py` 独立检查旧 EX 初始层的递推一致性，保留发现的旧数据异常。
大表实验的范围、结果和复现命令以 [FREE10_RESULTS.md](FREE10_RESULTS.md) 为准。
这些仍是独立实验后端，未接入生产文件格式或 UI。
