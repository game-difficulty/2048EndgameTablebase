# free12-4k 衔接计算脚本

这是一次性计算工具。默认输出 `D:/free12-4k`，源库只读；不需要转换或修改
`D:/free12`、`F:/free12` 中的 BC 数据。默认命令只打印计划，不启动计算。

## 已核实的设置与区间

| 项目 | 设置 |
| --- | --- |
| 算法 | EXAD |
| 源种子库 | `D:/free11-1k/free11_1024_` |
| 边界成功率库 | `G:/free11-2k/free11_2048_` |
| STSL | `min(108, 96) = 96` |
| 种子筛选 | 行内最大成功率严格 `> 0.4` |
| 回算剪枝 | 绝对阈值 `0.05`，相对阈值 `0` |
| 成功率格式 | `uint32`，比例尺 `4,000,000,000` |
| 出 4 概率 | `0.1`，与两个源库一致 |
| 压缩 | 结果、临时文件均不压缩 |
| 物理变换 | identity；D4 full canonical |

旧源库的 `config.txt` 没有保存 STSL，但其物理/逻辑签名包含 STSL。
`run.py` 调用现有 `_signature()`，穷举合法候选，分别唯一匹配出 108 和 96，
再通过文件头、LUT 和配置签名交叉核对；不会使用当前 UI 的 STSL 猜测历史设置。

层号以文件名中的编号为准，换算如下：

```text
S11(i) = 5 × 32768 + 20 + 2i
S12(k) = 4 × 32768 + 22 + 2k

初始：S12 = S11 - 32768 + 1024  => k = i + 511
边界：S11 = S12 + 32768 - 2048  => j = k - 1023
```

| 用途 | free11 层 | free12-4k 层 | free12 真实盘面和 |
| --- | ---: | ---: | ---: |
| 第一个种子 | 1k / 469 | 980 | 133054 |
| 第二个种子 | 1k / 470 | 981 | 133056 |
| 第一个边界 | 2k / 21 | 1044 | 133182 |
| 第二个边界 | 2k / 22 | 1045 | 133184 |

这里将“向后生成 64 层”定义为**两个种子之后新增 64 个完整层**：982–1045。
整个区间有 66 层，1044、1045 为注入边界，1043→980 为回算区间。
若希望**连种子在内共 64 层**，使用 `--new-layers 62`；边界相应变为
1042、1043，脚本会重新验证源层并打印换算结果。

## 运行

在项目 `src` 目录，使用现有 Python 环境和 MinGW/CMake：

```powershell
cmake -S tools/free12_bridge -B native_core/build-free12-bridge -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native_core/build-free12-bridge -j 6

python tools/free12_bridge/run.py plan
python tools/free12_bridge/run.py run --threads 16
```

`run` 依次执行四阶段；中断后使用**相同路径及区间参数**重跑该命令。
也可按顺序分阶段执行：

```powershell
python tools/free12_bridge/run.py prepare
python tools/free12_bridge/run.py generate
python tools/free12_bridge/run.py boundary
python tools/free12_bridge/run.py solve
```

`--chunked-solve` 使用项目现有 EXAD 分块回算；`--direct-io` 开启回算的
Direct I/O。这两项可在续算时更改。默认 16 线程，可用 `--threads` 调整。
向前生成使用 C++ 原生流程与缓冲 IO。

`--output`、`--source1`、`--source2` 可指定其他目录。不同区间或源数据必须使用
不同输出目录，已有 `bridge-plan.json` 的目录不会接受参数或源文件变化。
不要绕过 `run.py` 在正式输出目录手工调用底层 helper。

本机已经编译好 `native_core/build-free12-bridge/free12_bridge.exe`。
独立构建会将同一编译器的 OpenMP/GCC DLL 放到 executable 旁边，避免加载到
Anaconda 的旧 DLL；不替换项目正在使用的 `.pyd` 或 DLL。
回算通过当前安装的 `native_core.formation_core.run_pattern_solve_exad` 执行。

## 具体实现

**准备。** C++ 使用 `read_solved_layer_file<uint32_t>()` 读取源文件，按 slot 的
`value_base + row * row_width` 检查整行最大值。0.4 对应整数 `1,600,000,000`，
等于阈值的行不保留。保留行的 64 位 masked board 原样进入目标 builder，
不展开成真实盘面，不按每列成功率筛选。

不能直接复制源 bitmap 文件并改名：目标 LUT 的合法后缀集合、AD 参数、
盘面和以及物理签名不同。因此用目标 LUT 和 `build_layer_from_boards()` 重建
`.exadtmp`，并验证输入保留行数与重建行数严格相等。masked board 不变，目标
EXAD 的 lane 关系由现有代码决定。例如源宽度 42、336 对应目标 210、1680，
这来自 `num_free_32k` 从 5 变为 4；源成功率只用于筛选，不作为目标初值。

**生成。** 直接调用现有 `ensure_exad_temp_through_cpp()`，复用 carry、derive、
validate、STSL、容量预测和续算逻辑。两个种子作为本地 anchor，另外写一个有合法
头部的空 `0.exadtmp`，阻止原生入口生成真正的 layer0。Python 在调用前要求这三个
文件有效，缺种子时直接报错，不会从头启动 free12。

原生续算会重处理 980，将其 spawn-2 后继并入 981。这是完整续算 carry 的正常
行为，故生成后的 981 可能比最初筛出的集合大。此后 1044、1045 都是完整的
`arr1` 输出；最后剩余的 1046 `arr2` 只是部分 carry，不用于边界。

**边界。** free11-2k 的 21、22 层实际文件已验证均只有 slot 5，宽度 1。
对每个目标末层，只为 slot 5 分配成功率；其余 slot 流式读过并作为零删除。
slot 5 的每个盘面进一步验证：五个 F、四个永久 32768 之外的隐含大数恰好为
2048。直接使用相同 masked board，在源 LUT、源内存索引中查询；匹配则复制
源值到目标五列，缺失则保留零，最终通过现有 `compact_solved_layer()` 删除零行。
若延长区间导致源层已不再是标量，计划阶段拒绝该快捷算法。

**回算。** 设置 `steps = 1048`。现有 EXAD 在 `target_step >= steps-2` 时直接
返回空未来层，因此不能把 `steps` 设置为 1046。1044、1045 已有真实 `.exadbook`，
原生回算跳过它们并从 1043 开始，且能读取其成功率。脚本仅在回算期间为
0–979 创建零字节 `.exadbook`，利用已有文件名续算判断跳过区间之前的计算。
正常退出或异常时清理这些占位；进程强制结束留下的占位会在续跑后清理。
占位绝不用于未来层读取，也不会残留为可查询的假结果。

现有剪枝在层 i 回算后处理 i+2，不会改变尚需使用的未来概率。
由于 0–979 被跳过，最后另对 980、981 应用相同绝对阈值 0.05，补齐这两层的
存盘剪枝。剪枝仍是按 EXAD 行内最大值决定是否保留整行。

## 输出与续算约束

- `free12_4096_.exadlut`：目标 LUT。
- `free12_4096_980.exadtmp` 等：尚待回算的生成层。
- `free12_4096_980.exadbook` … `1045.exadbook`：最终区间结果。
- `free12_4096_config.txt`：含 STSL、格式、剪枝和物理签名。
- `bridge-plan.json`：固定计算计划及源文件大小、mtime。
- `bridge-progress.json`：完成的阶段。
- `bridge-native.log`：筛选、边界计数和 native helper 错误。
- 原生 `exad_generate_stats.csv` / `exad_solve_stats.csv`：逐层统计。

输出目录有进程锁，防止两个脚本同时操作同一批检查点。再次 `run` 会跳过已完成
阶段；单独执行某阶段时，其前置文件必须存在。回算开始后应续跑 `run` 或 `solve`。

工具生成的种子、LUT、边界和补充剪枝通过临时文件发布。原生生成/回算保留当前
仓库的写入行为；脚本会检查文件头、目录、长度、sum 和签名后才允许续算。
若强制断电造成中途写坏的层，脚本报出具体文件并停止，需先移走该坏检查点再
从对应阶段恢复。现有格式没有整文件校验和，长度检查不等于完整内容校验。

## 验证记录

```powershell
python tools/free12_bridge/test_bridge.py -v
```

已通过普通及分块 EXAD 的端到端测试：完整生成、标量转五列、源缺失、源零值、
未合出 2048 的行、严格阈值、读取注入边界、重复续算、占位清理、截断文件拒绝。
独立概率核对：两个未来层恒为 0.8 和 0.7 时，前一层每列应为
`0.9 × 0.8 + 0.1 × 0.7 = 0.79`，实际结果符合整数舍入误差。

真实源层全量筛选并重建的验证结果：

| 源层 | 原 masked 行数 | 最大值 > 0.4 的行数 | 目标重建行数 |
| --- | ---: | ---: | ---: |
| 469 | 1,808,187 | 432,320 | 432,320 |
| 470 | 1,642,495 | 414,008 | 414,008 |

另以各 128 行的真实种子样本运行现有生成器，成功生成目标 982 层。
这些验证只在 `native_core/build-free12-bridge/` 中进行；未启动正式 64 层计算。

结果沿用已接受的两项近似：源库历史剪枝可能低估；五种 2048 位置共用源值且
源库没有两个 2048 的合并机会，可能高估。脚本不试图校正这些误差。

## SSD 工作区与异步归档

`ssd.py` 复用现有 `RunOptions.pathname`（热路径）及 `cold_pathnames`
（冷路径），不修改 C++ 求解器、不改变格式/剪枝/压缩设置：

```powershell
C:/Anaconda/python.exe tools/free12_bridge/ssd.py plan
C:/Anaconda/python.exe -u tools/free12_bridge/ssd.py solve --threads 16 --hot-directory C:/2048_tables/tmp/free12-4k
```

`plan` 只读；`solve` 取得 D 盘原有 `bridge.lock`，旧计算仍运行时会拒绝启动。
切换需先结束旧进程，再校验检查点；只能恢复完整层，未完成层的分块需重算。
初次启动仅复制接下来需要的两层 `.exadbook` 和 LUT/config 到 SSD。
生成的 `.exadtmp` 留在 HDD，由核心冷热路径查找逻辑读取。
SSD 保留计算、分块、合并、成功率读取、剪枝重写涉及的文件。

独立求解子进程与后台归档线程并行。归档读取 SSD 上的完成统计 CSV，看到
第 i 层完整完成记录后才搬走 i+2 层。重启时也可归档两层活动未来层之外的
已完成 SSD 检查点。每个结果先复制为 HDD 上的 `.ssd-copying` 文件，flush/fsync，
检查源文件未变、长度和 EXAD 元数据一致，再原子替换 HDD 正式文件，最后删除
SSD 原文件。这是结构验证，不是全量内容校验和。被中断的复制可以重试，源文件保留。
最低两层等待原脚本补充剪枝后再归档。全部完成后验证 HDD 全部结果，再标记完成。

默认 SSD 预留 64 GiB（`--reserve-gib`）。归档失败或 SSD 达到预留线时，管理进程
只终止它自己启动的求解子进程，保留已完成的 SSD 检查点，报错退出；当前未完成层
可能需要重算。SSD 至少还需容纳两层结果及一个层的全部分块/合并峰值，预留线不是
对未来峰值的保证。归档无法腾出 HDD 的最终结果总容量，请同时关注 HDD 剩余空间。

切换后始终用 `ssd.py solve` 续算，不能直接用旧入口 `run.py run`，因为未归档的
检查点可能只在 SSD。逐层统计和 helper 日志位于 SSD，D 盘继续保存固定计算计划
及最终完成标记。不要在运行中手动搬动活动文件。测试：

```powershell
C:/Anaconda/python.exe tools/free12_bridge/test_ssd.py -v
```

## 扩展到 955（2026-09-24）

`extend.py` 是本次一次性迁移入口：旧回算停止后，从 free11-1k 的 444/445 层
筛选种子，独立生成目标 955–984，校验后安装到 D 盘。其中 980–984 的旧
`.exadtmp` 按要求删除替换，985 起的生成文件及全部已有非空 `.exadbook` 保留。
旧计划备份为 `bridge-before-955-plan.json`。冷热两份计划同步将最低层改为
955，原终点 1044/1045 和 solver_steps=1048 不变，随后自动调用 SSD 续算。
生成目录为 `C:/2048_tables/tmp/free12-extension-955`，计算热目录仍是
`C:/2048_tables/tmp/free12-4k`。新旧生成集合的截断衔接位于 984/985 附近，
不存在的后继依旧按零处理。

迁移完成后仍使用 `ssd.py solve` 续算；不要再次使用旧 `run.py run`，也不要再次
执行 `extend.py` 去覆盖已经回算完成的新区间。955/956 的补充剪枝和归档下限
由更新后的计划自动派生。
