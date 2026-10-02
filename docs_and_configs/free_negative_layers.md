# free10 及以上空小数起步与负编号 layer

## 约定

仅标准 `freeN` 且 N >= 10 改为枚举固定大数的位置、其余格为空；按完整八向
对称归一化，不执行原种子的两次移动或四角过滤。free8/free9 等较小 free、
带前缀的变体和其他定式保持现有逻辑。不承诺覆盖任意历史下所有可输入盘面。

保留 patterns_config.json 的旧 seed 和 nums_adjust，永久保持
`logical_layer = (board_sum - legacy_layer_zero_sum) / 2`。
标准 freeN 的旧小数和为 2(N-1)，因此新的起始逻辑层是 `1-N`：
free10=-9、free11=-10、free12=-11。既有非负文件不改名、不改盘面和、不强制删除。

## 实现方案

计算内核继续使用从零开始的内部序号，避免无符号索引、进度、环形缓存和终点
逻辑混入负数。新增 RunOptions.layer_offset（默认 0），统一在持久化路径处
执行 `logical_layer = internal_step + layer_offset`。Python 构建入口为新 free
设置 offset=1-N，steps 与 docheck_step 各增加 N-1，使终点盘面和、成功判定
盘面和与旧计算保持一致。真实初始盘面和仍由新种子计算。低级 API 默认 offset=0，
已有工具的显式种子/编号协议不变。

覆盖 Classic、AD、EX、EXAD（普通及分块）、BC。所有直接拼接生成、结果、
压缩、分块临时文件名的地方应用同一转换；LUT/config 名称及物理签名不变。
BC 保留内部非负 ordinal 和按绝对盘面和的文件内容，文件名逻辑编号可为负，
发现文件时反向转换到内部 ordinal；保持旧正层位置/成功率配对方式。

查询使用旧 nums_adjust 计算有符号逻辑层，允许新 free 范围内的负数，不能
无条件提前返回；读取与解码仍用真实盘面和。文件发现、随机局面、进度统计、
最优分支处理及压缩路径都须接受有符号文件编号。不能让 -1 与“未找到”的
内部哨兵混用，内部继续零起点。

## 续算与兼容

直接沿用现有断点逻辑，存在的层按当前规则跳过；不引入补算迁移流程，不预设
删除多少层，不因起点改变而批量删除旧层。用户自行删除需要重新生成/回算的文件；
正常回算时原有的已消费层剪枝、压缩、临时文件清理仍然执行。
允许旧生成集合边界处少量缺失带来的误差，缺失后继沿用原求解语义。
旧文件格式和 LUT 不升级；生成配置记录起始层/策略以便排查，但不能因此拒绝
读取旧表。生成和回算终点保持原编号；前段新增负层。

## 实施顺序与验证

1. 统一逻辑编号转换，修改 Python 种子/选项及 BC 种子入口。
2. 覆盖五类算法全部文件路径及 BC 发现/求解链路，保持内部算法零起点。
3. 修改查表、随机抽取、文件发现和进度统计。
4. 验证新旧种子规则、负层到旧 0 层的盘面和、所有算法路径、普通/分块、
   冷热盘、压缩/最优分支适用路径、重复续算与旧非负文件兼容。
5. 编译实际 native 模块并运行针对性测试，记录结果及未覆盖限制。

核心不变量：同一实际盘面查询旧正层路径不变；内部步数增加但终点和不变；
free9 种子集合不变；补算不覆盖仅因本功能而被判为无效的旧结果。

## 已实施的文件与链路

| 链路 | 修改位置 | 处理内容 |
| --- | --- | --- |
| Python 入口 | `engine_core/BookBuilder.py` | `generate_free_empty_inits` 枚举固定大数位置；四类 native 选项和 BC 入口对齐新起点；进度及配置记录负层 |
| 公用路径 | `native_core/include/FormationRuntime.h` | `RunOptions.layer_offset`、有符号文件编号解析、冷热目录路径统一转换 |
| Classic / AD | `BookGenerator.cpp`、`BookGeneratorAD.cpp`、`BookSolver.cpp`、`BookSolverAD.cpp` | 生成、普通/分块回算、结果压缩及最优分支标记（适用算法） |
| EX | `EXPrefix36Runtime.cpp` | `.exgen`、`.zbook`、`.exzbook`、最优分支续算与完成检测 |
| EXAD | `BookGeneratorEXAD.cpp`、`BookSolverEXAD.cpp` | `.exadtmp`、`.exadbook`、`.exadzbook`、普通/分块及冷热路径 |
| BC | `BCFamilyGenerationRunner.cpp`、`BCFamilySolveRunner.h` | 固定大数种子、生成文件发现、已有层保留、exact 前沿读取、归档压缩、checkpoint 和临时目录 |
| 查询 | `ReaderRuntime.cpp` | 五种 reader 的负层定位；BC 有符号盘面和差；随机抽样及文件扫描接受负编号 |
| 绑定 | `bindings_formation.cpp`、`formation_core.pyi` | 暴露 `layer_offset`，低级 API 默认仍为 0 |

旧函数 `generate_free_inits` 保留用于较小 free；BC 内部对应的
`generate_free_initial_boards` 对 N>=10 提前返回新种子集合。新逻辑枚举所有固定
大数布局并按八向对称取代表，并非只选一种摆法。free9 仍走原来的两次移动、
归一化和角落过滤。当前配置目录只定义到 free12；规则和种子辅助函数覆盖
free10..free16，未额外添加 free13..free16 的菜单/定式定义。将来添加这些标准
free 定义时仍须保留其旧基准 `2*(N-1)` 小数和。

| 定式 | 固定大数和 | 旧 layer 0 盘面和 | 新起始层 |
| --- | ---: | ---: | ---: |
| free10 | 196608 | 196626 | -9 |
| free11 | 163840 | 163860 | -10 |
| free12 | 131072 | 131094 | -11 |

计算内部 `step` / BC `ordinal` 及统计日志中的序号仍从 0 起；对应文件编号需加
`layer_offset`。例如 free12 日志内部 11 对应旧 layer 0。持久化的最优分支标记
和 BC checkpoint 保存逻辑层号，读取时还原内部序号，使旧标记继续对齐。
旧 EX 最优分支标记不能让新增负层跳过回算；缺少负层时先按原流程完成回算。

## 使用及边界

在原定式、原目录继续启动计算即可使用新起点。哪些已有层需要删掉重生成由
使用者决定，没有预设删除层数。正层的路径和二进制内容协议、AD/EXAD 分组与
成功率行宽、BC LUT/物理签名均保持不变；查旧表不要求先改 config 或升级文件。
新增负层与旧正层的集合可能不完全衔接，按既有缺失后继处理语义接受该误差。
最优分支已有处理范围由原标记继续决定，不强制重过滤全部旧层。

BC 回算延用既有 exact 前沿要求；本次允许读取归档目录里的未压缩
`.bcpos + .bcsuc` 配对作为衔接前沿。仅剩 `.bccmp/.bcraw` 的层依然可查表，
但不增加把这类剪枝归档还原为 exact 回算前沿的流程；这是现有续算能力的边界。
BC 已完成标记不能单独证明新增负层已存在。

低级 native 工具若自行提供种子和 `RunOptions`，仍由调用者显式设置 offset、
steps 和 docheck_step。标准 Python 构建入口自动完成这些调整。BC 构建链路从
标准 `freeN` 定式及 `freeN_target_` 文件前缀推导偏移。

## 验证记录（2026-10-03）

回归测试保存在 `tests/test_free_negative_layers.py`。测试只使用临时目录与小目标，
不修改实际定式目录。新增 10 项专项测试及现有测试合计 27 项，
`C:/Anaconda/python.exe -m pytest tests -q` 全部通过（17.88 秒）。

- free10..free16 种子只含 F/0、固定大数数量与对称归一化正确；新旧 layer 0
  盘面和相同；free9 原种子数量仍为 21283。
- Classic、AD、EX、EXAD 的同一 native 计算在 offset=0/-2 下按层号平移后的
  结果一致；AD/EXAD 普通及分块、重复续算通过。
- 五种 reader 的负层查询通过，Classic 旧正层路径对照一致；EX、EXAD、BC
  另覆盖压缩结果。查询测试证明路径和解码工作，不代表完整生产表的策略覆盖率。
- 非 BC 四类算法保留旧已完成非负结果，使用更早种子补负层通过；AD 旧两层
  exact 前沿继续执行原有消费后剪枝，比较对象为已完成归档层。
- BC 从固定大数种子生成/回算到旧 layer 0、保留旧非负未压缩结果后补算、
  已完成断点重复执行通过；BC 压缩负层查询与重复续算通过。
- 冷热路径、临时压缩、Classic/EX 最优分支适用路径通过。测试发现并修复了
  BC 内存缓存压缩分支在 Windows 上过早删除仍被 reader 占用源文件的问题。

编译使用 `native_core/build-formation`，CMake 增加可选
`NATIVE_OUTPUT_DIRECTORY`，默认仍输出到 `native_core`；测试时可输出到独立目录
而不干扰运行中的程序。编译产物已替换正常导入路径的
`native_core/formation_core.cp312-win_amd64.pyd`，旧模块另存为
`formation_core.previous-locked-negative-layers-20261003-002057.pyd`。
已经打开的进程仍持有旧模块，重新启动程序后使用新版。未重算生产定式。

复现命令（仓库根目录）：

```powershell
C:/Anaconda/python.exe tests/test_free_negative_layers.py -v
```

`FREE_LAYER_MODULE_DIR` 可指定独立编译目录；未设置时测试正常导入的模块。

## 完整 free10 测试发现的 BC 终止边界问题（2026-10-03）

实际 free10-128 无压缩、无剪枝测试中，Classic 与 EX 四方向结果完全相同，
BC 的上方向偏低 1.152e-6、右方向偏低 1.1e-8，超过 6e-9 容差。
把 BC 成功检查门槛改成与 EX 相同后，完整复算的四方向数值未改变，因此不能
用原有硬编码 docheck 门槛解释这次超差。

检查全部 109 个 BC 成功率文件的数据区发现，只有 layer 99 的载荷为空：
layer 98 有 76,779,067 行，layer 99 是 virtual_empty。EX 对应终止层分别有
76,779,067 和 53,984,387 行成功局面。原因是 BC 只生成到 final_sum-2，漏掉
最后一次扩展的 spawn-4 输出，又在这个本应有数据的盘面和上创建空哨兵。

修复让 BC 生成到 final_sum，并把最后两层都限制为成功局面。最后一层由已有
spawn-4 carry / source4 路径生成；前一层只含成功局面，正常的成功源跳过逻辑
不会从它额外扩展 spawn-2。求解器原有的 virtual_empty 保留在实际终点之外。
Python 的 BC 预计生成层数相应增加一层，与 EX 的真实层数一致。

新增 `test_bc_final_spawn4_terminal_matches_ex`：读取末层成功率文件实际载荷，
旧代码稳定得到空载荷而失败，修复版有非空且全为成功的载荷，并通过 EX 查询
对照。完整修复复算保留在
`C:/2048_tables/test/free10-validation-20261003/bc-terminal-recheck`，最终数值
一致性待该复算完成后确认；不得仅凭修复代码宣称完整验证通过。

旧结果不会因这个修改自动被判为无效。已有 BC 回算结果可能已传播了错误终端
边界，不能只补一个末层文件就视为全部修复；需重新回算受影响区间。此次实测
保留修复前证据，修复版使用新的独立目录完整计算。

随后将上述末端回归扩展为强制路由矩阵：生成 resident/single/family 与回算
resident/single/family 交叉共 9 组，全部通过（10.562 秒）。测试读取生成与
回算 CSV，断言实际采用的路由等于指定路由，避免小盘面自动回退到 resident
造成假覆盖。每组检查最后真实层的成功率载荷非空、行数跨路由一致且全部为 1，
并与 EX 的有效查询数值比较，容差 6e-9。此矩阵是小规模边界回归，不等同于
对三种路由各自完成一次生产规模 free10-128 测试。
