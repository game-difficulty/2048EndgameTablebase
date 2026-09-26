# 2048Verse 历史记录导入

正式导入器使用 2048Verse 个人历史页自身调用的公开只读接口。它不扫描其他路径，默认单请求串行分页，并在请求之间等待 250ms。

```powershell
Set-Location 'C:\Apps\2048endgameTablebase\src - cloud'
.\tools\verse-history\import-verse-history.ps1 -Username game_difficulty
```

可用 `-DelayMs` 调整分页间隔、`-Retries` 调整瞬时错误重试次数、`-Force` 强制重新采集。默认输出到 `data/verse-history/<用户名>/4x4.vhs`、`3x4.vhs`、`3x3.vhs` 和 `2x4.vhs`。

继承本站账号前，服务端会对已完整的 VHS2 缓存运行 `refresh-verse-history-api.mjs`：按 Verse 对局时间倒序读取至 2026-09-24 00:00（UTC+8，含当天），按对局 ID 合并，核对 `totalGames` 和 `hs`。若计数不闭合，会在临时结果中完整多排序复查；同一 ID 的时间、分数或盘面冲突时停止且不覆盖旧档。若 Verse 当前不再返回某些已完整采集的旧 ID，但其余交集记录完全一致，则保留旧记录、记录远端移除数量和 ID 并继续导入。批量采集器运行时继承任务等待，不并行修改缓存。此刷新只用于首次继承前，不代表继承成功后的持续同步。

## 获取与完整性

每个变体首先使用 `sort=date&desc=true`。以接口返回的对局 `id` 去重；若唯一 ID 少于 `totalGames`，则依次补充日期升序、分数降序和分数升序，并按 ID 取并集。只有唯一 ID 数量等于页面声明数、最高分一致且二进制往返验证通过时，分段才会替换正式文件。

每个变体完成后立即落盘。运行中断后再次执行同一命令，会验证并跳过完整的 VHS2 分段。旧 VHS1、损坏文件或唯一数不足的文件都会重新采集。

请求遇到 HTTP 429 或 5xx 时使用退避重试。四个变体和所有分页始终串行执行，不对 Verse 后端进行并发抓取。

## VHS2

VHS2 文件头保存：

- 变体编号和实际采用的检索排序组合。
- 采集 UTC 毫秒时间。
- 页面声明数、所有检索轮次实际读取数、最终唯一数和页面最高分。

每局保存：

- Verse 对局 ID，使用 ULEB128。
- 原始 `played_at` 的 UTC 毫秒时间；记录按时间排序，后续记录保存毫秒差值。
- 得分，使用 ULEB128。
- 每格 5 bit 的棋块指数。空格为 0、2 为 1、65536 为 16、131072 为 17，可表示至指数 31。

文件不重复保存盘面字符串、一维棋块和二维棋盘。查看概要并验证格式：

```powershell
node .\tools\verse-history\verse-history-codec.mjs .\data\verse-history\j89757\3x3.vhs
```

## 检查工具与旧浏览器采集器

只读接口检查工具可单独审计某种排序：

```powershell
node .\tools\verse-history\probe-verse-api.mjs j89757 3x3 date
```

`collector.playwright.js` 保留了旧的浏览器滚动与盘面展开实现，供接口结构变化时排查。它生成的 VHS1 只保留页面本地时间到秒，正式导入器不会把 VHS1 当作已完成的接口档案。

## 排行榜批量迁移

先采集榜单快照。此命令依次读取 4x4 前 600 名以及 3x4、3x3、2x4 各前 500 名，共 42 个请求；同一变体内若分页漂移导致用户名重复，采集会失败而不会保存一个不完整名单。

```powershell
Set-Location 'C:\Apps\2048endgameTablebase\src - cloud'
.\tools\verse-history\collect-verse-leaderboard.ps1
```

默认批次目录是 `data/verse-history/cohorts/leaderboard-expanded-2026-09-25/`：

- `leaderboard-selection.csv` 保存 600 个原始名次、得分和对局时间。
- `players-selected.csv` 是按不区分大小写用户名去重后的入选表，保留每种变体的入选名次与榜单分数。
- `cohort.json` 是批量导入队列及来源元数据。
- `players-history-summary.csv` 在历史导入过程中更新，保存每名玩家四个变体各自的总局数与最高分。
- `bulk-state.json` 每处理完一名玩家就原子更新，用于中断恢复和失败重试。

执行全部队列：

```powershell
.\tools\verse-history\import-verse-cohort.ps1
```

先检查任务但不请求个人历史：

```powershell
.\tools\verse-history\import-verse-cohort.ps1 -DryRun
```

也可用 `-Limit 10` 只处理队列前十名，或用 `-Only j89757` 检查单个玩家。重复执行会验证并跳过已经完整落盘的四个 VHS2 分段。历史仍统一保存在 `data/verse-history/<用户名>/<变体>.vhs`，多个榜单批次只引用同一份玩家档案，不会重复保存盘面和对局。

批量导入只有一个进行中的 HTTP 请求，同一批次还有进程锁阻止从两个终端重复启动。每次成功响应后至少等待 500ms；单变体声明总局数达到 250、1000、5000 时，页间等待下限分别为 500ms、750ms、1000ms。玩家之间至少等待 3 秒；玩家总局数达到 1000 或 5000 时，分别增加至 6 秒或 10 秒。HTTP 429、5xx 和网络错误继续按指数退避。每页固定 50 局，因此正常情况下个人历史请求数是四个变体的 `max(1, ceil(totalGames / 50))` 之和；只有日期降序出现缺 ID 时才追加其他排序的补取请求。
