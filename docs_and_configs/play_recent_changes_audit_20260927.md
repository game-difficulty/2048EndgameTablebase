# 2026-09-27 最近提交核查

范围：`ebeab2c` 公共分析库至 `ca94160` 删除弹窗，共同核查删除、个人页缓存、分析任务重试、公共分析库、阶段回放、实际发布配套。其他直播业务改动不在本轮修改范围。

## 已确认问题与修复

| 问题 | 证据 / 影响 | 修复 |
| --- | --- | --- |
| 删除按钮全禁用 | 空字符串传入 Vue disabled，真实页面按钮 disabled | ca94160 已改为空闲 null |
| 删除接口遗漏上线 | Nginx 记录真实 DELETE 返回 405；前端升级而后端没有路由 | v77 已补上，新增成套打包与发布验证工具 |
| 删除错误被弹窗挡住 | catch 写入页面级 error；页面“重试”只调用 loadHistory | 弹窗内展示独立错误，确认按钮可重复发起幂等删除 |
| 删除读取无用回放 | SELECT 包含 archive，删除只用身份与模式 | 删除读取移除 BLOB；写事务串行化；SQL authorizer 测试禁止读取 archive |
| 个人页缓存竞争 | 缓存命中在请求序号递增前返回，旧网络响应仍能覆盖当前页 | 每次读取先使旧请求失效，包含缓存命中 |
| 409 重试丢失请求身份 | analysis_request_in_progress 也清掉 request_id | 仅价格/目录变化使报价失效；排队与幂等映射在同一事务写入 |
| 异步开窗被拦截 | 四个入口在 await 请求之后 window.open | 点击时先保留窗口，请求成功再导航；失败关闭；被阻止时当前页导航 |
| 刷新阶段回放失败 | 游客线上实测第一次播放成功、刷新后 401、空棋盘、登录框 | 按 artifact ID 保存并恢复该标签页回放，确认保存成功后才清除地址凭证 |
| 阶段步号不一致 | 新选择器直接显示零基起始 index；旧历史为 index+1 | 统一为用户可见的一基起始步号 |
| 筛选条件与游标不匹配 | 翻页时取正在编辑的 filters，却沿用旧游标 | 翻页读取已提交筛选快照 |
| 回放滑块恢复不一致 | 初始化时先设置 value，再更新 max，浏览器会将 value 截到 0 | 先更新范围再设置 value；浏览器核对步号、盘面和滑块 |

## 验证原则

- 后端对临时数据库执行完整删除、重复删除、他人/游客权限、剩余 PB 晋升、周榜及统计刷新，不删除生产玩家数据。
- 回放首次打开和刷新必须通过浏览器验证；资源 200、健康接口 200、测试源码含特定字符串均不代替流程验证。
- 当前线上版本不能仅凭 frontend/release.json 判断后端版本。完整发布包包含 play-release-manifest.json，逐文件校验源代码和静态文件。
- `tools/package_play_release.py` 拒绝脏发布源码、前后端提交不一致或 HTML/gzip 不一致。
- `tools/verify_play_release.py --root RELEASE_DIR` 校验包内文件；`--base-url URL` 检查 DELETE 鉴权路由、评价筛选及阶段回放入口。405 会阻止通过。
- 保留旧 hashed 静态资源，以供已打开页面使用；发布时同时更新普通和 gzip 入口。

## 本轮明确的验证边界

- 真实用户付费分析不会被自动触发。扣费与请求去重在隔离数据库中验证，线上只验证公开分析读取和回放。
- 浏览器验证使用桌面 Chromium 与移动视口；不能替代真实百度/Safari/所有 WebView 的设备兼容测试。
- 本轮没有修改单局评分、Rating、综率计算公式或直播加注规则。

## 最终验收和发布

- 后端相关测试：112 项通过，涵盖历史、分析、公共库、榜单、Verse 导入、滚动榜、设置和发布检查。后续仅调整回放入口版本，分析历史和发布检查 6 项再次通过。
- 干净发布源码前端：542 项通过；生产构建通过。工作区 545 项包含未提交的其他直播功能测试，不能用该数字代表发布包。
- 本地隔离账号：真实浏览器验证取消/关闭、模拟 503 后错误在弹窗可见、重试 DELETE 200、删除行消失、次佳仍保留。验证历史缓存慢响应与海报往返。
- 公网游客：评价 A 筛选、阶段列表、签名回放首次打开、前进一步、刷新后保持步号和棋盘；刷新没有请求回放 API。移动端 390×844 深/浅色浮窗检查通过，关闭后解除滚动锁。
- 公网接口检查使用真实浏览器请求；Python urllib 默认 UA 遇到边缘 403，不能误报为源站故障。源站 `verify_play_release.py` 全部通过。
- 当前发布：`/opt/2048tables/play/releases/20260927-play-v79-audit`，源码 `918ee6689f3dc1d6d74370e3abcd965d87ede5c2`。主站静态回放入口同步，主站后端/直播入口没有替换。
- Play 包 SHA-256：`12ebee9b5e334c720532cf9aeae4eb77b9b3ea6178e4f8d60b743d06e0a1b335`；主站静态包：`111d588d045614c95bba6974583d3aa715d48fa6740353932c0b355aa30780ad`。
- 发布目录所有清单文件哈希一致，Play/主站服务 active，Nginx 配置检查通过，最后一次发布后的日志无 ERROR/Traceback。
- 核查中 v78 的刷新复测发现 Worker 转移了原始 ArrayBuffer；v79 改为保存解析器返回的缓冲区，并通过本地及公网刷新验证。v78 不作为最终验收版本。
- 回滚：Play current 切回 v77（本轮前版本）；主站 index.html、index.html.gz、release.json 从 `/var/lib/2048tables/backups/main-before-play-v78-audit` 恢复。保留新旧内容寻址资源，不需要回滚数据库。

## 后续版本升级前必须处理

`human_analysis_summaries` 目前的唯一键仍为 `(run_id, pattern, target, metric_version)`，没有包含 `analyzer_version`。当前分析器版本固定为 1，不改变现有结果；将来提高分析器版本前必须迁移此约束和 upsert，以免覆盖旧版本成果。本轮没有对线上摘要表做这项结构迁移。

截图位于 `output/playwright/audit-delete-failure.png`、`audit-delete-success.png`、`audit-public-replay-refresh.png`、`audit-library-mobile.png` 和 `audit-library-mobile-light.png`。
