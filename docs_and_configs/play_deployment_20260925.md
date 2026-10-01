# Play 站首发部署记录（2026-09-25）

## 2026-09-26 站长页审批事务查询

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v65-admin-approval-transactions`，以线上
`v64-live-immediate-sync` 为基线叠加发布。主站站长页用“用户查询／审批事务”页内标签切换，统一查询
Verse 历史继承和对局补录申请，支持类型、状态、用户身份、外站账号及事务编号筛选，并复用已有带审计的
批准、拒绝、重试和撤销入口。分页查询只读取申请摘要，不读取回放 BLOB；Verse 状态与更新时间增加组合索引。

主站 Nginx 的 HTTP／HTTPS 服务块新增精确的 `/api/admin/approval-transactions` 代理，指向 Play API；
主站只切换入口和新增内容寻址资源，Play 与 Live 入口未替换。上线前对局库、主站入口及 Nginx 配置备份在
`/var/lib/2048tables/backups/admin-approval-transactions-v61-20260926` 和
`/var/lib/2048tables/backups/admin-approval-transactions-v65-20260926`。生产库 `PRAGMA quick_check` 为 `ok`，
摘要查询当前读取 21 条 Verse 事务且补录事务为 0；未登录公网接口返回预期 401。Play、主站 Cloud 服务及
Nginx 均为 active，主站入口与审批页内容寻址资源返回 200，审批资源经 Cloudflare 命中 `HIT`。

发布包 SHA-256：后端 `F15E479BE65F84555634BBFB7AAFFA653918F5C7A5D6D538189DF684AF2E1545`；
主站前端 `7DD8647B88DAE2F47A079470487D1BC94A115E09D407952DBB2C7A8F47CC9484`。
回滚时将 Play `current` 切回 `20260926-play-v64-live-immediate-sync`，恢复 v65 备份目录中的主站入口和
Nginx 配置后重载 Nginx；新增索引可由旧版安全保留。

## 2026-09-26 旧浏览器与移动布局兼容

Play 静态资源当前为 `/opt/2048tables/play/releases/20260926-play-v57-page-compat`。桌面三栏原先依赖 CSS Grid `subgrid` 对齐棋盘、节点用时和排行榜；不支持该特性的浏览器会忽略行轨定义，左右栏因此落到页面底部。Play 入口现与主站共用渲染兼容检测器，检测 `subgrid`、Cascade Layers、`color-mix()`、OKLCH、`aspect-ratio` 和注册属性，在需要时加载无层叠、无新颜色语法的兼容 CSS；非 `subgrid` 环境使用明确的五行／四行网格回退，棋块正方形也有百分比内边距回退。Nginx 新增长期缓存的 `/compat/` 静态路径，Play 发布包会包含整个兼容目录。

棋盘测量和排行榜长用户名缩放不再强制依赖 `ResizeObserver`，缺失时会在初次渲染测量并监听窗口尺寸变化。真实 Chromium 中模拟同时缺少上述五类 CSS 能力和 `ResizeObserver`，在 1061×838 下棋盘、两侧内容均为 500px 高且顶底坐标一致；800px 双栏和 390px 单栏没有横向溢出，控制台无错误。前端 498 项测试和生产构建通过。静态增量包 SHA-256 为 `ee28eb441a649ec4438cec779a2205575abf9730c0c823534bdf4594de5ea74b`；本次不修改后端代码和数据库。

v55 初次上线时主站和 Play 共用同一个生成后的兼容 CSS，主站固定视口使用的高优先级 `body { overflow: hidden }` 因而在百度浏览器等兼容模式中覆盖了 Play 的长页面。v56 改为按入口分别收集依赖并生成两个内容寻址文件；Play 专用兼容包不含该全局规则。移动触摸仿真确认页面 `body/html` 均为 `overflow: visible`，棋盘外上划从 `scrollY=0` 滚到 896，棋盘内上划保持 `scrollY=0` 并继续交给棋盘。兼容专项 17 项通过，生产构建通过；v56 静态包 SHA-256 为 `4d748cb6edd4be58c355b5f848b93fac492278dedb8e95befb345300a0b95a4b`。

v57 继续检查完整榜单、个人主页、历史记录、统计、设置／规则／登录弹窗和 Verse 回放页。360×732 的旧能力移动仿真中，各长页面均可触摸滚动，`scrollWidth` 与 360px 视口一致；统计 SVG 上滑会继续滚动页面，Verse 回放页也可正常滚动。设置、分析海报和账号菜单新增 `vh` 高度回退，兼容包末尾显式覆盖 `dvh`，避免不认识动态视口单位的浏览器让长弹窗超出屏幕。兼容专项 18 项和生产构建通过；v57 静态包 SHA-256 为 `794b484479af85f279fcae6783edaff9e97e8c684ffbe6574e339b917bc058d7`。

## 2026-09-26 终局提示延迟与关闭

终局改动首次发布于 `/opt/2048tables/play/releases/20260926-play-v54-terminal-overlay-delay`，并已合并进后续发布。正式对局死亡后，封局、回放上传和榜单刷新仍立即在后台执行；棋盘上的终局提示改为等待 2 秒后以 360 ms 透明度动画淡入。提示右上角新增关闭按钮，关闭后同一局不再重复显示，玩家可直接查看或截图终盘；新对局使用独立状态，会再次正常提示。减少动画偏好的浏览器将淡入缩短为 1 ms。

新增独立时序控制器，回归测试覆盖延迟、同局关闭不复现、新局恢复提示及计时器取消。前端全量 498 项和生产构建通过。发布时线上已由另一项功能推进至 v53，因此 v54 从 `/opt/2048tables/play/releases/20260926-human-live-v53` 复制，只叠加本次构建的 Play 静态资源，未用旧基线覆盖同期功能；公网入口、内容资源与 Play 健康检查均通过。

## 2026-09-26 玩家直播、直播大厅与真人房间

Play 已切换至 `/opt/2048tables/play/releases/20260926-human-live-v53`，Live 主服务同步发布真人动态房间、玩家直播协议与大厅资源。对局页设置浮窗可为当前正式局开启直播，浏览器直接向 Live 子域发布经服务端规则重演验证的动作；`https://live.2048tables.online/lobby` 同时展示在线 AI 与玩家房间。真人房间采用随机 96 bit 地址、75 秒短期签名许可和 90 秒断线恢复窗口，同账号只保留一个活动房间，过期房间批量结束并释放容量。

真人房间沿用 Live 的聊天、礼物和活动外壳，主区域按左侧节点用时、中间棋盘、右侧当前变体前十榜排列。4×4、3×4、3×3、2×4 的主棋盘和大厅缩略棋块均保持正方形。礼物只按实际消耗的常驻 Token 部分向主播返还 50%，同一礼物请求在同一事务中写分成收据和 Token 流水；主播不能给自己的房间送礼。32768／65536 福袋只有在 Play 回查耐久进度验证通过后创建，并以正式对局 ID 保证幂等。

生产环境新增共享 `HUMAN_LIVE_SIGNING_KEY`，Play 设置 `HUMAN_LIVE_PUBLIC_ORIGIN`，Live 设置 Play 内部回查地址、可信发布 Origin、32 个真人房间上限、120 个全局观众连接上限及 90 秒恢复窗口。Live Nginx 新增 `/lobby`、真人发布 WebSocket 和只读 Play 前十榜代理。上线前备份为 `/var/lib/2048tables/backups/auth-before-human-live-v53-20260926.sqlite3`、`human-before-human-live-v53-20260926.sqlite3`、`live-app-before-human-live-v53-20260926.tar.gz`，环境文件和 Nginx 配置也保留同名 `before-human-live-v53-20260926` 副本。

Play 包 SHA-256 为 `6b968e9dd0972008b318be949f1e59ddf38b6dc136f0b050db46c43e2ef0470c`，Live 包为 `1c3170ab80a7725d511186881b10f0025d7c47db144ad2dfec8bce47d629eb80`。前端 496 项、直播相关后端 46 项通过，生产构建与服务器端导入检查通过；两个 SQLite `PRAGMA quick_check` 均为 `ok`。公网 Play、大厅、Live API 和排行榜代理返回预期状态，真实浏览器确认大厅卡片与既有 AI 房间正常，静态资源命中 CDN。

## 2026-09-26 B10 Rating 平均口径修正

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v52-rating-mean-board-sum-fixed`。四种变体的个人 Rating 改为先对当前准入 B10 的终盘盘面和取算术平均，再将平均盘面和代入该变体公式；不再先计算十个单局 Rating 后取平均。单局卡片仍保留按该局盘面和计算的单局 Rating。

评级持久化版本提升至 2，统计派生版本提升至 5；服务启动时只读取归档状态中的终盘，不读取或重演回放 BLOB，并重建玩家 Rating、RA 排名、统计摘要及成长轨迹。生产副本迁移和生产库 `PRAGMA quick_check` 均为 `ok`。mini114 四模式结果分别为 4×4 `2713.5789`、3×4 `2819.7754`、3×3 `2962.5511`、2×4 `2840.7114`，页面四舍五入后与 Verse 的 `2714 / 2820 / 2963 / 2841` 一致。上线前备份为 `/var/lib/2048tables/backups/human-before-play-v52-rating-20260926.sqlite3`；相关后端 49 项测试通过。

## 2026-09-26 对局补录申请

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v51-manual-archive-fixed`。本人设置页“游戏与记录”新增补录申请：玩家填写变体、结束时间、最终得分并上传回放；服务器在创建申请前完整解析回放、重放全部动作并核对变体与得分。通过自动校验的申请进入站长审批，审批、拒绝和撤销均保留审计记录；拒绝后允许重新提交。

批准的补录局以 `source='manual'` 进入正式归档、总榜、PB、B10、个人主页、统计及本人回放分析，历史记录标记“补录”。滚动榜候选的回填与单局写入均显式排除 `manual`，所以补录局始终不进入近 168 小时榜和每周 Token 结算。生产库检查结果为 `ok`，新表当前为空，`manual` 来源的滚动候选为 0。

主站管理后台新增“对局补录审核”面板；主站的 `/api/admin/archive-applications` 由 Nginx 转发到 Play API，与既有 Verse 审核采用相同隔离方式。Play 上传代理上限由 1 MiB 调整为 3 MiB，应用仍把回放严格限制为 2 MiB。上线前 SQLite 在线备份为 `/var/lib/2048tables/backups/human-before-play-v51-manual-archive-20260926.sqlite3`，主站旧静态目录保留在 `/opt/2048tables/app/frontend/dist.before-v51-manual-20260926`；Nginx 两份原配置也已单独备份。

发布包 SHA-256 为 `859082b1196aec79f9f774020a05fe9e68c9e5ca0072a03d5faf6467e0b8b4b5`。专项回归 12 项、后端全量 645 项、前端全量 496 项和生产构建通过。发布目录以 v50 为基线叠加 v51 包，以保留服务器上与 Python 3.10 匹配的 Linux `native_core`；首次使用空目录展开会缺少该平台二进制，已在切换主站静态文件之前回滚并按基线发布。公网用户接口与站长接口均返回预期鉴权响应，Play 健康检查通过。

## 2026-09-26 Verse 远端历史减少兼容

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v50-verse-remote-removal`。首次继承前的增量刷新若发现 Verse 当前总局数少于完整 VHS2 缓存，不再立即报 `verse_remote_count_decreased`；刷新器会完成多排序全量复查。同一 Verse 对局 ID 的时间、分数或盘面变化仍会拒绝导入；仅有旧 ID 从当前接口消失且其余交集完全一致时，保留此前完整采集的记录，与当前新增记录合并，并在 `human_external_audit` 写入 `remote_removed`，保存模式、远端数量、保留数量和移除 ID。刷新子进程的错误现在提取具体 `Error:` 行，不再因只截 stderr 尾部而向站长显示残缺路径堆栈。

真实数据复测 `mini114`：Verse 4×4 当前为 179 局，缓存为 183 局；完整复查确认交集 179 局均未变化，保留 ID `1865077`、`1865183`、`1865400`、`1865778`，同时新增 3×3 的 60 局和 2×4 的 2 局。失败申请 #8 已重试完成，最终公开导入 4×4 183、3×4 153、3×3 4,348、2×4 2,677，共 7,361 局；四模式全部可见，审核日志已记录四个远端移除 ID。上线前数据库备份为 `/var/lib/2048tables/backups/human-before-play-v50-verse-removal-20260926.sqlite3`，该用户缓存备份为 `/var/lib/2048tables/backups/mini114-cache-before-v50-20260926.tar.gz`。Node 刷新/编解码 10 项、后端继承 10 项通过；生产库 `PRAGMA quick_check` 为 `ok`，Play 健康检查通过。

## 2026-09-26 Verse 历史缓存预置

将本地 `data/verse-history` 中通过 VHS2 全量解码校验的完整档案预置到生产服务器 `HUMAN_VERSE_ARCHIVE_ROOT=/var/lib/2048tables/play/verse-history`。本地共有 951 个四模式完整用户、3,804 个 VHS2 分段、1,159,953 条记录；没有格式错误、非法用户名或大小写冲突。服务器原有 `game_difficulty` 使用服务器较新的四个分段保留，其余 950 个用户名以整目录原子移入，最终生产缓存仍为 951 个完整用户和 3,804 个分段。四个不完整本地目录未上传：`16royu65qplobk7l4izx`、`BryceYeh`、`mmmcccc`、`xyz`；不完整缓存不能免除完整抓取，预置反而可能让继承任务走额外恢复分支。

传输归档 SHA-256 为 `0916999e51b2cc4eb35c07035c443ddd24b6460626bc8b1df06655afad487715`。合并前缓存备份位于 `/var/lib/2048tables/backups/verse-history-before-seed-20260926-0955.tar.gz`，导入清单位于 `/var/lib/2048tables/backups/verse-history-seed-20260926-0955.json`。服务器档案逻辑占用约 17 MiB，4 KiB 文件系统块实际占用约 32 MiB；应用分区仍剩余约 19 GiB。服务环境下实际读取 `Hobbelweg` 与 `game_difficulty` 四模式成功，Play 健康检查和对局数据库 `PRAGMA quick_check` 均通过。预置缓存本身不认领外站账号、不导入成绩；站长批准归属后仍会先执行既有截止范围的增量刷新，再校验和导入。

## 2026-09-26 IPS／MPS 10 Hz 刷新

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v49-speed-refresh`。IPS／MPS 的显示刷新由 500 ms 一次提高至 100 ms 一次，即 10 Hz；一秒滚动统计窗口及 IPS／MPS 的计数口径不变。输入发生时仍立即更新，停止操作后最多约 100 ms 即反映样本离开一秒窗口。

工作区源码使用独立速度时钟，只在对局页、正式棋盘且开启速度显示时推进，普通连续用时仍保持 500 ms 更新。为避免夹带同期尚未发布的直播功能，线上静态文件以 v48 为基线，仅将原速度／连续时间时钟从 500 ms 提升到 100 ms；服务器请求频率、周期留档和后端负载均不受影响。速度专项 3 项、对局与会话相关 31 项测试以及生产构建通过，Play 服务健康。

首次切换 v49 时只替换了未压缩入口，沿用的 `human/index.html.gz` 仍指向 v48，启用 `gzip_static` 的公网浏览器因此继续加载 500 ms 脚本。现已在原发布目录内原子重建该预压缩入口；公网入口实际加载 `human-v49-speed100ms.js`，脚本含一处 100 ms 定时器且不含旧 500 ms 定时器。真实浏览器四次错相输入后分别在 1,037／1,052／1,036／1,007 ms 归零，符合一秒窗口加最多约 100 ms 刷新延迟。打包工具新增入口与 `.gz` 内容一致性检查，防止同类发布错误再次进入归档。

## 2026-09-26 IPS／MPS 实时显示修复

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v48-ips-mps`。此前输入与有效移动的时间样本保存在普通数组中，数组追加不会立即使 Vue 计算值失效；一秒窗口又短，页面通常只看到 `IPS 0 · MPS 0`。现在两组样本使用响应式快照，记录输入或成功移动时同步推进显示时钟，并继续由 500 ms 时钟清除超过一秒的样本。

IPS 统计正式棋盘收到的方向输入，MPS 只统计实际改变棋盘且已成功写入本地存档的移动。真实浏览器连续发送三次方向输入时立即显示 `IPS 3 · MPS 1`，停止后归零。新增时间窗口和样本上限测试；前端全量 493 项测试及生产构建通过。公网入口已加载 `human-DxdsEM4Z.js`，Play 服务健康；原高分监管阈值保持 800000／70000／5000／10000。

## 2026-09-26 高分监管阈值上调

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v47-monitor-thresholds`。新建对局进入联网高分监管的分数线调整为：4×4 严格超过 800,000、3×4 严格超过 70,000、3×3 严格超过 10,000、2×4 严格超过 5,000；等于阈值仍不进入监管。阈值在建局时写入对局记录，已有活跃局继续使用开局时冻结的旧值；上线时生产库中唯一一局旧 4×4 活跃局仍保留 360,000。

周期留档生命周期复查确认：正常死亡封存先用服务端分段与终局尾部生成并验证最终 gzip 回放，随后在同一写事务中设为 `sealed` 并删除临时 `human_chunks`；前后端均拒绝或停止该局的后续周期追加。仍为 `active` 的未死亡局没有按年龄清理路径，长期离线、在线许可过期和服务重启均保留已经接收的分段。新增回归测试覆盖封局压缩清理、封局后拒绝追加，以及模拟长期离线并重新初始化数据库后仍保留分段。

后端 38 项测试、前端监管会话 20 项通过；公网 `/api/human/config` 已返回四项新阈值并带 `Cache-Control: no-store`，服务健康。此次仅更新后端规则，无数据库迁移或前端资源重建。

## 2026-09-26 统计分享图

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v46-statistics-poster-final`。玩家统计页新增浏览器本地生成的 1600×1600 PNG 分享图，内容跟随当前变体及“按对局数／按时间”设置，包含玩家、PB、B10 分数线、B10 Rating、正式记录、两张成长曲线和终盘成就；4×4 额外包含 32K 综率成长。页脚按界面语言显示“生成于／Generated at https://play.2048tables.online/”。生成过程不上传图片，也不增加服务器端绘图负载。

切换变体时会清空上一模式的曲线与特征标签，并在新数据返回前禁用下载，避免极快操作生成混合变体图片。Rating 纵轴允许负值；不足两天的时间范围显示到分钟并只保留首、中、尾三个刻度。前端 489 项测试、分享图专项 6 项和生产构建通过；线上实测下载了 4×4／按对局数和 3×4／按时间两种图片，均为 1600×1600，页面控制台无错误。内容寻址脚本经 CDN 二次请求命中 `HIT`。

## 2026-09-26 残局实力榜定式目录

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v41-strength-catalog`。残局实力榜的定式下拉菜单现在只列出至少有一条有效正式分析记录的定式；当前生产数据因此只显示 `free10-256`（4 条）和 `free10-512`（2 条），不再显示无记录的可分析表库。

新增 `human_analysis_catalog` 持久化小表。排行榜目录请求只顺序读取该表，不扫描分析事实，也不查询实时表库目录；新增或撤销分析资格时仅对受影响定式通过既有覆盖索引计数并更新一行，全量评级重建和首次迁移则一次聚合回填。查询计划已确认读取路径只扫描目录主键索引，写入计数使用 `human_analysis_results_week` 覆盖索引。

上线前 SQLite 在线备份为 `/var/lib/2048tables/backups/human-before-play-v41-strength-catalog-20260926.sqlite3`，迁移副本与生产库 `PRAGMA quick_check` 均为 `ok`。后端专项 86 项通过；公网目录接口仅返回上述两个有记录定式，Play 服务健康。

## 2026-09-26 终盘成就扩展

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v40-terminal-achievements`。统计规则提升至版本 4：2×4 固定按“满盘、512、二阶满盘、512+256、三阶满盘”显示；3×3 增加“1K+512、三阶满盘”；3×4 增加“4K+2K+1K+512+256”及继续包含 128 的下一层。2×4 的 512 使用非数字内部键，避免 JavaScript 把整数键提前枚举而破坏界面顺序。

上线前备份为 `/var/lib/2048tables/backups/human-before-play-v40-achievements-20260926.sqlite3`，`PRAGMA quick_check` 为 `ok`。启动后 110 条既有派生事实已全部重建为版本 4；统计、Verse 导入、完整榜单和实况统计相关 51 项测试通过。公网接口和真实页面已核对三种变体的档位及顺序，页面控制台无错误；4×4 综率保持 `43/92`，成长曲线仍从第 10 局开始，Rating 榜 B10 分数线仍为 `795032`。

## 2026-09-26 32K 综率成长起点

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v39-rating-and-rate-threshold`，合并保留 Rating 榜 B10 分数线与统计规则 v3。32K 综率成长曲线只返回第 10 局及以后的累计节点；第 10 个点包含前十局的全部阶段事实，前九局不单独绘制。当前生产数据返回 40 个点，横轴从 10 到 49，首点为 9/19，最终总综率保持 43/92。后端相关 42 项测试通过；公网 Rating 榜仍返回 B10 线 795032，Play 服务正常。

## 2026-09-26 Rating 榜 B10 分数线

最终线上合并版本为 `/opt/2048tables/play/releases/20260926-play-v39-rating-and-rate-threshold`。Rating 榜详情列由“基于 N 局”改为“当前 B10 线 · 分数”，直接左连接已持久化的 `human_player_statistics.b10_score`，不扫描历史对局；不足十局时显示“—”。英文同步为 `Current B10 line`。该版本同时保留统计规则 v3 及同批发布的综率阈值调整，生产库 110 条派生事实仍全部为版本 3。

线上 4×4 第一名接口返回 Rating `2966.256...`、B10 线 `795032`；公网入口加载 `HumanLeaderboardPage-Cm9mn2Ek.js`，其中包含新中英文文案且不再包含“基于 N 局”。相关后端 62 项、前端专项 4 项和生产构建通过；最终服务进程健康，公网 API 与内容寻址脚本均已复核。

## 2026-09-26 玩家统计终盘综率定稿

Play 当前为 `/opt/2048tables/play/releases/20260926-play-v37-statistics-labels`，基于完整排行榜 v35 发布内容叠加统计修正。统计规则版本提升至 3：4×4 综率按终盘的 32K、16/32、24/32、28/32、30/32、31/32 六层计算五个递进阶段，不再读取回放或复用实况 `StageCounter`。无回放 Verse 导入局与本站局采用同一口径。

用户“游戏难度”的生产数据为 `49 / 22 / 11 / 7 / 3 / 0`，分子 `43`、分母 `92`，综率 `46.739%`。生产库 110 条派生事实已全部重建为版本 3，`PRAGMA quick_check` 为 `ok`。页面显示 `43 / 92 个残局阶段 · 49 局达到 32K`，终盘柱图使用简写层级；成长曲线横轴按真实首尾自适应，SVG 内部比例和刻度字号已放大。上线前备份为 `/var/lib/2048tables/backups/human-before-play-v36-statistics-20260926.sqlite3`。

验收：用户给出的首组及五组层级数据均写入固定测试，统计、导入、完整榜单和实况统计相关 50 项通过；前端 485 项和生产构建通过。公网接口返回版本 3、43/92 和完整七档成就，浏览器实际页面无控制台错误或警告。前端发布包为 `output/play-v36-statistics-final.tar.gz`，SHA-256 为 `0A93372172C6C10CE139E98DE12C9F4338A39A27D4DF191D2AC3F41BAE41320D`；v37 只在此基础上更新后端显示标签。回滚时将 `current` 原子切回 v35 并重启服务。

## 2026-09-26 完整排行榜与残局实力榜

Play 已切换至 `/opt/2048tables/play/releases/20260926-play-v35-full-leaderboards`。新增独立 `/leaderboard` 页面，提供四变体分数总榜／近 168 小时榜、四变体 B10 Rating 总榜、四变体主要成就数量榜、4×4 的 32K 综率榜，以及按定式筛选的 4×4 残局实力总榜／近 168 小时榜；所有榜单固定每页 50 人。综率榜要求至少 10 局达到 32K。实力榜每位玩家、每个定式只取最佳正式分析，内部精确加权分只用于排序，不进入公开 API；页面明确只计入从个人主页历史对局发起的“对局－定式”分析。

后端新增窄的分析事实、总榜最佳、周榜最佳和维护状态表。分析摘要完成时在同一事务内更新榜单事实，人工资格变更会同步刷新；近 168 小时实力榜由 Play 后台每分钟分批剔除过期结果。读取榜单不选择回放归档 BLOB。Rating、PB、成就数量和综率继续复用现有持久派生表，Verse 批量导入沿用已有刷新路径。

上线前备份为 `/var/lib/2048tables/backups/human-before-play-v35-leaderboards-20260926.sqlite3`。先在备份副本执行完整迁移并通过 `PRAGMA quick_check`，再原子切换发布目录；生产库迁移后同样为 `ok`。当前事实表有 6 条正式分析，两个定式拥有总榜结果。线上 catalog 列出 22 组可评级 4×4 定式，五类分页接口均返回正常；实力榜公开响应不含 `weighted_score`，综率接口返回最低 10 局门槛。Nginx 已增加 `/leaderboard` 的 SPA 入口，并为两个公开完整榜单接口保留后端的 30 秒公共缓存头；JSON 源站响应启用 gzip。入口和内容寻址脚本公网返回 200，静态脚本二次请求为 Cloudflare `HIT`，浏览器实际读取到中英文资格说明及 `free10-256` 榜单。

验收：后端正式测试目录 627 项通过，排行榜、分析、Verse 导入和滚动榜相关测试 84 项通过；前端 485 项及排行榜专项 4 项通过，生产构建通过。主站、Play 健康接口和 Play 公网页面均返回 200，Play 服务启动日志无本次发布错误。发布包为 `output/play-v35-full-leaderboards.tar.gz`，SHA-256 为 `60699C9365F610EDB467B0F294C7FCC04C5A6112490187EC8A74AD8D6694032E`。回滚时将 `current` 原子切回 v34，恢复 `/etc/nginx/play.2048tables.online.before-v35-20260926` 并重启 Play、重载 Nginx；新增窄表可由旧版忽略。

## 2026-09-26 玩家统计页显示与 32K 综率修正

Play 已切换至 `/opt/2048tables/play/releases/20260926-play-v34-statistics-corrections`。变体选择器不再重复显示 B10 Rating；成绩、Rating 与 32K 综率折线按实际数据首尾确定横坐标，时间轴不再被 Unix 时间 0 拉伸，对局数轴也不再强制从第 0 局开始；坐标刻度、图例和提示文字已放大。

v34 曾把 32K 综率误改为主站实况的合并事件口径并显示 35/49；该结果不符合 Play 已确定的终盘层级定义，已由后续 v35 纠正，不应作为历史统计结论引用。v34 上线前备份位于 `/var/lib/2048tables/backups/human-before-play-v34-statistics-20260926.sqlite3`。

验收：统计相关后端 49 项、前端 485 项及生产构建通过；公网统计接口返回版本 2 和新序列，入口加载 `human-stats34-ceb826de.js`，Play 服务正常。发布包为 `output/play-v34-statistics-corrections.tar.gz`，SHA-256 为 `3500792BD6C9EC1443D3DADFDEC0C7C34FF07CB877D2B471CF2DB10EEC7D7F83`。

## 2026-09-26 玩家统计页后端上线

Play 已切换至 `/opt/2048tables/play/releases/20260926-play-v33-player-statistics`。此前统计页前端已经包含在线上静态资源中，但 v32 后端没有 `/api/human/users/{username}/statistics`，页面因此固定显示“无法读取统计”。本次以 v32 运行目录为基线，仅发布统计表、统计接口以及本站封局、人工审核、Verse 导入、Verse 补回放的派生事实更新路径，没有带入工作树中尚未发布的完整排行榜后端。

首次启动将 110 局既有 Verse 记录回填到窄表；其中 19 局已有完整回放，迁移期间一次性重演并得到 16 局 32K+ 回放覆盖。统计页面日常读取不访问归档 BLOB。当前 4×4 共 91 局，49 局达到 32K，已覆盖局的累计残局阶段为 19/35。数据库 `PRAGMA quick_check` 通过；上线前备份位于 `/var/lib/2048tables/backups/human-before-play-v33-statistics-20260926.sqlite3`。

验收：统计相关后端 63 项、前端 481 项及生产构建通过；公网统计接口返回 200，浏览器从个人主页切换到统计页及直接刷新 `#statistics` 均能显示数据，控制台无错误或警告。主站、Play 首页和 Play 健康接口均返回 200。v32 目录已恢复为发布前内容，可通过原子切换 `current` 回滚；统计窄表保留不会影响旧版运行。

## 2026-09-26 表库分析桥接更新

Play 当前切到 `/opt/2048tables/play/releases/20260926-play-v21-tablebase-bridge`，以前一版 `20260926-play-v20-navigation-performance` 为基础，仅覆盖表库目录、远程分析读取器及新增的内部桥接模块。主站 `/opt/2048tables/app/backend/app.py` 增加内部表库路由，原文件备份为 `backend/app.py.before-play-tablebase-bridge-20260926`；新增 `backend/remote_workers/internal_bridge.py`。Play 环境文件新增 `REMOTE_TABLEBASE_PROXY_BASE_URL=http://127.0.0.1:8000`，原文件备份为 `/etc/2048tables/play.env.before-tablebase-bridge-20260926`。

Play 通过带现有工作器密钥的 loopback 接口读取主站远程工作器在线目录，并将批量分析查询交给主站连接的工作器；工作器离线时不列出对应定式。内部接口验证密钥及定式、目标是否在主站远程清单中。没有向公网添加新的匿名表库查询能力。

验收：本地桥接、远程工作器和人类分析相关测试 **32 项通过**；线上内部目录列出 **18 个远程表库组合**，实际 `free10_512` 批量查询成功。Play 服务环境中读到 **30 组、17 种定式**，其中 4×4 **26 组、13 种**，与主站实时目录一致。主站表库 API、Play 健康接口和公网首页均返回 200。没有扣费运行完整生产对局分析任务。

回滚：将 Play `current` 原子切回 `20260926-play-v20-navigation-performance`，恢复备份的 Play 环境文件并重启 `2048tables-play.service`。若同时回滚主站桥接，恢复 `backend/app.py.before-play-tablebase-bridge-20260926` 到原位、移除新增的 `internal_bridge.py` 并重启 `2048tables-cloud.service`；两边回滚顺序以先 Play 后主站为宜。对局数据库未迁移。

公开地址：`https://play.2048tables.online/`。Cloudflare 中 `play` 的 A 记录指向现有云服务器，开启代理。Let's Encrypt 证书通过 webroot 签发，到期日 2026-12-24；Certbot 的自动续期任务使用 Nginx 的 ACME 路径。

发布包 SHA-256：`f20ea5950439aa5ea77e079d39e113ae88af80a0804c4c46f3048080a9f0183a`。源自本地 `src - cloud` 工作树（HEAD `4b34f41`，包含尚未提交的 Play 改动）；`tools/package_play_release.py` 只包含 Play 所需的 Python 源码、前端 `human` 入口、静态资产与 Verse 回放器，不包含本地数据库或导入缓存。

线上文件与进程：

- `/opt/2048tables/play/current` 当前指向 `/opt/2048tables/play/releases/20260925-play-v8-account-redesign`；旧版本 v1–v7 留存供回滚。发布新版本时将旧哈希资产复制到新版本，保持已打开页面的资源可用。
- `2048tables-play.service` 只绑定 `127.0.0.1:8766`；主站与直播站仍用原有服务和前端目录。Nginx 直接提供静态文件，仅将 Play 的 `/api/` 代理到该进程。
- 账号与 Token 使用主站 `/var/lib/2048tables/auth.sqlite3`，Play 对局证据和压缩归档位于 `/var/lib/2048tables/play/human.sqlite3`。Verse 导入缓存与分析临时目录位于 `/var/lib/2048tables/play/`。
- 上线前用 SQLite 在线备份 API 保存账号库至 `/var/lib/2048tables/backups/auth-before-play-20260925.sqlite3`，备份已通过 `PRAGMA quick_check`。
- Play 独立执行人类周榜 Token 结算，采用统一奖励键与独立结算游标；未来主站升级为统一滚动榜结算时，重复奖励由唯一凭证避免。

验收：本地 Play 相关后端测试 44 项、前端测试 467 项通过，后续结算/导航修订后的后端测试 34 项、前端测试 467 项通过，构建通过。公网主站、直播站、Play 首页与 Play 配置接口均返回 200；Play 预览登录接口返回 404。浏览器已验证主站共享账号自动登录、四变体入口、个人主页/设置和返回棋盘。曾发现个人主页往返持有对局锁，现已在页面离开时释放，并在主页入口跳过对局激活；复测通过。Play 的 gzip 静态 JS 经 Cloudflare 二次请求返回 `cf-cache-status: HIT`。

当前范围：主站账号设置同步已于同日[独立发布](main_settings_deployment_20260925.md)。Play 的分析任务创建、结果轮询与下载已部署，但本次没有用真实归档和 Token 扣费执行一项生产分析。

设置文案补丁：v4 `releases/20260925-play-v4-settings-labels` 仅替换 Play 的 `human/index.html` 与新增的内容寻址脚本，补齐滑动灵敏度、成绩展示阈值、每秒输入／移动次数、出 4 比例等中英文标签，并将“游戏与AI”改为“游戏与记录”。v3 仍保留，可通过原子切换 `current` 回滚。公网 Play 新入口和脚本均返回 200；主站与直播页返回 200。

练习板提醒补丁：`current` 随后切换到 `releases/20260925-play-v5-practice-reminder`。练习板第 41 次有效移动弹出中英文提醒；撤销恢复此前移动计数，重置清零，同一来源局面只提醒一次。只替换 Play 静态入口和新增脚本；练习板测试 11 项、前端构建通过。公网 Play 新入口、脚本、主站与直播页均返回 200。

回滚设置文案补丁（不回滚对局数据库）：将 `/opt/2048tables/play/current` 原子切回 `releases/20260925-play-v4-settings-labels`，无需重启服务。账号库备份仅用于数据库级事故恢复；恢复前须另行保全上线后的新增账号和 Token 流水。

账号入口与导航补丁：`current` 随后切换到 `releases/20260925-play-v6-account-ui`。游戏页“设置”固定打开浮窗；登录用户名与未登录入口分别复用主站账号控制组件和登录／注册组件，并增加适合 Play 浅色、深色主题的样式。个人主页入口移至页头“回放”和“规则”之间，“设置”右侧增加新窗口打开主站的入口。前端构建、Play 交互测试 11 项、账号设置同步测试 2 项通过；本地浏览器检查了两种主题、个人主页路径上的设置行为和账号菜单，公网浏览器检查了新入口与登录／注册浮窗。新脚本的预压缩响应为 44,771 字节，Cloudflare 已命中 `HIT`。回滚此补丁可将 `current` 原子切回 `releases/20260925-play-v5-practice-reminder`。

规则文案补丁：`current` 随后切换到 `releases/20260925-play-v7-rules-copy`。规则浮窗简化练习、阈值联网、服务器留档和终局上传说明，明确服务器不能恢复本地对局，并提示为 C 盘预留空间；中英文同步更新。前端构建和 Play 交互测试 11 项通过，公网规则浮窗已逐项核对。回滚此补丁可将 `current` 原子切回 `releases/20260925-play-v6-account-ui`。

账号浮窗视觉重做：`current` 随后切换到 `releases/20260925-play-v8-account-redesign`。Play 专属样式将身份、额度和操作分组，采用圆角悬浮卡片、两列额度摘要和独立总余额层级；主要操作、安全操作与退出使用不同层级，并分别适配米白棕金浅色主题和炭灰暖金深色主题。桌面浅色、桌面深色和 390 px 窄屏均通过浏览器截图检查；前端构建和 Play 交互测试 11 项通过。回滚可将 `current` 原子切回 `releases/20260925-play-v7-rules-copy`。

导航与加载性能优化：`current` 于 2026-09-26 切换到 `releases/20260926-play-v20-navigation-performance`。对局页入口将个人主页、登录账号菜单和分析弹窗拆为按需加载，入口脚本由约 146 KB（gzip 53 KB）降至 62 KB（gzip 22 KB）；配置与身份请求并发执行，线上不再请求仅供本地预览的接口。对局、玩家主页和回放改用 History API 切换，榜单、个人最佳与历史摘要在当前页面会话内复用，返回页面不再整页刷新或重新生成 Best 10 海报。BFCache 返回时恢复会话检查，不再强制刷新。Verse 回放字体采用一年不可变缓存。公网冷启动实测配置与身份请求同一时刻发出，初始脚本传输约 22 KB；棋盘—玩家主页往返的 Navigation Entry 始终为 1，第二次进入同一主页没有新增 API 请求。前端构建以及 Play 交互、会话与回放导出测试 37 项通过。

排行榜对齐与海报名次配色：`current` 于 2026-09-26 先切换到 `releases/20260926-play-v21-eleven-row-leaderboard`，右栏固定为前十名加本人名次共 11 行；游客或本人无成绩时用空白 `-` 行占位。总榜在既有每人最佳查询内返回本人名次，近 168 小时榜从已维护的玩家最佳窄表读取，不查询回放。实测左右内容区均高 500 px、各 11 行、行高 38 px，排名块为 38×38 px。随后静态前端切换到 `releases/20260926-play-v22-poster-medals-practice-label`，Best 10 海报采用独立金银铜前三名标签及统一 B4–B10 标签色，并移除练习板文字后的外链箭头。

## 2026-09-26 4×4 分析展示图评级

本次发布将 4×4 分析展示图的内部评级接入分析摘要：服务器在原有付费分析完成时，把评级版本、残局级别所需的汇总指标和 SSS–F 结果一并持久化；旧摘要在读取时可根据已保存的小型摘要懒计算评级，不重新读取或分析回放。3×4、3×3、2×4 暂不生成评级展示图。展示图仍由前端生成，评级不足以判断时显示待评级。

线上 `current` 已原子切换至 `/opt/2048tables/play/releases/20260926-play-v24-rating`。v24 以健康的 v22 运行目录为底座，仅覆盖本次发布包内容，保留服务器 Linux 原生模块、字体和配置资源；此前单独的 v23 发布包因缺少原生模块未切换并已保留作审计。v24 远端编译、应用导入、评级样例和 `/api/human/health` 均通过，公网 Play 首页返回 200。没有使用真实归档和 Token 扣费执行生产分析。

本地验收：评级及摘要相关后端测试与既有 Play 后端测试共 71 项通过，前端测试 474 项通过，生产构建通过。发布包 SHA-256：`5fc52251864f3aad1daa625d97d72cbaf3b49549de1a64bc2d75c3476af0dc80`。

分析展示图静态资源热修复：线上随后切换到 `/opt/2048tables/play/releases/20260926-play-v25-poster-assets-route`。二维码两张 WebP 已存在于 `frontend/dist/human/`，但 Nginx 原先只允许入口页并让其余 `/human/` 请求落入 404，导致浏览器在 Canvas 绘制前报 `poster_assets_unavailable`。v25 增加只读、长期缓存的 `/human/` 静态资源位置，并用新的查询版本及内容寻址脚本绕过 Cloudflare 已缓存的旧 404；未带入尚未发布的统计页代码。两张资源公网响应分别为 35,402 与 93,782 字节、状态 200，新的展示图脚本和健康接口均返回 200。回滚时将 `current` 切回 v24，并恢复 `/etc/nginx/play.2048tables.online.before-poster-assets-20260926`。

内测时间与 Canvas 兼容热修复：线上随后切换到 `/opt/2048tables/play/releases/20260926-play-v26-poster-timing-compat`。内测阶段将 Verse 早期归一化回放中的 `100 ms` 间隔按真实记录参与速度评分；评级版本升至 2，既有付费分析摘要在首次读取时直接从持久化摘要升级并保存，无需重新分析或扣费。展示图脚本同时为缺少 `CanvasRenderingContext2D.roundRect` 的浏览器增加兼容路径。部署复核还发现入口 `index.html` 更新后遗漏重建 `index.html.gz`，Cloudflare 因而持续取得指向旧脚本的预压缩入口；v25、v26 的入口压缩文件均已重建。公网入口现指向 `human-posterfix26.js`，后者加载 `HumanAnalysisDialog-posterfix26.js`。生产摘要 `#371d6405 / free10-256` 已从待评级升级为 `S`，平均有效步间隔约 `1.162 s`；服务健康检查通过。回滚可将 `current` 切回 v25，但 v25 不会读取版本 2 评级，数据库中已升级的小型摘要无需删除。

分析弹窗文案热修复：线上随后切换到 `/opt/2048tables/play/releases/20260926-play-v27-analysis-copy`。对局概览不再显示对局内部短 ID，任务进度不再显示分析任务 UUID；费用确认区的 Token 余额固定显示一位小数。仅替换分析弹窗内容寻址脚本及入口，入口 `.gz` 同步重建。前端 474 项测试和生产构建通过；公网入口已确认加载 `human-posterfix27.js` 与 `HumanAnalysisDialog-posterfix27.js`，健康接口正常。回滚可将 `current` 原子切回 v26。

分析展示图信息与视觉更新：线上随后切换到 `/opt/2048tables/play/releases/20260926-play-v28-analysis-poster-stats`。非 PB 对局改为显示其在本人该变体公开有效历史中的当前名次；同分按较新的对局优先，与历史成绩排序一致。分数、MAX COMBO 和平均吻合度采用描边、渐变与柔和光影的新数字样式；吻合度标签增加 16K／32K／65K 残局级别；评价统计底部增加与主站回放页同色序的 5 px 七段比例条。既有分析摘要读取时即时计算当前个人名次，不重新分析回放。相关后端测试 34 项、前端测试 474 项及生产构建通过；样稿核对了个人名次和 PB 的中英文分支。公网入口加载 `human-posterfix28.js` 与 `HumanAnalysisDialog-C-RN6VhD.js`，生产摘要读取返回个人名次，健康接口正常。回滚可将 `current` 原子切回 v27。

对局页榜单精简：线上随后切换到 `/opt/2048tables/play/releases/20260926-play-v31-compact-ranking`。右侧常驻榜单的每行仅保留用户名和分数，用户名与分数的原有跳转行为不变；完整榜单浮窗不受影响。长用户名固定为单行省略，完整名称保留在悬停标题中，分数列不收缩。前端 479 项测试和生产构建通过；公网可访问树确认榜单行不再出现最大块及回放文字，静态 CSS 预压缩资源命中 CDN。

排行榜长用户名适配：线上随后切换到 `/opt/2048tables/play/releases/20260926-play-v32-ranking-name-fit`。桌面右侧排行榜从 210 px 增宽 30% 至 273 px，页面容器同步增宽；用户名根据浏览器实际排版宽度从默认 13 px 连续缩小，最低为 6.5 px，达到下限仍放不下时才显示单行省略号，悬停标题继续保留完整名称。公网模拟验证中，中等长度中文名缩至约 9.88 px 后完整显示，超长中英文混排名称在 6.5 px 下省略，固定得分列未被压缩。前端 481 项测试和生产构建通过。

## 2026-09-26 真人直播视觉与信息同步

Play 当前切换至 `/opt/2048tables/play/releases/20260926-play-v60-live-best-sync`，Live 使用同一 build id `b8a16f643a6d-20260926034747930`。本轮修复直播分享失败，主播主题与棋块颜色随直播同步；真人直播间的节点用时和排行榜采用 Play 对局页的三栏结构，棋盘 gap 颜色随 Live 的浅色／深色模式。大厅缩略棋盘改用共享 `getTileLabelStyle` 字号规则，2–64 放大 20%，四位与五位以上数字按与其他棋盘一致的比例缩小。直播间棋盘上方第二项由步数改为 BEST，主播个人最佳随握手发送，并在个人最佳接口返回后增量同步；显示值不低于当前局分数。

直播后端只持有当前房间的小型最佳分和配色快照，不增加逐步数据库查询。上线前 Live 文件备份位于 `/var/lib/2048tables/backups/human-live-ui-before-v57-20260926.tar.gz`，v59 后端单文件备份位于 `/var/lib/2048tables/backups/human_content-before-v59-20260926.py`。回滚 Play 可将 `current` 切回 v57-page-compat；Live 可恢复上述 tar 中的 `backend/live` 和 `frontend/dist` 对应文件后重启 `2048tables-cloud.service`。

验收：前端 501 项、真人直播后端 7 项测试及生产构建通过。Play 与 Cloud 服务均为 active，loopback 健康接口正常；公网 Play 和 Live 入口加载同一 v60 build id，Live 大厅 API 同时返回 AI 和重连后的真人房间。

真人直播即时同步与等高修复：Play 随后切换至 `/opt/2048tables/play/releases/20260926-play-v64-live-immediate-sync`，Live 静态入口同步使用 build id `b8a16f643a6d-20260926040744909`。主播端在更新响应式 `run.seq` 前先把新动作加入内存回放，修复观众端固定落后一步的问题；本地持久化失败时仍会撤销该内存事件。直播说明缩短为仅列出公开内容。

直播房间以 500 px 为 4×4 棋盘和两侧正文栏的共同高度；矩形变体根据正方形棋块实际计算棋盘高度，两侧正文同步采用该高度。固定 38 px 行与 6 px 间距按可用高度决定默认节点数，因此 4×4 默认展示 32K 在内的 11 个节点且榜单、节点栏不再初始滚动；实际达到默认范围以外节点时仍追加并启用滚动。前端 502 项测试及生产构建通过，公网入口、内容寻址 JS/CSS 和 Play 健康接口均返回 200。发布采用最小静态包，未包含同一工作区中并行开发的管理后台资源；Live 入口备份位于 `/var/lib/2048tables/backups/live-static-before-v64-20260926.tar.gz`。

直播间不可用页更新：Play 当前从已有 v66 发布目录切换至 `/opt/2048tables/play/releases/20260926-play-v67-live-empty-state`，Live 入口使用 build id `b8a16f643a6d-20260926122311783`。无效、已结束或主播不在线的房间现在显示与直播大厅一致的暗色空状态，以弱化 4×4 棋盘和离线标记作为主视觉；主操作改为返回直播大厅。404 房间不再提供无效的重试，临时网络故障和内容版本不匹配仍保留重试入口；中英文按浏览器语言显示，桌面与 390 px 窄屏均已检查。前端 502 项测试和生产构建通过，公网无效房间、直播大厅及两项服务状态正常。发布只覆盖 Live 入口 JS/CSS，未带入同一工作区的其他改动；Live 静态备份位于 `/var/lib/2048tables/backups/live-static-before-v67-20260926.tar.gz`。

真人直播主动结束事件：Play 当前切换至 `/opt/2048tables/play/releases/20260926-play-v69-room-ended`，Live 最终入口 build id 为 `ac7358c05fa1-202609261343-hotfix2`。主播主动关闭后，数据库房间状态确认结束，Live 后端在一秒维护周期内向现有观众广播一次 `room_ended`；观众页停止重连并由 `LiveApp` 直接切换到“房间不可用”空状态。普通发布连接断开仍显示离线并保留原有 90 秒恢复窗口；后台页面或已断开的观众在重连前只额外检查一次房间详情，只有明确 404 才结束页面。后端协议测试 9 项、前端全量 517 项、直播生命周期专项 5 项及生产构建通过；线上 AI 房间浏览器检查无控制台错误，部署代码的结束／普通离线分支复核通过。回滚备份位于 `/var/lib/2048tables/backups/live-before-v69-room-ended-20260926.tar.gz`。

对局中 BEST 即时更新：Play 随后切换至 `/opt/2048tables/play/releases/20260926-play-v68-live-best`。游戏页 BEST 始终取该变体历史记录与当前局分数的较大值，服务器返回较旧记录时不会降低浏览器已知值；直播发布握手同样使用该实时值。Live 服务端在接受每一步后单调抬高房间 `best_score`，快照和后续 BEST 消息都不能把它降到当前分数以下。主站普通游戏、小游戏、AI 直播原有实现已经采用相同语义，本次没有重复改动。前端相关测试 19 项、真人直播后端 8 项通过；公网 Play 实际走到 60 分时 SCORE 与 BEST 同步为 60，浏览器控制台无错误。回滚 Play 可将 `current` 原子切回 v67；Live 后端备份位于 `/opt/2048tables/app/backups/live-best-20260926T051956Z`。

真人直播空棋格配色修复：Live 入口于 2026-09-27 更新为 build id `493a99dccc5c-202609270140-empty-grid`。真人直播间不再使用主播从 Play 上传的空棋格颜色；空棋格与 AI 直播统一读取 Live 自身的 `--color-empty`，已有数字的棋块仍继承主播主题。相同规则覆盖房间主棋盘、直播大厅缩略图和画中画 Canvas。直播专项测试 4 项及生产构建通过；公网真人房间已核对为空棋格深海军蓝、非空棋块仍为主播经典主题。回滚备份位于 `/var/lib/2048tables/backups/live-before-v70-empty-grid-20260927014123.tar.gz`。

AI 直播最高分下注于 2026-09-27 升级为选项池匹配规则 `matched-option-pools-v2`。唯一胜方与每个落败选项分别匹配 `min(胜方池, 落败选项池)`，各选项内按个人本金比例分配收益或损失；并列第一的选项合并为获胜联合池，不互相赔付，且每个落败池只被匹配一次。因此拆分账户或追加次数不再改变选项间的匹配量。最小 Token 单位的整数尾差在每个“获胜池－落败选项”对内使用确定性最大余数法分配，保证收益与损失精确对称。

规则版本按批次持久化：已有本金入池的 `matched-accounts-v1` 批次继续按原规则结算，尚无下注的旧批次可安全升级，新批次使用 v2。上线时存在一个已封盘、已有一人下注的 v1 批次，已确认保留 v1 标记。前端 Live build id 为 `380a5c1eaa76-20260927-prediction-v2`，中英文规则浮窗已更新；公网入口、内容寻址脚本及房间 API 均返回 200。上线备份为 `/var/lib/2048tables/backups/live-before-v71-prediction-v2-20260927-020621.tar.gz` 和 `/var/lib/2048tables/backups/auth-before-prediction-v2-20260927-020621.sqlite3`。预测专项 29 项、直播相关合计 50 项及生产构建通过；另用 1,000 组任意账户和金额结构验证三个选项等概率时，每个账户三种唯一胜方结果的净收益之和精确为 0。

65536 独立加注于 2026-09-27 改为达标后本金不退：65536 档总到账为加注额的 8 倍，65536＋32768 档为 50 倍，升级仍只补 42 倍差额。技术性作废时尚未兑付的加注仍全额退款，已兑付奖励不重复退款。个人历史中的加注本金改为从原始 stake 读取，不依赖结算行中的本金返还字段。

加注规则使用独立持久版本：部署时当前已封盘批次有 3 位用户、共 6,100 Token 旧规则加注，已标记为 `target-principal-return-v1` 并继续履行原承诺；新批次使用 `target-no-principal-v2`。前端根据批次版本切换说明和预估到账金额。线上 Live build id 为 `08dc6d4ed811-20260927-target-rules-v2`，公网入口、静态资源和房间 API 均返回 200；当前批次页面已实测仍显示旧规则的 900／5100 到账说明。直播相关 52 项测试及生产构建通过。回滚备份为 `/var/lib/2048tables/backups/live-before-v73-target-rules-20260927-022628.tar.gz` 和 `/var/lib/2048tables/backups/auth-before-target-rules-v2-20260927-022628.sqlite3`。

最高分下注于 2026-09-27 进一步升级为固定两两对手盘规则 `matched-pairwise-pools-v3`。复核发现 v2 在并列第一且各获胜选项池大小不同时，虽然资金守恒且没有拆号收益，但不能保证“各选项命中率相等时每位用户的期望净收益为零”。v3 在封盘时仅按三个选项总额建立互不重复使用本金的 A-B、A-C、B-C 对手盘；结算时只有一方押中的对手盘发生等额转移，双方都押中或都未押中则退款。每组风险的正反方向完全镜像，因此严格公平性对唯一赢家和各类并列均成立。

规则继续按批次固化。部署时最新批次 `de7789f3-3065-4ee4-af0e-b7bbfec3c527` 已封盘且有 4 个账户、1,700 Token 本金，保留 `matched-option-pools-v2` 并按下注时承诺结算；之后尚未下注及新建批次使用 v3。线上 Live build id 为 `e3b0c869fbfa-20260927-pairwise-fairness`，当前 v2 批次的规则浮窗已实测显示清晰的旧规则说明，新批次脚本包含 v3 的两两对手盘说明。预测、批次和直播路由共 56 项测试通过；另以 1,000 组随机账户及金额结构遍历七种非空胜者集合，验证相等边际命中率下每个账户的期望净收益精确为 0。服务端同一反例也返回 `[0,0,0]`，账号库 `quick_check` 为 `ok`。回滚备份为 `/var/lib/2048tables/backups/live-before-v74-pairwise-fairness-20260927-052643.tar.gz` 和 `/var/lib/2048tables/backups/auth-before-pairwise-fairness-20260927-052643.sqlite3`。

直播间最高分下注与 65536 独立加注的四个单次金额档位随后统一调整为 100／1,000／5,000／10,000 Token。服务端金额白名单同步移除 500、加入 10,000，不能通过绕过按钮提交旧档；接口返回的新金额同时驱动两个下注区域，静态回退值和中英文规则说明也已更新。线上 Live build id 为 `e26b0175375b-20260927-prediction-amounts`，真实页面已核对两组按钮均显示新四档，规则浮窗显示同一金额列表。下注及直播相关 57 项测试、生产构建、账号库 `quick_check` 和公网资源检查通过。回滚备份为 `/var/lib/2048tables/backups/live-before-v75-prediction-amounts-20260927-162707.tar.gz` 和 `/var/lib/2048tables/backups/auth-before-prediction-amounts-20260927-162707.sqlite3`。

福袋份额算法于 2026-09-27 升级为固定份数总额守恒模型。开奖前先生成 10 份符合上下限的随机份额并统一打乱：32K 的 10 份严格合计 5,000，65K 的 10 份严格合计 20,000；满 10 位中奖者时必定发完整个奖池，中奖人数不足时只发放随机序列中对应数量的份额。创建活动时同时校验奖池必须能由 10 个上下限内的份额组成，防止配置到开奖时才失败。

红包与福袋的领取位置公平性已复核。随机红包先生成总额守恒的正整数拆分，福袋先生成带上下限的总额守恒拆分，两者都在分配前执行独立均匀乱序，因此每个领取／中奖位置对整批份额可交换，条件期望分别严格等于红包总额除以份数、福袋总额除以 10。线上后端单文件已更新，备份为 `/var/lib/2048tables/backups/lucky_bags-before-v76-20260927-171632.py`；账号库在线备份为 `/var/lib/2048tables/backups/auth-before-lucky-bags-v76-20260927-171511.sqlite3`，`PRAGMA quick_check` 为 `ok`。福袋、红包、批次、直播路由与真人直播相关 61 项测试通过；服务器随机校验 200 批均满足长度、边界和总额约束，部署时没有待开奖福袋，Cloud 服务与 Live API 正常。
