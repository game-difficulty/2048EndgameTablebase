# 比赛项目试玩站部署记录（2026-09-28）

项目 1「真华容道」已在后续版本开放 `/practice/1`，发布记录见
`competition_cargo_deployment_20260928.md`。以下为此前版本记录。

`tournament.2048tables.online` 首次上线时只公开了项目试玩页。
2026-09-28 后续已在 `/test` 接入完整比赛前后端，详见
`docs_and_configs/competition_full_test_deployment_20260928.md`。十一个稳定试玩地址为：

- `https://tournament.2048tables.online/practice/2`：50% 出 4（3×3）
- `https://tournament.2048tables.online/practice/3`：对抗出数（4×4）
- `https://tournament.2048tables.online/practice/4`：纯 2 满盘竞速（3×3）
- `https://tournament.2048tables.online/practice/5`：大满盘撤销竞速（3×3）
- `https://tournament.2048tables.online/practice/6`：骰子障碍（3×3）
- `https://tournament.2048tables.online/practice/7`：镜面 64 砖×10 竞速（4×4）
- `https://tournament.2048tables.online/practice/8`：256 砖（5×5）
- `https://tournament.2048tables.online/practice/9`：孤岛（困难，4×4）
- `https://tournament.2048tables.online/practice/10`：随机形状棋盘（困难，12 格）
- `https://tournament.2048tables.online/practice/11`：百步封锁（4×4，仅试玩）
- `https://tournament.2048tables.online/practice/12`：越来越大（4×4，仅试玩）

项目目录为 `https://tournament.2048tables.online/practice`，站点根路径会跳转到
该目录。旧的 `/projects/{project_ref}` 项目标识地址继续可用。

## 线上状态

- 发布目录：`/opt/2048tables/tournament/releases/20260928-growing-tiles-v16`
- 当前软链接：`/opt/2048tables/tournament/current`
- Nginx 配置：`/etc/nginx/sites-available/tournament.2048tables.online`
- 本地配置源：`deploy/tournament.nginx.conf`
- 发布包 SHA-256：`b29bd6af11fad614a568d0124d9955b96189a42597252ee30267f05f6fa721d0`

该版本移除方向缓存和 100ms 人工节流，普通项目的每次输入都立即计算并提交新棋盘帧，
由棋盘动画在收到更新帧时自行快进。真实浏览器中 40 次方向输入在约 95ms 内完成派发。
终局浮窗在最后一步后约 2 秒渐入，可关闭且不会重置棋盘。对抗出数页面不再公开搜索深度。

上线后逐一检查 `/practice/2` 至 `/practice/8`，均返回 HTTP 200；EvilGen WASM
返回 HTTP 200、长度 22689 字节。主站、Play、Live 均保持 HTTP 200，Nginx
配置检查通过且服务为 active。

同日后续版本修正了大满盘撤销竞速的随机游标：撤销只恢复棋盘、得分和步数，
不会回退随机序列，因而重走同一步会继续消耗新的出数。

项目 9、10 复用本地主站小游戏的困难规则，但明确不接入 Powerups。孤岛在无现存
孤岛时以 5% 概率生成，每个现存孤岛降低 2 个百分点；相邻孤岛只与同类收束合并且
不计分。随机形状在 7×7 范围生成、按比赛版困难模式最大完整矩形面积 4–8 筛选并裁切，
有效格总数固定为 12。前后端各用 200 组确定性样本验证了格数、来源边界、连通性和
矩形范围。

孤岛特殊块不再使用 CSS 模拟图案，改为直接引用小游戏正式资源
`https://2048tables.online/minigames-assets/portal.png?v=minigames-img-20260710b`，并沿用
小游戏的 cover 填充与蓝色辉光。线上返回的 PNG 与仓库 `pic/portal.png` 的 SHA-256
均为 `6fcd1cb6347e9ddd73f83d119765c01ba7242b36a551fb65ce92fafc0a20b046`。

随机形状裁切后可能形成非正方形棋盘。v7 将背景格从带百分比 margin 的浮动布局改为
与 tile 共用同一绝对坐标函数，消除了纵向百分比 margin 按棋盘宽度计算造成的错位。
线上 6×3 非正方形样本中，初始 tile 与对应背景格的 `x/y/width/height` 最大差值为 0。

v8 对随机形状棋盘做了独立视觉适配：恢复与其他项目一致的棕色圆角棋盘底板，以底板
区域表达不可用空洞，并改用“正方形格子 + 等宽间距”的单位几何计算外框比例。线上
非正方形样本的格子宽高比为 1，tile 与背景格最大坐标差仍为 0；桌面和 390px 窄屏
均完成真实浏览器复测。

v10 将共享比赛棋盘动画改为与主站 `BaseBoard.vue` / `useBoardAnimation.js` 相同的
持久 tile 模型：新帧中断旧动画时先无过渡快进到上一提交终点，经 `nextTick` 和强制
重排后再执行下一段滑动；合并与新出块随后分阶段显现。线上以 12ms 间隔各输入 48 次，
项目 9 捕获 330 个移动帧、项目 10 捕获 206 个移动帧，对角漂移次数与正交偏差均为 0。
另在项目 9 实际生成孤岛块后继续以 12ms 间隔输入 48 次，单独捕获 41 个孤岛移动帧，
孤岛 tile 的对角漂移次数和正交偏差同样为 0。

v12 将对抗出数的 EvilGen 搜索移入独立 Web Worker，搜索未返回时棋盘、键盘和滑动
输入均锁定且不会缓存；单步等待超过 300ms 后才显示“AI 思考中”胶囊，短计算不会
闪烁。空位自适应深度保持不变，但盘面数字之和小于 120 时深度额外封顶为 5，恰好
等于 120 时不触发封顶。线上 Nginx 同步补充 `application/wasm`，资源版本升级至
`tournament-v2`，避免浏览器因 MIME 错误放弃流式编译并重复下载 WASM。公网真实浏览器
验证 Worker、JS 与 WASM 均为 HTTP 200，首次有效移动完成后无控制台错误或随机出数回退。

v13 精简项目 9、10 的选手文案。孤岛只保留特殊块的合并与计分规则；随机形状只保留
随机 12 格、不可进入棋盘外区域和结算方式。生成概率、生成来源范围、完整矩形面积等
实现参数不再出现在试玩目录或详情页。

v14 修正大满盘撤销竞速的死亡恢复流程：未达到盘面和 2044 时，即使当前盘面无路可走，
项目仍保持进行中，不显示死亡浮窗并允许继续撤销或重开；仅在真正达到目标后进入完成状态。

v15 新增仅供试玩的项目 11“百步封锁”。4×4 棋盘每 100 个有效移动轮换封住 3 格，
解封上一轮格子；封住的数字保留但不能移动、合并或在该格出数，无路可走后按得分结算。
遮罩使用可透视数字的深色阴影及锁图标。比赛后端、正式建房项目池未改动。
前端 12/12 项测试、构建通过；线上 `/practice/11`、`/practice`、`/test`、`/api/health`
及主站、Play、Live 均返回 HTTP 200。浏览器确认项目 11 可加载并完成有效移动，控制台无错误。
Nginx 配置备份为 `/etc/nginx/sites-available/tournament.2048tables.online.before-hundred-step-seal-v15`。

v16 新增仅供试玩的项目 12“越来越大”：4×4 标准出数，64+64 合成双格 128；
两个 128 移动时优先取可达到的两格重合，否则单格重合生成三格直线或 L 型 256；
多格砖整块移动，256 不可再合并，死亡后按得分结算。百步封锁改为开局先随机封住
3 格再生成两块初始数字；每次百步轮换也先更新封锁格再出数，所有出数均排除当前封锁格。
比赛后端和正式建房项目池未改动。前端 21/21 项测试、构建通过；公网 `/practice/11`
与 `/practice/12` 均为 HTTP 200，浏览器确认前者开局 3 个封锁格与 2 块不重叠数字，
后者可操作且控制台无错误。Nginx 配置备份为
`/etc/nginx/sites-available/tournament.2048tables.online.before-growing-tiles-v16`。

## 回滚

如需回滚 v16，将 `/opt/2048tables/tournament/current` 原子切回
`/opt/2048tables/tournament/releases/20260928-hundred-step-seal-v15`，并从 v16 的 Nginx 备份
恢复站点配置、检查配置后重载 Nginx。发布目录保留用于审计，不需要删除。
