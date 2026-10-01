# 比赛站完整测试部署记录（2026-09-28）

项目 1 后续发布情况见 `competition_cargo_deployment_20260928.md`；以下线上结构
与验证数据保留为首次完整测试部署时的记录。

当前已实现的比赛站前后端已部署到独立测试入口：

- 比赛站入口：`https://tournament.2048tables.online/test`
- 房间地址：`https://tournament.2048tables.online/rooms/{room_code}`
- 项目试玩：`https://tournament.2048tables.online/practice`
- 新增试玩：`https://tournament.2048tables.online/practice/9`、`https://tournament.2048tables.online/practice/10`、`https://tournament.2048tables.online/practice/11`、`https://tournament.2048tables.online/practice/12`
- 健康检查：`https://tournament.2048tables.online/api/health`

`/test` 与现有试玩页共用同一份前端构建，但根路径仍跳转到
`/practice`，因此测试比赛流程不会替换已公开的试玩入口。

## 身份与权限

测试站使用主站真实 `tb_shared_session` / Bearer token，未开启
`COMPETITION_ALLOW_DEV_AUTH`。未登录请求 `/api/session` 返回 HTTP 401。
主站角色为 `admin` 或 `organizer` 的账户可创建房间；测试部署还通过
`COMPETITION_BOOTSTRAP_ORGANIZER_IDS=5` 将现有站长账号授权为比赛主办方。
这项授权只影响比赛系统，不修改主站用户角色。其他已登录账户可作为
队员、队长或观众进入授权范围内的房间。

## 线上结构

- systemd：`2048tables-competition-test.service`
- 服务端口：`127.0.0.1:8770`
- 后端发布目录：`/opt/2048tables/tournament-app/releases/20260928-undo-death-v9`
- 后端当前链接：`/opt/2048tables/tournament-app/current`
- 前端发布目录：`/opt/2048tables/tournament/releases/20260928-growing-tiles-v16`
- 前端当前链接：`/opt/2048tables/tournament/current`
- 独立数据库：`/var/lib/2048tables/competition-test/competition.sqlite3`
- 环境文件：`/etc/2048tables/competition-test.env`
- Nginx 站点：`/etc/nginx/sites-available/tournament.2048tables.online`
- Nginx 上线前备份：`/etc/nginx/sites-available/tournament.2048tables.online.before-full-20260928`

本地可复现配置：

- `deploy/2048tables-competition-test.service`
- `deploy/competition-test.env`
- `deploy/tournament.nginx.conf`

发布包：

- 前端 SHA-256：`b29bd6af11fad614a568d0124d9955b96189a42597252ee30267f05f6fa721d0`
- 后端 SHA-256：`804d2eede1b5f38f65ca73bca4f27a544d08258c3318c89fe17b8b5a07b1c8d3`

## 验证结果

- 比赛前端测试 21/21 通过。
- 比赛后端测试 56/56 通过。
- 比赛服务重启后健康检查返回 schema version 7。
- SQLite `PRAGMA quick_check` 返回 `ok`。
- `/test`、`/rooms/ABC123`、`/practice/2`–`/practice/12` 均返回 HTTP 200。
- WebSocket 反向代理已接通，未认证握手正确以 4401 关闭。
- Nginx 配置检查通过，比赛服务与 Nginx 均为 active。
- 主站、Play 和 Live 公网入口均保持 HTTP 200。
- 服务环境已确认加载生产 `native_core.ai_core.EvilGen`。
- 大满盘撤销竞速的撤销只回退棋盘、得分和步数，不回退 RNG 游标；真实浏览器验证中，同一局面第一次重走生成 4，撤销后再次重走生成 2。
- 项目 9 的孤岛图像直接使用主站小游戏的版本化 `portal.png` 资源；公网资源返回
  HTTP 200、`image/png`、10375 字节，内容 SHA-256 与仓库原图一致。
- 项目 10 的背景格与 tile 统一使用绝对坐标布局；Playwright 在 6×3 非正方形棋盘上
  测得两者 `x/y/width/height` 最大差值为 0，修复了原浮动布局的纵向累计错位。
- 项目 10 另有专用非正方形视觉适配：保留统一棋盘底板与配色，空洞显示为底板区域，
  单元格按方格单位和等宽间距布局。线上样本格子宽高比为 1，桌面和窄屏检查通过。
- 项目 10 的生成来源调整为 7×7，最大完整矩形面积调整为 4–8，有效格仍固定为 12。
  前后端各验证 200 组确定性样本；线上样本为裁切后 4×6、12 个有效格、矩形面积 5，
  且全部四向连通。
- 比赛棋盘动画统一采用主站 BaseBoard 的持久 tile 与中断快进方案，不再为每一输入帧
  重建临时 moving layer。项目 9、10 在线上以 12ms 间隔各输入 48 次，移动帧的对角
  漂移次数和正交偏差均为 0；镜面项目连续输入后全部 tile 正常收束，控制台无错误。
  项目 9 在实际生成孤岛后另捕获 41 个孤岛移动帧，专项漂移指标也均为 0。
- 对抗出数前后端统一增加低盘面和保护：数字和小于 120 时搜索深度最高为 5，数字和
  等于 120 时仍按空位数使用原自适应深度。浏览器试玩把同步 WASM 搜索移入 Worker；
  正式比赛请求等待与试玩搜索等待期间都立即锁定输入，不缓存后续按键。单步超过 300ms
  才显示“AI 思考中”胶囊。慢 Worker 实测中，350ms 时步数保持 0，等待期间连续三个方向
  输入均被丢弃，完成后步数仅增加到 1。公网 WASM 以 `application/wasm` 返回，真实浏览器
  加载 `tournament-v2` 资源后完成有效移动且控制台 0 错误。
- 项目 9、10 的公开说明已改为选手视角的简短规则，不再公开孤岛生成概率、随机形状
  来源范围或完整矩形面积等内部生成参数；前端试玩目录、详情页和后端新建项目目录保持一致。
- 大满盘撤销竞速在未达到 2044 前不再因无路可走而完成项目。死亡盘面保持 playing，
  因而不会显示终局浮窗、不会冻结计时，且撤销和重开操作继续可用；达到 2044 后仍立即完成。

## 回滚

1. 用上线前备份恢复 Tournament Nginx 配置，执行 `nginx -t` 后重载。
2. 停止并禁用 `2048tables-competition-test.service`。
3. 将 `/opt/2048tables/tournament/current` 原子切回
   `/opt/2048tables/tournament/releases/20260928-practice-v3-direct-input`。
4. 保留比赛数据库和发布目录供审计，不需要删除。
