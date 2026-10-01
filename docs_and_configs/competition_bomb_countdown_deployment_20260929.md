# 定时炸弹倒计时调整发布（2026-09-29）

练习项目 15「定时炸弹」生成时的初始倒计时由 16–40 改为 **12–32**（含端点，等概率整数）。规则、4×4 棋盘、5% 特殊块概率及按得分排名均未改变。页面说明与测试同步更新。

- 线上前端：`/opt/2048tables/tournament/releases/20260929-bomb-countdown-12-32-r1`。
- 比赛后端仍为 `20260929-special-practice-r3`，Nginx 与数据库均未改动。
- 可复现发布脚本：`tools/deploy_competition_bomb_countdown_20260929.py`。
- 发布前数据库只有两间 `SEATING` 房间，没有进行中的比赛；前端 50 项测试与生产构建通过。
- 公网 `/practice/13`、`/practice/14`、`/practice/15`、新 JS、项目 15 榜单和 `/api/health` 均返回 200；`/practice/15` 引用本次构建的 `index-fHLqNpnp.js`。比赛服务与 Nginx 均为 `active`。

如需回滚，仅将 `/opt/2048tables/tournament/current` 原子切回 `20260929-special-practice-r3`。旧资源和发布目录已保留；无需重启后端。
