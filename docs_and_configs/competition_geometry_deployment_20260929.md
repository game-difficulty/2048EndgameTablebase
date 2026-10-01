# 比赛棋盘尺寸与直播布局发布（2026-09-29）

本次发布将随机形状棋盘改为 6×6 来源并裁成最小外接矩形，将「越来越大」改为 4 行、5 列；同步更新比赛适配器、练习站、直播棋盘字号与华容道直播布局。旧比赛房间仍可通过 `tournament-v2` 适配器运行；新建项目 10、12 使用 `tournament-v3`。两个试玩榜单的规则版本升至 2，旧成绩保留但不混榜。

## 发布目标

- 比赛后端：`/opt/2048tables/tournament-app/releases/20260929-geometry-r1`
- 比赛前端：`/opt/2048tables/tournament/releases/20260929-geometry-r1`
- Live 静态入口：`/opt/2048tables/app/frontend/dist/live/index.html`，仅追加内容哈希资源，不替换主站或 Human 的 HTML 入口。
- Live 入口备份：`/var/lib/2048tables/backups/competition-geometry-20260929-r1/live-index.html`
- 可复现发布脚本：`tools/deploy_competition_geometry_20260929.py`；服务器端暂存于 `/opt/2048tables/deploy-stage/competition-geometry-20260929-r1`。

发布前后，比赛数据库只有两间 `SEATING` 房间，没有进行中的对局。比赛后端与前端原先均指向 `20260929-projects-11-12-r1`；发布脚本检查这些前置条件后才原子切换，并在健康检查失败时回滚。

## 验证

- 本地赛事后端测试 98 项、赛事前端测试 44 项通过；两套前端构建通过。
- 浏览器本地实测：4×5 试玩棋盘为 620×496px；随机棋盘已按局面裁剪。华容道直播棋盘在 368×436px 容器中为 204.56×357.98px，在 158×436px 窄容器中为 158×276.5px，均保持 4:7 且没有碰到底部步数区域。直播出口为中央 1、2 列。
- 服务器端注册检查通过：12 个项目均可解析；项目 10、12 的新适配器可初始化并生成正确棋盘尺寸。
- 公网 `/api/health`、`/practice/10`、`/practice/12`、Live 首页及新版 JS 资源均返回 HTTP 200；两个试玩榜单返回规则版本 2；比赛服务保持 `active`。
- 公网比赛前端引用 `index-wJTQwqgr.js`；Live 前端引用 `live-JDq-dvxT.js`。Live 入口 SHA-256：`4f0b1ea968f499a15661916161995c2897730c205e51e1127d2c772b775b4b74`。

## 独立发现

首次发布前冒烟检查尝试初始化旧的「对抗出数」适配器时，发现当前服务器上的 `native_core.ai_core.EvilGen` 只暴露 `gen_new_num`，而目前线上比赛代码调用 `gen_new_num_seeded`。这个问题在旧发布目录中同样存在，未由本次棋盘尺寸变更引入。发布前检查因此终止，**当时没有切换线上版本**；随后只对本次新规则做初始化冒烟检查，发布成功。该兼容问题后来通过独立比赛后端原生模块发布修复，见 `competition_seeded_ai_deployment_20260929.md`。

## 回滚

将比赛后端及前端的 `current` 符号链接分别指回 `20260929-projects-11-12-r1`，重启 `2048tables-competition-test.service`；将备份 `live-index.html` 还原至 Live 入口。新资源与发布目录保留，可供审计，无需删除。
