# 练习项目 13–15 发布（2026-09-29）

已将「出双入对」「化学反应」「定时炸弹」发布到独立赛事站的 `/practice/13`、`/practice/14`、`/practice/15`。三项均为 4×4、每次出数有 5% 概率产生特殊块，死亡后按标准得分排行。炸弹初始倒计时在 16–40 之间等概率取整数；从该次出数票据的 value 通道取值，不额外推进 RNG。三项尚未加入正式比赛 BP 项目池。

## 发布范围

- 前端：`/opt/2048tables/tournament/releases/20260929-special-practice-r3`，从线上前一版复制并覆盖新 `dist`。
- 后端：`/opt/2048tables/tournament-app/releases/20260929-special-practice-r3`，从线上前一版复制，仅覆盖 `competition/backend/practice_leaderboard.py`；原先的比赛 EvilGen 原生模块保持不变。
- Nginx：`/practice/{1–15}` 路由，原配置备份为 `/etc/nginx/sites-available/tournament.2048tables.online.before-20260929-special-practice-r3`。
- 可复现脚本：`tools/deploy_competition_special_practice_20260929.py`。主站、Play、Live 均未发布新文件。

## 发布前后验证

- 发布前数据库只有两间 `SEATING` 房间，无进行中的比赛。
- 本地前端 50 项测试、试玩榜单后端 6 项测试与生产构建通过。
- 线上三个新练习地址、三个试玩榜单接口、`/practice`、`/test`、`/api/health` 均返回 200；榜单返回 `metric: score`。
- 公网 `/practice/15` 引用与本次本地构建相同的 `index-BGXkzbJj.js` 和 `index-DdFxMQ6g.css`，JS 资源返回 200。
- 旧 `/practice/12` 以及主站、Play、Live 首页均返回 200；比赛服务和 Nginx 均为 `active`。

首次两次发布尝试因脚本对 Nginx 新路由的即时冒烟检查过早返回 404，均自动回滚到旧版；改为带正确 SNI 并等待重载生效后，`r3` 成功。旧发布目录和失败尝试留下的新目录均保留，未删除。

## 回滚

将比赛前端 `current` 指回 `20260929-geometry-r1`、比赛后端 `current` 指回 `20260929-evilgen-seeded-r1`，重启 `2048tables-competition-test.service`；恢复上述 Nginx 配置备份、执行 `nginx -t` 并重载 Nginx。无需修改数据库。
