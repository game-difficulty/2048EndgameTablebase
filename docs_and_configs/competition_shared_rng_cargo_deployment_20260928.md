# 比赛站随机流与真华容道出口修复部署（2026-09-28）

本次发布包括：所有试玩项目的固定种子出数流、正式项目 `tournament-v2` 适配器、困难孤岛与随机形状展示名称，以及真华容道特殊块部分移出时可在出口范围内横移的修复。旧版 `tournament-v1` 适配器保留，供已创建的房间继续解析。

## 发布位置

- 前端：`/opt/2048tables/tournament/releases/20260928-shared-rng-cargo-v19`
- 后端：`/opt/2048tables/tournament-app/releases/20260928-shared-rng-cargo-v12`
- 前端发布包 SHA-256：`5c8c54948d861d479a4ee542b6c0e6a9689bd57e88354728867a859766d74361`
- 后端覆盖包 SHA-256：`8e1d1bb9f0fd2728636280a18a6dea1223471eab07b345b742d9e6412251d455`

后端由上一版 v11 复制，仅覆盖 `service.py`、`projects/__init__.py`、`projects/tournament_variants.py` 和 `projects/cargo_transport.py`。未更新主站、Play、Live 或 Nginx。上线前数据库有一间候场房间，无进行中的游戏会话。

## 验证

- 本地前端 36 项、后端 70 项测试通过；前端构建成功。
- 新后端编译及导入成功，可注册 20 个新旧版本适配器；比赛服务重启后为 `active`，健康检查正常，近期错误日志无记录。
- 公网 `/practice/1`、`/practice`、`/test`、`/api/health` 均为 HTTP 200；公网 HTML 引用本次构建的 `index-MTtsqQXB.js`，该资源返回 HTTP 200。
- 线上 `cargo_transport.py` 和 `tournament_variants.py` 与本地发布文件 SHA-256 一致。

## 回滚

- 前端：`/opt/2048tables/tournament/releases/20260928-cargo-v18`
- 后端：`/opt/2048tables/tournament-app/releases/20260928-cargo-v11`

如需回滚，原子切回两个 `current` 软链接，并重启 `2048tables-competition-test.service`；无需数据库迁移。
