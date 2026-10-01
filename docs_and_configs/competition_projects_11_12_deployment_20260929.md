# 比赛项目 11、12 部署记录（2026-09-29）

本次将「百步封锁（4×4）」和「越来越大（4×4）」加入比赛建房项目池。试玩页此前已包含这两项；本次同步发布赛事前端的 12 项目录与比赛后端的项目元数据、正式适配器。已有房间在创建时固化项目池，不会自动增加新项目。

## 发布范围

- 比赛后端：仅更新 `competition/backend/projects/tournament_variants.py`、`competition/backend/projects/__init__.py`，新增 `competition/backend/projects/practice_variants.py`。其余服务代码继承线上当前版本。
- 赛事前端：发布当前 `competition/frontend/dist` 构建；未修改主站或直播站。
- 后端新发布目录：`/opt/2048tables/tournament-app/releases/20260929-projects-11-12-r1`。
- 前端新发布目录：`/opt/2048tables/tournament/releases/20260929-projects-11-12-r1`。
- 保留原发布目录，回滚只需恢复两个 `current` 符号链接并重启 `2048tables-competition-test.service`。

## 验证

- 本地赛事前端构建通过，前端测试 42 项通过；定向比赛后端测试 55 项通过。
- 新发布目录中后端注册检查：目录共 12 项，项目 11、12 可解析、初始化并生成公开视图。
- 发布时数据库仅有 1 个 `SEATING` 房间，无正在进行的比赛。
- 线上比赛服务 `active`，`/api/health` 返回 200；`/test`、`/practice/11`、`/practice/12` 和新前端资源均返回 200。
- 运行中的比赛进程工作目录指向新后端发布目录。

部署脚本位于 `tools/deploy_competition_projects_11_12.py`，包含版本前置检查、项目注册冒烟检查及健康检查失败时的自动回滚。
