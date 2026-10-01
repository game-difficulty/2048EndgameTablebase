# 真华容道项目 1 部署记录（2026-09-28）

后续 L 型绘制、死亡判定和第 10 步首块规则的发布见
`competition_cargo_v18_deployment_20260928.md`；本文保留首次上线记录。

项目 1「真华容道」已发布到现有独立比赛测试站：

- 单人试玩：`https://tournament.2048tables.online/practice/1`
- 比赛入口：`https://tournament.2048tables.online/test`
- 项目目录：`https://tournament.2048tables.online/practice`
- 健康检查：`https://tournament.2048tables.online/api/health`

本次发布的特殊块移动规则为整块沿输入方向滑至可达最远位置；开局有上方
2×2 入口、下方中央出口，10 分钟内以送出特殊块数量计分。比赛端使用
服务端适配器判定及独立限时，试玩端使用同规则的浏览器引擎。

## 发布版本

- 比赛前端：`/opt/2048tables/tournament/releases/20260928-cargo-v17`
- 比赛后端：`/opt/2048tables/tournament-app/releases/20260928-cargo-v10`
- systemd：`2048tables-competition-test.service`
- Nginx：`/etc/nginx/sites-available/tournament.2048tables.online`
- 前端发布包 SHA-256：`3f01acddecdad9a3d422a1517814b9c0ac94b7b65a51c2676ca4d88df37d211e`
- 后端覆盖包 SHA-256：`9cd2479f4e439a0e84b5db4fff317cb47e1aaafbe1b8732e7464bb43192eed11`

后端从此前 v9 发布目录复制，只覆盖 `service.py`、项目注册与目录文件，并新增
`cargo_transport.py`。本地 `schemas.py` 与上次发布的差异没有纳入此次上线，
以免捎带无关 API 变化。前端发布包是完整的比赛站 Vite 构建。Nginx 仅将
`/practice/1` 加入原有 SPA 路由白名单。

主站和 Live 服务没有随此次发布更新；本地实现的项目 1 Live 观众渲染器尚未
发布至共用 Live 站。

## 验证

- 发布前：比赛后端测试 60 项、比赛前端测试 26 项通过，比赛站及主站前端构建通过。
- 新发布后端导入目录时可识别 `project-01` 和 `tournament-cargo-transport-4x4`。
- Nginx 配置检查成功，比赛服务重启后为 `active`，新进程工作目录为 v10 发布目录。
- 公网 `/practice/1`、`/practice`、`/test`、`/api/health` 均为 HTTP 200；
  健康接口返回 schema version 7。
- 公网实际加载的 JS 资源包含新项目标识及 `/practice/1` 路由。
- 主站、Play、Live 公网首页均保持 HTTP 200；比赛服务最近错误日志为空。

## 回滚点

- 前端旧版：`/opt/2048tables/tournament/releases/20260928-growing-tiles-v16`
- 后端旧版：`/opt/2048tables/tournament-app/releases/20260928-undo-death-v9`
- Nginx 旧版：`/etc/nginx/sites-available/tournament.2048tables.online.before-cargo-v17`

回滚时原子切回两个 `current` 链接，恢复 Nginx 备份并执行 `nginx -t`，
随后重启比赛服务和重载 Nginx。数据库与新发布目录保留以便审计。
