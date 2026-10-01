# 真华容道后续规则部署（2026-09-28）

比赛站项目 1「真华容道」已更新：

- L 型特殊块的凹角保持空白；完整 2×2 方块仍填满中央接缝。
- 数字砖和特殊块四向均无有效移动时判定死亡，试玩页延迟 2 秒显示可关闭的终局浮窗，死亡后倒计时冻结；正式比赛停止该方计时。
- 开局只有两块初始数字，前 10 次有效移动后才出现首个特殊块；之后送出一块仍立即生成下一块。

## 发布版本

- 试玩和比赛前端：`/opt/2048tables/tournament/releases/20260928-cargo-v18`
- 比赛后端：`/opt/2048tables/tournament-app/releases/20260928-cargo-v11`
- 公网试玩：`https://tournament.2048tables.online/practice/1`
- 前端发布包 SHA-256：`33b2290d74cbf5287966c716da46141774fd736cbced5718af37ced73887f4a6`
- 后端覆盖包 SHA-256：`d9f0bf8ba943d071cf123d3673fc55631707e8ae9196d7f1e3f6534de2e62fbd`

后端从 v10 目录复制，只覆盖 `cargo_transport.py` 与项目目录文件
`tournament_variants.py`；本地其他未发布的主站/Live 改动不在此次发布中。
Nginx 路由配置没有变化。

## 验证

- 发布前比赛后端 64 项、试玩前端 30 项测试通过，比赛前端构建成功。
- 上线前数据库只有一间候场房间，没有进行中的对局。
- 服务器端新适配器初始状态 `cargo=None`，首块阈值为第 10 次有效移动。
- 公网 `/practice/1`、`/practice`、`/test`、`/api/health` 均返回 HTTP 200；实际加载的 JS 包含新版开局说明与死亡状态。
- 服务进程工作目录指向 v11，服务为 `active`，近期错误日志为空，SQLite `quick_check=ok`。
- 主站、Play、Live 首页均保持 HTTP 200。Live 观众渲染器仍未在共用 Live 站发布。

## 回滚点

- 前端旧版：`/opt/2048tables/tournament/releases/20260928-cargo-v17`
- 后端旧版：`/opt/2048tables/tournament-app/releases/20260928-cargo-v10`

如需回滚，原子切回两个 `current` 软链接并重启
`2048tables-competition-test.service`；无需修改数据库或 Nginx。
