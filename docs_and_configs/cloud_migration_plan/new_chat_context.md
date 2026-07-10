# 2048Tables 云服务项目新对话上下文

更新时间：2026-07-06  
基准代码副本：`C:/Apps/2048endgameTablebase/src - cloud`  
当前本地 checkpoint：`1a01d19 Migrate minigames to browser engine`  
当前分支：`codex/multipath-io-routing-checkpoint`

本文件用于新开 Codex 对话时快速恢复上下文。新对话开始前，优先阅读本文，然后运行 `git status --short` 确认工作区状态。

## 项目目标

`src - cloud` 是从本地桌面版 2048 endgame tablebase 项目复制出来的云服务版本。目标是把原本 pywebview 桌面壳中的前后端分离应用改造成公网可访问服务：

- 前端：Vue 3 单页应用。
- 后端：FastAPI + WebSocket。
- 云入口：`cloud_server.py` 直接启动 FastAPI/uvicorn，不启动 pywebview。
- 表库：只在服务器后端解析真实路径，前端只能看到安全 catalog。
- 用户体系：匿名用户可浏览页面，但 trainer/tester/replay/analysis 等查询、上传、下载、分析动作必须登录。
- 配额体系：已实现 token 余额模型，关键操作按配置扣 token。
- Minigames：已迁移为浏览器本地 JS engine，不再依赖后端 Python engine。

## 本地与服务器路径

- 本地云仓库：`C:/Apps/2048endgameTablebase/src - cloud`
- 原桌面仓库：`C:/Apps/2048endgameTablebase/src`
- 公网域名：`https://2048tables.online/`
- 服务器登录：`ssh ubuntu@43.135.117.69`
- 服务器应用目录：`/opt/2048tables/app`
- 服务器表库目录：`/opt/2048tables/tablebases`
- 服务器 SQLite auth DB：`/opt/2048tables/app/docs_and_configs/cloud_runtime/auth.sqlite3`
- systemd 服务：`2048tables-cloud.service`
- 云服务端口：后端通常监听 `127.0.0.1:8000`，由 Nginx/Cloudflare 对外代理。

注意：本地提交不会自动同步服务器。涉及上线时需要确认 `/opt/2048tables/app` 是否已同步最新本地代码，并重启 `2048tables-cloud.service`。

## 关键架构

### 后端入口

- `cloud_server.py`：云端启动入口。
- `backend/app.py`：FastAPI app、HTTP API、WebSocket action 分发。
- `backend/cloud_safety.py`：云模式禁用高风险/旧桌面 action，例如 `NOTEBOOK_*`、`MINIGAME_*`、本地路径/构建等。
- `backend/session.py`：WebSocket `GameSession`，记录 `user_id/auth_session_id/user_email/user_role`。

### 表库 catalog

- 配置：`docs_and_configs/cloud_tablebases.json`
- 服务：`backend/tablebase_catalog.py`
- HTTP：`GET /api/tablebases`
- 前端 client：`frontend/src/services/tablebases/catalogClient.js`

manifest 当前列出：

- `L3_128`、`L3_256`、`L3_512`、`L3_1024`
- `free9_128`
- `free10_256`、`free10_512`
- `442t_128`、`442t_256`、`442t_512`

catalog 对前端只返回 `pattern/target/full_pattern/dtype/spawn_rate` 等安全字段，不返回 root、relative path、绝对路径。

### 上传、下载、分析

- `backend/cloud_files.py`：上传/下载临时文件 registry、扩展名、大小限制、TTL。
- `backend/cloud_analysis_jobs.py`：analysis job registry、结果 zip、过期清理。
- HTTP：
  - `POST /api/uploads`
  - `GET /api/downloads/{download_id}`
  - `POST /api/analysis/jobs`
  - `GET /api/analysis/jobs/{job_id}`
  - `GET /api/analysis/jobs/{job_id}/download`
- Replay 上传后通过 WS action `REPLAY_LOAD_UPLOAD` 加载，不接受任意服务器路径。
- Analysis 结果文件不应长期持久化；按 job/upload TTL 和清理逻辑处理。

## 认证与 token 配额

### 认证模块

- DB 初始化：`backend/auth/db.py`
- 认证服务：`backend/auth/service.py`
- 认证 API：`backend/auth/routes.py`
- 依赖/guard：`backend/auth/dependencies.py`
- 管理脚本：`scripts/auth_admin.py`

主要表：

- `users`
- `sessions`
- `refresh_tokens`
- `invite_codes`
- `email_verification_codes`
- `usage_events`
- `token_accounts`
- `token_ledger`
- `uploads`
- `analysis_jobs`

当前行为：

- 未登录可以打开目录、帮助、各个页面和静态资源。
- 触发查询、上传、下载、analysis job、replay load 等受保护动作时要求登录。
- WebSocket 允许匿名连接，但 protected action 会返回 auth-required。
- 注册需要邀请码 + 邮箱验证码。
- SMTP 使用 Brevo 方向，但之前 Brevo Transactional SMTP 可能仍处于激活/验证等待状态。不要把 SMTP 密钥写入仓库。

### token 余额

- 配置：`docs_and_configs/cloud_token_costs.json`
- 表库倍数：`docs_and_configs/cloud_table_multipliers.json`
- 逻辑：`backend/quota/service.py`
- 单位：`1 token = 1000 units`

余额字段：

- `token_accounts.bonus_balance_units`：赠送 token，上限 2000 token。
- `token_accounts.paid_balance_units`：充值/站长加额 token，本轮无上限。
- 展示余额 = bonus + paid，消费时优先扣 bonus。

发放规则：

- 显式登录/注册登录时检查。
- 距离 `last_weekly_grant_at` 超过 7 天，则赠送 1000 token。
- 增加后赠送余额不超过 2000 token。

扣费规则：

- trainer/tester 查到成功率：`1 * table_multiplier`
- trainer/tester 没查到成功率：`0.2 * table_multiplier`
- analysis 每分析一个 replay 文件：`100 * table_multiplier`
- replay 载入 `.rpl`：`3`，倍率始终为 1，解析失败也扣。

倍率：

- `L3_*`、`442t_*`：1x
- `free9_*`：3x
- `free10_*`：5x

备注：`public-test` 已在 2026-07-06 通过服务器 DB 事务增加 20000 paid token，并写入 `token_ledger`；加额后总余额为 22529 token。

## 前端页面状态

### App、目录、账号区

- `frontend/src/App.vue`
- `frontend/src/app/tabRegistry.js`
- `frontend/src/components/MainMenuView.vue`
- `frontend/src/features/auth/AuthPage.vue`
- `frontend/src/app/useAuthState.js`

当前模式：

- 默认进入目录页，不再强制跳登录整页。
- 顶部右侧是账号入口：未登录显示登录/注册；已登录显示紧凑账号按钮，点击下拉显示余额和 logout。
- 受保护动作由前端 `requireAuth()` 和后端 guard 双重拦截。

### Gamer

- 页面：`frontend/src/features/gamer/pages/GamerPage.vue`
- 会话：`frontend/src/features/gamer/composables/useGamerSession.js`
- WASM client：`frontend/src/services/wasm/aiCoreClient.js`
- 静态 WASM：`frontend/public/wasm/`

当前状态：

- Gamer 已恢复为当前站内页面，不再跳转 Netlify。
- AIPlayer 走前端 WASM/worker 搜索，不查后端云表。
- EvilGen 走前端 WASM。
- 难度影响 EvilGen 出数概率。
- 加载 WASM 需要时间，页面会显示 loading，加载完成后步进和 AI 可用。

### Trainer

- 前端：`frontend/src/features/trainer/**`
- 后端：`backend/handlers/trainer.py`
- 结果扣费：`backend/trainer_helpers.py`

当前状态：

- 定式选择使用服务器 catalog，不暴露路径。
- 已移除本地路径选择和录制/回放入口。
- 选择表库后应自动加载并触发默认局面。
- 查成功率按 token 规则预扣/结算。

### Tester

- 前端：`frontend/src/features/tester/**`
- 后端：`backend/handlers/tester.py`、`backend/tester.py`

当前状态：

- 定式选择使用服务器 catalog。
- 移动、随机、结果查询需要登录并扣 token。
- 保存日志/回放改为浏览器下载。
- latest replay 绑定用户/session，避免多用户串数据。
- 错题本和 Notebook 云版已移除。

### Replay / Analysis

- Replay 前端：`frontend/src/features/replay/**`
- Replay 后端：`backend/handlers/replay.py`、`backend/replay.py`
- Analysis 后端：`backend/handlers/analysis.py`、`backend/analysis.py`

当前状态：

- Replay 打开文件通过浏览器选择 + `/api/uploads` + `REPLAY_LOAD_UPLOAD`。
- Analysis 创建 HTTP job，WS 订阅进度，完成后下载 zip。
- Analysis 下载按钮翻译已补齐。
- Analysis 结果文件名中过长随机 job id 已优化过；后续如再改命名，确保不泄露服务器路径。

### Minigames

- 前端会话：`frontend/src/features/minigames/composables/useMinigameSession.js`
- 前端 engine：`frontend/src/features/minigames/engine/**`
- 前端页面：`frontend/src/features/minigames/pages/**`
- 后端 Python minigames：已删除。

当前状态：

- Minigames 已纯前端化，不再发送 `MINIGAME_*` WebSocket action。
- 后端云模式拒绝 `MINIGAME_*`，防旧客户端或恶意请求占用服务器 CPU。
- 存档使用浏览器 localStorage，key 仍是 `2048tables:minigames`，schema version 2。
- Tricky Tiles 复用前端 WASM EvilGen；WASM 不可用时会 fallback 到随机 spawn。
- 后端 `/minigames-assets` 静态图片资源挂载仍保留。

## 部署与运维

常用服务器命令：

```bash
ssh ubuntu@43.135.117.69
cd /opt/2048tables/app
systemctl status 2048tables-cloud
sudo systemctl restart 2048tables-cloud
journalctl -u 2048tables-cloud -n 100 --no-pager
```

常用本地验证：

```bash
python -m compileall -q backend cloud_server.py scripts
cd frontend
npm run build
```

云端接口 smoke：

```bash
curl -s https://2048tables.online/api/tablebases
```

如要检查服务器 DB：

```bash
sqlite3 /opt/2048tables/app/docs_and_configs/cloud_runtime/auth.sqlite3
```

不要在仓库提交：

- `docs_and_configs/cloud_runtime/auth.sqlite3`
- SMTP/API 密钥
- 本地上传、analysis 临时结果
- `frontend/test-results/`
- `__pycache__/`

## 推荐新对话启动步骤

1. 确认工作目录使用 `C:/Apps/2048endgameTablebase/src - cloud`，不要误用原桌面仓库。
2. 运行 `git status --short`。
3. 阅读本文和相关专项文档：
   - `docs_and_configs/cloud_migration_plan/code_map.md`
   - `docs_and_configs/cloud_migration_plan/auth_quota_plan.md`
   - `docs_and_configs/cloud_migration_plan/ubuntu_deployment.md`
4. 如果任务涉及服务器，尽量一次 SSH 会话内连续完成，不要每个小步骤反复登录。
5. 如果任务涉及部署，先确认本地 commit 是否已经同步到 `/opt/2048tables/app`。
6. 如果任务涉及扣费/用户/邀请码，先备份或查询 SQLite 状态，再写入。

## 近期注意点

- 当前本地最新 commit 已完成 Minigames 纯前端化，但服务器是否已部署该 commit 需要单独确认。
- `public-test` 已完成一次 20000 paid token 加额；后续如需继续加额，仍应先查 DB 当前余额再写入。
- Brevo SMTP 验证曾处于等待状态，注册邮箱验证码可能仍需检查生产环境日志。
- token 加额建议写入 `paid_balance_units`，并同步写 `token_ledger` 作为审计。
- 旧文档中“匿名只能看登录页”或“Gamer 跳 Netlify”的描述已经过期，以本文为准。
