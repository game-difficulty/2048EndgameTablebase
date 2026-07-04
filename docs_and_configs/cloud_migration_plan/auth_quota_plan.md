# 注册登录与额度系统设计

更新时间：2026-07-04

当前判断基于 `C:/Apps/2048endgameTablebase/src - cloud` 副本。v1 采用 SQLite，目标是先完成“必须登录才能使用云服务、注册需要邀请码和邮箱验证码、关键请求记录用量但暂不扣费”的实用云服务版，并为后续限流、积分、付费预留数据结构。

## 业务边界

- 匿名用户只可加载静态前端资源和登录/注册页面；`/api/tablebases`、上传、下载、analysis job、WebSocket 都要求登录。
- 注册流程：邀请码校验 -> 邮箱验证码发送 -> 邮箱验证码消费 -> 创建用户 -> 消耗邀请码 -> 创建默认 quota -> 创建 session。
- 登录状态通过 httpOnly cookie `tb_session` 维护；WebSocket 握手读取同一 cookie。
- `GameSession` 记录 `user_id/auth_session_id/user_email/user_role`，便于 trainer/tester/replay/analysis 绑定用户。
- 关键请求当前 `cost=0`，只写 `usage_events`，不减少可用额度。后续只需调整 cost 和 quota 检查策略。

## SQLite 表

- `users`：用户主表，字段包括 `email/password_hash/display_name/role/status/email_verified_at/last_login_at`。
- `sessions`：登录 session，保存 session token hash、过期时间、UA、IP、撤销时间。
- `refresh_tokens`：预留刷新 token，目前创建但前端暂不使用。
- `invite_codes`：邀请码，保存 code hash、允许邮箱/域名、最大使用次数、已使用次数、过期和禁用状态。
- `email_verification_codes`：邮箱验证码，按邮箱和用途保存 hash、过期、尝试次数、消费时间。
- `user_quotas`：额度表，v1 默认 lifetime quota，`limit_value=0/used_value=0` 表示暂不限制。
- `usage_events`：使用记录，记录 `event_type/quota_key/cost/metadata/ip/session_id`。
- `uploads`：上传文件归属和 TTL 元数据，绑定 `user_id/session_id`。
- `analysis_jobs`：分析任务归属和 TTL 元数据，绑定 `user_id/session_id`。

## 后端接口

- `GET /api/auth/me`：返回当前 cookie 是否已认证。
- `POST /api/auth/send-email-code`：需要 `{ email, invite_code }`，生产环境依赖 SMTP。
- `POST /api/auth/register`：需要 `{ email, password, invite_code, verification_code, display_name? }`，成功后设置 cookie。
- `POST /api/auth/login`：成功后设置 cookie。
- `POST /api/auth/logout`：撤销 session 并清 cookie。
- `GET /api/tablebases`：已改为登录后可用。
- `POST /api/uploads`、`GET /api/downloads/{download_id}`、`POST /api/analysis/jobs`、analysis 查询/下载：已绑定当前用户。
- `WS /ws/{client_id}`：未登录直接 1008 关闭；已登录后将用户信息写入 `GameSession`。

## 用量记录映射

- Trainer：`TRAINER_SET_FILEPATH`、`TRAINER_GET_RESULTS`、`TRAINER_DEFAULT` -> `trainer_query`。
- Tester：`TESTER_SELECT_PATTERN`、`TESTER_RESET_RANDOM`、`TESTER_MOVE` -> `tester_move`。
- Replay：`REPLAY_LOAD_UPLOAD` -> `replay_load`。
- Analysis：`ANALYSIS_SUBSCRIBE` 和 HTTP 创建 job -> `analysis_job`。
- Upload：`kind=analysis` -> `upload_analysis`，`kind=replay` -> `upload_replay`。

## 运维脚本

`scripts/auth_admin.py`：

- `create-admin --email EMAIL --password PASSWORD [--display-name NAME]`
- `create-invite [--uses N] [--expires-days N] [--label TEXT] [--email EMAIL] [--email-domain DOMAIN] [--created-by-email EMAIL]`
- `list-users`
- `list-invites`

生成的邀请码明文只打印一次，数据库只保存 hash。

## 部署环境变量

- `CLOUD_AUTH_DB`：SQLite 路径，默认 `docs_and_configs/cloud_runtime/auth.sqlite3`。
- `AUTH_SESSION_DAYS`：session 天数，默认 14。
- `AUTH_COOKIE_SECURE`：默认 `1`，公网 HTTPS 应保持开启；本地 HTTP 调试可设为 `0`。
- `SMTP_HOST/SMTP_PORT/SMTP_USERNAME/SMTP_PASSWORD/SMTP_FROM/SMTP_TLS`：邮箱验证码 SMTP。
- `AUTH_ALLOW_DEV_EMAIL_CODES=1`：无 SMTP 时把验证码返回给前端，仅开发/临时测试使用，不建议公网开启。

## 后续计划

1. 部署前配置 SMTP，或明确接受临时开发验证码模式。
2. 生成初始管理员和首批邀请码。
3. 增加登录/注册页面翻译文案。
4. 将 quota 检查从“只记录”升级为“记录 + 可选拒绝”。
5. 增加按 IP + user_id 的轻量限流中间件。
6. 需要付费时新增支付订单表，将 `usage_events` 和 `user_quotas` 与支付结果关联。
