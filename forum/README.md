# 2048 社区站

独立 Vue 3 / FastAPI 服务，论坛业务数据使用 **PostgreSQL 17**。沿用站点账号的稳定用户 ID；现有账号 SQLite 仅作只读会话校验，不存储论坛内容，也不复制密码。

状态：2026-10-03 已完成第一批可运行功能和本地验收，尚未部署。完整首版范围见 [产品与技术方案](../docs_and_configs/design/forum_v1_plan_20261003.md)。下表明确当前实现与后续工作，不能把第一批视为完整首版验收通过。

## 当前实现

| 模块 | 已实现 |
| --- | --- |
| 基础服务 | PostgreSQL 专用连接、显式 Alembic 迁移、健康检查、独立构建和运行；启动检查 schema，不自动建表 |
| 身份 | 主站共享会话只读校验，撤销会话/停用账号即时失效；独立管理员配置和按板块的版主权限 |
| 讨论 | 6 个初始板块、主题与楼层、引用回复、游标分页、指定楼层定位、正文修订、删除回复保留占位 |
| 创作 | 文字正文、预览、自动/手动保存云端草稿、本机恢复、多端修订冲突提示与另存、发布重试去重 |
| 棋盘 | `[[board:变体:编码\|说明]]` 自动渲染、即时预览、语法速查与复制；兼容 Play 编码；4×4 / 3×4 / 3×3 / 2×4 手绘、撤销和 PNG 导出；标记为非正式成绩 |
| 互动 | 点赞、收藏、回复与引用通知、通知已读；基础标题/正文中文子串搜索 |
| 治理 | 管理员公告板发帖、举报队列、按板块授权、隐藏/恢复、锁定/解锁、处理原因及审计落库、禁言约束 |
| 前端 | 手机与桌面适配、浅色/深色/系统主题、键盘焦点、加载/空/错误状态、既有站点入口 |
| 一致性 | 用户级数据库事务锁、主题楼层串行分配、提交幂等键、乐观修订、通知与 outbox 同事务 |
| 边界 | 正文结构白名单与转义、精确 Origin 校验、请求体限制、账号写入限流、私有草稿与收藏、API 禁止共享缓存 |

数据表中的 outbox 目前仅写入事件，**尚无 worker 或实时推送**；禁言/版主授权有数据库约束和服务端判断，完整管理界面仍待实现。搜索当前为参数化 `ILIKE`，不宣称已具备大规模全文检索能力。纯文本编辑不是完整富文本编辑器。

棋盘文字语法详见 [BOARD_SYNTAX.md](BOARD_SYNTAX.md)。例如 `[[board:4x4:fedc/ba98/7654/3210]]` 可直接放在正文或回复中，无需手绘；原文会保留，支持继续编辑。

方块背景与字色复用主站 `themes.json`、颜色解析器及 `2048tables-tile-palette` 共享 Cookie；未收到共享主题时采用主站 Default 配色。打开论坛或从主站切回时刷新配色，帖子、预览、手绘和 PNG 导出均使用它。跨子域继承依赖线上父域 Cookie；本地预览以默认主题显示。论坛不覆盖主站的主题设置。

## 目录

```text
forum/
  backend/       API、共享会话适配、业务事务和结构化内容校验
  migrations/    Alembic 环境与冻结的 PostgreSQL schema
  frontend/      Vue 3 + Vue Router + Vite
  tests/         真实 PostgreSQL 集成测试
  compose.yaml   仅本地 PostgreSQL（绑定 127.0.0.1:55432）
  server.py      本地 API 入口（127.0.0.1:8002）
```

## 本地运行（PowerShell）

前提：Python 3.11+、Node.js 22.12+、Docker Desktop。以下命令从仓库根目录执行。`.env.example` 是配置参考；服务不会自动读取 `.env`，需显式设置环境变量或由进程管理器注入。

```powershell
docker compose -f forum/compose.yaml up -d --wait
python -m venv forum/.venv
& forum/.venv/Scripts/python.exe -m pip install -r forum/requirements-dev.txt -c forum/constraints-tested.txt

$env:FORUM_DATABASE_URL = 'postgresql+psycopg://forum:local-development-only@127.0.0.1:55432/forum'
$env:FORUM_ENV = 'development'
$env:FORUM_PUBLIC_ORIGIN = 'http://127.0.0.1:5175'
$env:FORUM_ALLOW_DEV_AUTH = '1'
$env:FORUM_ADMIN_IDS = '1'
& forum/.venv/Scripts/python.exe -m alembic -c forum/alembic.ini upgrade head
& forum/.venv/Scripts/python.exe -m forum.server
```

另开终端，从仓库根目录执行：

```powershell
Set-Location forum/frontend
npm ci
$env:VITE_FORUM_DEV_USER = '1:LocalAdmin'
npm run dev
```

访问 <http://127.0.0.1:5175>。这是显式启用的本地身份，不需要生产账号；页面会显示开发身份提示。可换成 `2:LocalPlayer` 验证普通用户权限，名字须使用 ASCII（HTTP 请求头要求）。生产构建不包含开发身份请求头，后端在生产模式拒绝启用开发认证。不要将上述本地配置暴露到公网。

数据库只初始化板块，不自动创建示例帖子。手工验收产生的帖子会保留在本地卷中。`docker compose -f forum/compose.yaml stop` 停止数据库并保留卷；正常重启无需删除数据或再次创建库。

## 真实账号接入

设置 `FORUM_ALLOW_DEV_AUTH=0`，并将 `CLOUD_AUTH_DB` 指向现有账号库的实际绝对路径。服务以 SQLite `mode=ro` 打开，校验共享 cookie 对应的哈希会话、到期时间、撤销状态和账号状态。沿用现有 `tb_shared_session` / `tb_session`；显式 Bearer 客户端也可校验同类会话。部署时由代理保持同源 API，并验证共享 cookie 的域、Secure 和 SameSite 设置。

管理员从 `FORUM_ADMIN_IDS` 显式配置，不能从赞助等级或前端参数推导。按板块版主存于 `forum_roles`。现阶段只能由受控数据库管理流程维护版主/禁言，管理 UI 在后续里程碑；不要授予普通账号数据库权限。

同机只读账号库是当前接入方式。未来论坛拆机应增加可信内部会话校验接口，不能跨网络挂载 SQLite 或复制用户密码。此批测试验证了本地会话适配和撤销行为，尚未验证生产跨子域登录。

## 校验

```powershell
# 需对这个专用本地 PostgreSQL 拥有 CREATEDB 权限。
$env:FORUM_TEST_ADMIN_URL = 'postgresql://forum:local-development-only@127.0.0.1:55432/forum'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
& forum/.venv/Scripts/python.exe -m pytest forum/tests -q

Set-Location forum/frontend
npm test
npm run build
npm audit
```

每次测试创建随机命名的隔离数据库 `forum_test_<uuid>`，运行迁移两次以检查可重复执行，结束后仅删除该测试库；不会清空连接 URL 指定的 `forum` 库。未设置测试 URL 时测试跳过，不允许退回 SQLite。不要传入生产连接。

当前 24 项 PostgreSQL 测试覆盖并发楼层、重复提交、正文修订、草稿隔离与冲突、删除占位、隐藏内容过滤、版主范围、通知已读隔离、共享会话撤销、Origin 校验、限流、内容校验和棋盘语法原文往返。另有 19 项前端语法测试，校验 Play 编码兼容、错误输入、转义和渲染限制。浏览器已验证实际发帖/回复/收藏/绘制/PNG 导出及语法插入、预览、发布、刷新、编辑，并检查手机与桌面布局。

健康检查：`GET /health/live`、`GET /health/ready`；接口 schema：`GET /openapi.json`。业务 API 前缀为 `/api/forum/v1`，写请求必须带配置的完整 `Origin`；创建主题/回复还需 UUID `Idempotency-Key`，重试保持键和内容一致。变更内容后使用新键。普通浏览器由前端自动处理。

生产静态文件由 `npm run build` 生成到 `frontend/dist`，FastAPI 可直接托管。API 错误不会回退到 SPA 首页。开发代理将 `/api` 和 `/health` 转发到 8002。

## 后续实施顺序（均仍在完整首版范围内）

1. **完善内容创作与作品模块**：富文本、图片上传与处理、标签/主题编辑、回复草稿；Play 对局来源接口、各录像协议解析、回放/步数锚点/分支分析、作品撤回和来源状态同步，可信成绩必须经服务端校验。
2. **完善社区互动与发现**：订阅、关注、提及、阅读位置、通知偏好与分页；outbox worker 的领取/重试/幂等消费、SSE 补拉；中文搜索索引与短词召回、排序、公开内容 SEO。
3. **完善运营和管理**：公告置顶/定时、投票与问答、版主和禁言管理 UI、审核/申诉/审计检索、账号屏蔽与隐私处理、赛事/直播主题联动。
4. **完整首版验收**：双语、可访问性复核、媒体配额、代理 IP 限流、保留策略、备份恢复演练、混合负载和故障测试、线上共享会话验证、监控告警与灰度发布。

上述步骤对应总方案的里程碑，第一批提供数据库和用户流程基础，不取消其他必要模块。大厅持久聊天按总方案作为独立开关建设，语音/无限私信不在默认首版范围。

## 部署与恢复边界

本地 Compose 不是生产部署模板。生产需独立数据库账号/连接限制/凭据、HTTPS、可信反向代理、服务管理和监控；migration 账号与运行账号的最小权限分离尚需部署配置落实。升级先备份，再显式运行 Alembic，最后启动 schema 匹配的版本。第一版迁移不提供破坏性 downgrade；恢复应先将已验证备份恢复至另一个数据库，核验后切换。

论坛 PostgreSQL 使用 `pg_dump`/`pg_restore` 或等价 PostgreSQL 原生备份方案，并演练实际恢复；账号等现有 SQLite 继续使用仓库的 `tools/backup_sqlite.py` 在线备份。不能用 SQLite 脚本备份论坛 PostgreSQL，也不能只复制运行中的 PostgreSQL 数据目录。媒体落地后需将对象清单、内容 hash 与数据库备份协调。

生产部署前必须阅读 [deployment_retention.md](../docs_and_configs/deployment_retention.md)，核对干净源代码基线及显式 overlay 清单，成功健康检查后才按规定做保留审查。保留每应用最新 3 个版本及全部运行/引用/pin 版本；数据库备份超过 7 天才可按策略淘汰，并始终保留每库至少 3 份已验证备份。当前没有执行生产迁移、发布或清理。
