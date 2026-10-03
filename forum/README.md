# 2048 社区站

独立 Vue 3 / FastAPI 服务，论坛业务数据使用 **PostgreSQL 17**。沿用站点账号的稳定用户 ID；现有账号 SQLite 仅作只读会话校验，不存储论坛内容，也不复制密码。

状态：2026-10-03 基础版本已提交为 `392261a`，第二、三批已提交为 `6414707`；第四批 `6ee17d5` 已实现投票、问答、排期公告、搜索索引、屏蔽、分类通知、隐私工单、假设回放分支和运维工具，尚未部署。数据库版本为 `0005_notice_category`。完整首版范围见 [产品与技术方案](../docs_and_configs/design/forum_v1_plan_20261003.md)，逐项差异见 [本轮验收清单](ACCEPTANCE.md)。目前不能视为完整首版全部交付。

前端已按现有功能重排全局导航、讨论列表、阅读页、创作台、个人区与管理工作台，支持 URL 恢复筛选和手机底部导航。页面分工、尺寸、状态与本地验证见 [前端版面设计与实施](../docs_and_configs/design/forum_frontend_layout_20261003.md)。

## 当前实现

| 模块 | 已实现 |
| --- | --- |
| 基础服务 | PostgreSQL 专用连接、显式 Alembic 迁移、健康检查、独立构建和运行；启动检查 schema，不自动建表 |
| 身份 | 主站共享会话只读校验，撤销会话/停用账号即时失效；独立管理员配置和按板块的版主权限 |
| 讨论 | 6 个初始板块、主题与楼层、引用回复、游标分页、指定楼层定位、正文修订、删除回复保留占位 |
| 创作 | Markdown 格式工具、标题/加粗/斜体/列表/引用/代码/链接/表格、预览、自动/手动保存云端草稿、本机恢复、多端修订冲突提示与另存、发布重试去重、主题标题与标签修订 |
| 棋盘 | `[[board:变体:编码\|说明]]` 自动渲染、即时预览、语法速查与复制；兼容 Play 编码；4×4 / 3×4 / 3×3 / 2×4 手绘、撤销和 PNG 导出；标记为非正式成绩 |
| 媒体与回放 | 图片校验与重新编码、私有附件、撤回/配额；HPR1/HPR2、Verse/RPL1 规则校验；Play 公开源引用与打开时重查；独立 worker 线程中的回放、进度拖动、步数引用 |
| 互动 | 点赞、收藏、主题/板块订阅、个人页与关注、`<@用户ID>` 提及、回复/订阅/关注/提及通知、通知偏好/已读/分页；outbox 后台任务与 SSE 更新、断线重读；基础标题/正文中文子串搜索 |
| 阅读与回复草稿 | 楼层末尾进入视口后记录私有阅读位置、跨设备继续阅读；回复自动保存、本机恢复、云端版本冲突对照与选择、发布后清理已发布版本 |
| 治理 | 举报上下文与结案、隐藏主题列表/恢复、锁定/解锁、置顶、板块设置、版主授权/撤销、限时禁言/解除、附件移除、申诉提交/处理/结果查询、审计检索、任务处理状态；服务端按板块授权 |
| 前端 | 手机与桌面适配、浅色/深色/系统主题、键盘焦点、加载/空/错误状态、既有站点入口 |
| 一致性 | 用户级数据库事务锁、主题楼层串行分配、提交幂等键、乐观修订、通知与 outbox 同事务 |
| 边界 | 正文结构白名单与转义、精确 Origin 校验、请求体限制、账号写入限流、私有草稿与收藏、API 禁止共享缓存 |

内容编辑采用 Markdown 源码 + 工具栏 + 预览，阅读时把解析结果转为 Vue 节点，不执行用户 HTML。仅显示论坛上传的图片，外部图片地址不会自动加载。支持粘贴一张 PNG/JPEG/WebP，异步上传后追加正文，保留上传期间的输入；`details` fenced block 渲染为可折叠纯文本，不触发提及/附件。搜索使用 PostgreSQL `pg_trgm` 和一、二字短词 GIN 索引辅助参数化子串匹配，支持作者/标签/板块/时间/主题 ID/类型过滤；迁移账号需有安装 `pg_trgm` 权限。仍需真实数据量的查询性能验收。

第四批新增：

- 投票：单/多选、改选、截止与提前关闭、始终/投票后/截止后显示结果；选项创建后不可修改。
- 问答：待解决/已解决/关闭、从已加载楼层选择采纳答案、关联重复问题、版本冲突检查；删除采纳回复自动恢复待解决。移帖要求同时拥有两个板块的管理权限。
- 发现与通知：最新回复/最新发布/精选/未回复/关注/未读视图；个人页公开回复；用户/板块屏蔽；分类偏好；管理处置、申诉、禁言和系统公告通知；按主题、类型、日期与已读状态聚合已加载通知，保留每条楼层链接。
- 公告：管理员草稿/排期/发布/到期/撤回，目标站点、禁止回复、全员通知和审计；worker 重新检查时间与权限，失效内容退回草稿，不阻塞其他事件。公开摘要 API 支持目标站点过滤；其他站点尚未接入。
- 社区运营页：公告、人工登记的赛事/直播摘要、隐私请求与运行指标。外部卡片严格递增来源版本，可撤回；不将人工摘要认证为官方成绩，尚无自动同步。
- 数据与隐私：导出自己的数据（每类最多 1000 条，文件明确标注是否截断）、私有处理请求、管理员结案与结果通知。工单不是自动删除引擎，管理员须先执行并核实实际处置。
- 回放：片段循环、从当前帧创建 FBR1 假设分支，手动指定每步生成块，最多 1000 步、撤销、保存并引用。始终标记非正式成绩。Play 引用绑定规范化录像 SHA-256；源站录像变化时返回 `SOURCE_REVISED`，不把旧步数引用映射到新录像；旧的无版本引用需重新导入。
- 运维：公开页面安全元信息、sitemap、结构化请求日志、可信代理 IP 限流、管理端队列指标、PostgreSQL 原生备份与恢复校验工具。

第三批交互规则：关注上限 200 人，仅通知关注后发布的新主题，不回补历史；关注清单仅本人可见。正文提及上限 10 人，按稳定用户 ID 定位，代码、转义、图片说明和链接文字不触发提及；同一用户在同一帖子中最多生成一条通知，反复编辑不会重复通知。通知暂停设置适用于这些新通知。

回复草稿最多保留 50 份非空内容，「我的社区」可查看、继续编辑或清空。主题隐藏后仍可清空自己的草稿，释放配额；隐藏主题标题不向无权限用户展示。云端清空后保留版本号，防止旧设备把已经清空的草稿重新写回；多端冲突由用户对照选择版本。阅读位置仅向后推进，「我的社区」列出最近 100 个仍公开的主题，不把加载但未进入视口的楼层自动标为已读。

申诉绑定具体处置记录，支持自己的主题隐藏、锁定及账号禁言；禁言不阻止私有草稿与申诉。版主只能处理所属板块的主题申诉，账号禁言申诉由管理员处理，不能处理自己的申诉。通过时检查相关处置是否仍为最新状态，过期申诉不能撤销后来的决定；每次处理必须填写原因并写入审计。结果可在「我的社区」查看，也通过持久化通知与 SSE 提示。

棋盘文字语法详见 [BOARD_SYNTAX.md](BOARD_SYNTAX.md)。例如 `[[board:4x4:fedc/ba98/7654/3210]]` 可直接放在正文或回复中，无需手绘；原文会保留，支持继续编辑。

方块背景与字色复用主站 `themes.json`、颜色解析器及 `2048tables-tile-palette` 共享 Cookie；未收到共享主题时采用主站 Default 配色。打开论坛或从主站切回时刷新配色，帖子、预览、手绘和 PNG 导出均使用它。跨子域继承依赖线上父域 Cookie；本地预览以默认主题显示。论坛不覆盖主站的主题设置。

## 目录

```text
forum/
  backend/       API、共享会话适配、业务事务和结构化内容校验
  migrations/    Alembic 环境与冻结的 PostgreSQL schema
  frontend/      Vue 3 + Vue Router + Vite
  tests/         真实 PostgreSQL 集成测试
  worker.py      持久化 outbox 消费与通知分发
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
$env:FORUM_IP_HASH_SECRET = 'local-forum-test-rate-key'
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

订阅通知还需要独立启动后台任务（另开 PowerShell，从仓库根目录）：

```powershell
$env:FORUM_DATABASE_URL = 'postgresql+psycopg://forum:local-development-only@127.0.0.1:55432/forum'
$env:FORUM_ENV = 'development'
$env:FORUM_ADMIN_IDS = '1' # 与 API 一致，公告发布时会重查权限
& forum/.venv/Scripts/python.exe -m forum.worker
```

升级本地已有库时先执行 `alembic upgrade head`（当前 `0005_notice_category`），再重启 API 和 worker。迁移不删除原有帖子。worker 可以多进程运行，以 `SKIP LOCKED` 领取事件并在同一事务内去重投递；失败事件保留错误类型，每分钟重试。未启动 worker 时，作者/引用回复仍可直接产生通知，订阅通知会留在 outbox，排期公告也不会发布。

数据库只初始化板块，不自动创建示例帖子。手工验收产生的帖子会保留在本地卷中。`docker compose -f forum/compose.yaml stop` 停止数据库并保留卷；正常重启无需删除数据或再次创建库。

## 真实账号接入

设置 `FORUM_ALLOW_DEV_AUTH=0`，并将 `CLOUD_AUTH_DB` 指向现有账号库的实际绝对路径。服务以 SQLite `mode=ro` 打开，校验共享 cookie 对应的哈希会话、到期时间、撤销状态和账号状态。沿用现有 `tb_shared_session` / `tb_session`；显式 Bearer 客户端也可校验同类会话。部署时由代理保持同源 API，并验证共享 cookie 的域、Secure 和 SameSite 设置。

管理员从 `FORUM_ADMIN_IDS` 显式配置，不能从赞助等级或前端参数推导。管理员在「社区管理 → 用户权限」维护板块版主与限时禁言；版主仅能处置自己板块的内容，不能授权其他版主。引导管理员账号受保护，不能在 UI 中禁言或降权。所有变更要求原因并记录审计；普通账号不需要数据库权限。

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

当前 69 项 PostgreSQL/服务测试通过，覆盖前三批能力以及投票隐私/改选/截止、采纳权限/修订/删除恢复、短词搜索/过滤/游标、屏蔽与通知偏好、公告排期/改期竞态/失败隔离/到期/撤回、隐私导出与工单、来源卡版本、移帖权限、修订历史隐私、FBR1 规则、Play 来源变化、SEO 隐藏过滤及 IP 限流。24 项前端测试覆盖语法、Markdown 安全边界和通知聚合；生产构建通过，生产依赖审计无已知漏洞。

浏览器已验证富文本与图片发布、录像播放/逐步定位、worker + SSE 实时通知、版主授予/撤销和审计；第三批验证回复草稿离开/恢复、楼层跳转保留输入、多端冲突对照、提及发布与个人页、关注、继续阅读及申诉提交—管理处理—主题恢复，并检查 320px 深色页面无横向溢出。公开 Play 源站通过受控替身验证撤回逻辑，尚未对生产账号/真实公开录像做线上联调。

第四批浏览器验收：投票改选/刷新保留，排期公告由 worker 发布并关闭回复，假设分支试走/出数/保存/发回复/重播，模拟剪贴板图片上传及折叠预览/发布，通知偏好持久化，屏蔽/解除，以及 320px 深色设置页无横向溢出。截图保存在本地 `output/playwright/forum-poll-desktop.png`、`forum-replay-branch.png`、`forum-settings-mobile-dark.png`。

## 媒体、录像和通知的当前边界

- 图片：PNG/JPEG/WebP 静态图，输入 5 MiB / 1600 万像素以内，限制长边 2560 后重新编码为 WebP 并剥离原始元数据；不接收 SVG 或动图。每人最多 100 个活跃附件 / 50 MiB。
- 附件元信息与二进制目前均存于 PostgreSQL `forum_media`，与内容权限和备份保持一致。以后扩展对象存储时需迁移授权读取方式，不可将私有附件改为公开桶。
- 未发布附件只对作者/管理员可见。已发布附件的匿名读取依赖当前公开帖子引用；隐藏/删除/编辑移除最后公开引用后会停止公开读取。代码示例不会绑定附件。复用附件到新帖要求归属当前作者，避免通过复制引用绕过原作者撤回。已下载到客户端的内容无法从用户设备收回。
- 录像上传最多 2 MiB，支持 HPR1/HPR2、Verse 文本与 RPL1；复用 Play 的 Python 规则/格式校验模块及前端移动引擎。规则校验不等于真人成绩认证。`[[replay:附件UUID@步数]]` 在正文中按需显示播放器，计算放在浏览器 Worker 中。
- 引用 Play：只接受对局 UUID，固定访问 `FORUM_PLAY_ORIGIN`（默认 `https://play.2048tables.online`）的公开录像接口，不转发账号凭据、不跟随重定向、不抓取任意 URL。每次加载重新请求源站；源站隐藏/撤回/停用/故障时不展示论坛缓存副本。重查后已经载入客户端的回放仍属于已下载内容。
- SSE 每次连接发送收件箱状态，每 2 秒检查变更和会话，每 60 秒重连；前端从数据库收件箱重新读取消息，不依赖易丢失的进程内消息队列。部署代理须关闭流式响应缓冲并允许相应读取超时。生产连接并发和源站限流仍需要压测。
- 内容与媒体的服务端依赖包含仓库 `backend/human_play` 及其 Python 依赖；前端构建复用 `frontend/src/human/engine.js`、共享工具和主题文件，不能只拷贝 `forum/` 目录部署。

健康检查：`GET /health/live`、`GET /health/ready`；接口 schema：`GET /openapi.json`。业务 API 前缀为 `/api/forum/v1`，写请求必须带配置的完整 `Origin`；创建主题/回复还需 UUID `Idempotency-Key`，重试保持键和内容一致。变更内容后使用新键。普通浏览器由前端自动处理。

生产静态文件由 `npm run build` 生成到 `frontend/dist`，FastAPI 可直接托管。API 错误不会回退到 SPA 首页。开发代理将 `/api` 和 `/health` 转发到 8002。

## 后续实施顺序（均仍在完整首版范围内）

1. **作品与跨站闭环**：Play 选局及“发到社区”入口、作品卡视图/对局条件搜索、标注与海报、其他录像协议、正式成绩资格说明；赛事/直播自动投影修订和撤回、跨站公告消费。
2. **阅读与治理完善**：楼中楼折叠、列表返回位置、独立运营/全局版主能力、自动审核与待审核队列、完整导出与经核验的数据处置工具。
3. **完整首版验收**：全量双语、可访问性、分页 sitemap、数据保留执行工具、生产权限与代理配置、混合负载/故障测试、线上共享会话联调、告警及灰度。本地备份恢复演练已完成，不等于生产演练。

上述步骤对应总方案的剩余里程碑，不取消其他必要模块。大厅持久聊天按总方案作为独立开关建设，语音/无限私信不在默认首版范围。

## 部署与恢复边界

本地 Compose 不是生产部署模板。生产需独立数据库账号/连接限制/凭据、HTTPS、可信反向代理、服务管理和监控；migration 账号与运行账号的最小权限分离尚需部署配置落实。升级先备份，再显式运行 Alembic，最后启动 schema 匹配的版本。第一版迁移不提供破坏性 downgrade；恢复应先将已验证备份恢复至另一个数据库，核验后切换。

论坛 PostgreSQL 使用 `pg_dump`/`pg_restore` 或等价 PostgreSQL 原生备份方案，并演练实际恢复；账号等现有 SQLite 继续使用仓库的 `tools/backup_sqlite.py` 在线备份。不能用 SQLite 脚本备份论坛 PostgreSQL，也不能只复制运行中的 PostgreSQL 数据目录。媒体落地后需将对象清单、内容 hash 与数据库备份协调。

本地演练命令（数据库账号需 CREATEDB，URL 仅接受 loopback）：

```powershell
& forum/.venv/Scripts/python.exe -m forum.tools.backup_restore_check --container tables2048-forum-local-postgres-1 --output-dir forum/runtime/backups
```

工具通过导出的一致性快照运行 `pg_dump`，恢复至新建随机库并比较全部论坛表的行数和内容哈希，完成后仅删除该临时恢复库。2026-10-03 最新演练验证了 34 张表，备份 82,701 字节，schema `0005_notice_category`；报告 `forum/runtime/backups/forum-20261003T073948Z-fb5a616c.verified.json`。本地备份与报告不提交 Git，也不自动删除已有备份。

IP 限流默认按实际 TCP 对端统计。反向代理接入时配置精确的 `FORUM_TRUSTED_PROXIES` CIDR，并由代理覆盖 `X-Real-IP`；服务入口关闭 Uvicorn 自动转发头解析，手动启动 Uvicorn 时同样需 `--no-proxy-headers`。所有 API 进程配置同一高熵 `FORUM_IP_HASH_SECRET`，数据库仅存每日 HMAC 标识；缺少配置时使用进程临时密钥，不能保证跨进程限流。当前写入 120/IP/分钟、读取 600/IP/分钟，同时保留用户写入限流。IP 窗口表过期清理与生产参数调优仍待部署运维实现。

生产部署前必须阅读 [deployment_retention.md](../docs_and_configs/deployment_retention.md)，核对干净源代码基线及显式 overlay 清单，成功健康检查后才按规定做保留审查。保留每应用最新 3 个版本及全部运行/引用/pin 版本；数据库备份超过 7 天才可按策略淘汰，并始终保留每库至少 3 份已验证备份。当前没有执行生产迁移、发布或清理。
