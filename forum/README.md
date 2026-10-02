# 2048 社区站

独立 Vue 3 / FastAPI 服务，论坛业务数据使用 **PostgreSQL 17**。沿用站点账号的稳定用户 ID；现有账号 SQLite 仅作只读会话校验，不存储论坛内容，也不复制密码。

状态：2026-10-03 基础版本已提交为 `392261a`；第二批内容、媒体、录像、订阅通知和管理功能，以及第三批关注、提及、阅读进度、回复草稿和申诉已实现，尚未部署。当前数据库版本为 `0003_social`。完整首版范围见 [产品与技术方案](../docs_and_configs/design/forum_v1_plan_20261003.md)。完整首版发布验收仍需后文列出的集成和运维工作。

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

内容编辑采用 Markdown 源码 + 工具栏 + 预览，阅读时把解析结果转为 Vue 节点，不执行用户 HTML。仅显示论坛上传的图片，外部图片地址不会自动加载。搜索仍为参数化 `ILIKE`，尚未实现大规模全文索引。管理后台具备常用处置和申诉流程；自动审核与定时公告仍待实现。

第三批交互规则：关注上限 200 人，仅通知关注后发布的新主题，不回补历史；关注清单仅本人可见。正文提及上限 10 人，按稳定用户 ID 定位，代码、转义、图片说明和链接文字不触发提及；同一用户在同一帖子中最多生成一条通知，反复编辑不会重复通知。通知暂停设置适用于这些新通知。

回复草稿最多保留 50 份非空内容，「我的社区」可查看、继续编辑或清空。主题隐藏后仍可清空自己的草稿，释放配额；隐藏主题标题不向无权限用户展示。云端清空后保留版本号，防止旧设备把已经清空的草稿重新写回；多端冲突由用户对照选择版本。阅读位置仅向后推进，「我的社区」列出最近 100 个仍公开的主题，不把加载但未进入视口的楼层自动标为已读。

申诉绑定具体处置记录，支持自己的主题隐藏、锁定及账号禁言；禁言不阻止私有草稿与申诉。版主只能处理所属板块的主题申诉，账号禁言申诉由管理员处理，不能处理自己的申诉。通过时检查相关处置是否仍为最新状态，过期申诉不能撤销后来的决定；每次处理必须填写原因并写入审计。目前处理结果在「我的社区」刷新查看，尚未接入审核结果的独立实时通知。

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
& forum/.venv/Scripts/python.exe -m forum.worker
```

第二批 schema 为 `0002_content`，升级本地已有库时同样先执行 `alembic upgrade head`，再启动 API 和 worker。迁移不删除原有帖子。worker 可以多进程运行，以 `SKIP LOCKED` 领取事件并在同一事务内去重投递；失败事件保留错误类型，每分钟重试。未启动 worker 时，作者/引用回复仍可直接产生通知，订阅通知会留在 outbox，不能算已送达。

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

当前 53 项 PostgreSQL/服务测试覆盖基础流程，以及附件可见性与撤回、图片重编码、HPR/RPL1 校验、公开来源撤回、订阅去重、通知偏好、SSE 会话撤销、版主授权和禁言、置顶分页及审计；第三批增加关注通知时间边界、提及去重与转义、私有阅读位置、并发草稿修订/清空、禁言期间申诉、跨板块/自审拒绝、旧申诉不能撤销新处置。23 项前端测试覆盖语法和 Markdown 的安全边界，生产构建通过。

浏览器已验证富文本与图片发布、录像播放/逐步定位、worker + SSE 实时通知、版主授予/撤销和审计；第三批验证回复草稿离开/恢复、楼层跳转保留输入、多端冲突对照、提及发布与个人页、关注、继续阅读及申诉提交—管理处理—主题恢复，并检查 320px 深色页面无横向溢出。公开 Play 源站通过受控替身验证撤回逻辑，尚未对生产账号/真实公开录像做线上联调。

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

1. **扩展创作与作品**：录像分支分析、其他历史分析录像/赛事协议、可信成绩证明。当前已具备富文本、图片、主题与回复草稿、主题元信息编辑和上述 Play 录像回放主流程。
2. **扩展互动与发现**：中文搜索索引与短词召回、关注/未读信息流、公开内容 SEO、分类型通知偏好及聚合。关注、提及、阅读位置、主题/板块订阅、基础通知偏好与分页、outbox worker 和 SSE 已实现。
3. **扩展运营**：定时公告、投票与问答、账号屏蔽与隐私流程、赛事/直播主题联动、审核结果与系统消息通知。常用版主和管理员处置、申诉界面已实现。
4. **完整首版验收**：双语、可访问性复核、媒体配额、代理 IP 限流、保留策略、备份恢复演练、混合负载和故障测试、线上共享会话验证、监控告警与灰度发布。

上述步骤对应总方案的剩余里程碑，不取消其他必要模块。大厅持久聊天按总方案作为独立开关建设，语音/无限私信不在默认首版范围。

## 部署与恢复边界

本地 Compose 不是生产部署模板。生产需独立数据库账号/连接限制/凭据、HTTPS、可信反向代理、服务管理和监控；migration 账号与运行账号的最小权限分离尚需部署配置落实。升级先备份，再显式运行 Alembic，最后启动 schema 匹配的版本。第一版迁移不提供破坏性 downgrade；恢复应先将已验证备份恢复至另一个数据库，核验后切换。

论坛 PostgreSQL 使用 `pg_dump`/`pg_restore` 或等价 PostgreSQL 原生备份方案，并演练实际恢复；账号等现有 SQLite 继续使用仓库的 `tools/backup_sqlite.py` 在线备份。不能用 SQLite 脚本备份论坛 PostgreSQL，也不能只复制运行中的 PostgreSQL 数据目录。媒体落地后需将对象清单、内容 hash 与数据库备份协调。

生产部署前必须阅读 [deployment_retention.md](../docs_and_configs/deployment_retention.md)，核对干净源代码基线及显式 overlay 清单，成功健康检查后才按规定做保留审查。保留每应用最新 3 个版本及全部运行/引用/pin 版本；数据库备份超过 7 天才可按策略淘汰，并始终保留每库至少 3 份已验证备份。当前没有执行生产迁移、发布或清理。
