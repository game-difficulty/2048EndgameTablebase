# 2048 Competition

## 小组单循环时刻表

- `event_fixtures.py` 独立保存小组阶段、固定队伍快照和对阵，复用正式房间的建房入口、项目契约、预约及结算，不写死赛事标识或队名。举办方在“赛事管理”锁定名单后设置分组、项目池及 BO3/BO5/BO7/自定义规则；四队一组生成六场，八队两组共十二场。奇数队小组自动轮空。
- `GET /api/events/{slug}/fixtures` 返回公开对阵与当前用户的操作能力；入房码仅向获准的参赛人员及举办方开放。`POST .../fixture-stages` 生成阶段，不提前创建空房间。分组、名单和房间模板生成后固定，名单实质变化时拒绝新的预约。
- `POST /api/events/{slug}/fixtures/{fixture_id}/actions` 接受 `schedule/confirm/reject/cancel_proposal/bind`，使用 `revision + command_id` 处理并发与幂等；严格拒绝多余请求字段。初次登记只允许本场队长或举办方，在同一个数据库事务内建房、绑定赛事与固定名单，房间归举办方所有，登记队长不获得工作人员或房主权限。
- 队长改期先保存提议，原开赛时间继续有效；另一方队长确认后修改同一房间的时间，保留席位和签到、清除双方准备状态。提议方可撤回，另一方可拒绝。举办方可以直接改期；原开赛时间已到或流程已开始时禁止修改。服务端统一存 UTC，前端 datetime-local 明确按北京时间 UTC+8 解读，不采用设备时区。
- 同队同一时间的有效预约拒绝保存，间隔不足一小时仅提醒，不保证前一场会按时结束。双方全部签到入座且队长准备后，现有服务端定时扫描到点进入抽签，缺席沿用房间十五分钟迟到规则；无需页面保持打开。
- 举办方可以绑定固定名单匹配的已有正式房间（包括已开始/已结束房间），不会重建或修改原对局；一个对阵/房间最多对应一个有效绑定。落位重赛时自动跟踪替代房间，比分从有效房间结算读取。取消房间保留历史，由举办方绑定替代房间，不允许队长反复建空房。
- 回归：`python -m pytest competition/tests/test_event_fixtures.py -q`、`npm test`（competition/frontend）。

## 锁定名单与赛程房间

- 创建房间时可选赛事的两支已锁定队伍（人数须与房间规则一致），并设置未来开战时间。请求字段为 `event_slug`、`yellow_team_id`、`white_team_id`、带时区的 `starts_at`。不填赛程字段时保留原自由落座房间。
- 保存队名、名单版本、选手和固定席位快照；按报名表队内序号落座，1 号为队长。赛事名单之后解锁不会改变已有赛程。小组对阵预约可按上文协商改期，普通手动建房仍需关闭未开赛房间后重新创建。
- 选手进入房间调用 `POST /api/competitions/{room_code}/check-in` 完成一次到场签到；GET 快照和广播不会代替别人签到。可提前进入与准备，不会消耗比赛时间。
- 开战时间指抽签/BP 流程开始的最早时间。到点且全员签到、双方队长准备后自动开始，仍保留原 BP、布阵、至少 60 秒下注窗口和首局就绪流程。包干时钟从首局实际开始计算。
- 开战时间后超过房间迟到宽限（默认 15 分钟），若仅一队已全部入座且队长准备，则另一队判负，胜方比分为本房间总局数，项目记为未实际进行的 `late_forfeit`。签到为一次到场，正常断线不追溯成迟到。
- 双方均未就位：标记 `both_late` 并结束为 0:0，无项目成绩；不猜测胜方。判定以服务端时间和持久化就位状态为准，重复执行不会重复结算。
- 赛事公开列表按时间升序展示对阵、北京时间、比分；只有参赛选手与管理人员可见进入码。统计型 14360 杯不创建对战赛程。

独立的团队比赛服务。当前实现范围为 M1 至 M8：

- 独立 FastAPI 应用、SQLite 数据库和 Vue/Vite 前端。
- 通过适配层复用主站用户会话，不依赖 Battle 或现有小游戏页面。
- 举办方创建比赛房间、配置裁判角色。
- 黄/白双方各三个席位，1 号位自动成为队长。
- 原子抢座、换座、离座、满员检测和席位锁定。
- 双方队长准备后进入 `DRAW`，为下一个 BP 里程碑提供稳定入口。
- 服务器抽签先后手，并保存随机种子承诺和算法版本供后续审计。
- 先手选择 A/BAN M、后手选择 B/BAN N、双方密封盲选和项目 C 抽签。
- BP 使用服务器绝对截止时间；后台 worker 和请求侧 catch-up 共用同一幂等推进逻辑。
- 超时严格按冻结的项目池顺序选择首个合法项目或组合。
- 盲选提交前不向对方、普通队员、举办方或事件日志暴露选择内容。
- 项目 C 展示结束后自动进入双方队长秘密布阵，每方独立计时 3 分钟。
- A/B/C 三场必须由 1/2/3 号位各出战一次；超时按 A=1、B=2、C=3 自动布阵。
- 单方提交后仅本队三名参赛者可回看本队完整布阵；双方完成后进入 `GAME_A_READY`，但完整 A/B/C 编排仍不向对手、观众、主办方或裁判公开。各场实际开始时只公开该场出战者，后续场次继续保密。
- 每局开始前执行“双方出战者就绪、双方队长确认”的开局检查；第四项就绪提交后，服务端在同一事务中自动启动对局和两队包干计时，无需赛事方在场。
- 两队各有服务器权威的 60 分钟包干时钟；刷新、掉线或关闭页面不会停钟。
- 标准 2048 测试适配器提供服务端确定性出数和动作判定，不与比赛状态机耦合。
- 单方达到测试项目条件后仅停止本方时钟；双方结束后按得分公开结果。
- 赛中六名参赛选手都能实时查看双方公开棋盘、得分和进度；只有本局出战者能操作自己的棋盘。
- 双方队长按结果 revision 确认后自动推进 A → B → C，三局后冻结全场结果。
- 任一方包干时间耗尽时，所有尚未完成项目立即按该方负场结算。
- 选手可按网络/设备、项目、规则或其他类别上报问题；同一选手同类未结问题自动去重。
- 裁判可暂停比赛；系统原子冻结当时仍运行的队伍计时器，仅在双方队长确认后由裁判恢复。
- 裁判可修订当前赛果；每次修订提升 result revision，并使旧的双方确认立即失效。
- 裁判可带原因强制推进结果确认阶段，或在异常情况下强制结束全场。
- 暂停、恢复、问题处置、赛果修订和强制裁决均进入不可变事件日志；结果确认默认不设超时。
- 房间 WebSocket 快照同步、命令幂等和不可变事件日志。
- 正式 `ProjectRegistry` 以 `project_ref + adapter_rules_version` 精确注册适配器；赛事规则版本与运行适配器版本彼此独立。
- 已注册 10 个 `tournament-v1` 正式项目适配器：项目 1“真华容道”以及 50% 出 4、EvilGen 对抗出数、纯 2 满盘竞速、大满盘撤销竞速、骰子障碍、镜面 64×10 竞速、256 砖、困难孤岛和困难随机形状 12 格。真华容道前 10 次有效移动无特殊块，随后出现首块；在无路可走或独立 10 分钟限时到达时结束，比较送出块数；死亡后计时停止，暂停比赛时限时亦暂停。
- 新建赛事使用 `tournament-v2`，旧赛事继续按 `tournament-v1` 解析。每个项目实例预先固定种子；每次有效移动的出数恰好推进一次出数流，位置、数值及孤岛出现与否由同一票据派生；即使没有空格也消耗该票据。重开继续出数流，撤销只恢复盘面、不回退游标。华容道特殊块形状、随机形状棋盘初盘和百步封锁的封格分别使用独立的确定性随机流；骰子障碍只在骰子/墙位使用阵营区分，双方后续出数仍可共享同一流。
- 比赛站公开提供 `/practice` 项目目录及 `/practice/1` 至 `/practice/12` 单人试玩页；项目 9、10 复用本地主站小游戏的困难规则且不含 Powerups，项目 11“百步封锁”和项目 12“越来越大”目前仅用于试玩、不进入正式建房项目池。试玩无需登录；已登录玩家的终局成绩会保存为每人每项目一条最佳记录，游客可查看各项目 Top 10。此“试玩榜”由前端上报，仅供练习参考，不作防作弊或赛事裁决；正式比赛仍由服务端权威适配器判定。各项目规则变更时可单独提高榜单规则版本，旧成绩不混排。`/projects/{project_ref}` 保留为项目标识兼容地址。
- 试玩目录与各项目页共用独立的浅色／深色切换；首次访问跟随系统配色，手动选择保存在浏览器本地，不影响比赛房间或主站主题。
- 正式对局和试玩页共用同一项目定义与动态棋盘视图；棋盘移动沿用主站 300ms 动画节拍，EvilGen 复用现有 WASM，所有项目显示厘秒级局内用时。
- 创建房间时冻结项目顺序、适配器描述和公开视图协议；每个项目实例固定实例 ID、适配器引用和版本。
- 项目公开画面按 `view_kind + view_protocol` 注册；标准 2048 目前仅作为流程测试适配器。
- 比赛站通过带内部凭据的 `PublicMatchProjection` 向 Live 提供公开目录和快照，Live 不直接连接比赛数据库。
- 比赛从 `DRAW` 进入既有直播大厅，复用 Live 房间、观众、聊天、点赞、音乐、画中画和 1280×720 舞台。
- `competition-match-v1` 覆盖 BP、密封布阵状态、三局项目公开画面、局间确认和全场结算；未知项目 renderer 只降级对应画面。
- BP 与秘密布阵向观众提供服务器校时的公开阶段倒计时；只公开行动方和截止时间，不公开操作凭据或密封内容。
- Live 轮询按 `generation + content_sequence` 验证完整公开快照；短暂断线保留最后一帧并显示重连状态，迟到的旧快照不会让比赛画面倒退。
- 比赛直播默认关闭礼物、红包、福袋、预测和历史 AI 统计，保留聊天、点赞、音乐和画中画。

## 目录

```text
competition/
  backend/       比赛领域、API、WebSocket、认证适配
  frontend/      独立候场站点
  tests/         比赛领域测试
  runtime/       本地数据库（不提交）
  server.py      独立服务入口
```

## 视觉衔接

比赛站保持独立路由和权限体系，但视觉基础与主站、对局站和直播站一致：使用
Clear Sans 字体、主站暖色浅底/金棕强调色，以及直播站的深海军蓝比赛舞台。
常规面板采用 6–8px 圆角、细边框和克制阴影；红色只用于暂停与不可逆裁判动作，
黄/白仅表达阵营，不作为全站装饰色。后续直播接入时，公开比赛内容应作为固定
16:9 `RoomStage` 内容组件渲染；裁判表单、问题处理和队长操作不进入观众画面。

## 本地启动

从仓库根目录启动后端：

```powershell
$env:COMPETITION_ALLOW_DEV_AUTH='1'
$env:COMPETITION_BOOTSTRAP_ORGANIZER_IDS='1'
python -m competition.server
```

另一个终端启动前端：

```powershell
Set-Location competition/frontend
$env:VITE_COMPETITION_DEV_USER='1:本地举办方:admin'
npm install
npm run dev
```

本地可直接访问 `http://127.0.0.1:5174/practice`，各项目拥有稳定的
`/practice/1` 至 `/practice/12` 地址。生产环境的 `tournament.2048tables.online`
需要将 `/practice/*`、`/projects/*` 与 `/rooms/*` 都回退到前端 `index.html`，并原样发布
`public/wasm/evil_core.js` 和 `evil_core.wasm`。

生产环境不要启用 `COMPETITION_ALLOW_DEV_AUTH`。比赛站读取主站签发的
`tb_shared_session`、`tb_session` 或 Bearer token。数据库默认位于
`competition/runtime/competition.sqlite3`，可以用 `COMPETITION_DB` 修改。

BP 默认操作时间为 60 秒，抽签展示为 4 秒，项目 C 展示为 5 秒，秘密布阵为
180 秒。本地测试可分别设置 `COMPETITION_DRAFT_TURN_SECONDS`、
`COMPETITION_DRAW_REVEAL_SECONDS`、`COMPETITION_C_DRAW_REVEAL_SECONDS` 和
`COMPETITION_LINEUP_SECONDS`。包干时长默认 3600 秒，可用
`COMPETITION_TEAM_CLOCK_SECONDS` 修改。标准 2048 仅为测试适配器，默认达到
2048 棋块结束；流程测试可用 `COMPETITION_TEST_PROJECT_TARGET_TILE` 调低目标。

Live 与比赛站部署时，两端必须配置相同的内部凭据。比赛站：

```powershell
$env:COMPETITION_LIVE_INTERNAL_TOKEN='replace-with-a-long-random-secret'
```

Live 主站：

```powershell
$env:COMPETITION_LIVE_API_ORIGIN='http://competition-service:8001'
$env:COMPETITION_LIVE_INTERNAL_TOKEN='replace-with-a-long-random-secret'
```

比赛结束后默认在直播大厅保留 30 分钟，可用
`COMPETITION_LIVE_RESULT_RETENTION_SECONDS` 调整。内部接口未配置凭据时保持关闭；
Live 侧未配置比赛服务地址时仅忽略比赛房间，不影响现有 AI 和玩家直播。

## 主要 API

### 赛事目录与统计型赛事

`/events/819984-cup-3` 复用团队选 Ban 房间；`/events/14360-cup-1` 是独立的
`team-top5-3x3-v1` 统计模块，通过 `event_formats.py` 的能力声明切换页面，不建立房间。
报名模块支持单人、自主组队、举办方分组三种策略；默认关闭报名，由管理员显式开放。
自由组队需本人接受邀请，队长可选择确认队伍。举办方锁定最终名单不要求队长先提交，
但仍须满足分组、人数、队长及队内序号等要求。离队会撤回队伍确认状态，解散队伍保留
各成员个人报名。每个账号在一届赛事内至多一支队伍；赛事队长没有房间管理权限。

按队伍导入每行 `队名,队长用户名,其余队员用户名…`，支持 Excel 制表符、逗号及可选表头，
每队人数按赛事配置校验，预览可调整队长及外援。高级导入仍支持逐人填写
`当前用户名或ID,队名（可空）,外援0或1,队长0或1,队内序号（可空）`；可选择用户名或 ID
匹配，仅填账号也可。用户名只匹配当前有效账号，优先匹配大小写完全一致的名字，
包括历史遗留的大小写冲突账号；只有不存在精确匹配且忽略大小写后唯一时才回退。
任一匹配失败或匹配到多个账号则整批拒绝。
先预览账号姓名再确认；整体替换并取消旧邀请，使用 revision
防止覆盖其他人的新操作。14360杯尚未分组时可只导入个人名单，个人成绩正常统计，
不生成临时队伍；最终锁定需四队各五人且一位外援。原统计名单会在 schema 13 中一次性迁移。

“锁定参赛人员”关闭自主报名、退出和成员变动，但举办方仍可为同一批用户调整分组；
队长仍可确认已有队伍。“锁定最终名单”要求所有成员分组且满足人数、队长及席位规则，
不要求队长确认。锁定后所有名单写接口（含旧统计导入入口）都拒绝修改。管理员解锁
必须填写原因，解锁不会自动开放报名。锁定快照、导入及其他操作写入
`tournament_enrollment_audit`。这些名单尚不自动绑定对战房间席位（下一阶段）。

统计进程需要只读访问对局站的数据库：`HUMAN_PLAY_DB`（默认与账号数据库同目录的
`human-play.sqlite3`）和 `CLOUD_AUTH_DB`。部署到独立机器时必须先提供可靠的数据源；
不能用空库代替。统计使用 SQLite `mode=ro`，不修改对局站数据、不读取回放 BLOB。
数据源异常会显示错误，不伪装成零成绩。公共统计共享 10 秒内存缓存，前台可见时每 30 秒刷新。

时间窗固定为北京时间 `[2026-10-01 00:00, 2026-10-08 00:00)`，同时限制开局和结束时间。
仅选 Table 原生、已封存且可排名的 3×3 局；外站导入不计入。按得分降序选最佳五局，
同分沿用对局站的结束时间、局 ID 降序。个人 rating 复用 `top_rating` 对平均盘面和的公式，
不平均单局 rating，也不读取个人主页 B10。团队盘面和与 rating 分别求和；不足五局
补零后除以五，计算个人 rating；零有效局的 rating 定义为零，避免对零取对数。

- `GET /api/events`、`GET /api/events/{slug}`：公共赛事目录与详情。
- `GET /api/events/{slug}/statistics`：进程、四队及个人最佳五局。
- `POST /api/events/{slug}/roster`：管理员预览/确认分组（`dry_run`、`revision`、`entries`）。
- `GET /api/events/{slug}/enrollment`：报名状态、公开名单及本人邀请（私有 no-store）。
- `POST /api/events/{slug}/enrollment/actions`：报名、组队、设置及锁定（`action`、`revision`）。
- `POST /api/events/{slug}/enrollment/import`：通用名单预览和整体替换。
- `POST /api/events/{slug}/rooms`：仅房间型赛事支持关联房间。

### 对战房间

- `GET /api/session`
- `GET /api/competitions`
- `POST /api/competitions`
- `GET /api/competitions/{room_code}`
- `POST /api/competitions/{room_code}/staff`
- `POST /api/competitions/{room_code}/seat`
- `POST /api/competitions/{room_code}/seat/leave`
- `POST /api/competitions/{room_code}/ready`
- `POST /api/competitions/{room_code}/draft/pick-ban`
- `POST /api/competitions/{room_code}/draft/blind`
- `POST /api/competitions/{room_code}/lineup`
- `POST /api/competitions/{room_code}/games/current/readiness`
- `POST /api/competitions/{room_code}/games/current/start`
- `POST /api/competitions/{room_code}/games/current/move`
- `POST /api/competitions/{room_code}/games/current/action`（当前用于项目内重开与撤销）
- `POST /api/competitions/{room_code}/games/current/result/confirm`
- `POST /api/competitions/{room_code}/issues`
- `POST /api/competitions/{room_code}/issues/{issue_id}/resolve`
- `POST /api/competitions/{room_code}/suspension/start`
- `POST /api/competitions/{room_code}/suspension/readiness`
- `POST /api/competitions/{room_code}/suspension/resume`
- `POST /api/competitions/{room_code}/games/current/result/override`
- `POST /api/competitions/{room_code}/games/current/result/force-advance`
- `POST /api/competitions/{room_code}/force-finish`
- `GET /api/internal/live/rooms`（仅 Live 内部凭据）
- `GET /api/internal/live/rooms/{public_key}`（仅 Live 内部凭据）
- `WS /ws/rooms/{room_code}`

对战房间写操作由服务层执行并写入 `competition_events`。客户端命令携带
`command_id`，相同命令重试不会重复执行。秘密布阵提交事件在公开前只记录
队伍与提交方式，不记录场次、席位或用户映射。

## 连续观战传输（2026-10-01）

选手本机执行游戏，每次有效移动、撤销、重开或结束生成一个递增序号。
使用独立的 `WS /ws/projects/{room_code}` 上传，握手协议为 `project-stream-v2`。
`payload` / `checkpoint` 负责恢复和结算，`frames` 保存尚未确认的
`{sequence, payload}`。每帧只含公开画面与真实移动轨迹，不含随机种子、撤销栈或秘密布阵。

- 选手端按 80 ms 窗口批量发送，最多 4 批未确认数据在途，不逐批等待网络往返。
  本机操作不等待网络。`stream.ack.accepted_sequence` 是累计、已经持久化的序号，
  收到后才能删除对应帧；服务器按序号幂等。ACK 超过 8 秒未到或连接失活时重连，
  通过 `stream.ready` 取得服务器游标并补发。一个对局实例只允许一个上传连接，
  被新页面接管的旧连接以 4409 关闭并停止重连，防止两个页面互相抢写。
- 普通批次以 `delta_base` 指向上一已发送批次，撤销栈、回头看看历史、指标历史
  只传保留前缀长度和追加后缀；棋盘、RNG 游标等当前恢复状态仍随批次完整发送。
  首次、重连、结束和至少每 2 秒发送完整 checkpoint。服务端按连接顺序还原并保存
  完整存档后才确认，因此这里只降低传输量，不降低落盘频率或放宽 ACK 可靠性。
  基准不符时要求重连，首包重新发送完整存档；不尝试猜测或拼接错误的撤销栈。
- 服务器在原有 session 状态中保存连续的最近 128 帧，帧缓存上限 512 KiB，
  无需新表。批次必须递增，最后一帧必须与本次完整状态一致。
  `frame_start` 表示当前可补齐范围；长断线留下缺口时，只保留连续后缀。
- 比赛页通过原有房间 WebSocket 推送本批新增帧。每个连接只有一个写入协程，
  发送队列限制 128 条/4 MiB，发送超时或溢出必须明确关闭连接，不能仅从广播集合移除。
- 直播后端通过带内部令牌的 `WS /ws/internal/live/{public_key}` 订阅赛事服务，
  通知可以合并，但依据双方游标从保留窗口取齐中间帧，不再每 250 ms HTTP 轮询画面。
  初次建立仍可通过 HTTP 取快照，大厅目录仍每秒更新。直播进程保留最近 128 帧，
  普通广播只发增量；重连重新取完整窗口。直播观众仍沿用已有的有界发送队列。
- 两个观看入口共用 `ProjectPlayback` 队列与正式棋盘组件。乱序帧先等待缺口，
  重复帧不重复播放；正常情况下按源 `elapsed_ms` 时间轴显示每一步，初始缓冲 60 ms，
  积压时最短间隔 16 ms 追赶，不反复移动时间轴锚点，
  沿用棋盘快速连续输入的动画打断机制，不改变 slide/merge/pop 参数。
  正常结算前等队列播完并给最后一步 300 ms 动画时间。
- 初次进入直接显示当前盘面。短断线在保留范围内逐步补齐；超过 128 帧、字节窗口，
  或更换项目/代际时，明确恢复到最新快照，不伪造已丢失步骤。每次出队都检查缺口，
  请求补取后最多等 800 ms；没有后续消息也必须恢复，不能卡在缺失序号永久停播。
  浏览器休眠超过 5 秒、播放积压超过 128 步时同样恢复快照。参赛者本人始终显示本机实时状态。

上线需要同时更新赛事后端、赛事前端与直播后端/前端；比赛裁决和计时不等待观看队列。
不需要数据库迁移，已有比赛状态仍可恢复。旧 HTTP 上传端点和旧准备请求返回 426，
要求选手刷新；不能混用新前端与旧后端。应在无进行中比赛时一起切换四端并要求刷新。
赛事服务目前依赖单进程内的房间 Hub，继续使用单 worker；扩容多 worker 前必须增加跨进程通知。
生产 Nginx 的 `/ws/` 已支持 Upgrade，覆盖新路由；内部直播连接仍使用已有的
`COMPETITION_LIVE_API_ORIGIN` / `COMPETITION_LIVE_INTERNAL_TOKEN`。

### 定向验证

- `python -m pytest competition/tests/test_stream_transport.py competition/tests/test_client_runtime.py competition/tests/test_project_registry_and_live.py tests/test_competition_live.py tests/test_live_routes.py -q`
- `cd competition/frontend` 后执行 `node --test tests/projectStream.test.js tests/projectPlayback.test.js tests/matchRuntime.test.js`。
- 浏览器联调仅用临时数据库：仓库根目录执行 `python -m competition.tests.stream_fixture_server`，
  `frontend` 中执行 `npm run dev -- --host 127.0.0.1 --port 5198 --config ../competition/frontend/tests/streamQa.vite.config.mjs`。
  浏览器打开 `http://127.0.0.1:5198/live/`，导入
  `/@fs/<仓库绝对路径>/competition/frontend/tests/streamBrowserHarness.js` 并运行
  `const qa = await mountStreamQA(); await qa.run(120); qa.close()`。
  联调使用真实两端 Runtime、上传连接、房间观众、内部直播订阅及直播棋盘组件；
  白方上下行各额外延迟 300 ms，第 40 次操作断线，记录两个入口实际显示的序号，
  不仅比较最后收到的快照。该临时服务仅绑定 loopback，不用于生产部署。
