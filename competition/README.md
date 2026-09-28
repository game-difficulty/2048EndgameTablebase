# 2048 Competition

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
- 比赛站公开提供 `/practice` 项目目录及 `/practice/1` 至 `/practice/12` 单人试玩页；项目 9、10 复用本地主站小游戏的困难规则且不含 Powerups，项目 11“百步封锁”和项目 12“越来越大”目前仅用于试玩、不进入正式建房项目池。试玩无需登录、成绩不入库，正式比赛仍由服务端权威适配器判定。`/projects/{project_ref}` 保留为项目标识兼容地址。
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

所有写操作由服务层执行并写入 `competition_events`。客户端命令携带
`command_id`，相同命令重试不会重复执行。秘密布阵提交事件在公开前只记录
队伍与提交方式，不记录场次、席位或用户映射。
