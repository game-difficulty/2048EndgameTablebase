# 独立比赛系统后端设计

状态：设计草案，未实施  
更新时间：2026-09-27  
适用仓库：`src - cloud`

## 1. 文档目的

本文定义一个独立于主站、Play 对局站和现有 Battle 模块的 2048 比赛系统后端。

比赛系统承载：

- 举办方创建和管理比赛房间；
- 裁判控制比赛、处理异常和执行判罚；
- 黄、白两队各三个固定选手席位；
- 队长准备、先后手抽签、项目选择与 BAN、双方盲选；
- 双方秘密排兵布阵；
- A、B、C 三个项目依次进行；
- 每队 60 分钟包干计时；
- 每局结果确认和全场比分结算；
- 向现有 Live 系统发布严格脱敏的公开比赛状态。

本文只讨论后端领域、数据、权限、状态机、协议、计时、审计和 Live 集成。
前端布局、视觉样式和具体交互不在本文范围内。

## 2. 已确认的产品边界

### 2.1 独立系统

比赛系统不是新的 Battle mode，也不使用 Battle 房间作为运行时宿主。

最终上线时应具有：

- 独立域名；
- 独立后端应用和进程；
- 独立数据库；
- 独立 HTTP 与 WebSocket 协议；
- 独立发布、回滚和扩容边界。

现有 Battle 可以作为房间状态机、按查看者脱敏、幂等命令和实时广播的设计参考，
但比赛系统不得依赖 Battle 数据表、Battle WebSocket 消息或 Battle 页面生命周期。

### 2.2 主站账号只提供身份

比赛系统复用主站的用户身份和公开资料，但不把赛事权限写入主站账号表。

主站应只向比赛系统提供：

- 稳定用户 ID；
- 当前账号状态；
- 昵称、头像等公开资料；
- 可验证的登录凭据或一次性授权码。

举办方、裁判等比赛权限由比赛系统自己保存。

### 2.3 项目规则暂不实施

具体项目池尚未确定，且未来项目肯定不是现有小游戏池规则。因此本阶段：

- 不迁移现有 Minigames 项目；
- 不提取现有小游戏共享内核；
- 不冻结具体棋盘、动作、胜负或计分协议；
- 不实现项目专属的直播渲染协议。

但比赛编排模块与未来项目运行器之间必须从一开始保持解耦。比赛编排只能通过稳定端口
操作项目，不得 import 或查询项目内部实现。

### 2.4 观众复用现有 Live 系统

观众继续从现有直播大厅进入房间，使用现有 Live 房间外壳、观众连接、聊天、点赞、在线人数、
断线恢复和固定舞台。比赛系统不建立第二套观众房间或观众入口。

Live 侧只增加：

- 一种动态比赛房间来源；
- 一种比赛直播内容类型；
- 一套只包含公开数据的比赛直播协议。

## 3. 总体架构

```text
Main Identity Service
  └─ 身份认证、账号状态、公开资料

Competition Service
  ├─ Administration       举办方、裁判、房间配置
  ├─ Identity Gateway     主站身份交换和资料快照
  ├─ Roster               队伍与六个固定席位
  ├─ Draft                抽签、选择、BAN、盲选
  ├─ Lineup               双方秘密布阵
  ├─ Match Orchestrator   整场比赛状态机
  ├─ Team Clock           两队包干计时
  ├─ Project Port         未来项目运行器接口
  ├─ Result Settlement    单局与全场结算
  ├─ Audit                不可变操作记录
  └─ Public Projection    严格脱敏的直播投影
            │
            │ signed internal publish stream
            ▼
Existing Live Service
  ├─ Competition Dynamic Room Provider
  ├─ CompetitionMatchContent
  ├─ LiveHub / audience / reconnect
  └─ Existing Live Lobby
```

比赛数据库是比赛事实的唯一权威来源。Live 只保存必要的公开快照缓存和直播社交数据，
不得成为比赛恢复或判定的事实来源。

## 4. 后端模块拆分

### 4.1 Administration

职责：

- 创建赛事、比赛和房间；
- 配置黄白队名称；
- 配置项目池引用、规则版本和比赛时间；
- 任命或撤销举办方、裁判；
- 开放落座、锁定阵容、取消比赛；
- 执行暂停、恢复、改判、强制推进等裁判操作；
- 查询审计记录和异常状态。

Administration 不能直接改写各业务表。所有会改变比赛语义的管理操作都应调用
Match Orchestrator，并产生标准领域事件和审计记录。

### 4.2 Identity Gateway

职责：

- 将主站授权结果交换为比赛站自己的 Session；
- 验证账号是否仍可用；
- 获取并缓存公开资料；
- 在落座或比赛开始时生成资料快照。

建议使用一次性授权码或短期签名令牌交换 Session，而不是让独立域名直接依赖主站 Cookie。
比赛系统只保存主站稳定用户 ID，不保存主站密码、设备令牌或完整认证数据库副本。

### 4.3 Roster

房间固定六个选手席位：

| seat_key | 队伍 | 队内位置 | 比赛角色 |
| --- | --- | --- | --- |
| `yellow-1` | 黄 | 1 | 队长 |
| `yellow-2` | 黄 | 2 | 队员 |
| `yellow-3` | 黄 | 3 | 队员 |
| `white-1` | 白 | 1 | 队长 |
| `white-2` | 白 | 2 | 队员 |
| `white-3` | 白 | 3 | 队员 |

举办方和裁判不是选手席位，不占六个位置。

Roster 负责：

- 自主落座；
- 离座和换座；
- 并发抢座；
- 锁定和解锁阵容；
- 维护在线状态；
- 根据当前席位派生 captain/player 权限。

同一用户在同一比赛中最多占一个席位。落座必须在数据库事务内完成，并由
`UNIQUE(match_id, seat_key)` 与 `UNIQUE(match_id, user_id)` 双重约束兜底。

进入 BP 后普通选手不得自行换座。裁判解锁或替换选手必须记录原因，并按规则决定是否重置准备、
BP 或布阵状态。

### 4.4 Draft

Draft 独立管理：

- 系统抽签决定先手队；
- 先手选择项目 A，并 BAN 项目 M；
- 后手选择项目 B，并 BAN 项目 N；
- 双方同时秘密选择 X、Y；
- X、Y 同时公开；
- 系统从 X、Y 中确定项目 C。

Draft 不读取前端倒计时，不信任前端可选项目列表。所有合法性检查基于比赛创建时冻结的项目池快照。

建议约束：

- A 与 M 不同；
- B 与 N 不同；
- A、B、M、N 互不相同；
- X、Y 不得为 A、B、M、N；
- X 与 Y 可以相同；
- X 与 Y 相同时直接得到 C，无需执行无意义的随机二选一；
- 项目池至少包含五个当场可用项目。

这些建议在赛事规则最终确认前仍属于待冻结项。

每个阶段保存绝对 `deadline_at`。到期时，服务器按冻结项目池中的稳定排序选择第一个合法项目。
自动选择必须和人工选择走同一事务与校验路径，并标记 `automatic=true`。

盲选数据在双方提交前不得进入普通比赛快照、日志消息、Live outbox 或可被观众访问的查询结果。

### 4.5 Lineup

Lineup 负责双方对 A、B、C 三局的秘密出战安排。

默认约束：

- 三位选手各出战一次；
- 每个项目恰好一位选手；
- 双方独立拥有 180 秒；
- 超时按队内 1、2、3 号位依次对应 A、B、C；
- 提交后队长不能自行修改；
- 完整 A/B/C 编排只向本队三名已落座参赛者返回；
- 对手、观众、主办方和裁判只能看到是否已经提交，不能查看密封编排；
- 双方提交完毕也不公开全套阵容；仅在每局实际开始时公开该局出战者，后续场次继续保密；
- 裁判的异常处置通过暂停、改判或强制结束完成，不以查看密封阵容为前提。

### 4.6 Match Orchestrator

Match Orchestrator 是所有比赛状态转换的唯一入口。

建议主状态：

```text
DRAFTING      比赛配置尚未开放
SEATING       开放落座
READY         六席已满，等待双方队长准备
DRAWING       生成并提交先后手抽签结果
DRAFT_FIRST   先手选择 A / BAN M
DRAFT_SECOND  后手选择 B / BAN N
DRAFT_BLIND   双方秘密选择 X / Y
DRAFT_REVEAL  公开 X / Y 并确定 C
LINEUP        双方秘密布阵
GAME_A        项目 A 进行中
CONFIRM_A     A 局结果等待双方队长确认
GAME_B        项目 B 进行中
CONFIRM_B     B 局结果等待双方队长确认
GAME_C        项目 C 进行中
SETTLEMENT    全场结算事务
FINISHED      结果冻结
CANCELLED     比赛取消
```

暂停不是另一个主状态，而是附加的 `suspension`：

```text
suspension.active
suspension.reason
suspension.started_at
suspension.actor_id
```

这样暂停不会丢失原阶段信息。

每次状态转换必须：

1. 锁定比赛行；
2. 将所有已到期的前置阶段先补算完成；
3. 校验操作者权限、当前 revision 和当前 phase token；
4. 在同一事务中写领域状态、审计事件和 outbox；
5. 增加比赛 revision；
6. 提交后再广播新快照。

任何其他模块不得绕过 Orchestrator 直接修改 `matches.phase`。

### 4.7 Team Clock

每队初始拥有 60 分钟包干时间。数据库不应每秒更新。

每队保存：

```text
remaining_ms_base
running_since
state = stopped | running | expired
revision
```

当前剩余时间计算：

```text
running: max(0, remaining_ms_base - (server_now - running_since))
stopped: remaining_ms_base
```

时钟规则：

- 项目 A 正式开始时两队同时运行；
- 某方项目结果被服务器接受后，只停止该方时钟；
- 另一方继续计时；
- 双方都结束后进入确认阶段，双方均停止；
- 下一局正式开始时双方继续运行；
- 赛事暂停时停止所有正在运行的时钟；
- 恢复时只恢复暂停前本应运行的时钟；
- 普通掉线、刷新或关闭页面不停止时钟；
- 任一队剩余时间归零时，该队所有尚未完成项目立即判负。

时钟停止、超时判负和局状态转换必须在同一数据库事务中完成。

客户端提交“完成”不能直接暂停时钟。必须由 Project Session Port 返回服务器接受的完成事实。
未来项目运行器应定义完成证明何时算作接受，以及验证排队时间是否计入队伍时间。

### 4.8 Result Settlement

负责：

- 将项目运行器的完成事实转换为比赛单局结果；
- 等待双方都完成或出现超时/判罚；
- 公开单局结果；
- 接受双方队长确认；
- 推进下一局；
- 计算全场比分和胜者；
- 冻结结果及回放引用。

队长确认只能确认已经公开且 revision 未变化的结果。确认阶段是否设置超时、超时后是否自动推进，
仍需赛事规则确认。裁判必须拥有有理由的强制推进能力。

需求原文第 8 条再次写了“项目 B 对战”，本文按项目 C 理解，最终规则文档应显式修正。

### 4.9 Audit

所有影响比赛公平性或最终结果的动作必须生成不可变审计事件，包括：

- 创建、开放、锁定、取消比赛；
- 举办方和裁判任免；
- 落座、离座、换座、裁判替换选手；
- 队长准备或取消准备；
- 抽签；
- 所有人工和自动 BP 提交；
- 布阵提交和超时默认（公开审计负载不得包含具体编排）；
- 时钟启动、停止、暂停、恢复和归零；
- 项目启动、完成、验证失败和判罚；
- 队长确认；
- 裁判改判和强制推进；
- 最终结算。

审计事件至少包含：

```text
event_id
match_id
sequence
event_type
actor_type
actor_user_id
request_id
phase
revision_before
revision_after
public_payload_json
private_payload_encrypted_or_restricted
created_at
```

普通观众接口只读取公开事件。秘密盲选和布阵不得因为写入审计表而被普通查询泄露。

## 5. 权限模型

### 5.1 持久角色

| 角色 | 范围 | 主要能力 |
| --- | --- | --- |
| `platform_admin` | 全平台 | 管理赛事、任命人员、紧急处置 |
| `organizer` | 指定赛事 | 创建和配置比赛、任命裁判、开放房间 |
| `referee` | 指定赛事或比赛 | 暂停、恢复、替换选手、判罚、强制推进 |

### 5.2 派生角色

| 派生角色 | 来源 |
| --- | --- |
| `yellow_captain` | 当前占据 `yellow-1` |
| `white_captain` | 当前占据 `white-1` |
| `yellow_player` | 当前占据任一黄方席位 |
| `white_player` | 当前占据任一白方席位 |
| `spectator` | 已认证但没有管理或选手权限的访问者 |

客户端传入的角色声明不能作为授权依据。每个命令都必须根据数据库中的当前授权和席位重新计算权限。

### 5.3 权限矩阵摘要

| 操作 | 举办方 | 裁判 | 队长 | 队员 | 观众 |
| --- | --- | --- | --- | --- | --- |
| 创建/配置比赛 | 是 | 否 | 否 | 否 | 否 |
| 开放落座 | 是 | 可授权 | 否 | 否 | 否 |
| 自主落座 | 可作为选手 | 可作为选手 | 是 | 是 | 否 |
| 队长准备 | 否 | 否 | 是 | 否 | 否 |
| BP 提交 | 否 | 异常处置 | 是 | 否 | 否 |
| 布阵提交 | 否 | 异常处置 | 是 | 否 | 否 |
| 项目操作 | 否 | 否 | 当前出战者 | 当前出战者 | 否 |
| 确认单局结果 | 否 | 强制推进 | 是 | 否 | 否 |
| 暂停/恢复 | 否或配置授权 | 是 | 否 | 否 | 否 |
| 判罚/改判 | 否或配置授权 | 是 | 否 | 否 | 否 |
| 查看公开状态 | 是 | 是 | 是 | 是 | 是 |
| 查看秘密状态 | 限配置 | 是并审计 | 仅本队 | 仅必要本队信息 | 否 |

## 6. 项目运行器解耦端口

本节只冻结比赛系统对未来项目规则的依赖方向，不规定项目实现。

### 6.1 Project Catalog Port

```text
snapshot_pool(project_refs, requested_versions) -> ProjectPoolSnapshot
validate_pool(snapshot, competition_config) -> ValidationResult
public_descriptor(project_ref, ruleset_version) -> ProjectPublicDescriptor
```

`ProjectPoolSnapshot` 必须在比赛开放前冻结，并包含稳定排序。后续项目服务升级不能改变进行中比赛的
默认超时选择结果。

### 6.2 Project Session Port

```text
create_game(match_id, game_key, project_ref, ruleset_version, competitors, config)
start_game(game_instance_id, authoritative_start_at)
submit_action(game_instance_id, actor_id, sequence, action)
get_state(game_instance_id, viewer_context)
suspend_game(game_instance_id, reason)
resume_game(game_instance_id)
forfeit_game(game_instance_id, side, reason)
get_completion(game_instance_id) -> pending | accepted | rejected
get_result(game_instance_id) -> ProjectResult
get_replay_ref(game_instance_id) -> ReplayRef
```

上述名称是逻辑契约，不要求最终采用进程内调用、HTTP、RPC 或消息队列。

Match Orchestrator 只认识：

- 项目实例 ID；
- 生命周期状态；
- 哪一方已被接受为完成；
- 统一结果；
- 回放引用；
- 公开视图引用。

它不能读取棋盘、随机数、动作细节或项目内部计分字段来自行判定胜负。

### 6.3 Project Public View Port

```text
get_public_view(game_instance_id, side, after_sequence) -> PublicProjectView
```

返回值必须包含项目自有的：

```text
view_kind
view_protocol
generation
sequence
snapshot_or_events
```

未来 Live 项目渲染器按 `view_kind + view_protocol` 注册。比赛直播外壳不理解其内容。

### 6.4 故障原则

- 项目服务暂不可用时，比赛保持在当前阶段并向裁判报警；
- 不因项目服务超时直接把某一方判负；
- 重试创建项目实例必须幂等；
- 项目实例 ID 与规则版本一经启动不得替换；
- 项目公开视图故障不能影响权威比赛状态；
- 没有直播渲染器时，Live 降级显示项目名称和生命周期状态。

## 7. 数据模型

以下是逻辑模型，具体数据库 DDL 在实现阶段确定。正式赛事优先使用支持可靠事务、行锁和备份恢复的
关系数据库；若首版使用 SQLite，必须保持单写者部署，并避免依赖只在 SQLite 中成立的行为。

### 7.1 competitions

```text
competition_id
public_key
name
status
config_json
created_by
created_at
updated_at
```

### 7.2 competition_staff

```text
competition_id
user_id
role = organizer | referee
granted_by
granted_at
revoked_at
```

### 7.3 matches

```text
match_id
competition_id
public_match_key
room_code
phase
revision
phase_token
yellow_team_name
white_team_name
first_side
project_pool_snapshot_json
selected_projects_json
lineup_visibility_policy
suspension_json
result_json
opened_at
started_at
finished_at
created_at
updated_at
```

`room_code` 只用于选手加入，不应直接作为 Live 内部主键。`public_match_key` 用于公开 URL，且不能暴露
数据库自增信息。

### 7.4 match_seats

```text
match_id
seat_key
side
position
user_id
display_name_snapshot
avatar_snapshot
joined_at
locked_at
left_at
revision
```

同一 `match_id + seat_key` 只能有一个当前占用者，同一用户只能占一个当前席位。

### 7.5 match_team_ready

```text
match_id
side
captain_user_id
ready
ready_at
seat_revision
```

队长席发生变化时，相关 ready 必须自动失效。

### 7.6 match_draft_actions

```text
action_id
match_id
stage
side
pick_project_ref
ban_project_ref
automatic
submitted_by
submitted_at
sealed_payload
revealed_at
```

盲选记录在公开前必须由查询层隔离。是否对密封内容做数据库级加密可在威胁模型确定后决定，
但至少不能和公开 JSON 共用一个无差别序列化路径。

### 7.7 match_lineups

```text
match_id
side
game_key = A | B | C
seat_key
player_user_id
automatic
submitted_by
submitted_at
revealed_at
```

### 7.8 match_games

```text
game_id
match_id
game_key
order_index
project_ref
ruleset_version
project_instance_id
status
yellow_player_user_id
white_player_user_id
started_at
ended_at
result_json
result_revision
```

### 7.9 team_clocks

```text
match_id
side
remaining_ms_base
running_since
state
resume_after_suspension
revision
updated_at
```

### 7.10 match_confirmations

```text
match_id
game_key
side
captain_user_id
result_revision
confirmed_at
```

结果 revision 改变时旧确认自动失效。

### 7.11 match_events

保存有序领域事件和审计数据。`UNIQUE(match_id, sequence)`。

### 7.12 command_deduplication

```text
match_id
actor_user_id
request_id
command_type
request_fingerprint
response_json
created_at
```

相同 request ID 与相同 fingerprint 返回原响应；相同 request ID 携带不同内容必须拒绝。

### 7.13 integration_outbox

```text
outbox_id
aggregate_type
aggregate_id
event_sequence
destination
payload_json
created_at
delivered_at
attempt_count
next_attempt_at
```

Public Match Projection 通过事务 outbox 发布，避免数据库已经推进但 Live 永远没有收到阶段变化。

## 8. 命令和快照协议

### 8.1 HTTP 职责

HTTP 适合：

- 身份登录和 Session；
- 创建、配置和查询赛事；
- 创建、开放和查询比赛；
- 查询历史、审计和最终结果；
- 下载回放或导出结果；
- 首次加载完整快照。

### 8.2 WebSocket 职责

选手和裁判连接比赛服务自己的 WebSocket，用于：

- 订阅按当前查看者脱敏的比赛快照；
- 落座、准备、BP、布阵和确认命令；
- 项目动作的控制通道；
- 在线状态和恢复；
- 接收 revision 更新与命令回执。

观众不连接该 WebSocket，而是连接现有 Live 房间。

### 8.3 命令信封

```json
{
  "type": "command",
  "command": "draft.submit_first",
  "match_id": "public-or-session-bound-id",
  "request_id": "uuid",
  "expected_revision": 42,
  "phase_token": "opaque-token",
  "payload": {}
}
```

服务端不得信任 payload 中的 user ID、side、seat 或 captain 标记。

### 8.4 回执

```json
{
  "type": "command_result",
  "request_id": "uuid",
  "accepted": true,
  "revision": 43,
  "server_time": "UTC timestamp",
  "data": {}
}
```

冲突响应应返回当前 revision、当前 phase 和可机器识别的错误码，使客户端能够刷新快照而不是盲目重试。

### 8.5 按查看者快照

内部先构建完整领域状态，再分别投影：

```text
OrganizerProjection
RefereeProjection
YellowCaptainProjection
WhiteCaptainProjection
YellowPlayerProjection
WhitePlayerProjection
PublicProjection
```

本赛事的项目对局不采用 Battle 的对手棋盘隔离规则。项目开始后，Yellow/White
PlayerProjection 都包含双方经 Project Public View Port 生成的公开棋盘、得分和进度，供六名
参赛选手实时查看；操作命令仍只授权给当前场次对应侧的出战者。项目内部状态、控制凭据、校验数据和
尚未被运行器接受的完成声明不属于该公开视图。

禁止先生成一个包含秘密字段的通用 JSON，再依靠前端隐藏。投影函数必须有字段白名单测试。

## 9. 超时调度与恢复

### 9.1 绝对期限

BP、盲选、布阵和确认阶段都保存绝对 UTC `deadline_at`。前端倒计时只是展示。

### 9.2 双重推进机制

超时不能只依赖一个常驻定时器。每个到期阶段通过两种方式推进：

1. 后台 deadline worker 主动扫描并推进；
2. 任何读取、命令或重连在处理前先执行 `catch_up(match_id, now)`。

两条路径调用相同幂等事务。因此进程重启、任务丢失或短暂停机后，比赛仍能恢复到正确阶段。

### 9.3 服务器时间

持久化使用 UTC 时间；进程内等待可使用 monotonic clock，但不能把 monotonic 值写入数据库。
所有快照返回 `server_time`，客户端据此显示倒计时。

### 9.4 重连

- 用户 Session 与席位分离，断开 WebSocket 不离座；
- 重连后按数据库恢复当前权限和阶段；
- 项目动作使用 sequence 拒绝重复或乱序；
- 掉线是否允许裁判暂停属于赛事规则，系统不自动暂停；
- 房间在线状态不是比赛结果事实。

## 10. 抽签与可审计随机

先后手抽签和从 X/Y 中确定 C 必须由服务器完成。

建议比赛开放时生成随机种子并公布 commitment：

```text
commitment = SHA-256(match_id || seed)
```

抽签使用带领域标签的确定性派生：

```text
HMAC(seed, "first-side")
HMAC(seed, "blind-project-c")
```

比赛结束后可公开 seed，第三方据此验证抽签。未结束比赛的 seed 不进入普通快照或 Live。

如果未来赛事不要求公开验证，也应保留 seed、算法版本和抽签输入用于内部审计。

## 11. Live 集成

### 11.1 集成原则

Live 只接收 `PublicMatchProjection`。它不能：

- 查询秘密盲选；
- 查询未公开布阵；
- 读取裁判备注；
- 读取项目控制令牌；
- 直接计算胜负；
- 反向修改比赛状态。

### 11.2 动态房间来源

当前 Live 静态房间通过 `RoomDefinition` 注册，真人直播动态房间则在路由中专门查询
`human_rooms`。比赛接入前应将动态房间来源通用化，而不是继续增加第三组硬编码分支。

建议接口：

```text
DynamicRoomProvider
  list_active_rooms()
  resolve_room(room_id)
  authorize_publisher(room_id, credential)
  generation(room_id)
  is_ended(room_id)
```

注册：

- `HumanPlayRoomProvider`
- `CompetitionMatchRoomProvider`

比赛 provider 可以通过有认证的内部 API 读取公开房间目录，或消费 Competition Service 的公开房间事件。
Live 不应直接连接比赛数据库。

### 11.3 房间生命周期

推荐：

- 两位队长准备并进入抽签后，比赛直播房间进入 Live 大厅；
- BP、布阵、三局和结算均在同一 Live 房间内；
- 比赛 FINISHED 后保留一段结算展示期；
- 展示期结束后房间离线并从大厅撤下；
- 比赛重启或回退不得复用旧 generation 的实时事件。

### 11.4 内容注册

新增：

```text
content_kind = competition-match
protocol     = competition-match-v1
```

后端注册 `CompetitionMatchContent`，前端注册 `CompetitionMatchContent.vue`。

LivePage、LiveHub、观众连接、聊天、点赞、在线人数、重连和 1280×720 RoomStage 保持不变。

### 11.5 Public Match Projection

公开投影至少包含：

```text
match_public_key
generation
content_sequence
phase
phase_timing（公开阶段开始时间、截止时间、行动方）
yellow_team / white_team
public_roster
first_side（公开后）
public_draft
lineup_submission_status
revealed_players
games A/B/C
current_game
score
team_clocks
captain_confirmation_status
public_result
project_public_views
server_time
```

绝不包含：

- 尚未公开的 X/Y；
- 任一队尚未开赛的布阵及完整 A/B/C 映射；
- 项目内部私密状态；
- 作弊检测；
- 裁判私密备注；
- 用户 Session 或操作凭据；
- 未经权威接受的完成声明。

### 11.6 发布通道

Competition Service 通过独立内部发布凭据或短期发布 lease 连接 Live。

推荐使用：

- 首次 JSON 全量快照；
- 有序 JSON 阶段事件；
- 项目公开视图所需的独立子协议事件；
- `generation + content_sequence` 检测重启、重复和断序；
- 断序时由 Live 重新请求比赛公开快照。

比赛状态事件量很低，不必为了 BP、比分或时钟使用二进制协议。未来项目棋盘若有高频步骤，再由对应
`view_protocol` 定义二进制格式。

当前 M8 首版由 Live 定时拉取完整公开投影，并用 `generation + content_sequence` 去重和防倒退。
比赛服务不可达时，Live 保留最后一份已验证投影供观众辨认现场状态，同时将房间标记为离线/重连；
恢复后直接以最新完整投影续播，不要求补齐中间事件。目录请求短暂失败也保留上一次目录缓存，只有
比赛服务成功返回的新目录才会撤下房间。后续如果项目画面频率显著提高，可在不改变
`competition-match-v1` 外壳的前提下增加事件推送或项目子协议流。

### 11.7 直播协议分层

`competition-match-v1` 只负责稳定比赛外壳：

- 阶段；
- 队伍、选手和比分；
- BP 公开状态；
- A/B/C；
- 两队时钟；
- 确认状态；
- 全场结果。

项目画面作为子协议：

```json
{
  "side": "yellow",
  "view_kind": "future-project-view",
  "view_protocol": "future-project-view-v1",
  "generation": 1,
  "sequence": 18,
  "payload": {}
}
```

未知项目视图不得使直播房间失败。Live 应降级显示项目名称、出战者和进行中/已完成状态。

### 11.8 直播大厅

观众仍使用现有直播大厅。大厅只需把当前硬编码的“玩家/AI”卡片描述改为房间提供的通用元数据：

```text
category_label
badge
title
subtitle
preview
started_at
```

比赛房间可显示“赛事直播”、双方队名、当前比分或阶段和内容提供的预览。
这属于房间目录通用化，不是另建观众交互。

### 11.9 Live 社交能力

聊天、点赞和在线人数可以直接复用。礼物、红包、福袋和预测是否对比赛房间开放应由 capability 配置决定，
不能因为复用 Live 外壳而自动继承 AI 房间的全部活动。

首版建议：

- 聊天：开启；
- 点赞：开启；
- 音乐和画中画：沿用 Live 默认能力；
- 礼物：由运营决定；
- 红包、福袋、预测：默认关闭。

## 12. 并发、幂等和一致性

### 12.1 Revision

每场比赛拥有单调递增 revision。所有会修改比赛状态的命令带 `expected_revision`。

对于项目高频动作，可在项目实例内部使用独立 sequence，避免每个方向操作都竞争整场比赛 revision。

### 12.2 Phase Token

进入每个新阶段生成新的不可预测 `phase_token`。命令必须携带当前 token，避免旧页面在阶段切换后提交
仍然格式合法的延迟操作。

### 12.3 Request ID

所有有副作用的 HTTP 和 WebSocket 命令必须带 request ID。幂等记录保留时间至少覆盖一场比赛的最长生命周期。

### 12.4 事务 Outbox

数据库状态和对 Live 的公开事件不能进行不受保护的“双写”。比赛事务只写状态和 outbox，独立发布器重试投递。

Live 收到重复 `event_sequence` 必须幂等忽略；出现缺口时重新获取完整公开快照。

## 13. 安全要求

- 比赛控制和 Live 发布使用不同凭据；
- 裁判和举办方接口执行 CSRF/Origin 或等价保护；
- 每个命令在服务端重新计算身份和权限；
- 房间码不可作为授权凭据；
- 公开 match key 不可推导数据库 ID；
- 秘密 BP 和布阵不能进入普通日志、错误追踪上下文或 Live outbox；
- 裁判读取秘密数据必须审计；
- 上传或回放数据设置严格大小和动作数上限；
- 项目运行器响应视为不受信任输入，需要 schema 和版本校验；
- Live 只能消费签名的 Public Match Projection；
- 比赛结束后撤销项目控制令牌和 Live 发布 lease。

## 14. 故障与恢复

### 14.1 Competition Service 重启

启动时：

- 扫描未结束比赛；
- 补算所有已到期阶段；
- 恢复队伍时钟；
- 恢复或重新连接项目实例；
- 重放未投递 outbox；
- 为 Live 发布新的 generation 快照。

### 14.2 Live 不可用

Live 故障不暂停比赛，也不影响判定。公开 outbox 保留并重试；Live 恢复后获取最新完整快照，
不要求逐条补播所有过期棋盘动画。

### 14.3 项目运行器不可用

项目运行器故障不得自动判负。Orchestrator 将比赛标记为需要裁判处理，并保留权威时钟快照。
是否自动暂停由赛事配置决定，所有处置写审计日志。

### 14.4 身份服务不可用

已有比赛 Session 可按短期缓存继续工作；新的登录、换人或管理授权变更失败关闭。
不能因为暂时无法获取头像而把已落座选手移出比赛。

## 15. 可观测性

建议指标：

- 活跃比赛数和各 phase 数量；
- deadline worker 延迟；
- 命令接受、冲突、拒绝和重复率；
- 比赛 WebSocket 在线人数和重连率；
- 项目端口调用 P50/P95/P99；
- 时钟停止确认延迟；
- outbox 待投递数量和最老年龄；
- Live 发布断线、断序和快照恢复；
- 裁判暂停、改判和强制推进次数；
- 秘密投影泄露测试失败数必须始终为零。

日志使用 match public key、event sequence 和 request ID 关联，不记录盲选或布阵的未公开明文。

## 16. 测试策略

### 16.1 状态机测试

- 非法阶段命令全部拒绝；
- 每条合法路径能到达 FINISHED；
- 取消、暂停和恢复不丢失原阶段；
- A、B、C 顺序不可跳过；
- 第 8 条按 C 项目执行；
- 重复状态转换幂等。

### 16.2 席位与权限测试

- 两人同时抢一个座位只有一人成功；
- 同一用户不能占两个席位；
- 1 号位变化立即改变队长权限并清除旧准备；
- 六席未满不能开始 BP；
- 普通队员不能提交 BP 或布阵；
- 裁判操作全部写审计。

### 16.3 秘密数据测试

为每种 Projection 做字段白名单快照测试：

- 黄队长看不到白方 X/Y 和布阵；
- 白队长看不到黄方 X/Y 和布阵；
- 队员只看到规则允许的本队信息；
- PublicProjection 永远不含密封字段；
- Live outbox 与日志不含密封字段；
- 双方提交后只公开规则允许公开的数据。

### 16.4 超时测试

- 人工提交与 timeout worker 同时执行只产生一个结果；
- 进程停机跨过 deadline 后能 catch up；
- 默认项目严格按冻结排序选择；
- 布阵超时采用 1/2/3 顺序；
- 队伍时钟归零与项目完成并发时只结算一次；
- 暂停期间不消耗时钟；
- 普通掉线继续消耗时钟。

### 16.5 Live 集成测试

- 活跃比赛出现在现有大厅；
- 未知房间返回 404；
- BP 未公开数据不会出现在 Live；
- Live 断线不影响比赛；
- outbox 重试不重复推进内容 sequence；
- Live 断序后获取完整公开快照；
- 未知项目 view protocol 安全降级；
- FINISHED 展示期结束后房间从大厅撤下。

### 16.6 故障注入

- Competition Service 在每个阶段重启；
- Live 服务不可用；
- 项目运行器超时或返回非法响应；
- 身份服务不可用；
- 数据库事务冲突；
- outbox 重复和乱序投递。

## 17. 建议实施顺序

该顺序不包含前端视图和具体项目规则。

### 阶段一：领域核心

- 数据模型；
- 权限与席位；
- Match Orchestrator；
- Draft；
- Lineup；
- Team Clock；
- 审计、revision、request ID；
- 纯后端状态机测试。

### 阶段二：身份与控制协议

- Identity Gateway；
- 举办方和裁判 HTTP API；
- 选手 WebSocket；
- 按查看者 Projection；
- 重连和 timeout catch-up。

### 阶段三：项目端口

- 冻结 Project Catalog / Session / Public View 契约；
- 使用测试用假项目运行器验证 A/B/C 编排；
- 不实现正式项目规则。

### 阶段四：Live 内容接入

- 通用 DynamicRoomProvider；
- CompetitionMatchRoomProvider；
- Public Match Projection outbox；
- CompetitionMatchContent；
- 现有大厅通用房间卡片元数据；
- Live 故障与恢复测试。

正式项目规则系列在上述边界稳定后单独设计和实施。

## 18. 尚待产品确认

下列规则会影响后端状态和判定，实施前必须冻结：

1. A 与 M、B 与 N 是否明确要求互不相同；
2. X=Y 时是否直接确定 C；
3. 三位选手是否必须各出战一次；
4. 已确定：完整布阵仅本队参赛者可见，单场出战者在该场实际开始时公开；
5. 每个项目的完成条件、比分、平局和异常判定；
6. 双方是否共享相同初始随机条件；
7. 项目完成证明在何时停止队伍时钟；
8. 队长确认是否有超时，超时后如何推进；
9. 选手掉线是否允许裁判暂停以及允许多久；
10. 举办方是否同时拥有裁判权限；
11. 已确定：主办方和裁判不能查看密封布阵；盲选的查看权限仍需单独确认；
12. 比赛直播从抽签、BP、布阵还是项目 A 开始进入大厅；
13. FINISHED 后直播结算画面保留多久；
14. Live 聊天、礼物和活动能力对比赛房间的开放范围；
15. 项目池配置是赛事级、比赛级，还是两者组合。

这些问题不影响系统独立、项目端口解耦和复用现有 Live 房间模块三个已经确认的架构决定。
