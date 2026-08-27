# 小游戏排位记录与验证

更新时间：2026-08-27

本文描述 `src - cloud` 中小游戏排位记录 v1。目标是在不逐步联网、不保存全部普通对局的前提下，对可能刷新个人纪录或榜单的成绩进行可复算验证。

## 运行流程

1. 已登录用户开始一局新游戏时调用 `POST /api/minigame-rankings/runs`。
2. 服务端创建一次性 `run_id`，绑定 `user_id`、小游戏、难度、开始时间和开始 IP。
3. 服务端生成 128 位随机 salt，并用服务端密钥执行 HMAC，派生 128 位 xoshiro 种子。
4. 浏览器使用该种子运行纯前端小游戏，同时只记录语义操作，不记录每帧或完整棋盘。
5. 游戏自然结束后，浏览器先提交分数、奖杯、终盘和动作摘要到 `qualify`。
6. 服务端只在成绩可能刷新已验证个人 PB、奖杯纪录或 TOP100 时签发一次性提交 token。
7. 浏览器编码并上传 MGO1 操作流。服务端先做廉价结构校验，再进入单 worker 重放队列。
8. Node 重放器使用同一 JS 规则和 RNG 从头复算。结果一致时更新 PB；否则拒绝。

匿名用户、旧 localStorage 对局、run 创建失败的对局仍可正常游玩，但不参与排名。

## MGO1 v1

前缀：`MINIGAME_v1MGO_B64_`

二进制采用小端和紧凑变长整数：

```text
magic             4 bytes   "MGO1"
format_version    1 byte    1
rules_version     ULEB128
game_code         1 byte
difficulty        1 byte    0/1
flags             1 byte    v1 必须为 0
run_id            16 bytes  UUID bytes
seed              16 bytes
actions           variable
crc32             4 bytes   little-endian
```

操作码：

```text
0x00..0x03  move up/right/down/left
0x10        bomb target
0x11        glove source + target
0x12        twist target
0x20        custom action + phase
0x30        timer tick
0x70        FNV-1a 64-bit state digest
0x7f        terminal reason
```

每条操作在 opcode 后保存相对上一条操作的 `delta_ms` ULEB128。每 128 个可变操作插入一次完整状态摘要，用于尽早发现流被修改。`END` 必须是最后一条记录。

限制：解码后不超过 256 KB，最多 50,000 条记录。当前稳定 game code 顺序由前后端 catalog 共同测试锁定。

关键实现：

- `frontend/src/features/minigames/protocol/mgo1.js`
- `frontend/src/features/minigames/engine/rankedRecorder.js`
- `frontend/src/features/minigames/engine/rankedReplay.js`
- `frontend/scripts/minigameVerifier.mjs`
- `backend/minigame_rankings/mgo1.py`

## 确定性规则

- 所有小游戏随机行为统一使用 `Xoshiro128StarStar`，不直接依赖 `Math.random()`。
- snapshot 保存 RNG state，刷新后可继续同一操作流。
- Blitzkrieg 保存剩余时间和计时状态；有效时长取操作流 delta 总和。
- Tricky Tiles 客户端和验证器都使用 WASM EvilGen。客户端 WASM 不可用时该局转为不参与排名，不静默提交随机降级结果。
- `rules_version` 发生行为不兼容变更时必须递增；旧规则验证器需要在升级期保留或明确下线。

## API

```text
POST /api/minigame-rankings/runs
POST /api/minigame-rankings/runs/{run_id}/qualify
POST /api/minigame-rankings/runs/{run_id}/submit
GET  /api/minigame-rankings/runs/{run_id}
```

旧的 `POST /api/minigame-rankings/scores` 返回 410，不再接受浏览器直接声明的成绩。

run token 和 submission token 都由 HMAC 签名。submission token 绑定终局摘要、有效期 10 分钟且只能消费一次。重复网络请求只有在 record hash 相同时才按幂等处理。

## 数据库存储

`minigame_ranked_runs` 保存短期 run 状态、salt、seed、声明摘要和待验证记录。单用户最多一个 pending/validating，全局最多 32 个。

`minigame_high_scores` 仍以 `(user_id, game_id, difficulty)` 为主键，只保留每用户每榜一行。只有刷新分数 PB 时保存已验证紧凑记录；低分候选验证后立即清除上传内容。

旧成绩迁移为 `verification_level = legacy`。它们可暂时展示，但不阻止用户用一份较低的真实成绩建立首个已验证 PB，也不参与候选 TOP100 阈值计算。

完成、拒绝、过期和非候选 run 元数据保留 30 天，每小时清理一次。PB 记录随个人最高分原子替换，不累积历史普通对局。

## 成本与故障边界

- `qualify` 只做 SQLite 查询和整数比较，不上传记录。
- `submit` 先在 Python 校验 Base64、大小、CRC、header、run/seed、动作数、时长和 END。
- 只有通过预筛的候选进入单个常驻 Node verifier，避免并发占满 CPU。
- Node 不可用或协议响应损坏时任务退回队列，并有 30 秒进程级退避；不把基础设施故障记为作弊。
- 明确的规则重放不一致才标记 `rejected`。
- verifier stderr 丢弃，不写逐步日志，避免磁盘增长。

环境变量：

```text
MINIGAME_RANKING_SECRET
MINIGAME_NODE_BINARY
MINIGAME_VERIFY_TIMEOUT_SECONDS
MINIGAME_VERIFY_RETRY_DELAY_SECONDS
```

未配置 `MINIGAME_RANKING_SECRET` 时，服务端在认证数据库旁生成权限为 0600 的持久 secret 文件。

## 回归命令

```text
cd frontend
npm run test:minigame-ranked
npm run build

python -m unittest tests.test_minigame_rankings -v
python -m compileall -q backend cloud_server.py scripts/auth_admin.py
```
