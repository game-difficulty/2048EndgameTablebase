# 前端盘面所有权

更新时间：2026-08-28。本文仅描述 `src - cloud` 云版。

## 不变量

- Gamer、Trainer、Tester 和 Replay 的当前盘面由浏览器独占维护。
- WebSocket/HTTP 响应不得写回当前盘面、历史、撤回栈、出数模式或动画状态。
- 服务器只返回定式挂载状态、随机初始局面、查询结果、预取结果，以及 Trainer 最优/最差出数建议。
- 所有异步响应必须携带请求标识；可能改变本地状态的建议还必须携带本地 `revision` 和查询盘面。
- 查询结果仅在 `pattern + catalog_version + board_hex` 仍与当前上下文一致时展示。
- 动画由不可变 `transition` 驱动；每个 transition id 只消费一次，撤回和快照同步不播放旧动画。

## 关键代码

- 通用 reducer：`frontend/src/features/practice/engine/practiceSession.js`
- 动画消费：`frontend/src/components/BaseBoard.vue`
- Trainer：`frontend/src/features/trainer/composables/useTrainerSession.js`
- Tester：`frontend/src/features/tester/composables/useTesterSession.js`
- 云端 stateless 查询：`backend/handlers/tablebase_query.py`
- 云端旧 action 拒绝：`backend/cloud_safety.py`

## 协议边界

- `TABLEBASE_QUERY`：传入前端当前盘面，返回该盘面结果和预取；不修改 session 盘面。
- `TRAINER_SPAWN_QUERY`：传入移动后盘面、模式和 revision，只返回一个出数建议。
- `TRAINER_TABLEBASE_READY` / `TESTER_TABLEBASE_READY`：只确认定式挂载。
- `TRAINER_DEFAULT` / `TESTER_BOARD_SEED`：显式请求时返回一个新局面种子；请求和响应都携带
  `client_revision`，仅当本地 revision 未变化时由前端创建新本地会话。

云模式禁用旧的移动、摆盘、撤回、旋转和逐步回放 action，避免旧客户端重新引入双重状态源。

对仍需保留的挂载、初始局面和查询 action，云端分发层强制要求
`client_local_board: true`。缺少该标志的旧客户端请求会在进入 handler 前被拒绝，
因此桌面兼容分支中的 `GameSession.board_encoded/history/spawn_mode/played_length`
在云服务中不可达。云端断线也不会持久化这些兼容字段。

云前端不消费遗留 `UPDATE_STATE` / `TESTER_STATE` 棋盘消息；迟到的随机局面响应、
查询结果和出数建议均只能通过 request id、revision、pattern 和 board 校验后生效。
