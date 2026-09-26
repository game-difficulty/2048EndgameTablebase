# Play 补录申请设计

状态：方案已确定，尚未实施。

## 1. 用户流程

在本人设置页“游戏与记录”增加“补录申请”区块。按钮打开独立浮窗，填写：

- 变体：4×4、3×4、3×3、2×4；
- 对局结束时间：按用户当前时区输入，提交时转为 UTC；
- 最终得分：非负整数；
- 回放文件：首版接受本站导出的 `.vrs`／回放代码、Verse 旧文本回放和本站下载的 `.hpr`。所有格式在服务器统一规范化为 RPL1 后保存。

文件选择后前端可以显示文件名和大小，但前端解析仅用于改善体验，不能作为验证结果。提交成功后浮窗显示变体、时间、得分、步数、状态和站长备注。状态包括 `pending`、`approved`、`rejected`、`revoked`。被拒绝后允许修正资料并重新申请；每次申请及决定都是独立记录。

## 2. 服务端验证

上传沿用大文件并发槽和频率限制，采用原始二进制请求体，避免 multipart 依赖和额外拷贝。建议首版限制：单文件 2 MiB、20 秒上传超时、每用户最多 3 个待审申请、24 小时最多 10 次申请。读取采用流式上限，不能先无界读入内存。

解析器从现有 `verse_replay` 拆出纯函数，返回规范化 RPL1、变体、初始盘面、终盘、计算所得分、步数、时间统计、是否死亡和 32K 阶段事实。服务器逐步执行每次移动，检查移动有效、出数位置为空、出数只能为 2／4、结束标记和 CRC 正确。HPR 先限量解压并解析，然后同样规范化为 RPL1。

只有同时满足以下条件才创建待审申请：

- 回放完整可解析且未超过步数、文件和解压上限；
- 回放变体与表单一致；
- 服务器重算得分与填写得分一致；
- 时间合法，不晚于服务器当前时间的允许误差；
- 未命中本站已有对局或另一个申请的相同规范化回放。

失败直接返回具体的中英文错误，不创建待审申请。重复检测不需要为归档增加 SHA-256：使用 RPL1 自带 CRC、规范化长度、变体、得分和步数先筛出极少量候选，再比较完整规范化字节。

合法回放仍不等于本站原生过程证明。RPL1 中的出数由文件携带，HPR 中的种子也不一定由本站签发，因此申请必须进入人工审批，来源永久标记为 `manual`。

## 3. 数据结构

新增窄表 `human_archive_applications`：

```text
id, user_id, variant, claimed_ended_at, claimed_score,
status, replay_crc, replay_size, moves, final_board_json,
is_game_over, timing_summary_json, warning_flags_json,
original_filename, requested_at, updated_at,
approved_by, approved_at, review_note, run_id
```

回放 BLOB 放入一对一的 `human_archive_application_payloads(application_id, archive)`，避免用户列表和站长待审列表误读大字段。审批操作另写只增不改的 `human_archive_application_audit`，记录申请、批准、拒绝、撤销、操作者、时间和备注。

待审和被拒绝的载荷与审批记录分开管理：审批元数据及审计永久保留；批准后载荷转入 `human_runs.archive` 并删除待审副本；被拒绝载荷建议保留 30 天后删除，审批记录仍保留。

批准后在现有 `human_runs` 创建一条记录：

- `source='manual'`、`status='sealed'`、`reason='imported'`；
- `archive` 保存 gzip 压缩的规范化 RPL1，`has_replay=1`；
- `state` 保存服务器重建出的终盘、得分、步数、用时和出 4 统计；
- `ended` 使用申请时间，`visible=1`，并通过申请表的 `run_id` 关联审批证据；
- 不伪造本站种子、防回档前缀或周期上传记录。

资格 SQL 仅承认存在 `approved` 申请关联的 `source='manual'` 对局。不能只依赖 `visible=1` 或前端隐藏。

## 4. 审批与派生数据事务

站长页增加独立“补录申请”面板，并纳入现有待审批用户标记。列表只读取摘要，展示玩家、变体、时间、分数、步数、终盘棋盘、最大块、是否死亡、回放格式和风险提示；提供只读回放预览。

批准或拒绝必须写备注。批准前在写事务之外再次解析保存的规范化回放；进入短写事务后重新核对申请仍为 `pending`、载荷和摘要没有变化、没有重复归档，然后：

1. 创建 `human_runs` 归档；
2. 更新申请为 `approved` 并绑定 `run_id`；
3. 写审批审计；
4. 写单局统计事实；
5. 刷新玩家 PB、B10 Rating、终盘成就、32K 综率、总榜和数量榜；
6. 删除待审载荷副本。

任何一步失败则整笔事务回滚，不能出现“审批成功但未上榜”或“已上榜但无审批记录”。撤销批准时不删除对局，改为不可见／无资格，并同步撤销榜单事实、重建个人派生统计，写入审计。

用户填写的对局时间无法单靠回放证明。补录局进入总榜、PB、B10、主页和统计，但始终不进入近 168 小时榜及每周 Token 结算，避免用自填时间获取时间敏感排名或奖励。

## 5. API 与现有功能兼容

用户接口：

- `GET /api/human/me/archive-applications`：读取本人申请摘要；
- `POST /api/human/me/archive-applications?variant=...&ended_at=...&score=...`：二进制上传；
- 可选 `POST /api/human/me/archive-applications/{id}/cancel`：只允许撤回待审申请。

站长接口：

- `GET /api/admin/archive-applications`：按状态或用户筛选；
- `GET /api/admin/archive-applications/{id}/replay`：限流只读预览；
- `POST /api/admin/archive-applications/{id}/decision`：批准或拒绝，并传备注及时间是否已核验；
- `POST /api/admin/archive-applications/{id}/revoke`：撤销已批准补录。

现有回放读取、分析和下载不能再通过 `source == 'verse'` 判断编码。首版可把 `verse` 与 `manual` 都固定为 RPL1 分支；更稳妥的后续迁移是给 `human_runs` 增加 `replay_format='hpr'|'rpl1'`，让来源与编码解耦。补录批准后应能直接观看、下载并发起本人单局多定式分析。

历史记录和海报显示“补录”来源。成绩地位与其他已认可归档一致，但内部保留来源，站长可以追溯审批记录。Score Logging Threshold 不适用于主动补录申请：批准即表示用户明确要求将该局公开归档。

## 6. 必测场景

- 四种变体的合法 RPL1、旧 Verse 文本和 HPR；
- 错误变体、伪造得分、非法移动／出数、截断、CRC 错误、压缩炸弹和超大文件；
- 同一回放重复提交、跨账号重复提交、并发双重批准；
- 非死亡回放仍可待审但明确显示警告；
- 批准事务失败完整回滚；拒绝、重新申请、批准后撤销均保留审计；
- 批准后主页、PB、Best 10、Rating、数量、综率、回放和分析同步更新；
- 补录局始终不进入近 168 小时榜和每周 Token；
- 所有摘要查询不读取回放 BLOB，上传和下载受单用户频率与全局并发限制。
