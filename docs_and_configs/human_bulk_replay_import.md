# 对局站批量补齐历史回放

工具：`tools/human_play_replay_import.py`

它只为已经存在于 `human_runs` 的历史成绩补齐回放，不新增成绩记录，也不改变成绩来源、结束时间、排行榜资格或近 168 小时榜资格。默认只做只读预检；只有显式传入 `--apply` 才写库。

## 支持范围

- Verse 旧版 `replay_` 单字符格式；
- Verse 的 `4x4-1_...` 等格式和 RPL1 格式；
- 2048next 的 `REPLAY_v1RPL_B64_...` 格式；
- 文件名提供玩家、变体、日期、分数、来源标签，例如：
  `XiaoLongBao_4x4_32k_#151_2026_05_03_733724_2048next.txt`。

2048next 的解析只存在于此后台工具中，没有加入对局站的公开“补录申请”上传接口。工具拒绝非 `pow2` 规则、AI 标记和中途难度变化，并把有效动作转换为站内统一 RPL1 后再保存。

`_pku` 文件默认记为 `skipped_pku`，不会解析或写入。

## 匹配规则

脚本按以下条件寻找唯一的既有记录：

1. 指定本站用户；
2. 相同变体和分数；
3. 文件名日期在所选时区内；
4. 回放重算的终盘与数据库终盘完全一致；
5. 记录已封存、尚无回放，且是支持统一 RPL1 归档的外部/人工记录。

零条候选记为 `no_match`，多条记为 `ambiguous`，已有不同回放记为 `archive_conflict`；这些状态不会写入。可用 overrides JSON 处理少量特殊情况：键是相对文件名，值是明确的 `run_id`，值为 `null` 表示主动跳过。

```json
{
  "bundle/special-file.txt": "exact-run-id",
  "bundle/already-handled.txt": null
}
```

## 使用

先设置线上数据库路径，执行只读预检并保存报告：

```powershell
$env:CLOUD_AUTH_DB = "C:/path/auth.sqlite3"
$env:HUMAN_PLAY_DB = "C:/path/human.sqlite3"
python tools/human_play_replay_import.py `
  --input "C:/Users/Administrator/Downloads/XLB 32ks - replay only - site tagged" `
  --username XLB `
  --report "C:/temp/xlb-replay-plan.json" `
  --require-all
```

检查报告后，使用同样参数写入：

```powershell
python tools/human_play_replay_import.py `
  --input "C:/Users/Administrator/Downloads/XLB 32ks - replay only - site tagged" `
  --username XLB `
  --report "C:/temp/xlb-replay-applied.json" `
  --require-all `
  --apply
```

Linux 上使用相同参数和服务器绝对路径即可。日期默认按 `Asia/Shanghai`（UTC+8）解释，可用 `--timezone` 修改。

## 写入与审计

实际写入在一个 SQLite 事务内完成。提交前再次核对用户、状态、来源、变体、分数和终盘；任何目标在预检后发生变化都会使整个批次回滚。成功后：

- 只更新原记录的压缩回放、步数、用时、节点和 `has_replay`；
- 更新滚动榜缓存中的回放入口；
- 以完整回放重算依赖回放的 32K 综率事实；
- 每个玩家/变体只在批次末尾重建一次统计与 Rating；
- 在 `human_bulk_replay_import_batches` 和 `human_bulk_replay_import_items` 留下批次与逐文件审计记录。

重复执行是幂等的：相同归档显示为 `already_attached`，不会重复写入或新增对局。

若核对线上库后确认历史成绩尚未导入，可显式增加
`--create-missing --operator-id <站长用户ID>`。这会为每份无法匹配的回放创建一条已经审批的补录申请及归档局，进入总榜、PB、B10 和个人统计，但沿用补录局规则，不进入近 168 小时榜和每周 Token 结算。文件只有日期而没有时刻时，以该时区当天 12:00 作为结束时间；2048next 自带可信时间戳时使用回放时间戳。原始来源标签仍写入归档状态、审批说明和批次审计中。
