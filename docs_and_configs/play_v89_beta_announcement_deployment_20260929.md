# Play v89 长局响应与内测公告发布记录

发布时间：2026-09-29。

## 发布内容

- Play 从 `/opt/2048tables/play/releases/20260928-play-v88-daily-activity`
  切换至 `/opt/2048tables/play/releases/20260929-play-v89-latency-beta`。
- 发布源码为 `f72fbff34434c0d99ee20596fc76cd0eb9376409`。长局输入延迟修复提交为
  `5b09409`，内测公告为 `38b48a1`，站长页陈旧测试修正为 `f72fbff`。
- 每步动画的同步布局读取由整个 `document.body` 收窄到当前棋盘，并使用布局隔离；
  真人直播增量发送只复制尚未发送的动作尾部，不再随整局历史长度反复复制全部动作。
- Play 同时补齐此前已经配置 Nginx、但后端尚未上线的主站分析任务与历史入口，
  使主站“测试－分析”与 Play 使用同一份耐久分析历史。
- 主站静态入口发布“对局站开放内测”公告，沿用上一份公告的首页提示、历史公告和站点直达入口，
  正文介绍四种棋盘、完整回放档案、个人主页与分享图、直接回放分析、Verse 历史继承、练习与直播工具，
  并保留本地存档及高分联网提醒。

## 验证

- 干净工作树前端全量测试：557 项通过。
- Play 后端与主站分析上传专项：45 项通过。
- 前端生产构建通过；Play 发布清单共校验 519 个文件。
- 公网 Play 健康接口返回 `{"ok":true}`；Play、主站 Cloud 与 Nginx 均为 active，
  `sudo nginx -t` 通过，Play 启动后日志没有 ERROR 或 Traceback。
- 主站分析历史与任务接口未登录访问均返回预期 401，确认请求已代理到 Play。
- Chromium 公网验收：公告摘要、正文、功能清单与 Play 链接正常；Play H/K 输入后本地保存步数增加，
  两个页面均无控制台错误或警告。
- 主站普通入口与 Play 入口均返回 gzip，`release.json` 与 Play manifest 均指向 `f72fbff`。

## 发布包与备份

- Play 包 SHA-256：`B89CF3F1070D6DE76C91416C47975DA9B42702ECCE195EECF800C3D3099C0A84`。
- 主站静态包 SHA-256：`41C744D15D7151FE2E357E8CF374401441232FC494647D166ED34340D9BABDB3`。
- Play 数据库备份：`/var/lib/2048tables/backups/human-before-play-v89-20260929.sqlite3`，
  `PRAGMA quick_check` 为 `ok`。
- 主站旧入口备份：`/var/lib/2048tables/backups/main-before-play-beta-20260929`。

## 回滚

Play 可将 `/opt/2048tables/play/current` 原子切回
`/opt/2048tables/play/releases/20260928-play-v88-daily-activity` 并重启
`2048tables-play.service`。主站公告可从 `main-before-play-beta-20260929` 恢复
`index.html` 与 `release.json`；旧发布原本没有 `index.html.gz`，回滚时应同时移除本次新增的该文件。
内容寻址资源可保留，供仍打开的页面继续使用。
