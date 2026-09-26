# 分析结果直达回放与历史分析设计

## 目标

主站“测试－分析”和对局站“本人历史对局－分析”共用同一套分析任务。分析完成后，用户应当可以：

1. 不下载 ZIP、不解压、不重新上传，直接打开分析生成的 `.rpl`；
2. 在账号级“历史分析”中重新找到过去的分析；
3. 当一个输入产生多个残局阶段、一个任务包含多个输入或多个定式时，先看到明确的阶段列表，再选择具体回放；
4. 继续下载现有完整 ZIP，但 ZIP 下载与在线回放互不依赖。

历史分析和阶段回放只对任务所属账号可见。打开自己已付费生成的分析回放不再次扣除“加载回放”Token。

## 现状与不能直接复用 ZIP 的原因

- `analysis_jobs`、任务目录、报告和 ZIP 是运行态数据，任务完成后保留 1 小时，清理时运行态数据库行和目录一起删除。
- 对局站已经把分析摘要写入 `human_analysis_summaries`，摘要可以长期读取；其中只有阶段文件名，没有可长期访问的 `.rpl`。
- 主站上传分析没有长期摘要，只在短期任务 manifest 中保存任务状态。
- 一个工作项可以多次进入、离开同一定式，因此会产生多个 `.rpl`；一个任务还可以包含多个输入文件，或在对局站包含同一局的多个定式目标。
- ZIP 只是下载容器。若用一个“查看回放”按钮指向 ZIP，网站无法知道用户要看哪个阶段；在线按 ZIP member 读取还会增加路径校验、重复解压和并发访问复杂度。
- 当前多个同名上传文件使用同一定式分析时，产物有文件名碰撞的可能。新实现不能把文件名当作身份。

因此应把 ZIP 内的 `.rpl` 提升为有独立 ID、权限和有效期的“分析阶段回放”，ZIP 仍是附加下载产物。

## 用户界面

### 分析完成后的入口

每个成功工作项显示：

- 输入来源：对局日期、分数，或上传文件名；
- 定式和目标；
- 找到的阶段数量；
- `查看回放`、`生成展示图`（符合条件时）和 `下载全部结果`。

阶段数为 1 时，`查看回放`直接打开该阶段。阶段数大于 1 时，按钮显示为`查看回放（N）`，打开阶段选择浮层。每个阶段至少显示：

- 第几段；
- 原始回放中的步数范围，例如“第 7,342–7,448 步”；
- 被评价的决策数；
- 最终吻合度和最大连击；
- `打开回放`按钮。

没有匹配阶段时明确显示“未找到可分析的残局阶段”，不能把“分析任务成功”误写成“存在回放”。已过期阶段保留元数据，按钮置灰并显示过期时间。

### 历史分析列表

主站在“测试－分析”弹窗增加`历史分析`入口；对局站在本人个人页增加`分析记录`入口。两处读取同一账号数据，但默认筛选不同：

- 主站默认显示全部，可筛选“上传分析／对局分析”；
- 对局站默认显示“对局分析”，可切换全部。

列表按创建时间倒序、游标分页，每页 20 条。一级按分析任务分组，显示时间、来源、状态、成功/失败工作项数量。展开后显示工作项和阶段。支持按来源、状态、变体、定式筛选；不读取输入回放、报告或任何 BLOB。

对局站条目可以返回原历史对局。主站上传条目只显示安全化后的原文件名。历史列表属于账号而非浏览器会话，在另一设备登录后也能看到。

### 回放页行为

使用地址：

`/replay?analysis_replay=<artifact_id>`

回放页识别参数后，通过受保护接口读取二进制 `.rpl`，沿用现有回放解析、棋盘、步进、关键失误和分析数据展示，不创建临时上传记录，也不走本地文件授权接口。

回放页显示来源信息：定式、目标、原对局/原文件、阶段步数范围，并提供“返回历史分析”。对局站跳到主站时，由对局站先申请一个仅用于该阶段的短期签名跳转凭据；URL 不携带回放正文。

分析器当前会略过阶段开头用于稳定评价的预热步，因此“原始阶段范围”和“实际写入 `.rpl` 的范围”不能混成一个字段。界面主标题显示原始阶段范围，次要信息显示“分析回放包含 N 步”；数据层分别记录 source range 和 replay range。

## 数据模型

运行态 `analysis_jobs` 和 `analysis_queue` 保持职责不变，新增三张窄表。不要把历史功能绑在会被清理的 manifest 或输出目录上。

### `analysis_history_jobs`

- `job_id`：现有随机任务 ID，主键；
- `user_id`；
- `origin`：`main_upload` 或 `human_archive`；
- `status`：`queued/running/finished/partial/failed`；
- `total/done/failed`；
- `created_at/completed_at`；
- `source_run_id`：仅单局对局分析时填写；
- `metadata_json`：只放小型展示信息和版本号，不放文件路径或回放。

### `analysis_history_items`

- `id`：整数主键；
- `job_id/work_index`：唯一；
- `source_run_id`：可空；
- `source_filename`：安全化显示名；
- `pattern/target/variant`；
- `status/error_code`；
- `stage_count`；
- `summary_id`：对局站已有长期摘要时关联。

所有工作项始终使用独立输出子目录，例如 `items/0001/`，不再根据“定式组合是否唯一”决定目录，以消除同名输入覆盖。

### `analysis_replay_artifacts`

- `artifact_id`：不可猜测 ID，主键；
- `history_item_id/segment_index`：唯一；
- `relative_path`：只能是受控根目录下的相对路径；
- `byte_size`；
- `source_start_index/source_end_index`：原回放中的完整残局阶段；
- `replay_start_index/replay_end_index/replay_move_count`：实际写入分析回放的范围；
- `evaluated_moves`；
- `goodness_of_fit/max_combo/performance_counts_json`；
- `pattern/target/variant/use_variant`；
- `created_at/expires_at/deleted_at`；
- `format_version`。

不保存 SHA-256。身份和幂等性由 `job_id + work_index + segment_index` 的唯一约束保证；文件在发布前使用现有 `.rpl` 解析器校验 sentinel、记录长度和终盘局面。

## 产物登记流程

1. 创建分析任务时立即写 `analysis_history_jobs/items`，因此排队、运行和失败任务也能在历史中看到。
2. `Analyzer` 返回结构化的 `segment_summaries`，其中必须包含实际 `replay_path`，后续逻辑不再从展示文件名反推路径。
3. 每个工作项完成后，逐个校验阶段 `.rpl`。
4. 将有效文件发布到共享的 `analysis_replays/YYYY/MM/<artifact_id>.rpl`。同一文件系统优先使用硬链接，任务目录清理后数据块仍由 artifact 链接持有；不支持硬链接时再原子复制。
5. 文件先写临时名，成功后 rename，再在事务中 UPSERT artifact 和 item 状态。失败时移除临时文件。重启重跑依靠唯一键保持幂等。
6. 工作项没有阶段时写 `stage_count=0`，仍可标记分析成功。
7. 所有工作项完成后，将历史任务标记为 `finished`；有成功也有失败时标记 `partial`。
8. ZIP、TXT、输入副本继续由现有短期任务清理；artifact 不位于任务目录，不会被一并删除。

对局站现有 `human_analysis_summaries.segments` 中补充 `artifact_id`。摘要长期存在，artifact 到期后仍可生成成绩单和查看统计，只是不能再打开阶段回放。

## 接口

- `GET /api/analysis/history?cursor=&limit=20&origin=&status=&variant=&pattern=`：账号历史列表；
- `GET /api/analysis/history/{job_id}`：工作项和全部阶段；
- `GET /api/analysis/replays/{artifact_id}`：返回 `.rpl` 二进制；
- `POST /api/analysis/replays/{artifact_id}/open-link`：对局站生成短期主站跳转信息；
- `DELETE /api/analysis/history/{job_id}`：用户删除历史元数据并提前删除仍存在的阶段文件，运行中任务不可删除。

现有 `GET /api/analysis/jobs/{job_id}` 在任务完成后增加每个 item 的 `history_item_id`、`stage_count`，并在阶段只有一个时返回 `replay_artifact_id`。当前任务弹窗因此无需等待历史页刷新即可打开回放。

所有读取都校验 `user_id`。artifact 响应使用 `Cache-Control: private, no-store`，限制为每账号每分钟 30 次、最多 5 个并发读取。历史查询只查窄表。打开自己的分析产物不扣 Token。

短期跳转凭据包含 `artifact_id/user_id/purpose/exp`，有效期建议 60 秒。凭据放入 URL fragment，由主站前端读取并立即用 `history.replaceState` 移除，再提交给 artifact 接口；fragment 不会进入普通 HTTP 请求日志。该凭据只授权读取一个 artifact，不创建登录会话，也不能用于其他接口。若主站和对局站最终确认共享父域登录态稳定，可优先使用普通账号鉴权，签名凭据只作为跨域兼容路径。

## 保留与容量控制

建议首版：

- 上传原文件：任务完成后立即删除；
- 用户上传的原文件：任务完成或失败进入终态后立即删除；
- TXT 与完整 ZIP：完成后保留 1 小时；
- 独立阶段 `.rpl`：不设固定到期日，按账号数量和全站磁盘上限淘汰；
- 历史元数据和对局站分析摘要：长期保留；
- 单账号 artifact 上限：普通账号 50 个，赞助账号 100 个；
- 全站 artifact 上限：4 GiB，并保留至少 4 GiB 磁盘安全余量；达到限制时按到期时间和创建时间清理最旧 artifact，不删除运行中任务产物。

主站与对局站两个 API 进程必须把 `CLOUD_ANALYSIS_REPLAY_ROOT` 指向同一持久目录；推荐 `/var/lib/2048tables/analysis-replays`。签名密钥沿用两个进程共有的服务密钥。

实测 89 步的阶段 `.rpl` 约 2.5 KiB，而对应 TXT 约 27 KiB。只长期一些保留 `.rpl` 比延长整个 ZIP/报告目录更节省空间。`.rpl` 保持原始格式落盘，便于直接传输和解析；HTTP 层可以使用源站压缩，但首版不增加新的磁盘压缩格式。

清理任务每小时运行，并在新任务创建前做一次轻量容量检查。数据库先标记 `deleted_at`，再删除文件；文件缺失时接口返回明确的 `ANALYSIS_REPLAY_EXPIRED`，不会把它当作权限错误。

## 与两个入口的兼容

### 对局站本人历史分析

- 继续按归档对局 ID 在服务器内部读取源回放，不增加下载再上传；
- 继续支持一次选择最多 6 个定式；每个定式是一个 history item，各自可含多个阶段；
- 保持 `human_analysis_summaries`、成绩单、实力榜口径不变；
- 隐藏对局仍不能创建或读取分析，沿用现有 `visible=1` 和归属校验；
- artifact 过期后允许重新分析，首版不承诺免 Token 恢复。

### 主站测试－分析

- 保持一次选择多个输入文件、一个定式目标的行为；
- 每个输入是一个 history item；即使文件同名也不会覆盖；
- 上传源文件在任务结束后删除，因此 artifact 到期后不能恢复，只能由用户重新上传原回放分析；
- 当前 ZIP 下载保留，用于离线保存全部 TXT 和 `.rpl`。

## 实施顺序

1. 先实现数据表、独立 item 输出目录、artifact 登记和清理，补充重启幂等、同名输入、多阶段、无阶段、过期和越权测试。
2. 增加历史与 artifact 接口，在主站回放页实现同源直接加载。
3. 改主站分析弹窗，加入阶段选择和历史分析列表。
4. 接入对局站分析弹窗与本人个人页，并实现跨站短期跳转。
5. 最后验证 1 小时临时产物清理、50/100 个账号配额，以及 4 GiB 与 4 GiB 安全余量两条全站限制。

首版不把回放正文或 ZIP 存入 SQLite，不从历史列表读取 BLOB，不公开分享分析阶段，也不自动永久保留用户上传的源文件。
