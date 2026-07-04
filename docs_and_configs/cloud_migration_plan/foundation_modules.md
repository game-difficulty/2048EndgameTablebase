# 云化基础模块索引

本文基于 `src - cloud` 副本，记录第一批云化基础模块。后续逐页改造前优先阅读本文，避免重复搜索路径。

## 前端模块

### `frontend/src/services/runtime/capabilities.js`

职责：
- 统一判断运行环境：desktop/browser。
- 统一探测 `pywebview`、桌面文件对话框、浏览器文件选择、浏览器下载、localStorage。

后续接入点：
- `frontend/src/services/runtime/desktopDialogs.js` 已复用 `getPywebviewApi()`。
- `frontend/src/components/MainMenuView.vue` 后续可用 `canOpenExternalUrl()` 替代直接访问 `window.pywebview`。
- trainer/tester/replay/analysis 改造时，用该模块决定走桌面 fallback 还是浏览器上传下载。

### `frontend/src/services/storage/localStorageStore.js`

职责：
- 提供 namespace + version 的 localStorage 封装。
- 支持默认值、迁移函数、读写 envelope、reset/remove。

后续接入点：
- `frontend/src/app/useAppSettings.js`：界面语言、主题、动画、速度等用户设置迁移到浏览器本地。
- `frontend/src/features/minigames/**`：小游戏进度迁移到浏览器本地。
- 可缓存 `/api/tablebases` 返回的只读 catalog。

### `frontend/src/services/files/browserFiles.js`

职责：
- 浏览器选择文件：`pickBrowserFiles()`、`pickSingleBrowserFile()`、`tryPickBrowserFiles()`。
- 读取文件：text、ArrayBuffer、JSON。
- 浏览器下载：Blob、text、JSON、HTTP response。

后续接入点：
- tester 保存日志/回放：改为下载 Blob。
- replay 加载 rpl：改为选择浏览器文件并上传。
- analysis 输入文件：改为选择浏览器文件并上传，输出改为下载。
- trainer record 功能当前计划移除，不再接入。

### `frontend/src/services/tablebases/catalogClient.js`

职责：
- 调用 `GET /api/tablebases`。
- 将后端 snake_case payload 归一成前端 camelCase。
- 提供按 pattern 分组和 target 列表工具。

后续接入点：
- trainer/tester 的定式选择 UI。
- settings 中只展示可用云端表库，不展示 builder。
- analysis 启动前校验所需表库是否存在。

## 后端模块

### `backend/cloud_safety.py`

职责：
- 统一云模式判断：`CLOUD_MODE=1` 且 `APP_MODE != desktop`。
- 统一 WebSocket action denylist。
- 提供禁用提示文案。

当前接入：
- `backend/app.py` 在 handler 分发前调用 `is_cloud_action_blocked()`。

后续接入点：
- 注册/登录、限流、积分/付费可接在同一分发前置层。
- tester/replay/analysis 的本地文件 action 已先被 deny，等浏览器上传下载实现后再按新 action/token 放行。

### `backend/cloud_files.py`

职责：
- 云端临时上传目录：`CLOUD_UPLOAD_ROOT`，默认系统 temp 下 `2048tables-cloud/uploads`。
- 上传大小限制：`CLOUD_MAX_UPLOAD_BYTES`，默认 50MB。
- TTL 清理：`CLOUD_UPLOAD_TTL_SECONDS`，默认 6 小时。
- 文件名清洗、扩展名白名单、路径越界检查。
- 文件/bytes 下载响应封装。

后续接入点：
- 新增 HTTP upload endpoint 后复用 `save_upload_file()`。
- analysis/replay 通过 upload token 解析临时文件。
- tester 保存日志/回放通过 download response 返回浏览器下载。

### `backend/tablebase_catalog.py`

职责：
- 从 `docs_and_configs/cloud_tablebases.json` 读取服务器端表库 manifest。
- 支持 `CLOUD_TABLEBASE_MANIFEST` 和 `CLOUD_TABLEBASE_ROOT` 环境变量覆盖。
- 只在后端解析真实路径，HTTP payload 不返回 root、relative_path 或绝对路径。

当前接入：
- `backend/app.py` 暴露 `GET /api/tablebases`。
- `backend/handlers/trainer.py` 的 `TRAINER_SET_FILEPATH` 已改为通过 catalog 加载。

后续接入点：
- tester 选表逻辑改为 catalog。
- settings 的 categories/target_tiles 后续进一步以 catalog 为准。
