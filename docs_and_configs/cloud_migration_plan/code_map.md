# 代码位置和改动方向

本文件记录后续云化实现时优先查看的代码位置。路径均相对于 `C:/Apps/2048endgameTablebase/src - cloud`。

## 顶层入口

- `frontend/src/App.vue`
  - 当前职责：顶层 tab 容器，导入所有页面组件，挂载全局 replay analysis dialog。
  - 改动方向：删除 Gamer 和 Notebook 页面挂载；保留 Trainer、Tester、Minigames、Replay、Settings、Help；保留或调整 analysis dialog。
- `frontend/src/app/tabRegistry.js`
  - 当前职责：定义 tab id、顺序、标题和 closable。
  - 改动方向：移除 Gamer/Notebook tab；Game 菜单点击改为外链跳转，不再创建本地 tab。
- `frontend/src/components/MainMenuView.vue`
  - 当前职责：目录页入口卡片。
  - 改动方向：保留目录页；Play Game 卡片点击跳转 `https://2048-endgame-tablebase.netlify.app/`；移除 Notebook 入口相关联动。
- `backend/app.py`
  - 当前职责：FastAPI app、WebSocket `/ws/{client_id}`、静态资源挂载、handler 分发。
  - 改动方向：移除或禁用 `handle_game_action`、`handle_notebook_action`；增加 HTTP 上传/下载接口；保留 trainer/tester/replay/analysis/minigames/settings/help 所需 action。

## settings/config

- `frontend/src/app/useAppSettings.js`
  - 当前职责：全局设置 store，WebSocket 拉取 `GET_SETTINGS`，通过 `UPDATE_SETTING` 写后端全局 config。
  - 改动方向：改为 localStorage 优先；仅从后端读取只读 metadata；`saveSetting` 不再向服务器写用户偏好。
- `frontend/src/features/settings/pages/SettingsPage.vue`
  - 当前职责：设置页，包含 builder/game/theme 三个 tab。
  - 改动方向：删除 builder tab；保留 game/theme 中适合本地存储的 UI 设置；固定或隐藏会影响服务器表库命中的设置，例如 `4_spawn_rate`。
- `frontend/src/features/settings/composables/useSettingsSession.js`
  - 当前职责：builder 状态、构建参数、文件夹选择、START_BUILD。
  - 改动方向：删除 builder 相关状态和 action；保留设置页本地偏好管理。
- `backend/handlers/settings.py`
  - 当前职责：`GET_SETTINGS`、`UPDATE_SETTING`、`START_BUILD`、help 加载。
  - 改动方向：`GET_SETTINGS` 只返回只读 metadata；禁用 `UPDATE_SETTING` 对普通用户写全局 config；删除或拒绝 `START_BUILD`；保留 `GET_HELP`。
- `Config.py`
  - 当前职责：加载全局 pickle config，包含表库路径、UI 设置、小游戏状态等。
  - 改动方向：拆分概念。云版只把服务器表库路径和服务端固定参数留在后端；用户偏好不再写入这里。

## trainer

- `frontend/src/features/trainer/pages/TrainerPage.vue`
  - 当前职责：训练页 UI，包括 pattern/target、本地路径、load、录制/回放控件。
  - 改动方向：移除 path 按钮和 recording 区块；选择 pattern/target 后自动加载服务器表库。
- `frontend/src/features/trainer/composables/useTrainerSession.js`
  - 当前职责：训练页 WebSocket、文件夹选择、record open/save、结果刷新、移动逻辑。
  - 改动方向：删除 `selectFolder`、record open/save/step 相关逻辑；`applyTablebase` 只发送 pattern/target。
- `backend/handlers/trainer.py`
  - 当前职责：`TRAINER_SET_FILEPATH` 支持用户传入 filepath 并写入 `filepath_map`，同时处理录制/回放。
  - 改动方向：禁止用户传 filepath；从服务器 catalog 查表库；删除或拒绝 `RECORD_*`、`TRIGGER_*` 录制/路径 action。

## tester / notebook

- `frontend/src/features/tester/pages/TesterPage.vue`
  - 当前职责：测试页 UI，包括保存日志/回放、跳转 Replay、跳转 Notebook、打开 Analysis。
  - 改动方向：移除 Notebook 按钮；保存日志/回放改为浏览器下载。
- `frontend/src/features/tester/composables/useTesterSession.js`
  - 当前职责：测试页 WebSocket、保存文件对话框、pattern 选择、移动和状态同步。
  - 改动方向：删除桌面保存对话框；保存日志/回放走 HTTP 下载或 Blob 下载；pattern/target 只从服务器可用列表选择。
- `backend/handlers/tester.py`
  - 当前职责：tester action，错误时写 `mistakes_book_store`，保存日志/回放到服务器本地路径。
  - 改动方向：移除 `mistakes_book_store.add_mistake`；保存日志/回放改为返回下载内容或 download token；禁用服务器路径写入。
- `backend/notebook.py`、`backend/handlers/notebook.py`、`frontend/src/features/notebook/*`
  - 当前职责：错题本持久化和页面。
  - 改动方向：云版整体移除或不注册入口；相关 action 不再分发。

## replay / analysis

- `frontend/src/features/replay/pages/ReplayPage.vue`
  - 当前职责：回放页面，菜单支持打开 rpl 文件和加载最新 tester replay。
  - 改动方向：打开 rpl 文件改为浏览器文件选择并上传；保留加载最新 tester replay。
- `frontend/src/features/replay/composables/useReplaySession.js`
  - 当前职责：回放 WebSocket、桌面打开文件、滑条阈值设置。
  - 改动方向：删除桌面打开文件对话框；上传后通过 server token 或 session id 加载 replay；滑条阈值存 localStorage。
- `backend/handlers/replay.py`
  - 当前职责：从服务器本地 path 读取 replay。
  - 改动方向：新增从上传临时文件或 bytes 加载 replay 的接口；禁用 `REPLAY_LOAD_FILE` 的任意服务器路径。
- `frontend/src/features/replay/components/ReplayAnalysisDialog.vue`
  - 当前职责：分析弹窗，用户输入服务器本地路径，或桌面选择文件。
  - 改动方向：改为浏览器多文件选择和上传；分析完成后提供下载。
- `backend/handlers/analysis.py`、`backend/analysis.py`、`backend/analysis_core.py`
  - 当前职责：解析服务器本地输入路径，分析并把输出写到输入文件所在目录。
  - 改动方向：HTTP 上传到临时 job 目录；输出写 job 目录；完成后打包下载；限制文件大小、数量、并发和清理。

## minigames

- `frontend/src/features/minigames/composables/useMinigameSession.js`
  - 当前职责：小游戏菜单和状态 WebSocket 客户端。
  - 改动方向：从 localStorage 读取难度、每个游戏存档、powerups；收到后端 state 后写回 localStorage。
- `backend/handlers/minigames.py`
  - 当前职责：小游戏 action，难度和关闭时保存到全局 config。
  - 改动方向：不再写全局 config；start/new/back/close 接受和返回浏览器存档 snapshot。
- `backend/minigames/service.py`、`backend/minigames/session.py`、`backend/minigames/engine/base.py`、`backend/minigames/powerups.py`
  - 当前职责：从 `SingletonConfig` 读取/保存小游戏状态和 powerups。
  - 改动方向：给 engine 和 powerups 增加 snapshot import/export；菜单 summary 从前端传入的 localStorage 汇总生成，或由前端自行合并。

## game

- `frontend/src/features/gamer/*`
  - 当前职责：内置 2048 游戏和 AI。
  - 改动方向：云版删除本地页面和逻辑；目录页直接跳转 Netlify。
- `backend/handlers/game.py`
  - 当前职责：内置游戏、AI step、保存 game_state。
  - 改动方向：云版不注册此 handler，避免暴露服务器 AI 资源。

## help

- `frontend/src/features/help/*`
  - 当前职责：帮助页。
  - 改动方向：保留。
- `backend/handlers/settings.py` 中 `GET_HELP`
  - 当前职责：按 `SingletonConfig().config["language"]` 选择 help 文件。
  - 改动方向：改为按前端 localStorage 语言传参选择，或由前端请求时携带 language。
