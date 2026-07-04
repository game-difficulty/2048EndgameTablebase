# 架构评估

本评估基于 `C:/Apps/2048endgameTablebase/src - cloud` 代码副本。

## 现有架构

- 前端是 Vue 3 + Vite，功能按 `frontend/src/features/*` 拆分，顶层 tab 管理在 `frontend/src/App.vue` 和 `frontend/src/app/tabRegistry.js`。
- 后端是 FastAPI + WebSocket，WebSocket 分发集中在 `backend/app.py`，各功能 handler 位于 `backend/handlers/*`。
- 桌面壳入口是 `backend_server.py`，负责启动后端、创建 pywebview 窗口、处理桌面文件对话框。云部署可以绕开桌面壳，直接运行 `backend.app:app` 或保留 `run_backend_server` 作为 uvicorn 入口。
- 核心算法和表库读取在 `engine_core/` 和 `native_core/`，训练、测试、分析、回放都依赖 `BookReaderDispatcher` 和 `SingletonConfig().config["filepath_map"]` 找表库。

## 云化可行点

- 前端已有浏览器运行基础，`frontend/src/services/runtime/backendUrl.js` 会从当前 origin 推导 HTTP/WebSocket 地址，适合同源部署到 `https://2048tables.online`。
- 功能后端已按 handler 分离，删除或屏蔽某页功能时可从 `backend/app.py` 的 action 分发层和对应 handler 两侧控制。
- trainer/tester/replay 的核心逻辑主要在会话对象 `GameSession` 中，不强依赖桌面窗口。
- 帮助页和静态资源已通过 FastAPI StaticFiles 暴露，适合云端直接服务。

## 核心风险

- `SingletonConfig` 是全局进程级配置，当前同时存服务器表库路径、用户 UI 设置、小游戏进度、错题本阈值、构建参数等。云版必须避免普通用户写入全局配置。
- 现有很多操作接收或返回服务器本地路径，例如 trainer 定式路径、replay 文件路径、analysis 输入路径和输出目录。浏览器不能也不应该知道这些路径。
- 桌面文件对话框通过 `backend/webview_api.py` 和 `frontend/src/services/runtime/desktopDialogs.js` 触发。云版应改为 `<input type="file">`、`FormData` 上传和 HTTP 下载。
- 小游戏和错题本当前写全局 pickle/config 文件。多人云服务下会互相污染数据。
- settings 页的 `START_BUILD` 会使用服务器资源构建定式，云版必须移除入口并在后端禁用 action。

## 目标架构方向

- 新增服务器表库 catalog：后端启动时读取只读 manifest，前端只看到可用 pattern/target，不看到真实路径。
- 新增浏览器本地偏好层：`useAppSettingsStore` 从 `localStorage` 读写 UI 设置，并只从后端读取只读 metadata，例如 categories、theme_map、target_tiles、available_tablebases。
- 新增文件服务层：上传 replay/analysis 输入到服务端临时目录，处理完成后返回 download token 或直接返回文件流；定期清理临时文件。
- 后续登录/限流/积分功能预留 `UserContext`、`RateLimiter`、`EntitlementService` 之类边界，但 v1 先使用匿名会话。
