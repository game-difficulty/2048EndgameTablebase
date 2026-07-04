# 下一步工作计划

本计划可作为下一轮实现任务清单。默认工作目录为 `C:/Apps/2048endgameTablebase/src - cloud`。

## 1. 建立 cloud runtime 和服务器 tablebase catalog

- 新增云端启动方式，直接运行 FastAPI/uvicorn，不启动 `pywebview`。
- 新增只读 tablebase catalog/manifest，列出服务器已有 pattern、target、spawn rate、dtype 和后端真实路径。
- 替换当前依赖 `docs_and_configs/config` 中 Windows 路径的逻辑，前端只接收可用定式列表，不接收路径。
- 验证 Cloudflare 反代 WebSocket：`wss://2048tables.online/ws/{client_id}`。

## 2. 移除不适合云端开放的功能

- game：目录页 Play Game 直接跳转 `https://2048-endgame-tablebase.netlify.app/`，移除本地 Gamer tab 和后端 game action 分发。
- notebook：移除入口、页面和后端 action 分发；tester 中不再写错题本。
- settings builder：移除 builder tab；后端拒绝或删除 `START_BUILD`。
- trainer record/path：移除录制、加载 rec、回放、选择本地定式路径；pattern/target 选择后自动加载服务器表库。

## 3. 增加浏览器上传和下载接口

- tester 保存日志：浏览器直接下载 `.txt`。
- tester 保存回放：浏览器下载 `.rpl`。
- replay 加载文件：浏览器选择 `.rpl` 后上传到后端临时目录，再在当前 session 中加载。
- analysis：浏览器选择 `.txt`、`.vrs` 或 replay 文件上传；分析输出写入 job 目录，完成后打包为 zip 下载。
- 后端增加文件大小、文件数量、扩展名、临时目录清理和并发限制。

## 4. localStorage 迁移

- `useAppSettingsStore` 改为 localStorage 存储 UI 设置：语言、主题、深色模式、动画、字体缩放、UI 缩放、demo speed、回放滑条阈值等。
- 固定或隐藏影响服务器表库匹配的设置，例如 `4_spawn_rate`。
- 小游戏进度、难度和 powerups 改为浏览器本地 localStorage；后端只做规则计算和状态序列化。
- 帮助页语言由前端本地设置决定，请求帮助内容时携带 language。

## 5. 部署和验证

- 在 Ubuntu 22.04.5 LTS 上安装 Python、Node、native build 依赖和运行依赖。
- 构建前端 `frontend/dist`，构建或部署 `native_core` 运行所需模块。
- 用 systemd 管理 uvicorn 服务，用 Nginx 或 Caddy 反代域名。
- 验证场景：
  - 目录页和 help 正常。
  - trainer/tester 只能选择服务器已有定式，且不显示服务器路径。
  - tester log/replay 可下载。
  - replay 可上传 `.rpl` 并播放。
  - analysis 可上传、分析、下载结果。
  - settings 刷新页面后保持浏览器本地设置。
  - minigames 不同浏览器互不影响进度。

## 架构预留

- 预留 `UserContext`：匿名用户先用浏览器生成 client id，后续可接入登录用户 id。
- 预留 `RateLimiter`：按 IP、匿名 id、用户 id 限制 trainer/tester/analysis 调用频率。
- 预留 `EntitlementService`：未来按积分/付费判断某个用户是否可使用指定 pattern/target。
- 预留 job/audit 记录：分析任务、下载任务和高成本查询后续可落库统计。
