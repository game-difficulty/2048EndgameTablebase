# 当前云化状态

更新时间：2026-07-04

本文档基于 `C:/Apps/2048endgameTablebase/src - cloud` 副本，记录最近一轮已完成事项和下一步落点。

## 本轮新增

- 新增云端冒烟脚本：`scripts/cloud_smoke_test.py`
  - 检查 `/api/tablebases` 不泄露 `root`、`relative_path`、绝对路径等字段。
  - 检查 replay 上传允许 `.rpl`，拒绝错误扩展名。
  - 检查 download registry 只允许临时下载目录内文件，并清洗下载文件名。
  - 检查 WebSocket 云 denylist 会拒绝 `START_BUILD`、`REPLAY_LOAD_FILE`、`TESTER_SAVE_LOG`、`NOTEBOOK_GET_INIT`。
- 新增部署模板目录：`docs_and_configs/deploy/`
  - `cloud.env.example`
  - `2048tables-cloud.service`
  - `nginx_2048tables.online.conf`
- 新增 Ubuntu 部署文档：`docs_and_configs/cloud_migration_plan/ubuntu_deployment.md`

## 已验证

本地云服务地址：`http://127.0.0.1:8000/`

已执行：

```bash
python -m compileall scripts backend cloud_server.py
npm run build
python scripts/cloud_smoke_test.py
git diff --check
```

结果：

- 后端编译通过。
- 前端构建通过。
- 首页 HTTP 200。
- 冒烟脚本通过。
- `git diff --check` 无 whitespace error。
- SSH 只读探测成功：`ubuntu@43.135.117.69`，系统为 Ubuntu 22.04.5 LTS。
- 服务器 `127.0.0.1:8000` 原由 `/data/app` 的 `cb-pricing.service` 占用。用户确认该平台已迁移到 `8081` 后，已执行 `systemctl disable --now cb-pricing.service`，释放 8000。
- 2048 云服务已部署到 `/opt/2048tables/app`，`2048tables-cloud.service` 当前监听 `127.0.0.1:8000`。
- 服务器仍保留 `8081` 监听，用于可转债定价分析平台。
- Linux native 模块已低并发补构建完成：`ai_core`、`mover_core`、`formation_core`、`bookgen_native.so` 均已生成。
- 公网浏览器访问已配置完成：
  - `https://2048tables.online/` 返回 `2048 Endgame Tablebase` 页面。
  - `https://2048tables.online/api/tablebases` 返回 `L3_128`、`L3_256`、`L3_512`。
  - `wss://2048tables.online/ws/...` 可连接。
- 云模式 `GET_SETTINGS` 已清理服务器私有字段，不再返回 `filepath_map` 或旧 Windows 本地路径。
- 重要约束：后续 `native_core/make.sh` 或任何 `*_core` 原生模块构建必须跳出沙箱/使用非沙箱执行环境。这里的“申请权限”指工具执行权限，不是向用户二次确认；无需再向用户询问。

服务器 `/api/tablebases` 已返回三项真实表库：

- `L3_128`
- `L3_256`
- `L3_512`

这些表库来自本地 Windows 目录 `C:/2048_tables/cloudtables/L3-128`、`L3-256`、`L3-512`，已作为二进制文件原样上传到服务器 `/opt/2048tables/tablebases/`，并按 manifest 目录名改为下划线格式。服务器当前表库目录约 `1.1G`，共 `530` 个文件。

## 下一步建议

1. 在服务器上创建目录和 venv，但先不切域名流量。
2. 服务器当前状态检查：

   ```bash
   ss -ltnp | grep -E ':(8000|8081)\b'
   ls -lh /opt/2048tables/app/native_core/*.so
   systemctl status 2048tables-cloud
   ```

3. 服务验证命令：

   ```bash
   cd /opt/2048tables/app
   /opt/2048tables/venv/bin/python scripts/cloud_smoke_test.py --base-url http://127.0.0.1:8000 --ws-url ws://127.0.0.1:8000
   curl -s http://127.0.0.1:8000/api/tablebases
   ```

4. 下一步做 Trainer/Tester/Replay/Analysis 端到端测试，优先使用 `L3_128` 小表验证流程。
5. 继续检查页面级真实操作，尤其 Trainer/Tester 读取 Linux tablebase 的兼容性。

## 暂不处理

- 注册、登录、限流、积分/付费。
- 多进程或多实例部署。
- 上传/analysis job 的数据库持久化。
- Cloudflare 大文件上传限制规避。
