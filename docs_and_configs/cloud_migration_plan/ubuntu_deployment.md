# Ubuntu 云部署步骤

当前文档基于 `src - cloud` 副本。目标服务器为 Ubuntu 22.04，v1 使用匿名访问、进程内 upload/job registry、本地临时目录和服务器本地 tablebase catalog。

## 目录约定

- 应用代码：`/opt/2048tables/app`
- Python venv：`/opt/2048tables/venv`
- 表库目录：`/opt/2048tables/tablebases`
- 上传临时目录：`/var/lib/2048tables/uploads`
- 环境变量文件：`/etc/2048tables/cloud.env`
- systemd：`/etc/systemd/system/2048tables-cloud.service`
- Nginx site：`/etc/nginx/sites-available/2048tables.online`

## 首次部署

1. 安装系统依赖：

   ```bash
   sudo apt update
   sudo apt install -y python3.10-venv python3-pip nginx rsync \
     build-essential cmake ninja-build meson p7zip-full liblzma-dev
   ```

2. 创建目录：

   ```bash
   sudo mkdir -p /opt/2048tables/app /opt/2048tables/tablebases /var/lib/2048tables/uploads /etc/2048tables
   sudo chown -R ubuntu:ubuntu /opt/2048tables /var/lib/2048tables
   ```

3. 同步 `src - cloud` 代码到 `/opt/2048tables/app`，排除本地构建缓存：

   ```bash
   rsync -av --delete \
     --exclude .git \
     --exclude frontend/node_modules \
     --exclude frontend/dist \
     --exclude __pycache__ \
     ./ /opt/2048tables/app/
   ```

4. 安装 Python 依赖：

   ```bash
   cd /opt/2048tables/app
   python3 -m venv /opt/2048tables/venv
   /opt/2048tables/venv/bin/python -m pip install --upgrade pip
   /opt/2048tables/venv/bin/python -m pip install -r requirements.txt
   ```

5. 放置表库文件到 `/opt/2048tables/tablebases`。首批 manifest 位于 `docs_and_configs/cloud_tablebases.json`，例如 `L3_128`、`L3_256`、`free9_128`。

6. 构建 Linux native 模块。不要把 Windows `native_core/build-*` 缓存目录带到服务器；如果已经带过去，先删除。

   注意：`native_core/make.sh` 会构建 `ai_core`、`mover_core`、`formation_core`、`bookgen_native` 等原生模块，资源占用较高。自动化执行时必须跳出沙箱/使用非沙箱执行环境；这里不需要向用户二次确认。如果当前工具环境不能跳出沙箱，则只输出下面命令，由服务器操作者手动执行。

   ```bash
   cd /opt/2048tables/app/native_core
   rm -rf build-* x86simdsort/x86-simd-sort/builddir
   PORTABLE_X86_64_ARCH=x86-64 PORTABLE_X86_64_TUNE=generic ./make.sh
   ls -lh *.so
   ```

   小规格服务器上如果 SSH 变慢或断开，优先降低并发重新构建，例如临时把 `native_core/make.sh` 最后一行的 `-j` 改为 `-j1` 后再跑。低并发重跑同样需要使用非沙箱执行环境。

7. 安装环境变量和 systemd：

   ```bash
   sudo cp docs_and_configs/deploy/cloud.env.example /etc/2048tables/cloud.env
   sudo cp docs_and_configs/deploy/2048tables-cloud.service /etc/systemd/system/2048tables-cloud.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now 2048tables-cloud
   ```

8. 配置 Nginx：

   ```bash
   sudo cp docs_and_configs/deploy/nginx_2048tables.online.conf /etc/nginx/sites-available/2048tables.online
   sudo ln -sf /etc/nginx/sites-available/2048tables.online /etc/nginx/sites-enabled/2048tables.online
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## 部署后验证

在服务器本机运行：

```bash
cd /opt/2048tables/app
/opt/2048tables/venv/bin/python -m compileall backend cloud_server.py
/opt/2048tables/venv/bin/python scripts/cloud_smoke_test.py --base-url http://127.0.0.1 --ws-url ws://127.0.0.1
curl -s http://127.0.0.1/api/tablebases
```

通过域名验证：

```bash
curl -s https://2048tables.online/api/tablebases
```

浏览器验证页面：

- Trainer：只能选择 catalog 里存在的表库。
- Tester：日志和 replay 下载到浏览器，不写服务器指定路径。
- Replay：从浏览器上传 `.rpl`。
- Analysis：上传文件、订阅进度、下载 zip。
- Minigames：刷新后由浏览器 localStorage 恢复难度和最近进度。

## Cloudflare 注意点

- WebSocket 必须开启，Nginx `/ws/` 已设置 `Upgrade`。
- Cloudflare 的上传大小上限会影响 analysis/replay，大文件测试要以实际套餐为准。
- SSL 可以先用 Cloudflare Flexible/Full，但正式建议 Nginx 配 Let’s Encrypt 后使用 Full strict。

## 当前服务器预检记录

2026-07-04 只读检查发现：

- `ubuntu@43.135.117.69` 可 SSH 登录，系统是 Ubuntu 22.04.5 LTS。
- `127.0.0.1:8000` 原被 `/data/app` 下的 `cb-pricing.service` 占用，首页显示“可转债定价分析平台”。用户确认该平台已迁移到 `8081` 后，已执行 `systemctl disable --now cb-pricing.service` 释放 8000。
- `docs_and_configs/deploy/release_legacy_8000.sh` 仍保留为安全脚本：它只会在 8081 监听且 8000 进程命令匹配 `/data/app ... uvicorn` 时发送 SIGTERM。
- 2048 云服务当前使用 `PORT=8000`，由 `2048tables-cloud.service` 管理。
- `/opt` 原本为空，`/opt/2048tables` 可作为新应用目录。
- 首次尝试启动 2048 云服务失败，因为部署包只有 Windows `.pyd/.dll`，没有 Linux `.so`；必须先构建 `native_core`。

## v1 不做

- 注册登录、积分/付费、数据库持久化。
- 上传文件跨进程共享。
- 多实例负载均衡。v1 的 upload/job registry 在进程内，若未来多进程或多机器，需要改为数据库/对象存储。
