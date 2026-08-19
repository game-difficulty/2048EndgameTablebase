# Windows 本机 Tablebase Worker

该 Worker 从 Windows 本机主动连接云服务器，只允许查询本地配置中明确列出的定式。服务器不会收到本地盘符或路径，也不能通过协议指定文件或目录。

## 当前定式与并发

示例配置位于 `docs_and_configs/remote_worker.local.example.json`：

| full_pattern | 默认路径 | resource_group | 表并发 |
| --- | --- | --- | --- |
| `free11_512` | `D:/free11-512` | `disk-d` | 1 |
| `free11_1024` | `D:/free11-1k` | `disk-d` | 1 |
| `4442f_2048` | `F:/4442f-2k` | `disk-f` | 1 |

`free11` 两张表共享 `disk-d` 的单个槽位，因此机械盘同一时间只执行一次读取。`F:` 也暂时限制为 1。后续迁移数据时只需修改私有配置中的 `path` 和 `resource_group`。

## 安装与配置

1. 在项目 Python 3.12 环境安装根目录 `requirements.txt`。必须包含 `numpy`、`websockets`，并确保 `native_core/formation_core.cp312-win_amd64.pyd` 可导入。
2. 复制示例配置，但不要把带有本机信息的私有配置提交到 Git：

   ```powershell
   Copy-Item docs_and_configs/remote_worker.local.example.json docs_and_configs/remote_worker.local.json
   ```

3. 设置与服务器 `REMOTE_TABLEBASE_WORKER_SECRET` 相同的高强度随机密钥。临时运行可使用：

   ```powershell
   $env:TABLEBASE_WORKER_TOKEN = "replace-with-server-secret"
   ```

   需要开机/登录后运行时，应通过 Windows 凭据管理方案或用户级环境变量注入，不要把密钥写入 JSON 或启动脚本。

4. 只检查配置、路径和表文件，不初始化 reader，也不执行真实查询：

   ```powershell
   py -3.12 -m tools.tablebase_worker `
     --config docs_and_configs/remote_worker.local.json `
     --check-config
   ```

5. 启动。脚本会优先使用项目同级的 `myenv/Scripts/python.exe`，不存在时回退到 `py -3.12`：

   ```powershell
   tools/tablebase_worker/start_worker.ps1
   ```

   使用指定 Python 时：

   ```powershell
   tools/tablebase_worker/start_worker.ps1 -PythonExe C:/path/to/python.exe
   ```

### Windows 常驻运行

生产环境建议为当前 Windows 用户创建“登录时启动”的计划任务，动作使用隐藏窗口运行 `start_worker.ps1`，并配置失败后每分钟重启。密钥可保存为用户级环境变量；启动脚本会依次读取当前进程、用户级和机器级 `TABLEBASE_WORKER_TOKEN`：

```powershell
[Environment]::SetEnvironmentVariable(
  "TABLEBASE_WORKER_TOKEN",
  "replace-with-server-secret",
  "User"
)
```

计划任务应设置为“已有实例运行时不启动新实例”，避免同一 `worker_id` 重复连接。用户未登录时 Worker 不运行；若要求无人登录也提供服务，应改用专用 Windows 服务账号和 Windows 服务管理器，并将密钥放入该服务账号的安全环境。

## 运行行为

- 连接地址默认为 `wss://2048tables.online/worker-ws/tablebase`，不需要开放家庭网络入站端口。
- 建连发送 `HELLO`，收到服务器 `HELLO_ACK` 后开始处理请求。
- 每 10 秒复查一次目录/表文件标记并发送 `HEARTBEAT` 及 `tables:[{full_pattern,ready}]`；不打开数据文件或执行真实查询。服务器 30 秒未收到消息会判定离线，能力集合变化时才更新 catalog epoch。
- 断线后按 1、2、4 秒递增并带少量随机抖动重连，最长等待 60 秒。
- 启动时只检查路径中是否存在匹配的表文件或 AD 层目录。每张表的 `BookReaderDispatcher` 在首次实际请求时才创建和 `dispatch`。
- `LOOKUP_BATCH` 在一个资源组槽位内顺序完成，避免机械盘随机寻道并发放大。
- `CANCEL` 会取消尚未开始的请求。正在执行的 native 读取不能强制终止，但结果会被丢弃，且资源槽会保持到读取真正结束。
- 日志默认写入 `tools/tablebase_worker/logs/worker.log`，单文件 5 MiB，保留 3 个轮转文件。

## 协议 v1

协议使用顶层扁平 JSON。Worker 不接受任何 `path`、命令或文件内容字段。

```json
{"type":"HELLO","protocol_version":1,"worker_id":"home-main","auth_token":"...","tables":[{"full_pattern":"free11_512","ready":true}]}
```

```json
{"type":"LOOKUP","request_id":"...","full_pattern":"free11_512","pattern":"free11","target":"512","board":"0011223344556677","use_variant":false,"board_is_lookup":false}
```

`board_is_lookup=true` 表示服务器已经执行 board replacement；否则 Worker 复用项目的 `replace_board_for_lookup()` 后再查表。批量响应的 `items` 与请求 `boards` 保持相同顺序。

## 测试

配置和协议测试不访问真实定式，也不会连接服务器：

```powershell
py -3.12 -m unittest discover -s tools/tablebase_worker/tests -v
```
