# 比赛对抗出数原生模块修复（2026-09-29）

## 根因与修复

比赛后端的对抗出数项目调用 `EvilGen.gen_new_num_seeded`，但服务器从主站目录加载的旧 `ai_core.so` 只暴露 `gen_new_num`。仓库中的 C++ 实现与 Python 绑定已有带种子的接口；缺失的是服务器上的编译产物。

在 `/opt/2048tables/deploy-stage/evilgen-seeded-20260929-r1/native_core` 中复制旧原生源码、替换三个已有的带种子源码文件并重新编译 `ai_core`。新版模块仅放入比赛后端独立发布目录的 `native_core` 包，由比赛服务优先加载；主站原生模块和主站服务均未修改。

- 新比赛后端：`/opt/2048tables/tournament-app/releases/20260929-evilgen-seeded-r1`
- 原比赛后端：`/opt/2048tables/tournament-app/releases/20260929-geometry-r1`
- 可复现部署脚本：`tools/deploy_competition_seeded_ai_20260929.py`
- 新比赛模块 SHA-256：`387c5695f50f0a1c692a04292e425c4031de84fee60dd120f9fb3be7eba78d77`
- 主站旧模块 SHA-256：`f7a0061e91e822ea39830510a97fd158586b2edaf77f95516b4f0dcf73120b19`，发布后未变。

## 验证

- 编译使用 Python 3.10、Release 模式和仓库自带 nanobind；只构建 `ai_core`。
- 对 `0x123456789abcde0f` 棋盘，深度 5 的带种子输出：种子 1 为 `(14, 2)`，种子 2 为 `(14, 1)`；旧非种子方法仍为 `(14, 1)`。三项与浏览器 WASM 既有测试一致。
- 发布前适配器冒烟检查：比赛目录中的 `ai_core.so` 被实际导入；`tournament-v2` 对抗出数项目可初始化，双方使用相同种子得到相同初始棋盘和 RNG 游标 2。
- 发布前数据库仅有两间 `SEATING` 房间，无进行中的对局。
- 发布后比赛服务为 `active`，比赛后端 `current` 指向新目录，公网 `/api/health` 返回 HTTP 200，项目 3 试玩入口返回 HTTP 200。比赛进程的 Python 搜索路径先指向新发布目录，因此不会调用主站旧模块。

## 回滚

将 `/opt/2048tables/tournament-app/current` 符号链接原子切回 `20260929-geometry-r1`，重启 `2048tables-competition-test.service`。但旧版缺少带种子方法，会重新暴露本问题；回滚只适用于处理新模块引起的更严重故障。新旧模块及暂存源码均保留供检查。
