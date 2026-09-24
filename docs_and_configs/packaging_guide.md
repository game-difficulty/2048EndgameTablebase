# Packaging and Release Guide

本文档是 2048 Endgame Tablebase 的发布规范。目标是让 Windows 和 Linux
安装包来自同一个干净的 Git 提交，使用固定命名，统一放入发布目录，并附带可核验的
英文 Release Note 和 SHA-256。

## 1. 固定路径与命名

以下命令均假定源码目录为：

```text
C:\Apps\2048endgameTablebase\src
```

最终发布文件只放在：

```text
C:\Apps\2048endgameTablebase\dist
```

每次发布使用语义化版本号 `vMAJOR.MINOR.PATCH`，例如 `v13.2.0`。最终文件名必须为：

```text
2048EndgameTablebase-windows-v13.2.0.7z
2048EndgameTablebase-linux-v13.2.0.tar.xz
```

临时构建目录和 Release Note 放在：

```text
C:\Apps\2048endgameTablebase\src\tmp\release-v13.2.0
```

`tmp/`、构建目录、压缩包和 `frontend/dist/` 都是本地产物，不提交到 Git。

## 2. 发布提交规范

发布包必须对应一个已提交的干净 `main` HEAD。提交前明确暂存文件，不使用
`git add .`：

```powershell
cd C:\Apps\2048endgameTablebase\src
git status --short
git add -- <需要发布的源码、配置和文档>
git diff --cached --check
git diff --cached --name-only
git commit -m "<简短、具体的英文提交信息>"
```

可以提交：

- 应用源码：`backend/`、`engine_core/`、`frontend/src/` 及入口文件。
- 原生实现源码：`native_core/src/`、`native_core/include/` 及必要构建配置。
- 用户需要的配置、帮助文档和静态资源。
- 与当前改动直接相关的维护文档。

不得提交：

- 根目录 `tests/` 中的一次性 Python 回归测试。测试可以在本地执行，但不进入发布提交。
- `frontend/dist/`、`build/`、`dist/`、`tmp/`、虚拟环境和各类缓存。
- PyInstaller 输出、原生编译产物、安装包和 SHA 文件。
- `logger.txt`、本地配置、tablebase、回放样本及其他用户数据。
- 与本次发布无关的改动。

`native_core/tests_src/` 与根目录 `/tests/` 均为本地测试目录，不提交到远端，
已在 `.gitignore` 中忽略。提交前执行
`git ls-files tests`，应无输出。

提交完成后确认：

```powershell
git status --short
git rev-parse HEAD
```

`git status --short` 必须为空。后续两个平台都从这个 HEAD 打包；打包过程中不得修改源码。

## 3. 发布前构建

设置本次版本和目录：

```powershell
$Version = "v13.2.0"
$Root = "C:\Apps\2048endgameTablebase"
$Source = "$Root\src"
$ReleaseDir = "$Source\tmp\release-$Version"
$FinalDir = "$Root\dist"
New-Item -ItemType Directory -Force $ReleaseDir, $FinalDir | Out-Null
```

构建前端。发布时使用 `npm ci`，以 `package-lock.json` 锁定依赖：

```powershell
cd "$Source\frontend"
npm ci
npm run build
```

每次发布都必须执行 Windows 原生发布目标构建，不能根据提交内容判断后跳过：

```powershell
cd $Source
cmake -S .\native_core -B .\native_core\build-formation -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DNATIVE_BUILD_TESTS=OFF
cmake --build .\native_core\build-formation --config Release --target `
  ai_core mover_core formation_core bookgen_native `
  bc_family_generation_full bc_family_solve_full -j
```

打包前确认 `native_core/` 中有 `ai_core*.pyd`、`mover_core*.pyd`、
`formation_core*.pyd`、`bookgen_native.dll` 和三个 MinGW 运行库；
`native_core/build-formation/` 中还必须有 `bc_family_generation_full.exe` 与
`bc_family_solve_full.exe`。缺少任一必要文件时停止打包。

Windows 的发布来源是固定的：三个 `.pyd`、`bookgen_native.dll` 和 MinGW 运行库只从
`native_core/` 收集，两个 BC helper 只从 `native_core/build-formation/` 收集。不得从
`build-bc-release`、`build-bc-relwithdeb` 或其他历史构建目录回退。spec 会在文件缺失、
或同一模块存在多个 ABI 文件时直接终止，不能临时修改搜索顺序绕过检查。

Linux 同样只使用本次容器内构建产生的 `native_core/` 模块和
`native_core/build-formation/` BC helper，不复用宿主机 Windows 产物。

## 4. Windows 打包

以仓库内 `2048EndgameTablebase.spec` 为唯一 PyInstaller 配置，不复制旧的长命令：

```powershell
cd $Source
$WinVenv = "$ReleaseDir\windows-venv"
$WinBuild = "$ReleaseDir\windows-build"
$WinDist = "$ReleaseDir\windows-dist"
python -m venv $WinVenv
& "$WinVenv\Scripts\python.exe" -m pip install --upgrade pip
& "$WinVenv\Scripts\python.exe" -m pip install -r .\requirements.txt pyinstaller
& "$WinVenv\Scripts\python.exe" -m PyInstaller --noconfirm --clean `
  --workpath $WinBuild --distpath $WinDist .\2048EndgameTablebase.spec
```

PyInstaller 完成后，必须逐项确认源模块与包内模块完全一致。模块没有可靠的语义化版本字段，
因此以完整 SHA-256 作为发布版本标识：

```powershell
$BundleRoot = "$WinDist\2048EndgameTablebase\_internal"

function Get-SingleFile([string]$Directory, [string]$Filter) {
    $Items = @(Get-ChildItem -LiteralPath $Directory -Filter $Filter -File)
    if ($Items.Count -ne 1) {
        throw "Expected one $Filter in $Directory, found $($Items.Count)"
    }
    return $Items[0]
}

$ModulePairs = @(
    @{ Source = (Get-SingleFile "$Source\native_core" "ai_core*.pyd").FullName; Bundle = (Get-SingleFile "$BundleRoot\native_core" "ai_core*.pyd").FullName },
    @{ Source = (Get-SingleFile "$Source\native_core" "mover_core*.pyd").FullName; Bundle = (Get-SingleFile "$BundleRoot\native_core" "mover_core*.pyd").FullName },
    @{ Source = (Get-SingleFile "$Source\native_core" "formation_core*.pyd").FullName; Bundle = (Get-SingleFile "$BundleRoot\native_core" "formation_core*.pyd").FullName },
    @{ Source = "$Source\native_core\bookgen_native.dll"; Bundle = "$BundleRoot\native_core\bookgen_native.dll" },
    @{ Source = "$Source\native_core\libgcc_s_seh-1.dll"; Bundle = "$BundleRoot\native_core\libgcc_s_seh-1.dll" },
    @{ Source = "$Source\native_core\libgomp-1.dll"; Bundle = "$BundleRoot\native_core\libgomp-1.dll" },
    @{ Source = "$Source\native_core\libwinpthread-1.dll"; Bundle = "$BundleRoot\native_core\libwinpthread-1.dll" },
    @{ Source = "$Source\native_core\libgcc_s_seh-1.dll"; Bundle = "$BundleRoot\libgcc_s_seh-1.dll" },
    @{ Source = "$Source\native_core\libgomp-1.dll"; Bundle = "$BundleRoot\libgomp-1.dll" },
    @{ Source = "$Source\native_core\libwinpthread-1.dll"; Bundle = "$BundleRoot\libwinpthread-1.dll" },
    @{ Source = "$Source\native_core\build-formation\bc_family_generation_full.exe"; Bundle = "$BundleRoot\native_core\bc_family_generation_full.exe" },
    @{ Source = "$Source\native_core\build-formation\bc_family_solve_full.exe"; Bundle = "$BundleRoot\native_core\bc_family_solve_full.exe" }
)

$ModuleVersions = foreach ($Pair in $ModulePairs) {
    if (-not (Test-Path -LiteralPath $Pair.Source) -or -not (Test-Path -LiteralPath $Pair.Bundle)) {
        throw "Missing module pair: $($Pair.Source) / $($Pair.Bundle)"
    }
    $SourceHash = (Get-FileHash -LiteralPath $Pair.Source -Algorithm SHA256).Hash
    $BundleHash = (Get-FileHash -LiteralPath $Pair.Bundle -Algorithm SHA256).Hash
    if ($SourceHash -ne $BundleHash) {
        throw "Packaged module mismatch: $($Pair.Bundle)"
    }
    [pscustomobject]@{
        BundlePath = $Pair.Bundle
        SHA256    = $BundleHash
    }
}
$ModuleVersions | Format-Table -AutoSize
```

前端也必须逐文件比对，防止 `frontend/dist` 没有同步最新源码：

```powershell
function Get-TreeHashes([string]$RootPath) {
    @(Get-ChildItem -LiteralPath $RootPath -Recurse -File | Sort-Object FullName | ForEach-Object {
        $Relative = $_.FullName.Substring($RootPath.TrimEnd('\').Length + 1)
        "$Relative`t$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)"
    })
}

$FrontendDiff = Compare-Object `
    (Get-TreeHashes "$Source\frontend\dist") `
    (Get-TreeHashes "$BundleRoot\frontend\dist")
if ($FrontendDiff) {
    $FrontendDiff | Format-Table -AutoSize
    throw "Packaged frontend does not match frontend/dist"
}
```

先从 `$WinDist\2048EndgameTablebase` 启动程序做基本冒烟验证，再压缩整个 onedir
目录。压缩包内必须只有一个顶层 `2048EndgameTablebase/` 目录：

```powershell
$WinArchive = "$FinalDir\2048EndgameTablebase-windows-$Version.7z"
Push-Location $WinDist
& "$Source\7zip\7z.exe" a -t7z -mx=7 $WinArchive .\2048EndgameTablebase
Pop-Location
& "$Source\7zip\7z.exe" t $WinArchive
```

## 5. Linux 打包

Linux 发布包使用 Docker 工作流构建，不能用 Windows PyInstaller 输出代替：

```powershell
& "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
  -ExecutionPolicy Bypass -File "$Root\linux_build\build_linux_docker.ps1"
```

仅当 `linux_build/docker/` 没有变化且本机 builder image 已存在时，才可加
`-SkipImageBuild`。脚本会重新构建 Linux 前端和原生模块，并把结果写到：

```text
C:\Apps\2048endgameTablebase\linux_build\artifacts
```

选择本次命令刚生成的最新 `.tar.xz`，复制并按版本重命名到最终目录：

```powershell
$LinuxArtifacts = "$Root\linux_build\artifacts"
$LatestLinux = Get-ChildItem "$LinuxArtifacts\2048EndgameTablebase-linux-*.tar.xz" |
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$LinuxArchive = "$FinalDir\2048EndgameTablebase-linux-$Version.tar.xz"
Copy-Item -LiteralPath $LatestLinux.FullName -Destination $LinuxArchive
tar -tJf $LinuxArchive | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Invalid Linux tar.xz archive" }
```

不得复用旧版本的 Linux 压缩包。应核对 `$LatestLinux.LastWriteTime` 属于当前构建。

Linux 原生模块必须动态链接系统 `libstdc++.so.6`，不得分别静态链接 C++ 运行库。
用 `readelf -d` 检查 `ai_core`、`mover_core`、`formation_core` 和 `bookgen_native.so`
均包含此 `NEEDED` 依赖；不要将构建机的 libstdc++ 打进包内。
验证时必须在同一进程先导入 `ai_core`、`mover_core` 再运行 `formation_core` 的 BC
生成和回算，检查 CSV 数值字段、每行列数及完成状态，不能只单独导入计算模块测试。
分别验证临时压缩开启和关闭；不得通过忽略统计写入异常规避失败。

## 6. 产物验证

两个包都必须存在、可列出且非空：

```powershell
$WinArchive = "$FinalDir\2048EndgameTablebase-windows-$Version.7z"
$LinuxArchive = "$FinalDir\2048EndgameTablebase-linux-$Version.tar.xz"
Get-Item $WinArchive, $LinuxArchive | Select-Object Name, Length, LastWriteTime
Get-FileHash $WinArchive, $LinuxArchive -Algorithm SHA256 |
  Select-Object Path, Hash
```

发布前至少验证：

- Windows 包可以解压、启动，并能打开主要页面。
- Linux 包归档结构完整，顶层目录为 `2048EndgameTablebase/`。必须在不同于构建容器的
  较新 Linux 桌面环境中验证实际 GUI：启动页、主页面和至少一次页面操作正常。
  Xvfb 可用于自动化检查，但不能替代 Wayland/GPU 的目标机验证。仅 backend child、
  HTTP 200 或原生模块 import 成功，不代表桌面包通过；无法验证时必须明确列出缺口。
  保持默认隐私模式进行测试：GTK WebKit 可能不提供 localStorage，布局等可选偏好必须
  能在存储缺失或读写失败时退化为会话内设置，不能因此阻断主页面挂载。
- 两个平台均包含 `frontend/dist`、帮助文档、默认 pattern、主题、MathJax、图片和必要
  原生运行库。
- 包内不含根目录 `tests/`、源码缓存、日志、本地配置、tablebase 或回放样本。
- 两个文件名中的版本号与 Git tag 完全一致。
- Windows 包内 `.pyd`、`bookgen_native`、BC helper 和 MinGW DLL 的 SHA-256 与本次构建
  源文件逐项一致；根目录和 `native_core/` 下的同名 MinGW DLL 都要检查，不得仅比较文件
  大小或修改时间。
- 启动 Windows 包的 backend child 做冒烟测试时，检查进程已加载的 `formation_core`、
  `libgcc_s_seh-1.dll`、`libgomp-1.dll` 和 `libwinpthread-1.dll` 均来自包内
  `_internal/native_core/`，不能来自 Anaconda、PATH 或旧的解压目录。
- Linux manifest 中必须包含 `ai_core`、`mover_core`、`formation_core`、
  `bookgen_native` 和两个 BC helper，并记录各自 SHA-256。
- Linux spec 使用 `exclude_system_libraries`，只额外保留嵌入式 Python 必需的旧 ABI
  库（当前 Bullseye/Python 3.9 为 libffi.so.7、libssl.so.1.1、libcrypto.so.1.1、
  libmpdec.so.3）。更换构建基线时重新审核此列表及 ELF 依赖。GTK/WebKit 和通用系统库
  必须来自目标系统，不能只剔除 GTK 主库后继续混用旧的传递依赖。
- 检查包内没有 libreadline、libtinfo、libgnutls、libpcre2、libstdc++ 等系统库。
  目标机需安装 GTK3/WebKit2GTK、GObject introspection，以及 libgomp、liblzma 等原生
  依赖；对实际包内 ELF 执行依赖检查，确认没有 `not found` 或版本/符号错误。
  在较新 Linux 上设置 `LD_LIBRARY_PATH` 指向包内 `_internal` 后执行 `/bin/sh -c
  'exit 0'` 也必须成功，以检查启动外部程序时的库冲突。

## 7. Release Note 规范

Release Note 必须使用英文，保存为：

```text
C:\Apps\2048endgameTablebase\src\tmp\release-v13.2.0\release-notes.md
```

内容以用户可观察到的变化为主，不照抄 commit，不写内部排查过程，不把测试文件或构建
细节列为功能。推荐格式：

```markdown
# 2048EndgameTablebase v13.2.0

## Highlights

- Fixed ...
- Added ...
- Improved ...

## Compatibility

- Existing `.rec` and `.rpl` files remain supported.

## SHA-256

- `2048EndgameTablebase-windows-v13.2.0.7z`: `<sha256>`
- `2048EndgameTablebase-linux-v13.2.0.tar.xz`: `<sha256>`
```

小型修复版本通常写 1 至 4 条 `Highlights`。只有确实存在兼容性、迁移或已知限制时才保留
对应章节。SHA-256 必须从最终 `dist` 文件重新计算后填写。

## 8. 推送与 GitHub Release

所有构建和验证通过后，推送 `main`，再给同一个提交打 annotated tag：

```powershell
cd $Source
git status --short
git push origin main
git tag -a $Version -m "2048EndgameTablebase $Version"
git push origin $Version
```

如果已安装 GitHub CLI，创建 Release 并上传两个最终包：

```powershell
gh release create $Version `
  $WinArchive $LinuxArchive `
  --title "2048EndgameTablebase $Version" `
  --notes-file "$ReleaseDir\release-notes.md"
```

没有 GitHub CLI 时，在 GitHub 网页中用同一个 tag 创建非 draft、非 prerelease Release，
粘贴同一份英文 Release Note，并上传这两个文件。上传完成后核对网页显示的文件名和大小；
不要上传解压目录、`.tar.gz`、构建报告、测试文件或临时源码归档。

## 9. 最终检查表

- `main` 工作区干净，发布改动已提交，`git ls-files tests` 无输出。
- Windows 和 Linux 包来自同一个 HEAD。
- 两个最终包均位于 `C:\Apps\2048endgameTablebase\dist`。
- 文件名、Release 标题、Release Note 和 tag 使用同一个版本号。
- Release Note 为英文，包含最终两个文件的 SHA-256。
- `main` 和 tag 已推送，GitHub Release 已发布且只附带两个正式包。

执行发布的 Agent 在结束前必须明确报告以下信息，不能只写“构建成功”：

- 用于打包的 Git HEAD/tag，以及工作区是否干净。
- Windows 模块版本表：三个 `.pyd`、`bookgen_native.dll`、两个 BC helper、三个 MinGW
  DLL 的包内路径和完整 SHA-256，并确认与源文件一致。
- Windows 冒烟进程实际加载的 native 模块/DLL 路径。
- Linux manifest 中对应六个 native 文件的完整 SHA-256。
- `frontend/dist` 与 Windows 包逐文件一致，Linux manifest 中前端入口资源名属于本次构建。
- 最终两个归档的文件名、大小和 SHA-256。
