
# 2048 AI Core - WebAssembly 迁移技术文档

## 1. 项目概述
本项目成功将原有的 C++ 2048 AI 核心（基于 `nanobind` 的 Python 扩展）迁移至 **WebAssembly (WASM)** 平台。目标是实现在不依赖 Python 环境的情况下，通过静态网页提供高性能的 AI 运算能力。

## 2. 核心技术决策

### 2.1 编译架构隔离
* **实施方案**：创建了独立的 `wasm_project/CMakeLists.txt`。
* **原因**：避免污染根目录用于生成 `.pyd` 的 `nanobind` 配置。WASM 编译需要完全不同的工具链（Emscripten）和链接参数。

### 2.2 绑定技术转换 (Embind)
* **转换逻辑**：将原有的 `NB_MODULE` 逻辑映射到了 Emscripten 的 `EMSCRIPTEN_BINDINGS`。
* **接口保持**：导出的 JS 接口与 Python 端的类名、方法名、成员变量保持 100% 一致（如 `find_best_egtb_move`, `AIPlayer` 类等），降低了前端适配成本。

### 2.3 OpenMP 兼容性保障 (关键点)
* **挑战**：浏览器原生运行多线程（OpenMP）需要开启 `SharedArrayBuffer`，这要求服务器配置严格的 COOP/COEP 响应头，会限制分发灵活性。
* **解决方案**：引入了 **Stub OpenMP (单线程存根)**。
    * 在 `wasm_project/` 中编写了一个虚拟的 `omp.h`。
    * 将 `pragma omp` 操作安全地退化为单线程执行。
* **结果**：生成的 WASM 文件可以在任何普通静态服务器（如 GitHub Pages）上直接运行，无需特殊配置。

## 3. 编译环境与命令

### 3.1 环境要求
* **EMSDK**: 已安装并激活 (`latest` 版本)。
* **编译器**: `emcc` / `em++`。
* **构建工具**: `CMake` + `MinGW Makefiles` (Windows 环境)。

### 3.2 编译流程
在 PowerShell 中执行以下序列：
```powershell
# 1. 激活环境
& "C:\Apps\2048endgameTablebase\emsdk\emsdk_env.ps1"

# 2. 进入构建目录
cd build_wasm

# 3. 配置并编译
emcmake cmake .. -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
emmake mingw32-make -j4
```

## 4. 产出物说明
文件路径：`build_wasm/`
* **`ai_core.js`**: 胶水代码。负责加载 WASM 模块、处理内存映射以及暴露 Embind 接口给 JavaScript。
* **`ai_core.wasm`**: 核心二进制。包含了 2.5MB 的硬编码权重及经过 `-O3` 优化的搜索算法。

## 5. JavaScript 调用示例
```javascript
import Module from './ai_core.js';

Module().then((instance) => {
    // 实例化 AI
    const ai = new instance.AIPlayer(BigInt("0x123456789abcdef0")); 
    
    // 调用接口
    const bestMove = instance.find_best_egtb_move(ai.board, 1);
    console.log("最佳移动:", bestMove);
    
    // 属性读取示例
    console.log("当前节点数:", ai.node);
});
```

## 6. 后续迭代建议 (给 Agent 的备忘)
1.  **SIMD 优化**：当前版本为通用编译。若追求极致速度，可尝试开启 `-msimd128` 并针对 WASM 重新编写 `BoardMover` 的位运算逻辑（替代原有的 AVX2）。
2.  **异步 Worker 化**：由于目前是单线程同步执行，建议前端在 `Web Worker` 中运行 `start_search`，避免 AI 思考时导致网页 UI 卡死。
3.  **持久化缓存**：3MB 的体积可以通过浏览器 `IndexedDB` 缓存，进一步提升二次加载速度。

---

**Agent 检查总结**：
此次任务解决了跨平台编译中最核心的“环境冲突”和“并发协议”问题。虽然牺牲了多线程性能，但换取了极高的**分发普适性**，这对于“静态网页版 2048 AI”来说是一个极优的权衡。