# SlangKernelLauncher

[English](README.md) | [中文](README_zh.md)

一个基于 [Slang](https://github.com/shader-slang/slang) 的跨平台 GPU Compute Kernel 启动框架。
只需编写一份 `.slang` 着色器，即可在 **CPU、Vulkan、Metal、CUDA** 四个后端上运行，无需为每个平台维护独立的 shader 代码。

## 特性

- **一次编写，四端运行** — Slang 编译器在构建期将 `.slang` 自动编译到 CPU (C++)、Vulkan (SPIR-V)、Metal (MSL)、CUDA (PTX)
- **类型安全** — 全部通过 C++ 模板在编译期绑定 kernel、backend 和 buffer 类型，无运行期字符串查找
- **统一异步模型** — 所有 backend 均提供 `dispatch()` / `dispatchAsync()` / `sync()` / `syncAll()` 接口
- **自适应内存** — Vulkan 自动检测集成显卡（统一内存）与独显（staging buffer），Metal 始终为统一内存（Apple Silicon）
- **自动下载依赖** — 首次 `cmake` 时自动从 GitHub 下载 Slang SDK，无需手动安装

## 平台与依赖

| 组件 | 要求 |
|------|------|
| 编译器 | C++17，CMake ≥ 3.20 |
| CPU backend | 无额外依赖（始终启用） |
| Vulkan backend | Vulkan SDK（可选，自动检测） |
| Metal backend | macOS（自动启用，需 Xcode Command Line Tools） |
| CUDA backend | CUDA Toolkit（可选，自动检测） |
| Slang SDK | 自动下载 v2026.3.1，无需手动安装 |

## 编译

```bash
# 1. 克隆仓库
git clone <repo-url>
cd SlangKernelLauncher

# 2. 配置（首次运行会自动下载 Slang SDK，约 145 MB）
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. 编译
cmake --build build -j$(nproc)

# 4. 运行测试
cd build && ctest --output-on-failure
```

> **Vulkan SDK**：若已安装 Vulkan SDK，CMake 会自动检测并启用 Vulkan backend。
> 也可以通过 `-DVULKAN_SDK=/path/to/VulkanSDK/x.x.x/macOS` 手动指定路径。

生成的测试可执行文件位于 `build/` 下：
- `test_kernel_factory` — CPU backend 基础测试
- `test_vulkan_kernel` — Vulkan backend 测试（需要 Vulkan SDK）
- `test_metal_kernel` — Metal backend 测试（仅 macOS）

## 快速上手

### 1. 编写 Slang Shader

在 `compiler/shaders/` 下新建 `myKernel.slang`：

```hlsl
// myKernel.slang
RWStructuredBuffer<float> inputBuffer;
RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void myKernel(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    outputBuffer[idx] = inputBuffer[idx] * 3.0f;
}
```

重新运行 `cmake --build build` 后，框架会自动生成：
- `myKernel.h` — 各 backend 的二进制 blob（SPIR-V / MSL / PTX）
- `myKernel_cpu.h / .cpp` — CPU 端 C++ 实现
- `myKernel_factory.h` — `KernelFactory` 模板特化

### 2. 在 C++ 中使用

```cpp
#define SLANG_KERNEL_NAMES_IMPL
#include "slang_kernels.h"    // 聚合所有生成的 kernel 头文件
#include "MemBuffer.h"        // 类型化 buffer 封装

// 声明要用的 kernel 组合
using Reg = skl::KernelRegistry<SlangKernelID::myKernel>;
```

#### CPU Backend

```cpp
Reg registry;
registry.initAllCPU();

const uint32_t N = 256;
skl::MemBuffer<float, skl::CPU> inBuf(N), outBuf(N);
inBuf.upload(std::vector<float>(N, 1.0f));

myKernel_Params p;
p.inputBuffer  = { inBuf.data(),  inBuf.size()  };
p.outputBuffer = { outBuf.data(), outBuf.size() };
registry.get<SlangKernelID::myKernel>().bindCPU(p);

skl::KernelLauncher<SlangKernelID::myKernel, skl::CPU>::dispatch(registry, N / 64);

std::vector<float> result;
outBuf.download(result);  // result[i] == 3.0f
```

#### Metal Backend（macOS）

```cpp
skl::MetalContext ctx = skl::createMetalContext();

Reg registry;
registry.initAllMetal(ctx);
registry.initAllCPU();

const uint32_t N = 256;
skl::MemBuffer<float, skl::Metal> inBuf(ctx, N), outBuf(ctx, N);
inBuf.upload(std::vector<float>(N, 1.0f));

registry.get<SlangKernelID::myKernel>().bindMetal(
    { inBuf.metalBuffer(), outBuf.metalBuffer() });

skl::KernelLauncher<SlangKernelID::myKernel, skl::Metal>::dispatch(registry, N / 64);

std::vector<float> result;
outBuf.download(result);  // result[i] == 3.0f

skl::destroyMetalContext(ctx);
```

#### Vulkan Backend

```cpp
skl::VulkanContext ctx = skl::createVulkanContext();

Reg registry;
registry.initAllVulkan(ctx);

const uint32_t N = 256;
skl::MemBuffer<float, skl::Vulkan> inBuf(ctx, N), outBuf(ctx, N);
inBuf.upload(std::vector<float>(N, 1.0f));

registry.get<SlangKernelID::myKernel>().bindVulkan(
    { inBuf.vulkanBuffer(), outBuf.vulkanBuffer() });

skl::KernelLauncher<SlangKernelID::myKernel, skl::Vulkan>::dispatch(registry, N / 64);

std::vector<float> result;
outBuf.download(result);  // result[i] == 3.0f

skl::destroyVulkanContext(ctx);
```

### 3. 异步 Dispatch

所有 GPU backend 均支持异步 dispatch，可在 GPU 计算期间并行执行 CPU 工作：

```cpp
// 提交异步任务（立即返回）
skl::KernelLauncher<SlangKernelID::myKernel, skl::Metal>::dispatchAsync(registry, N / 64);

// CPU 侧并行计算...
doCpuWork();

// 等待 GPU 完成
skl::KernelLauncher<SlangKernelID::myKernel, skl::Metal>::sync(registry);
```

同时提交多个 kernel 后统一等待：

```cpp
using Reg2 = skl::KernelRegistry<SlangKernelID::kernel, SlangKernelID::addVec>;
Reg2 registry;
registry.initAllMetal(ctx);

// 绑定 + 异步提交两个 kernel
registry.get<SlangKernelID::kernel>().bindMetal({ ... });
registry.get<SlangKernelID::addVec>().bindMetal({ ... });

skl::KernelLauncher<SlangKernelID::kernel,  skl::Metal>::dispatchAsync(registry, groupCount);
skl::KernelLauncher<SlangKernelID::addVec,  skl::Metal>::dispatchAsync(registry, groupCount);

// 一次等待所有 kernel 完成
skl::syncAll<skl::Metal>(registry);
```

## 项目结构

```
SlangKernelLauncher/
├── CMakeLists.txt               # 顶层构建配置
├── compiler/
│   ├── CMakeLists.txt           # SlangCompiler 构建 + SDK 自动下载
│   ├── shaders/                 # .slang 着色器源文件（在此添加自定义 kernel）
│   │   ├── kernel.slang         # 示例：input[i] * 2
│   │   └── addVec.slang         # 示例：a[i] + b[i]
│   └── src/main.cpp             # SlangCompiler 工具（构建期运行）
├── include/
│   ├── SlangKernelLauncher.h    # 核心框架（KernelManager / Registry / Launcher）
│   ├── MemBuffer.h              # 类型化 buffer（CPU / Vulkan / Metal / CUDA）
│   ├── MetalBackend.h           # Metal C++ API（纯 C++，无 ObjC 依赖）
│   └── VulkanBackend.h          # Vulkan C++ API
├── src/
│   ├── MetalBackend.mm          # Metal ObjC++ 实现（-fobjc-arc）
│   └── VulkanBackend.cpp        # Vulkan 实现
└── tests/
    ├── test_kernel_factory.cpp  # CPU backend 测试
    ├── test_vulkan_kernel.cpp   # Vulkan backend 测试
    └── test_metal_kernel.cpp    # Metal backend 测试
```

## 核心 API 速查

### Backend 标签类型

```cpp
skl::CPU     // CPU 软件执行
skl::Vulkan  // Vulkan compute（需 SKL_HAS_VULKAN）
skl::Metal   // Metal compute（需 SKL_HAS_METAL，仅 macOS）
skl::CUDA    // CUDA Driver API（需 SKL_HAS_CUDA_DRIVER）
```

### KernelRegistry

```cpp
// 声明一组 kernel
using Reg = skl::KernelRegistry<SlangKernelID::kernelA, SlangKernelID::kernelB>;
Reg registry;

// 批量初始化
registry.initAllCPU();
registry.initAllMetal(ctx);
registry.initAllVulkan(ctx);

// 获取单个 KernelManager
auto& mgr = registry.get<SlangKernelID::kernelA>();
```

### KernelLauncher

```cpp
// 同步 dispatch（N 个线程组）
skl::KernelLauncher<SlangKernelID::kernelA, skl::Metal>::dispatch(registry, N);

// 异步 dispatch
skl::KernelLauncher<SlangKernelID::kernelA, skl::Metal>::dispatchAsync(registry, N);

// 等待单个 kernel 完成
skl::KernelLauncher<SlangKernelID::kernelA, skl::Metal>::sync(registry);

// 等待 registry 中所有 kernel 完成
skl::syncAll<skl::Metal>(registry);
```

### MemBuffer

```cpp
// 构造
skl::MemBuffer<float, skl::CPU>    cpuBuf(count);
skl::MemBuffer<float, skl::Metal>  metalBuf(ctx, count);
skl::MemBuffer<float, skl::Vulkan> vulkanBuf(ctx, count);

// 上传 / 下载
buf.upload(std::vector<float>{ ... });
buf.upload(ptr, n);

std::vector<float> out;
buf.download(out);

// 异步传输（CUDA 真异步，其他 backend 等同同步）
buf.uploadAsync(ptr, n);
buf.downloadAsync(out);
buf.sync();

// Backend 专属访问器（供 bind 使用）
cpuBuf.data()            // T*
metalBuf.metalBuffer()   // MetalBuffer*
vulkanBuf.vulkanBuffer() // VulkanBuffer*
```

## 添加新 Kernel

1. 在 `compiler/shaders/` 下新建 `<name>.slang`，入口函数名与文件名保持一致
2. 重新运行 `cmake -B build`（CMake 会检测到新文件并更新构建规则）
3. 重新编译：`cmake --build build`
4. 在代码中将 `SlangKernelID::<name>` 加入 `KernelRegistry` 模板参数列表

## 已知限制

- 每个 `.slang` 文件只支持一个入口函数，且入口函数名必须与文件名相同
- `[numthreads]` 只支持一维（X 轴），Y/Z 固定为 1
- Metal backend 仅支持 Apple Silicon（统一内存），不支持 Intel Mac 独显
- CUDA backend 仅使用 Driver API，不依赖 CUDA Runtime（`libcuda.so` / `cuda.dll`）
