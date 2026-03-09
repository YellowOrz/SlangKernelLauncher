# SlangKernelLauncher

[English](README.md) | [中文](README_zh.md)

A cross-platform GPU compute kernel launch framework built on [Slang](https://github.com/shader-slang/slang).
Write a single `.slang` shader and run it on **CPU, Vulkan, Metal, and CUDA** — no per-platform shader maintenance required.

## Features

- **Write once, run everywhere** — the Slang compiler automatically transpiles `.slang` shaders at build time to CPU (C++), Vulkan (SPIR-V), Metal (MSL), and CUDA (PTX)
- **Type-safe** — kernels, backends, and buffer types are all bound at compile time via C++ templates; no runtime string lookups
- **Unified async model** — every backend exposes the same `dispatch()` / `dispatchAsync()` / `sync()` / `syncAll()` interface
- **Adaptive memory** — Vulkan automatically detects integrated GPUs (unified memory) vs. discrete GPUs (staging buffers); Metal always uses unified memory on Apple Silicon
- **Zero-install dependency** — the Slang SDK is downloaded automatically from GitHub on the first `cmake` run

## Requirements

| Component | Requirement |
|-----------|-------------|
| Compiler | C++17, CMake ≥ 3.20 |
| CPU backend | No extra dependencies (always enabled) |
| Vulkan backend | Vulkan SDK (optional, auto-detected) |
| Metal backend | macOS with Xcode Command Line Tools (auto-enabled) |
| CUDA backend | CUDA Toolkit (optional, auto-detected) |
| Slang SDK | Auto-downloaded (v2026.3.1), no manual install needed |

## Building

```bash
# 1. Clone the repository
git clone <repo-url>
cd SlangKernelLauncher

# 2. Configure (downloads Slang SDK ~145 MB on first run)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 3. Build
cmake --build build -j$(nproc)

# 4. Run tests
cd build && ctest --output-on-failure
```

> **Vulkan SDK**: if installed, CMake detects it automatically and enables the Vulkan backend.
> You can also point to it explicitly: `-DVULKAN_SDK=/path/to/VulkanSDK/x.x.x/macOS`

Test executables are placed under `build/`:
- `test_kernel_factory` — CPU backend tests
- `test_vulkan_kernel` — Vulkan backend tests (requires Vulkan SDK)
- `test_metal_kernel` — Metal backend tests (macOS only)

## Quick Start

### 1. Write a Slang Shader

Create `myKernel.slang` in `compiler/shaders/`:

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

After re-running `cmake --build build`, the framework generates:
- `myKernel.h` — per-backend binary blobs (SPIR-V / MSL / PTX)
- `myKernel_cpu.h / .cpp` — CPU-side C++ implementation
- `myKernel_factory.h` — `KernelFactory` template specializations

### 2. Use It in C++

```cpp
#define SLANG_KERNEL_NAMES_IMPL
#include "slang_kernels.h"    // aggregates all generated kernel headers
#include "MemBuffer.h"        // typed buffer wrapper

// Declare the kernel set you need
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

#### Metal Backend (macOS)

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

### 3. Async Dispatch

All GPU backends support async dispatch so CPU work can overlap with GPU execution:

```cpp
// Submit and return immediately
skl::KernelLauncher<SlangKernelID::myKernel, skl::Metal>::dispatchAsync(registry, N / 64);

// Do CPU work in parallel...
doCpuWork();

// Wait for GPU to finish
skl::KernelLauncher<SlangKernelID::myKernel, skl::Metal>::sync(registry);
```

Submit multiple kernels and wait for all of them at once:

```cpp
using Reg2 = skl::KernelRegistry<SlangKernelID::kernel, SlangKernelID::addVec>;
Reg2 registry;
registry.initAllMetal(ctx);

// Bind and submit both kernels asynchronously
registry.get<SlangKernelID::kernel>().bindMetal({ ... });
registry.get<SlangKernelID::addVec>().bindMetal({ ... });

skl::KernelLauncher<SlangKernelID::kernel,  skl::Metal>::dispatchAsync(registry, groupCount);
skl::KernelLauncher<SlangKernelID::addVec,  skl::Metal>::dispatchAsync(registry, groupCount);

// Wait for all kernels in one call
skl::syncAll<skl::Metal>(registry);
```

## Project Structure

```
SlangKernelLauncher/
├── CMakeLists.txt               # Top-level build configuration
├── compiler/
│   ├── CMakeLists.txt           # SlangCompiler build + SDK auto-download
│   ├── shaders/                 # .slang shader sources (add your kernels here)
│   │   ├── kernel.slang         # Example: input[i] * 2
│   │   └── addVec.slang         # Example: a[i] + b[i]
│   └── src/main.cpp             # SlangCompiler tool (runs at build time)
├── include/
│   ├── SlangKernelLauncher.h    # Core framework (KernelManager / Registry / Launcher)
│   ├── MemBuffer.h              # Typed buffer abstraction (CPU / Vulkan / Metal / CUDA)
│   ├── MetalBackend.h           # Metal C++ API (pure C++, no ObjC headers exposed)
│   └── VulkanBackend.h          # Vulkan C++ API
├── src/
│   ├── MetalBackend.mm          # Metal ObjC++ implementation (-fobjc-arc)
│   └── VulkanBackend.cpp        # Vulkan implementation
└── tests/
    ├── test_kernel_factory.cpp  # CPU backend tests
    ├── test_vulkan_kernel.cpp   # Vulkan backend tests
    └── test_metal_kernel.cpp    # Metal backend tests
```

## API Reference

### Backend Tag Types

```cpp
skl::CPU     // Software execution on CPU
skl::Vulkan  // Vulkan compute (requires SKL_HAS_VULKAN)
skl::Metal   // Metal compute  (requires SKL_HAS_METAL, macOS only)
skl::CUDA    // CUDA Driver API (requires SKL_HAS_CUDA_DRIVER)
```

### KernelRegistry

```cpp
// Declare a set of kernels
using Reg = skl::KernelRegistry<SlangKernelID::kernelA, SlangKernelID::kernelB>;
Reg registry;

// Batch initialization
registry.initAllCPU();
registry.initAllMetal(ctx);
registry.initAllVulkan(ctx);

// Access a single KernelManager
auto& mgr = registry.get<SlangKernelID::kernelA>();
```

### KernelLauncher

```cpp
// Synchronous dispatch (N thread groups)
skl::KernelLauncher<SlangKernelID::kernelA, skl::Metal>::dispatch(registry, N);

// Asynchronous dispatch
skl::KernelLauncher<SlangKernelID::kernelA, skl::Metal>::dispatchAsync(registry, N);

// Wait for a single kernel to finish
skl::KernelLauncher<SlangKernelID::kernelA, skl::Metal>::sync(registry);

// Wait for all kernels in the registry to finish
skl::syncAll<skl::Metal>(registry);
```

### MemBuffer

```cpp
// Construction
skl::MemBuffer<float, skl::CPU>    cpuBuf(count);
skl::MemBuffer<float, skl::Metal>  metalBuf(ctx, count);
skl::MemBuffer<float, skl::Vulkan> vulkanBuf(ctx, count);

// Upload / download
buf.upload(std::vector<float>{ ... });
buf.upload(ptr, n);

std::vector<float> out;
buf.download(out);

// Async transfer (truly async on CUDA; equivalent to sync on other backends)
buf.uploadAsync(ptr, n);
buf.downloadAsync(out);
buf.sync();

// Backend-specific accessors (used with bind calls)
cpuBuf.data()            // T*
metalBuf.metalBuffer()   // MetalBuffer*
vulkanBuf.vulkanBuffer() // VulkanBuffer*
```

## Adding a New Kernel

1. Create `<name>.slang` in `compiler/shaders/` — the entry function name must match the file name
2. Re-run `cmake -B build` (CMake detects the new file and updates the build rules)
3. Rebuild: `cmake --build build`
4. Add `SlangKernelID::<name>` to the `KernelRegistry` template argument list in your code

## Known Limitations

- Each `.slang` file supports exactly one entry point, and its name must match the file name
- `[numthreads]` is 1D only (X axis); Y and Z are fixed at 1
- The Metal backend requires Apple Silicon (unified memory); Intel Mac discrete GPUs are not supported
- The CUDA backend uses the Driver API only and does not depend on the CUDA Runtime (`libcuda.so` / `cuda.dll`)
