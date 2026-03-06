#include "SlangKernelLauncher.h"

// 构建时由 SlangCompiler 生成的聚合头文件，包含所有 kernel 的 blob 与 CPU 接口
#include "slang_kernels.h"

#include <cstdio>
#include <cstring>

namespace skl {

bool launchKernel(const KernelDesc& kernel,
                  uint32_t groupCountX,
                  uint32_t groupCountY,
                  uint32_t groupCountZ)
{
    // TODO: 根据当前平台选择合适的后端（Vulkan / Metal / CUDA）提交 dispatch
    const std::vector<uint8_t>* code = nullptr;
    const char* backend = nullptr;

    if (kernel.vulkan && !kernel.vulkan->empty()) { code = kernel.vulkan; backend = "Vulkan"; }
    else if (kernel.metal && !kernel.metal->empty()) { code = kernel.metal; backend = "Metal";  }
    else if (kernel.cuda  && !kernel.cuda->empty())  { code = kernel.cuda;  backend = "CUDA";   }
    else if (kernel.cpp   && !kernel.cpp->empty())   { code = kernel.cpp;   backend = "C++";    }

    if (!code)
    {
        fprintf(stderr, "[SKL] No valid kernel code available\n");
        return false;
    }

    fprintf(stdout, "[SKL] Launching '%s' via %s: %zu bytes, dispatch(%u,%u,%u)\n",
            kernel.entryPoint ? kernel.entryPoint : "?",
            backend, code->size(),
            groupCountX, groupCountY, groupCountZ);
    return true;
}

// 内置默认 kernel（来自 kernel.slang 的编译结果）
bool launchDefaultKernel(uint32_t groupCountX)
{
    KernelDesc desc{};
#ifdef SLANG_HAS_KERNEL_VULKAN
    desc.vulkan = &kernel_vulkan_kernel;
#endif
#ifdef SLANG_HAS_KERNEL_METAL
    desc.metal  = &kernel_metal_kernel;
#endif
#ifdef SLANG_HAS_KERNEL_CUDA
    desc.cuda   = &kernel_cuda_kernel;
#endif
#ifdef SLANG_HAS_KERNEL_CPP
    desc.cpp    = &kernel_cpp_kernel;
#endif
    desc.entryPoint = "kernel";   // 与 kernel.slang 入口函数名一致
    return launchKernel(desc, groupCountX);
}

// ── CPU backend ───────────────────────────────────────────────────────────────

CPUFunction toCPUFunction(void (*groupFn)(ComputeVaryingInput*, void*, void*))
{
    return CPUFunction(groupFn);
}

// ── CUDA backend ──────────────────────────────────────────────────────────────
#ifdef SKL_HAS_CUDA_DRIVER

static const char* cuResultString(CUresult r)
{
    const char* str = nullptr;
    cuGetErrorString(r, &str);
    return str ? str : "unknown";
}

bool loadCUDAFunction(const std::vector<uint8_t>& ptx,
                      const char*                  entryName,
                      CUDAFunction&                out)
{
    CUresult r = cuModuleLoadData(&out.module, ptx.data());
    if (r != CUDA_SUCCESS)
    {
        fprintf(stderr, "[SKL] cuModuleLoadData failed: %s\n", cuResultString(r));
        out.module = nullptr;
        return false;
    }

    r = cuModuleGetFunction(&out.function, out.module, entryName);
    if (r != CUDA_SUCCESS)
    {
        fprintf(stderr, "[SKL] cuModuleGetFunction('%s') failed: %s\n",
                entryName, cuResultString(r));
        cuModuleUnload(out.module);
        out.module   = nullptr;
        out.function = nullptr;
        return false;
    }

    return true;
}

void unloadCUDAFunction(CUDAFunction& fn)
{
    if (fn.module)
    {
        cuModuleUnload(fn.module);
        fn.module   = nullptr;
        fn.function = nullptr;
    }
}

#endif // SKL_HAS_CUDA_DRIVER

} // namespace skl
