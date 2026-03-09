#include "SlangKernelLauncher.h"

#include <cstdio>

namespace skl {

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
