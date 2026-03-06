// test_kernel_factory.cpp
// 验证 KernelFactory<SlangKernelID::kernel, skl::CPU>::get() 返回的 CPUFunction
// 能正确执行 kernel.slang：outputBuffer[i] = inputBuffer[i] * 2.0f

#define SLANG_KERNEL_NAMES_IMPL
#include "slang_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;

#define CHECK_MSG(expr, msg)                                                \
    do {                                                                    \
        if (expr) {                                                         \
            fprintf(stdout, "  [PASS] %s\n", msg);                         \
            ++g_passed;                                                     \
        } else {                                                            \
            fprintf(stderr, "  [FAIL] %s  (%s:%d)\n", msg,                \
                    __FILE__, __LINE__);                                    \
            ++g_failed;                                                     \
        }                                                                   \
    } while (0)

// 辅助：用 CPUFunction 手动逐 group dispatch
static void dispatchWithFactory(skl::CPUFunction& fn,
                                 kernel_Params&    params,
                                 uint32_t          groupCountX)
{
    for (uint32_t g = 0; g < groupCountX; ++g)
    {
        ComputeVaryingInput vi{};
        vi.startGroupID = { g, 0, 0 };
        vi.endGroupID   = { g + 1, 1, 1 };
        fn(&vi, nullptr, &params);
    }
}

// ── 1. 基础正确性 ─────────────────────────────────────────────────────────────
static void test_factory_basic()
{
    fprintf(stdout, "\n[TEST] KernelFactory<kernel, CPU>::get() basic (output[i] == input[i] * 2)\n");

    auto fn = skl::KernelFactory<SlangKernelID::kernel, skl::CPU>::get();

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount), output(elemCount, 0.f);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i + 1);

    kernel_Params params;
    params.inputBuffer  = { input.data(),  input.size()  };
    params.outputBuffer = { output.data(), output.size() };

    dispatchWithFactory(fn, params, groupCount);

    bool ok = true;
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        if (std::fabs(output[i] - input[i] * 2.f) > 1e-6f)
        {
            fprintf(stderr, "  Mismatch at [%u]: got %.6f, expected %.6f\n",
                    i, output[i], input[i] * 2.f);
            ok = false;
        }
    }
    CHECK_MSG(ok, "256 elements (4 groups x 64): all output[i] == input[i] * 2");
}

// ── 2. factory 结果与 kernel_dispatch 一致 ────────────────────────────────────
static void test_factory_matches_dispatch()
{
    fprintf(stdout, "\n[TEST] KernelFactory result matches kernel_dispatch\n");

    const uint32_t groupCount = 3;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i) * 0.5f - 64.f;

    std::vector<float> out_factory(elemCount, 0.f);
    std::vector<float> out_dispatch(elemCount, 0.f);

    // factory 路径：逐 group 调用 CPUFunction
    {
        auto fn = skl::KernelFactory<SlangKernelID::kernel, skl::CPU>::get();
        kernel_Params p;
        p.inputBuffer  = { input.data(), input.size() };
        p.outputBuffer = { out_factory.data(), out_factory.size() };
        dispatchWithFactory(fn, p, groupCount);
    }

    // 直接 dispatch 路径
    {
        kernel_Params p;
        p.inputBuffer  = { input.data(), input.size() };
        p.outputBuffer = { out_dispatch.data(), out_dispatch.size() };
        kernel_dispatch(p, groupCount);
    }

    bool match = true;
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        if (std::fabs(out_factory[i] - out_dispatch[i]) > 1e-6f)
        {
            fprintf(stderr, "  Diverge at [%u]: factory=%.6f, dispatch=%.6f\n",
                    i, out_factory[i], out_dispatch[i]);
            match = false;
        }
    }
    CHECK_MSG(match, "factory and kernel_dispatch produce identical results");
}

// ── 3. 枚举名称表 ─────────────────────────────────────────────────────────────
static void test_kernel_names()
{
    fprintf(stdout, "\n[TEST] slang_kernel_names[] table\n");
    CHECK_MSG(static_cast<int>(SlangKernelID::Count) >= 1, "at least one kernel registered");
    CHECK_MSG(slang_kernel_names[static_cast<int>(SlangKernelID::kernel)] != nullptr,
              "slang_kernel_names[kernel] is non-null");
    CHECK_MSG(std::string(slang_kernel_names[static_cast<int>(SlangKernelID::kernel)]) == "kernel",
              "slang_kernel_names[kernel] == \"kernel\"");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    fprintf(stdout, "=== KernelFactory Tests ===\n");

    test_factory_basic();
    test_factory_matches_dispatch();
    test_kernel_names();

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n",
            g_passed, g_failed);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
