// test_kernel_factory.cpp
// 验证 KernelManager / KernelRegistry / KernelLauncher 的 CPU dispatch

#define SLANG_KERNEL_NAMES_IMPL
#include "slang_kernels.h"
#include "MemBuffer.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
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

// ── 1. KernelManager 基础正确性 ───────────────────────────────────────────────
static void test_manager_cpu_basic()
{
    fprintf(stdout, "\n[TEST] KernelManager CPU basic\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    skl::MemBuffer<float, skl::CPU> inBuf(elemCount);
    skl::MemBuffer<float, skl::CPU> outBuf(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        inBuf.data()[i] = static_cast<float>(i + 1);

    kernel_Params params;
    params.inputBuffer  = { inBuf.data(),  inBuf.size()  };
    params.outputBuffer = { outBuf.data(), outBuf.size() };

    skl::KernelManager<SlangKernelID::kernel> mgr;
    mgr.initCPU();
    mgr.bindCPU(params);
    mgr.dispatch<skl::CPU>(groupCount);

    bool ok = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(outBuf.data()[i] - inBuf.data()[i] * 2.f) > 1e-6f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, outBuf.data()[i], inBuf.data()[i]*2.f); ok = false; }
    CHECK_MSG(ok, "256 elements: output[i] == input[i] * 2");
}

// ── 2. KernelRegistry + KernelLauncher ───────────────────────────────────────
static void test_launcher_via_registry()
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, CPU>::dispatch(registry, gx)\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    skl::MemBuffer<float, skl::CPU> inBuf(elemCount);
    skl::MemBuffer<float, skl::CPU> outBuf(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        inBuf.data()[i] = static_cast<float>(i + 1);

    kernel_Params params;
    params.inputBuffer  = { inBuf.data(),  inBuf.size()  };
    params.outputBuffer = { outBuf.data(), outBuf.size() };

    // Registry 统一管理所有 kernel 的资源
    skl::KernelRegistry<SlangKernelID::kernel> registry;
    registry.initAllCPU();
    registry.get<SlangKernelID::kernel>().bindCPU(params);

    // KernelLauncher 通过 registry 查找 manager 并 dispatch
    skl::KernelLauncher<SlangKernelID::kernel, skl::CPU>::dispatch(registry, groupCount);

    bool ok = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(outBuf.data()[i] - inBuf.data()[i] * 2.f) > 1e-6f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, outBuf.data()[i], inBuf.data()[i]*2.f); ok = false; }
    CHECK_MSG(ok, "KernelLauncher via registry: output[i] == input[i] * 2");
}

// ── 3. KernelManager rebind ───────────────────────────────────────────────────
static void test_manager_rebind()
{
    fprintf(stdout, "\n[TEST] KernelManager CPU rebind buffers\n");

    const uint32_t elemCount = kernel_THREAD_GROUP_X;

    skl::MemBuffer<float, skl::CPU> in1(elemCount);
    skl::MemBuffer<float, skl::CPU> in2(elemCount);
    skl::MemBuffer<float, skl::CPU> out(elemCount);
    in1.upload(std::vector<float>(elemCount, 1.f));
    in2.upload(std::vector<float>(elemCount, 5.f));

    skl::KernelManager<SlangKernelID::kernel> mgr;
    mgr.initCPU();

    kernel_Params p1;
    p1.inputBuffer  = { in1.data(), in1.size() };
    p1.outputBuffer = { out.data(), out.size() };
    mgr.bindCPU(p1);
    mgr.dispatch<skl::CPU>(1);
    CHECK_MSG(std::fabs(out.data()[0] - 2.f) < 1e-6f, "first bind: in=1 -> out=2");

    kernel_Params p2;
    p2.inputBuffer  = { in2.data(), in2.size() };
    p2.outputBuffer = { out.data(), out.size() };
    mgr.bindCPU(p2);
    mgr.dispatch<skl::CPU>(1);
    CHECK_MSG(std::fabs(out.data()[0] - 10.f) < 1e-6f, "rebind: in=5 -> out=10");
}

// ── 4. MemBuffer upload/download 往返 ────────────────────────────────────────
static void test_membuffer_roundtrip()
{
    fprintf(stdout, "\n[TEST] MemBuffer<float, CPU> upload/download roundtrip\n");

    const size_t n = 128;
    std::vector<float> src(n);
    for (size_t i = 0; i < n; ++i) src[i] = static_cast<float>(i) * 1.5f - 64.f;

    skl::MemBuffer<float, skl::CPU> buf(n);
    buf.upload(src);

    std::vector<float> dst;
    buf.download(dst);

    bool ok = (dst.size() == n);
    for (size_t i = 0; i < n && ok; ++i)
        ok = std::fabs(dst[i] - src[i]) < 1e-6f;
    CHECK_MSG(ok, "upload then download returns identical data");
}

// ── 5. 枚举名称表 ─────────────────────────────────────────────────────────────
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
    fprintf(stdout, "=== KernelManager / KernelLauncher CPU Tests ===\n");

    test_manager_cpu_basic();
    test_launcher_via_registry();
    test_manager_rebind();
    test_membuffer_roundtrip();
    test_kernel_names();

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
