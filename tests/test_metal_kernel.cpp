// test_metal_kernel.cpp
// 验证 KernelRegistry / KernelLauncher 的 Metal dispatch

#define SLANG_KERNEL_NAMES_IMPL
#include "slang_kernels.h"
#include "MetalBackend.h"
#include "MemBuffer.h"

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

using Reg = skl::KernelRegistry<SlangKernelID::kernel, SlangKernelID::addVec>;

// ── 测试 1：基础正确性 ────────────────────────────────────────────────────────
static void test_basic(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, Metal> basic (output[i] == input[i] * 2)\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i + 1);

    skl::MemBuffer<float, skl::Metal> inBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);
    inBuf.upload(input);

    registry.get<SlangKernelID::kernel>().bindMetal({ inBuf.metalBuffer(), outBuf.metalBuffer() });
    bool ok = skl::KernelLauncher<SlangKernelID::kernel, skl::Metal>::dispatch(registry, groupCount);
    CHECK_MSG(ok, "dispatch returned true");

    std::vector<float> output;
    outBuf.download(output);

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - input[i] * 2.f) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], input[i]*2.f); correct = false; }
    CHECK_MSG(correct, "256 elements: output[i] == input[i] * 2");
}

// ── 测试 2：重新绑定 buffer 后再次 dispatch ───────────────────────────────────
static void test_rebind(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, Metal> rebind buffers\n");

    const uint32_t elemCount = kernel_THREAD_GROUP_X;

    skl::MemBuffer<float, skl::Metal> inBuf1(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> inBuf2(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);

    inBuf1.upload(std::vector<float>(elemCount, 3.f));
    inBuf2.upload(std::vector<float>(elemCount, 7.f));

    registry.get<SlangKernelID::kernel>().bindMetal({ inBuf1.metalBuffer(), outBuf.metalBuffer() });
    skl::KernelLauncher<SlangKernelID::kernel, skl::Metal>::dispatch(registry, 1);
    std::vector<float> out;
    outBuf.download(out);
    CHECK_MSG(std::fabs(out[0] - 6.f) < 1e-5f, "first bind: in=3 -> out=6");

    registry.get<SlangKernelID::kernel>().bindMetal({ inBuf2.metalBuffer(), outBuf.metalBuffer() });
    skl::KernelLauncher<SlangKernelID::kernel, skl::Metal>::dispatch(registry, 1);
    outBuf.download(out);
    CHECK_MSG(std::fabs(out[0] - 14.f) < 1e-5f, "rebind: in=7 -> out=14");
}

// ── 测试 3：Metal 结果与 CPU 结果一致 ────────────────────────────────────────
static void test_matches_cpu(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel> Metal result matches CPU\n");

    const uint32_t groupCount = 3;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i) * 0.3f - 48.f;

    // CPU
    skl::MemBuffer<float, skl::CPU> cpuIn(elemCount);
    skl::MemBuffer<float, skl::CPU> cpuOut(elemCount);
    cpuIn.upload(input);
    {
        kernel_Params p;
        p.inputBuffer  = { cpuIn.data(),  cpuIn.size()  };
        p.outputBuffer = { cpuOut.data(), cpuOut.size() };
        registry.get<SlangKernelID::kernel>().bindCPU(p);
        skl::KernelLauncher<SlangKernelID::kernel, skl::CPU>::dispatch(registry, groupCount);
    }

    // Metal
    skl::MemBuffer<float, skl::Metal> inBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);
    inBuf.upload(input);

    registry.get<SlangKernelID::kernel>().bindMetal({ inBuf.metalBuffer(), outBuf.metalBuffer() });
    skl::KernelLauncher<SlangKernelID::kernel, skl::Metal>::dispatch(registry, groupCount);

    std::vector<float> gpuOut;
    outBuf.download(gpuOut);

    bool match = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(gpuOut[i] - cpuOut.data()[i]) > 1e-4f)
            { fprintf(stderr, "  [%u] Metal=%.6f CPU=%.6f\n", i, gpuOut[i], cpuOut.data()[i]); match = false; }
    CHECK_MSG(match, "Metal and CPU produce identical results");
}

// ── addVec 测试：基础正确性 ───────────────────────────────────────────────────
static void test_addvec_basic(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<addVec, Metal> basic (output[i] == a[i] + b[i])\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * addVec_THREAD_GROUP_X;

    std::vector<float> a(elemCount), b(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(elemCount - i);
    }

    skl::MemBuffer<float, skl::Metal> aBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> bBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);
    aBuf.upload(a);
    bBuf.upload(b);

    registry.get<SlangKernelID::addVec>().bindMetal({ aBuf.metalBuffer(), bBuf.metalBuffer(), outBuf.metalBuffer() });
    bool ok = skl::KernelLauncher<SlangKernelID::addVec, skl::Metal>::dispatch(registry, groupCount);
    CHECK_MSG(ok, "dispatch returned true");

    std::vector<float> output;
    outBuf.download(output);

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - (a[i] + b[i])) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], a[i]+b[i]); correct = false; }
    CHECK_MSG(correct, "256 elements: output[i] == a[i] + b[i]");
}

// ── addVec 测试：Metal 与 CPU 结果一致 ────────────────────────────────────────
static void test_addvec_matches_cpu(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<addVec> Metal result matches CPU\n");

    const uint32_t groupCount = 3;
    const uint32_t elemCount  = groupCount * addVec_THREAD_GROUP_X;

    std::vector<float> a(elemCount), b(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        a[i] = static_cast<float>(i) * 0.5f - 10.f;
        b[i] = static_cast<float>(elemCount - i) * 1.5f;
    }

    // CPU
    skl::MemBuffer<float, skl::CPU> cpuA(elemCount);
    skl::MemBuffer<float, skl::CPU> cpuB(elemCount);
    skl::MemBuffer<float, skl::CPU> cpuOut(elemCount);
    cpuA.upload(a);
    cpuB.upload(b);
    {
        addVec_Params p;
        p.aBuffer      = { cpuA.data(),   cpuA.size()   };
        p.bBuffer      = { cpuB.data(),   cpuB.size()   };
        p.outputBuffer = { cpuOut.data(), cpuOut.size() };
        registry.get<SlangKernelID::addVec>().bindCPU(p);
        skl::KernelLauncher<SlangKernelID::addVec, skl::CPU>::dispatch(registry, groupCount);
    }

    // Metal
    skl::MemBuffer<float, skl::Metal> aBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> bBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);
    aBuf.upload(a);
    bBuf.upload(b);

    registry.get<SlangKernelID::addVec>().bindMetal({ aBuf.metalBuffer(), bBuf.metalBuffer(), outBuf.metalBuffer() });
    skl::KernelLauncher<SlangKernelID::addVec, skl::Metal>::dispatch(registry, groupCount);

    std::vector<float> gpuOut;
    outBuf.download(gpuOut);

    bool match = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(gpuOut[i] - cpuOut.data()[i]) > 1e-4f)
            { fprintf(stderr, "  [%u] Metal=%.6f CPU=%.6f\n", i, gpuOut[i], cpuOut.data()[i]); match = false; }
    CHECK_MSG(match, "addVec: Metal and CPU produce identical results");
}

// ── 测试：kernel dispatchAsync + sync ────────────────────────────────────────
static void test_async_kernel(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, Metal>::dispatchAsync + sync\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i + 1);

    skl::MemBuffer<float, skl::Metal> inBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);
    inBuf.upload(input);

    registry.get<SlangKernelID::kernel>().bindMetal({ inBuf.metalBuffer(), outBuf.metalBuffer() });

    bool ok = skl::KernelLauncher<SlangKernelID::kernel, skl::Metal>::dispatchAsync(registry, groupCount);
    CHECK_MSG(ok, "dispatchAsync returned true");

    // 模拟 CPU 侧并行工作
    volatile float dummy = 0.f;
    for (uint32_t i = 0; i < elemCount; ++i) dummy += input[i];

    skl::KernelLauncher<SlangKernelID::kernel, skl::Metal>::sync(registry);

    std::vector<float> output;
    outBuf.download(output);

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - input[i] * 2.f) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], input[i]*2.f); correct = false; }
    CHECK_MSG(correct, "async: 256 elements output[i] == input[i] * 2");

    (void)dummy;
}

// ── 测试：addVec dispatchAsync + sync ────────────────────────────────────────
static void test_async_addvec(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<addVec, Metal>::dispatchAsync + sync\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * addVec_THREAD_GROUP_X;

    std::vector<float> a(elemCount), b(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(elemCount - i);
    }

    skl::MemBuffer<float, skl::Metal> aBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> bBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Metal> outBuf(ctx, elemCount);
    aBuf.upload(a);
    bBuf.upload(b);

    registry.get<SlangKernelID::addVec>().bindMetal({ aBuf.metalBuffer(), bBuf.metalBuffer(), outBuf.metalBuffer() });

    bool ok = skl::KernelLauncher<SlangKernelID::addVec, skl::Metal>::dispatchAsync(registry, groupCount);
    CHECK_MSG(ok, "dispatchAsync returned true");

    volatile float dummy = 0.f;
    for (uint32_t i = 0; i < elemCount; ++i) dummy += a[i] + b[i];

    skl::KernelLauncher<SlangKernelID::addVec, skl::Metal>::sync(registry);

    std::vector<float> output;
    outBuf.download(output);

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - (a[i] + b[i])) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], a[i]+b[i]); correct = false; }
    CHECK_MSG(correct, "async: 256 elements output[i] == a[i] + b[i]");

    (void)dummy;
}

// ── 测试：同时 dispatchAsync 两个 kernel，再 syncAll ──────────────────────────
static void test_async_multi_kernel(const skl::MetalContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] dispatchAsync(kernel) + dispatchAsync(addVec) then syncAll\n");

    const uint32_t groupCountK = 4;
    const uint32_t elemCountK  = groupCountK * kernel_THREAD_GROUP_X;
    const uint32_t groupCountA = 4;
    const uint32_t elemCountA  = groupCountA * addVec_THREAD_GROUP_X;

    // ── 准备 kernel 数据 ──────────────────────────────────────────────────────
    std::vector<float> kIn(elemCountK);
    for (uint32_t i = 0; i < elemCountK; ++i)
        kIn[i] = static_cast<float>(i + 1);

    skl::MemBuffer<float, skl::Metal> kInBuf(ctx, elemCountK);
    skl::MemBuffer<float, skl::Metal> kOutBuf(ctx, elemCountK);
    kInBuf.upload(kIn);

    // ── 准备 addVec 数据 ──────────────────────────────────────────────────────
    std::vector<float> aVec(elemCountA), bVec(elemCountA);
    for (uint32_t i = 0; i < elemCountA; ++i)
    {
        aVec[i] = static_cast<float>(i + 1);
        bVec[i] = static_cast<float>(elemCountA - i);
    }

    skl::MemBuffer<float, skl::Metal> aBuf(ctx, elemCountA);
    skl::MemBuffer<float, skl::Metal> bBuf(ctx, elemCountA);
    skl::MemBuffer<float, skl::Metal> aOutBuf(ctx, elemCountA);
    aBuf.upload(aVec);
    bBuf.upload(bVec);

    // ── 绑定两个 kernel ───────────────────────────────────────────────────────
    registry.get<SlangKernelID::kernel>().bindMetal({ kInBuf.metalBuffer(), kOutBuf.metalBuffer() });
    registry.get<SlangKernelID::addVec>().bindMetal({ aBuf.metalBuffer(), bBuf.metalBuffer(), aOutBuf.metalBuffer() });

    // ── 先提交两个 async dispatch ────────────────────────────────────────────
    bool okK = skl::KernelLauncher<SlangKernelID::kernel,  skl::Metal>::dispatchAsync(registry, groupCountK);
    bool okA = skl::KernelLauncher<SlangKernelID::addVec,  skl::Metal>::dispatchAsync(registry, groupCountA);
    CHECK_MSG(okK, "kernel  dispatchAsync returned true");
    CHECK_MSG(okA, "addVec  dispatchAsync returned true");

    // ── 模拟 CPU 侧并行工作 ───────────────────────────────────────────────────
    volatile float dummy = 0.f;
    for (uint32_t i = 0; i < elemCountK; ++i) dummy += kIn[i];
    for (uint32_t i = 0; i < elemCountA; ++i) dummy += aVec[i] + bVec[i];

    // ── 全局同步：等待所有 kernel 完成 ────────────────────────────────────────
    bool syncOk = skl::syncAll<skl::Metal>(registry);
    CHECK_MSG(syncOk, "syncAll returned true");

    // ── 验证 kernel 结果 ──────────────────────────────────────────────────────
    std::vector<float> kOut;
    kOutBuf.download(kOut);

    bool kCorrect = true;
    for (uint32_t i = 0; i < elemCountK; ++i)
        if (std::fabs(kOut[i] - kIn[i] * 2.f) > 1e-5f)
            { fprintf(stderr, "  kernel[%u] got %.6f exp %.6f\n", i, kOut[i], kIn[i]*2.f); kCorrect = false; }
    CHECK_MSG(kCorrect, "multi-async: kernel output[i] == input[i] * 2");

    // ── 验证 addVec 结果 ──────────────────────────────────────────────────────
    std::vector<float> aOut;
    aOutBuf.download(aOut);

    bool aCorrect = true;
    for (uint32_t i = 0; i < elemCountA; ++i)
        if (std::fabs(aOut[i] - (aVec[i] + bVec[i])) > 1e-5f)
            { fprintf(stderr, "  addVec[%u] got %.6f exp %.6f\n", i, aOut[i], aVec[i]+bVec[i]); aCorrect = false; }
    CHECK_MSG(aCorrect, "multi-async: addVec output[i] == a[i] + b[i]");

    (void)dummy;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    fprintf(stdout, "=== KernelLauncher / KernelRegistry Metal Tests ===\n");

    skl::MetalContext ctx = skl::createMetalContext();
    if (ctx.device == nullptr)
    {
        fprintf(stderr, "[SKIP] Metal context creation failed\n");
        return EXIT_SUCCESS;
    }

    {
        Reg registry;
        registry.initAllMetal(ctx);
        registry.initAllCPU();

        fprintf(stdout, "\n── kernel (multiply×2) ──\n");
        test_basic(ctx, registry);
        test_rebind(ctx, registry);
        test_matches_cpu(ctx, registry);

        fprintf(stdout, "\n── addVec (a + b) ──\n");
        test_addvec_basic(ctx, registry);
        test_addvec_matches_cpu(ctx, registry);

        fprintf(stdout, "\n── async dispatch ──\n");
        test_async_kernel(ctx, registry);
        test_async_addvec(ctx, registry);
        test_async_multi_kernel(ctx, registry);
    }

    skl::destroyMetalContext(ctx);

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
