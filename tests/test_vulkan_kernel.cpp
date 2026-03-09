// test_vulkan_kernel.cpp
// 验证 KernelRegistry / KernelLauncher 的 Vulkan dispatch

#define SLANG_KERNEL_NAMES_IMPL
#include "slang_kernels.h"
#include "VulkanBackend.h"
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
static void test_basic(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, Vulkan> basic (output[i] == input[i] * 2)\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i + 1);

    skl::MemBuffer<float, skl::Vulkan> inBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);
    inBuf.upload(input);

    registry.get<SlangKernelID::kernel>().bindVulkan({ inBuf.vulkanBuffer(), outBuf.vulkanBuffer() });
    bool ok = skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::dispatch(registry, groupCount);
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
static void test_rebind(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, Vulkan> rebind buffers\n");

    const uint32_t elemCount = kernel_THREAD_GROUP_X;

    skl::MemBuffer<float, skl::Vulkan> inBuf1(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> inBuf2(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);

    inBuf1.upload(std::vector<float>(elemCount, 3.f));
    inBuf2.upload(std::vector<float>(elemCount, 7.f));

    registry.get<SlangKernelID::kernel>().bindVulkan({ inBuf1.vulkanBuffer(), outBuf.vulkanBuffer() });
    skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::dispatch(registry, 1);
    std::vector<float> out;
    outBuf.download(out);
    CHECK_MSG(std::fabs(out[0] - 6.f) < 1e-5f, "first bind: in=3 -> out=6");

    registry.get<SlangKernelID::kernel>().bindVulkan({ inBuf2.vulkanBuffer(), outBuf.vulkanBuffer() });
    skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::dispatch(registry, 1);
    outBuf.download(out);
    CHECK_MSG(std::fabs(out[0] - 14.f) < 1e-5f, "rebind: in=7 -> out=14");
}

// ── 测试 3：Vulkan 结果与 CPU 结果一致 ───────────────────────────────────────
static void test_matches_cpu(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher Vulkan result matches CPU\n");

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

    // Vulkan
    skl::MemBuffer<float, skl::Vulkan> inBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);
    inBuf.upload(input);

    registry.get<SlangKernelID::kernel>().bindVulkan({ inBuf.vulkanBuffer(), outBuf.vulkanBuffer() });
    skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::dispatch(registry, groupCount);

    std::vector<float> gpuOut;
    outBuf.download(gpuOut);

    bool match = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(gpuOut[i] - cpuOut.data()[i]) > 1e-4f)
            { fprintf(stderr, "  [%u] Vulkan=%.6f CPU=%.6f\n", i, gpuOut[i], cpuOut.data()[i]); match = false; }
    CHECK_MSG(match, "Vulkan and CPU produce identical results");
}

// ── addVec 测试：基础正确性 ───────────────────────────────────────────────────
static void test_addvec_basic(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<addVec, Vulkan> basic (output[i] == a[i] + b[i])\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * addVec_THREAD_GROUP_X;

    std::vector<float> a(elemCount), b(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(elemCount - i);
    }

    skl::MemBuffer<float, skl::Vulkan> aBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> bBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);
    aBuf.upload(a);
    bBuf.upload(b);

    registry.get<SlangKernelID::addVec>().bindVulkan({ aBuf.vulkanBuffer(), bBuf.vulkanBuffer(), outBuf.vulkanBuffer() });
    bool ok = skl::KernelLauncher<SlangKernelID::addVec, skl::Vulkan>::dispatch(registry, groupCount);
    CHECK_MSG(ok, "dispatch returned true");

    std::vector<float> output;
    outBuf.download(output);

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - (a[i] + b[i])) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], a[i]+b[i]); correct = false; }
    CHECK_MSG(correct, "256 elements: output[i] == a[i] + b[i]");
}

// ── addVec 测试：Vulkan 与 CPU 结果一致 ──────────────────────────────────────
static void test_addvec_matches_cpu(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<addVec> Vulkan result matches CPU\n");

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

    // Vulkan
    skl::MemBuffer<float, skl::Vulkan> aBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> bBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);
    aBuf.upload(a);
    bBuf.upload(b);

    registry.get<SlangKernelID::addVec>().bindVulkan({ aBuf.vulkanBuffer(), bBuf.vulkanBuffer(), outBuf.vulkanBuffer() });
    skl::KernelLauncher<SlangKernelID::addVec, skl::Vulkan>::dispatch(registry, groupCount);

    std::vector<float> gpuOut;
    outBuf.download(gpuOut);

    bool match = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(gpuOut[i] - cpuOut.data()[i]) > 1e-4f)
            { fprintf(stderr, "  [%u] Vulkan=%.6f CPU=%.6f\n", i, gpuOut[i], cpuOut.data()[i]); match = false; }
    CHECK_MSG(match, "addVec: Vulkan and CPU produce identical results");
}

// ── 测试：kernel dispatchAsync + sync ────────────────────────────────────────
static void test_async_kernel(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<kernel, Vulkan>::dispatchAsync + sync\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i + 1);

    skl::MemBuffer<float, skl::Vulkan> inBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);

    // 异步 upload：CPU memcpy → staging，GPU DMA → device（离散卡），立即返回
    inBuf.uploadAsync(input);
    inBuf.sync();   // 等 upload 完成，data 已进入 GPU

    registry.get<SlangKernelID::kernel>().bindVulkan({ inBuf.vulkanBuffer(), outBuf.vulkanBuffer() });

    // 异步 dispatch：录制 + submit，立即返回
    bool ok = skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::dispatchAsync(registry, groupCount);
    CHECK_MSG(ok, "dispatchAsync returned true");

    // 模拟 CPU 侧并行工作（此处用简单计算代替）
    volatile float dummy = 0.f;
    for (uint32_t i = 0; i < elemCount; ++i) dummy += input[i];

    // 等待 GPU kernel 完成
    skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::sync(registry);

    // 异步 download：GPU DMA → staging（离散卡），立即返回
    std::vector<float> output;
    outBuf.downloadAsync(output);
    outBuf.sync();   // 等 download 完成

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - input[i] * 2.f) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], input[i]*2.f); correct = false; }
    CHECK_MSG(correct, "async: 256 elements output[i] == input[i] * 2");

    (void)dummy;
}

// ── 测试：addVec dispatchAsync + sync ────────────────────────────────────────
static void test_async_addvec(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] KernelLauncher<addVec, Vulkan>::dispatchAsync + sync\n");

    const uint32_t groupCount = 4;
    const uint32_t elemCount  = groupCount * addVec_THREAD_GROUP_X;

    std::vector<float> a(elemCount), b(elemCount);
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(elemCount - i);
    }

    skl::MemBuffer<float, skl::Vulkan> aBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> bBuf(ctx, elemCount);
    skl::MemBuffer<float, skl::Vulkan> outBuf(ctx, elemCount);

    // 异步 upload（可并发提交，各自持有独立 fence）
    aBuf.uploadAsync(a);
    bBuf.uploadAsync(b);
    aBuf.sync();
    bBuf.sync();

    registry.get<SlangKernelID::addVec>().bindVulkan({ aBuf.vulkanBuffer(), bBuf.vulkanBuffer(), outBuf.vulkanBuffer() });

    bool ok = skl::KernelLauncher<SlangKernelID::addVec, skl::Vulkan>::dispatchAsync(registry, groupCount);
    CHECK_MSG(ok, "dispatchAsync returned true");

    // 模拟 CPU 侧并行工作
    volatile float dummy = 0.f;
    for (uint32_t i = 0; i < elemCount; ++i) dummy += a[i] + b[i];

    skl::KernelLauncher<SlangKernelID::addVec, skl::Vulkan>::sync(registry);

    std::vector<float> output;
    outBuf.downloadAsync(output);
    outBuf.sync();

    bool correct = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (std::fabs(output[i] - (a[i] + b[i])) > 1e-5f)
            { fprintf(stderr, "  [%u] got %.6f exp %.6f\n", i, output[i], a[i]+b[i]); correct = false; }
    CHECK_MSG(correct, "async: 256 elements output[i] == a[i] + b[i]");

    (void)dummy;
}

// ── 测试：同时 dispatchAsync 两个 kernel，再 sync 两个 ─────────────────────────
static void test_async_multi_kernel(const skl::VulkanContext& ctx, Reg& registry)
{
    fprintf(stdout, "\n[TEST] dispatchAsync(kernel) + dispatchAsync(addVec) then sync both\n");

    const uint32_t groupCountK = 4;
    const uint32_t elemCountK  = groupCountK * kernel_THREAD_GROUP_X;
    const uint32_t groupCountA = 4;
    const uint32_t elemCountA  = groupCountA * addVec_THREAD_GROUP_X;

    // ── 准备 kernel 数据 ──────────────────────────────────────────────────────
    std::vector<float> kIn(elemCountK);
    for (uint32_t i = 0; i < elemCountK; ++i)
        kIn[i] = static_cast<float>(i + 1);

    skl::MemBuffer<float, skl::Vulkan> kInBuf(ctx, elemCountK);
    skl::MemBuffer<float, skl::Vulkan> kOutBuf(ctx, elemCountK);
    kInBuf.upload(kIn);

    // ── 准备 addVec 数据 ──────────────────────────────────────────────────────
    std::vector<float> aVec(elemCountA), bVec(elemCountA);
    for (uint32_t i = 0; i < elemCountA; ++i)
    {
        aVec[i] = static_cast<float>(i + 1);
        bVec[i] = static_cast<float>(elemCountA - i);
    }

    skl::MemBuffer<float, skl::Vulkan> aBuf(ctx, elemCountA);
    skl::MemBuffer<float, skl::Vulkan> bBuf(ctx, elemCountA);
    skl::MemBuffer<float, skl::Vulkan> aOutBuf(ctx, elemCountA);
    aBuf.upload(aVec);
    bBuf.upload(bVec);

    // ── 绑定两个 kernel ───────────────────────────────────────────────────────
    registry.get<SlangKernelID::kernel>().bindVulkan({ kInBuf.vulkanBuffer(), kOutBuf.vulkanBuffer() });
    registry.get<SlangKernelID::addVec>().bindVulkan({ aBuf.vulkanBuffer(), bBuf.vulkanBuffer(), aOutBuf.vulkanBuffer() });

    // ── 先提交两个 async dispatch，GPU queue 收到两个独立提交 ─────────────────
    bool okK = skl::KernelLauncher<SlangKernelID::kernel,  skl::Vulkan>::dispatchAsync(registry, groupCountK);
    bool okA = skl::KernelLauncher<SlangKernelID::addVec,  skl::Vulkan>::dispatchAsync(registry, groupCountA);
    CHECK_MSG(okK, "kernel  dispatchAsync returned true");
    CHECK_MSG(okA, "addVec  dispatchAsync returned true");

    // ── 模拟 CPU 侧并行工作 ───────────────────────────────────────────────────
    volatile float dummy = 0.f;
    for (uint32_t i = 0; i < elemCountK; ++i) dummy += kIn[i];
    for (uint32_t i = 0; i < elemCountA; ++i) dummy += aVec[i] + bVec[i];

    // ── 全局同步：等待所有 kernel 完成（等价于 cudaDeviceSynchronize）──────────
    bool syncOk = skl::syncAll<skl::Vulkan>(registry);
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
    fprintf(stdout, "=== KernelLauncher / KernelRegistry Vulkan Tests ===\n");

    skl::VulkanContext ctx = skl::createVulkanContext();
    if (ctx.device == VK_NULL_HANDLE)
    {
        fprintf(stderr, "[SKIP] Vulkan context creation failed\n");
        return EXIT_SUCCESS;
    }

    {
        // Registry 初始化一次，所有测试共用
        // 注意：registry 必须在 destroyVulkanContext 之前析构，
        // 因为析构时需要通过 vkCtx_->device 释放 Vulkan 资源。
        Reg registry;
        registry.initAllVulkan(ctx);
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
    } // registry 在此析构，ctx 仍然有效

    skl::destroyVulkanContext(ctx);

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
