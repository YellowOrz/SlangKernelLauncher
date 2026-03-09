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
    } // registry 在此析构，ctx 仍然有效

    skl::destroyVulkanContext(ctx);

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n", g_passed, g_failed);
    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
