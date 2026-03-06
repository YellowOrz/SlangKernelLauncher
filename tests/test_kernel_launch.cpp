#include "SlangKernelLauncher.h"

// 构建时生成的聚合头文件，包含所有 kernel 的 blob 与 CPU 接口
#include "slang_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

// ── 简易测试框架 ──────────────────────────────────────────────────────────────

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(expr)                                                     \
    do {                                                                \
        if (expr) {                                                     \
            fprintf(stdout, "  [PASS] %s\n", #expr);                   \
            ++g_passed;                                                 \
        } else {                                                        \
            fprintf(stderr, "  [FAIL] %s  (%s:%d)\n", #expr,          \
                    __FILE__, __LINE__);                                \
            ++g_failed;                                                 \
        }                                                               \
    } while (0)

// ── 测试用例 ──────────────────────────────────────────────────────────────────

// 1. 验证各 target 的字节码非空（有该 target 才检查）
static void test_kernel_blobs_not_empty()
{
    fprintf(stdout, "\n[TEST] kernel blob sizes\n");

#ifdef SLANG_HAS_KERNEL_VULKAN
    CHECK(!kernel_vulkan_kernel.empty());
    fprintf(stdout, "         vulkan : %zu bytes\n", kernel_vulkan_kernel.size());
#else
    fprintf(stdout, "         vulkan : (not compiled)\n");
#endif

#ifdef SLANG_HAS_KERNEL_METAL
    CHECK(!kernel_metal_kernel.empty());
    fprintf(stdout, "         metal  : %zu bytes\n", kernel_metal_kernel.size());
#else
    fprintf(stdout, "         metal  : (not compiled)\n");
#endif

#ifdef SLANG_HAS_KERNEL_CUDA
    CHECK(!kernel_cuda_kernel.empty());
    fprintf(stdout, "         cuda   : %zu bytes\n", kernel_cuda_kernel.size());
#else
    fprintf(stdout, "         cuda   : (not compiled)\n");
#endif

#ifdef SLANG_HAS_KERNEL_CPP
    CHECK(!kernel_cpp_kernel.empty());
    fprintf(stdout, "         cpp    : %zu bytes\n", kernel_cpp_kernel.size());
#else
    fprintf(stdout, "         cpp    : (not compiled)\n");
#endif
}

// 2. Vulkan SPIR-V 魔数校验（0x07230203）
static void test_vulkan_spirv_magic()
{
    fprintf(stdout, "\n[TEST] Vulkan SPIR-V magic number\n");

#ifdef SLANG_HAS_KERNEL_VULKAN
    const auto& blob = kernel_vulkan_kernel;
    CHECK(blob.size() >= 4);
    if (blob.size() >= 4)
    {
        uint32_t magic = 0;
        // SPIR-V 以小端序存储魔数
        magic |= (uint32_t)blob[0];
        magic |= (uint32_t)blob[1] << 8;
        magic |= (uint32_t)blob[2] << 16;
        magic |= (uint32_t)blob[3] << 24;
        CHECK(magic == 0x07230203u);
    }
#else
    fprintf(stdout, "  [SKIP] vulkan not compiled\n");
#endif
}

// 3. Metal MSL 内容校验（文本应包含 "kernel" 关键字）
static void test_metal_msl_content()
{
    fprintf(stdout, "\n[TEST] Metal MSL content\n");

#ifdef SLANG_HAS_KERNEL_METAL
    const auto& blob = kernel_metal_kernel;
    CHECK(!blob.empty());
    std::string msl(reinterpret_cast<const char*>(blob.data()), blob.size());
    CHECK(msl.find("kernel") != std::string::npos);
#else
    fprintf(stdout, "  [SKIP] metal not compiled\n");
#endif
}

// 4. CUDA PTX 内容校验（PTX 文本应以 ".version" 开头）
static void test_cuda_ptx_content()
{
    fprintf(stdout, "\n[TEST] CUDA PTX content\n");

#ifdef SLANG_HAS_KERNEL_CUDA
    const auto& blob = kernel_cuda_kernel;
    CHECK(!blob.empty());
    std::string ptx(reinterpret_cast<const char*>(blob.data()), blob.size());
    CHECK(ptx.find(".version") != std::string::npos);
#else
    fprintf(stdout, "  [SKIP] cuda not compiled\n");
#endif
}

// 5. launchKernel 正常路径
static void test_launch_kernel()
{
    fprintf(stdout, "\n[TEST] launchKernel with available targets\n");

    skl::KernelDesc desc{};
    desc.entryPoint = "kernel";   // 与 kernel.slang 入口函数名一致

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

    CHECK(skl::launchKernel(desc, 64));
    CHECK(skl::launchKernel(desc, 128, 4, 1));
}

// 6. launchKernel 异常路径：空 KernelDesc 应返回 false
static void test_launch_empty_kernel()
{
    fprintf(stdout, "\n[TEST] launchKernel with empty descriptor (expect false)\n");

    skl::KernelDesc empty{};
    CHECK(!skl::launchKernel(empty, 1));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main()
{
    fprintf(stdout, "=== SlangKernelLauncher Tests ===\n");

    test_kernel_blobs_not_empty();
    test_vulkan_spirv_magic();
    test_metal_msl_content();
    test_cuda_ptx_content();
    test_launch_kernel();
    test_launch_empty_kernel();

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n",
            g_passed, g_failed);

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
