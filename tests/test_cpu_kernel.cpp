// test_cpu_kernel.cpp
// 验证 Slang 生成的 CPU kernel 运算结果是否正确
// kernel.slang: outputBuffer[i] = inputBuffer[i] * 2.0f

#include "slang_kernels.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;

#define CHECK(expr)                                                         \
    do {                                                                    \
        if (expr) {                                                         \
            fprintf(stdout, "  [PASS] %s\n", #expr);                       \
            ++g_passed;                                                     \
        } else {                                                            \
            fprintf(stderr, "  [FAIL] %s  (%s:%d)\n", #expr,              \
                    __FILE__, __LINE__);                                    \
            ++g_failed;                                                     \
        }                                                                   \
    } while (0)

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

// ── 1. 基础正确性：N 个元素，每个值 * 2 ────────────────────────────────────────
static void test_basic_multiply()
{
    fprintf(stdout, "\n[TEST] CPU kernel basic correctness (output[i] == input[i] * 2)\n");

    // 使用 3 个 work group，每组 64 线程 → 192 个元素
    const uint32_t groupCount = 3;
    const uint32_t elemCount  = groupCount * kernel_THREAD_GROUP_X;

    std::vector<float> input(elemCount), output(elemCount, 0.f);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = static_cast<float>(i);

    kernel_Params params;
    params.inputBuffer  = { input.data(),  input.size()  };
    params.outputBuffer = { output.data(), output.size() };
    kernel_dispatch(params, groupCount);

    bool allCorrect = true;
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        float expected = input[i] * 2.f;
        if (std::fabs(output[i] - expected) > 1e-6f)
        {
            fprintf(stderr, "  Mismatch at [%u]: got %.6f, expected %.6f\n",
                    i, output[i], expected);
            allCorrect = false;
        }
    }
    CHECK_MSG(allCorrect, "all 192 elements == input * 2");
}

// ── 2. 单 group 验证 ──────────────────────────────────────────────────────────
static void test_single_group()
{
    fprintf(stdout, "\n[TEST] CPU kernel single group (%u threads)\n",
            kernel_THREAD_GROUP_X);

    const uint32_t elemCount = kernel_THREAD_GROUP_X; // 64
    std::vector<float> input(elemCount), output(elemCount, -1.f);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = 1.f + static_cast<float>(i) * 0.5f;

    kernel_Params params;
    params.inputBuffer  = { input.data(),  input.size()  };
    params.outputBuffer = { output.data(), output.size() };
    kernel_dispatch(params, 1); // 1 group

    bool allCorrect = true;
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        if (std::fabs(output[i] - input[i] * 2.f) > 1e-6f)
        {
            allCorrect = false;
            break;
        }
    }
    CHECK_MSG(allCorrect, "single-group 64 elements correct");
}

// ── 3. 零值输入 ───────────────────────────────────────────────────────────────
static void test_zero_input()
{
    fprintf(stdout, "\n[TEST] CPU kernel zero input\n");

    const uint32_t elemCount = kernel_THREAD_GROUP_X;
    std::vector<float> input(elemCount, 0.f), output(elemCount, 99.f);

    kernel_Params params;
    params.inputBuffer  = { input.data(),  input.size()  };
    params.outputBuffer = { output.data(), output.size() };
    kernel_dispatch(params, 1);

    bool allZero = true;
    for (uint32_t i = 0; i < elemCount; ++i)
        if (output[i] != 0.f) { allZero = false; break; }

    CHECK_MSG(allZero, "zero input -> zero output");
}

// ── 4. 负数输入 ───────────────────────────────────────────────────────────────
static void test_negative_input()
{
    fprintf(stdout, "\n[TEST] CPU kernel negative values\n");

    const uint32_t elemCount = kernel_THREAD_GROUP_X;
    std::vector<float> input(elemCount), output(elemCount, 0.f);
    for (uint32_t i = 0; i < elemCount; ++i)
        input[i] = -static_cast<float>(i + 1);

    kernel_Params params;
    params.inputBuffer  = { input.data(),  input.size()  };
    params.outputBuffer = { output.data(), output.size() };
    kernel_dispatch(params, 1);

    bool allCorrect = true;
    for (uint32_t i = 0; i < elemCount; ++i)
    {
        if (std::fabs(output[i] - input[i] * 2.f) > 1e-6f)
        {
            allCorrect = false;
            break;
        }
    }
    CHECK_MSG(allCorrect, "negative values doubled correctly");
}

// ── 5. 线程组常量符合预期 ─────────────────────────────────────────────────────
static void test_thread_group_constants()
{
    fprintf(stdout, "\n[TEST] Thread group size constants\n");
    CHECK(kernel_THREAD_GROUP_X == 64);
    CHECK(kernel_THREAD_GROUP_Y == 1);
    CHECK(kernel_THREAD_GROUP_Z == 1);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    fprintf(stdout, "=== CPU Kernel Correctness Tests ===\n");

    test_thread_group_constants();
    test_basic_multiply();
    test_single_group();
    test_zero_input();
    test_negative_input();

    fprintf(stdout, "\n=== Results: %d passed, %d failed ===\n",
            g_passed, g_failed);

    return g_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
