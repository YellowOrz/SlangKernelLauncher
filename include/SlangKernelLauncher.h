#pragma once

#include <cstdint>
#include <functional>
#include <vector>

// ── CPU dispatch types（与 kernel_cpu.h 共享，guard 防止重定义）──────────────
#ifndef SLANG_CPU_VECTOR_TYPES
#define SLANG_CPU_VECTOR_TYPES
struct uint2 { uint32_t x, y; };
struct uint3 { uint32_t x, y, z; };
struct uint4 { uint32_t x, y, z, w; };
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
#endif // SLANG_CPU_VECTOR_TYPES

#ifndef SLANG_CPU_DISPATCH_TYPES
#define SLANG_CPU_DISPATCH_TYPES
struct ComputeThreadVaryingInput { uint3 groupID; uint3 groupThreadID; };
struct ComputeVaryingInput       { uint3 startGroupID; uint3 endGroupID; };
#endif // SLANG_CPU_DISPATCH_TYPES

// ── CUDA Driver API（仅在 CMake 检测到 CUDAToolkit 时启用）─────────────────
#ifdef SKL_HAS_CUDA_DRIVER
#include <cuda.h>
#endif

namespace skl {

// ── 已编译的 kernel 描述符 ────────────────────────────────────────────────────
struct KernelDesc
{
    const std::vector<uint8_t>* vulkan = nullptr;   // SPIR-V
    const std::vector<uint8_t>* metal  = nullptr;   // MSL source
    const std::vector<uint8_t>* cuda   = nullptr;   // PTX
    const std::vector<uint8_t>* cpp    = nullptr;   // C++ source（已废弃，CPU 端用 toCPUFunction）
    const char* entryPoint = nullptr;
};

bool launchKernel(const KernelDesc& kernel,
                  uint32_t groupCountX,
                  uint32_t groupCountY = 1,
                  uint32_t groupCountZ = 1);

// ── Backend tags ──────────────────────────────────────────────────────────────
struct CPU    {};   // CPU 软件执行
struct CUDA   {};   // CUDA Driver API（需要 CUDAToolkit）
struct Vulkan {};   // Vulkan compute（预留）
struct Metal  {};   // Metal compute（预留）

// ── CPU backend ───────────────────────────────────────────────────────────────
// kernel_Group 签名：一次调用运行一个 work group（内含 THREAD_GROUP_X 个线程）
// 用户可在外部开启多线程，对每个 group 并行调用 CPUFunction
using CPUFunction = std::function<void(ComputeVaryingInput*, void*, void*)>;

// 将 kernel_Group 函数指针包装为 CPUFunction
// 用法：auto fn = skl::toCPUFunction(kernel_Group);
CPUFunction toCPUFunction(void (*groupFn)(ComputeVaryingInput*, void*, void*));

// ── KernelFactory ─────────────────────────────────────────────────────────────
// 特化在构建时生成的 slang_kernels.h 中，通过 SlangKernelID 枚举 + Backend tag
// 选取对应的函数
//
//   CPU:  skl::KernelFactory<SlangKernelID::kernel, skl::CPU>::get()
//           → skl::CPUFunction
//
//   CUDA: skl::KernelFactory<SlangKernelID::kernel, skl::CUDA>::get()
//           → skl::CUDAFunction（.module == nullptr 表示加载失败）
//
// 未提供特化的组合会在编译期报错（incomplete type）
template<auto KernelID, typename Backend>
struct KernelFactory;  // 主模板故意不定义

// ── CUDA backend（需要 CUDAToolkit）─────────────────────────────────────────
#ifdef SKL_HAS_CUDA_DRIVER

struct CUDAFunction
{
    CUmodule   module   = nullptr;
    CUfunction function = nullptr;
};

// 从 PTX 字节码加载内核函数（需提前调用 cuInit / cuCtxCreate）
// entryName 对应 .slang 的入口函数名（如 "kernel"）
// 返回 false 并打印错误信息时 out 保持未初始化状态
bool loadCUDAFunction(const std::vector<uint8_t>& ptx,
                      const char*                  entryName,
                      CUDAFunction&                out);

// 释放 CUmodule（CUfunction 随之失效）
void unloadCUDAFunction(CUDAFunction& fn);

#endif // SKL_HAS_CUDA_DRIVER

} // namespace skl
