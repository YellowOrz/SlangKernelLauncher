#pragma once

#include <cstdint>
#include <functional>
#include <type_traits>
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

// ── Backend tags ──────────────────────────────────────────────────────────────
struct CPU    {};   // CPU 软件执行
struct CUDA   {};   // CUDA Driver API（需要 CUDAToolkit）
struct Vulkan {};   // Vulkan compute

// ── CPU backend ───────────────────────────────────────────────────────────────
using CPUFunction = std::function<void(ComputeVaryingInput*, void*, void*)>;

CPUFunction toCPUFunction(void (*groupFn)(ComputeVaryingInput*, void*, void*));

// ── CUDA backend（需要 CUDAToolkit）─────────────────────────────────────────
#ifdef SKL_HAS_CUDA_DRIVER
struct CUDAFunction
{
    CUmodule   module   = nullptr;
    CUfunction function = nullptr;
};

bool loadCUDAFunction(const std::vector<uint8_t>& ptx,
                      const char*                  entryName,
                      CUDAFunction&                out);

void unloadCUDAFunction(CUDAFunction& fn);
#endif // SKL_HAS_CUDA_DRIVER

// ── KernelFactory 主模板（特化由 kernel_factory.h 生成）──────────────────────
template<auto KernelID, typename Backend>
struct KernelFactory;  // intentionally undefined

// ── KernelRunner - 纯静态 dispatch，无状态 ───────────────────────────────────
// 每个 Backend 特化只提供一个静态 dispatch() 函数，
// 所有运行时状态由 KernelManager 持有并传入。
template<auto KernelID, typename Backend>
struct KernelRunner;   // intentionally undefined

template<auto KernelID>
struct KernelRunner<KernelID, CPU>
{
    static void dispatch(CPUFunction& fn, void* params,
                         uint32_t gx, uint32_t gy = 1, uint32_t gz = 1)
    {
        for (uint32_t z = 0; z < gz; ++z)
        for (uint32_t y = 0; y < gy; ++y)
        for (uint32_t x = 0; x < gx; ++x)
        {
            ComputeVaryingInput vi{};
            vi.startGroupID = { x,     y,     z     };
            vi.endGroupID   = { x + 1, y + 1, z + 1 };
            fn(&vi, nullptr, params);
        }
    }
};

#ifdef SKL_HAS_CUDA_DRIVER
template<auto KernelID>
struct KernelRunner<KernelID, CUDA>
{
    static bool dispatch(const CUDAFunction& fn, void** kernelArgs,
                         uint32_t gx, uint32_t gy = 1, uint32_t gz = 1,
                         uint32_t blockX = 64, uint32_t blockY = 1, uint32_t blockZ = 1)
    {
        CUresult r = cuLaunchKernel(fn.function,
                                     gx, gy, gz,
                                     blockX, blockY, blockZ,
                                     0, nullptr,
                                     kernelArgs, nullptr);
        return r == CUDA_SUCCESS;
    }
};
#endif // SKL_HAS_CUDA_DRIVER

} // namespace skl

// ── Vulkan include（必须在 namespace skl 之外，避免 skl::skl:: 嵌套）─────────
#ifdef SKL_HAS_VULKAN
#include "VulkanBackend.h"
#endif

namespace skl {

// ── Vulkan KernelRunner ───────────────────────────────────────────────────────
#ifdef SKL_HAS_VULKAN
template<auto KernelID>
struct KernelRunner<KernelID, Vulkan>
{
    static bool dispatch(const VulkanContext& ctx, const VulkanFunction& fn,
                         VkDescriptorSet set,
                         uint32_t gx, uint32_t gy = 1, uint32_t gz = 1)
    {
        return dispatchVulkanBound(ctx, fn, set, gx, gy, gz);
    }
};
#endif // SKL_HAS_VULKAN

// ── KernelManager ─────────────────────────────────────────────────────────────
// 统一管理所有 backend 的资源生命周期与 buffer 绑定。
// 构造时自动加载 CPU backend；Vulkan/CUDA 通过 initXxx() 按需初始化。
// buffer 可随时通过 bindXxx() 绑定或重新绑定。
// dispatch<Backend>(gx, gy, gz) 在所有 backend 上签名完全统一。
//
// 用法示例:
//   skl::KernelManager<SlangKernelID::kernel> mgr;
//
//   // CPU
//   mgr.initCPU();
//   mgr.bindCPU(params);
//   mgr.dispatch<skl::CPU>(groupCount);
//
//   // Vulkan（ctx 由外部创建/销毁，多个 KernelManager 可共享同一 ctx）
//   skl::VulkanContext ctx = skl::createVulkanContext();
//   mgr.initVulkan(ctx);
//   mgr.bindVulkan({ &inBuf, &outBuf });
//   mgr.dispatch<skl::Vulkan>(groupCount);
//   mgr.bindVulkan({ &newBuf1, &newBuf2 });   // 重新绑定
//   mgr.dispatch<skl::Vulkan>(groupCount);
//   skl::destroyVulkanContext(ctx);
//
//   // CUDA（需要外部已调用 cuInit / cuCtxCreate）
//   mgr.initCUDA();
//   mgr.bindCUDA({ d_input, d_output });
//   mgr.dispatch<skl::CUDA>(groupCount);
//
template<auto KernelID>
class KernelManager
{
public:
    KernelManager() = default;

    ~KernelManager()
    {
#ifdef SKL_HAS_VULKAN
        if (vkCtx_ && vkPool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(vkCtx_->device, vkPool_, nullptr);
        if (vkCtx_)
            unloadVulkanFunction(vkFn_, vkCtx_->device);
        // vkCtx_ 非拥有，不销毁
#endif
#ifdef SKL_HAS_CUDA_DRIVER
        unloadCUDAFunction(cudaFn_);
#endif
    }

    KernelManager(const KernelManager&)            = delete;
    KernelManager& operator=(const KernelManager&) = delete;

    // ── CPU ──────────────────────────────────────────────────────────────────
    bool initCPU()
    {
        cpuFn_ = KernelFactory<KernelID, CPU>::get();
        return static_cast<bool>(cpuFn_);
    }

    bool cpuReady() const { return static_cast<bool>(cpuFn_); }

    template<typename Params>
    void bindCPU(Params& params) { cpuParams_ = &params; }

    // ── Vulkan ───────────────────────────────────────────────────────────────
#ifdef SKL_HAS_VULKAN
    // 初始化 Vulkan backend，持有外部 ctx 的非拥有指针，ctx 生命周期由调用方保证
    bool initVulkan(const VulkanContext& ctx)
    {
        vkCtx_ = &ctx;
        vkFn_  = KernelFactory<KernelID, Vulkan>::get(ctx);
        return vkFn_.pipeline != VK_NULL_HANDLE;
    }

    // 绑定（或重新绑定）buffer，内部重建 VkDescriptorPool + VkDescriptorSet
    void bindVulkan(const std::vector<VulkanBuffer*>& buffers)
    {
        if (vkPool_ != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(vkCtx_->device, vkPool_, nullptr);
        bindVulkanBuffers(vkFn_, *vkCtx_, buffers, vkPool_, vkSet_);
    }

    bool vulkanReady() const { return vkCtx_ && vkFn_.pipeline != VK_NULL_HANDLE; }
#endif // SKL_HAS_VULKAN

    // ── CUDA ─────────────────────────────────────────────────────────────────
#ifdef SKL_HAS_CUDA_DRIVER
    // 加载 CUDA 模块（需外部已完成 cuInit / cuCtxCreate）
    bool initCUDA()
    {
        cudaFn_ = KernelFactory<KernelID, CUDA>::get();
        return cudaFn_.module != nullptr;
    }

    // 绑定（或重新绑定）显存指针列表，顺序与 kernel 参数顺序一致
    void bindCUDA(std::initializer_list<void*> ptrs)
    {
        cudaDevicePtrs_.assign(ptrs);
        cudaKernelArgs_.resize(cudaDevicePtrs_.size());
        for (size_t i = 0; i < cudaDevicePtrs_.size(); ++i)
            cudaKernelArgs_[i] = &cudaDevicePtrs_[i];
    }

    bool cudaReady() const { return cudaFn_.module != nullptr; }
#endif // SKL_HAS_CUDA_DRIVER

    // ── 统一 dispatch 接口 ────────────────────────────────────────────────────
    // 所有 backend 签名完全一致：dispatch<Backend>(gx, gy, gz)
    template<typename Backend = CPU>
    bool dispatch(uint32_t gx, uint32_t gy = 1, uint32_t gz = 1)
    {
        if constexpr (std::is_same_v<Backend, CPU>)
        {
            KernelRunner<KernelID, CPU>::dispatch(cpuFn_, cpuParams_, gx, gy, gz);
            return true;
        }
#ifdef SKL_HAS_VULKAN
        else if constexpr (std::is_same_v<Backend, Vulkan>)
        {
            return KernelRunner<KernelID, Vulkan>::dispatch(
                *vkCtx_, vkFn_, vkSet_, gx, gy, gz);
        }
#endif
#ifdef SKL_HAS_CUDA_DRIVER
        else if constexpr (std::is_same_v<Backend, CUDA>)
        {
            return KernelRunner<KernelID, CUDA>::dispatch(
                cudaFn_, cudaKernelArgs_.data(), gx, gy, gz);
        }
#endif
        return false;
    }

private:
    // CPU
    CPUFunction cpuFn_;
    void*       cpuParams_ = nullptr;

#ifdef SKL_HAS_VULKAN
    const VulkanContext* vkCtx_  = nullptr;       // 非拥有，生命周期由调用方保证
    VulkanFunction       vkFn_{};
    VkDescriptorPool     vkPool_ = VK_NULL_HANDLE;
    VkDescriptorSet      vkSet_  = VK_NULL_HANDLE;
#endif

#ifdef SKL_HAS_CUDA_DRIVER
    CUDAFunction       cudaFn_{};
    std::vector<void*> cudaDevicePtrs_;
    std::vector<void*> cudaKernelArgs_;
#endif
};

// ── KernelRegistry ────────────────────────────────────────────────────────────
// 编译期注册一组 kernel，每个 kernel 对应一个 KernelManager 实例。
// 通过 get<KernelID>() 获取对应的 manager，initAllXxx() 批量初始化。
//
// 用法示例:
//   using Reg = skl::KernelRegistry<SlangKernelID::kernel /*, 更多 kernel... */>;
//   Reg registry;
//   registry.initAllCPU();
//   registry.initAllVulkan(ctx);
//   registry.get<SlangKernelID::kernel>().bindCPU(params);
//

namespace detail {
// 编译期查找 Target 在 Pack 中的下标（找不到触发 static_assert）
// 使用 if constexpr 避免 ternary 两个分支同时实例化的问题
template<size_t I, auto Target, auto First, auto... Rest>
constexpr size_t indexOfHelper()
{
    if constexpr (Target == First)
        return I;
    else if constexpr (sizeof...(Rest) > 0)
        return indexOfHelper<I + 1, Target, Rest...>();
    else {
        static_assert(sizeof...(Rest) + 1 == 0,
                      "KernelID not found in KernelRegistry");
        return I; // unreachable
    }
}

template<auto Target, auto... Pack>
struct IndexOf {
    static constexpr size_t value = indexOfHelper<0, Target, Pack...>();
};
} // namespace detail

template<auto... KernelIDs>
class KernelRegistry
{
public:
    // 获取指定 kernel 的 KernelManager
    template<auto KernelID>
    KernelManager<KernelID>& get()
    {
        return std::get<detail::IndexOf<KernelID, KernelIDs...>::value>(managers_);
    }

    template<auto KernelID>
    const KernelManager<KernelID>& get() const
    {
        return std::get<detail::IndexOf<KernelID, KernelIDs...>::value>(managers_);
    }

    // 批量初始化所有 kernel 的 CPU backend
    void initAllCPU()
    {
        std::apply([](auto&... mgrs) { (mgrs.initCPU(), ...); }, managers_);
    }

#ifdef SKL_HAS_VULKAN
    // 批量初始化所有 kernel 的 Vulkan backend（共享同一 ctx）
    void initAllVulkan(const VulkanContext& ctx)
    {
        std::apply([&ctx](auto&... mgrs) { (mgrs.initVulkan(ctx), ...); }, managers_);
    }
#endif

#ifdef SKL_HAS_CUDA_DRIVER
    // 批量初始化所有 kernel 的 CUDA backend
    void initAllCUDA()
    {
        std::apply([](auto&... mgrs) { (mgrs.initCUDA(), ...); }, managers_);
    }
#endif

private:
    std::tuple<KernelManager<KernelIDs>...> managers_;
};

// ── KernelLauncher ────────────────────────────────────────────────────────────
// 纯静态入口：从 KernelRegistry 中查找对应 KernelManager 并 dispatch。
// 使用方无需直接持有 KernelManager，只需传入 registry 引用。
//
// 用法示例:
//   skl::KernelLauncher<SlangKernelID::kernel, skl::CPU>::dispatch(registry, gx);
//   skl::KernelLauncher<SlangKernelID::kernel, skl::Vulkan>::dispatch(registry, gx);
//
template<auto KernelID, typename Backend>
struct KernelLauncher
{
    template<auto... KernelIDs>
    static bool dispatch(KernelRegistry<KernelIDs...>& registry,
                         uint32_t gx, uint32_t gy = 1, uint32_t gz = 1)
    {
        return registry.template get<KernelID>().template dispatch<Backend>(gx, gy, gz);
    }
};

} // namespace skl
