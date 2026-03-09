#pragma once

// IMemBuffer<T>     — 抽象基类，定义跨 backend 的统一传输接口。
// MemBuffer<T, B>   — 各 backend 的具体实现，继承 IMemBuffer<T>。
//
// 统一接口（纯虚）：
//   同步：upload / download          — 调用返回时数据传输已完成
//   异步：uploadAsync / downloadAsync — 调用立即返回，传输在后台进行
//   同步点：sync()                   — 等待所有异步操作完成
//
//   CPU / 当前 Vulkan（host-visible mapped memory）：
//     async 版本等同于同步版本，sync() 为空操作。
//   CUDA：
//     async 版本使用 cuMemcpyHtoDAsync / cuMemcpyDtoHAsync + 内部 CUstream，
//     sync() 调用 cuStreamSynchronize。
//
// Backend 专属访问器（非虚）：
//   CPU    → data()          返回 T*
//   Vulkan → vulkanBuffer()  返回 VulkanBuffer*
//   CUDA   → devicePtr()     返回 void*
//
// 生命周期约束：
//   MemBuffer<T, Vulkan> 持有 VulkanContext 的非拥有指针，
//   必须在 VulkanContext 销毁前析构（与 KernelManager 相同约定）。

#include "SlangKernelLauncher.h"   // CPU / Vulkan / CUDA tags，以及 VulkanBackend.h

#include <algorithm>
#include <cstring>
#include <vector>

namespace skl {

// ── IMemBuffer<T>：抽象基类 ───────────────────────────────────────────────────
template<typename T>
class IMemBuffer
{
public:
    virtual ~IMemBuffer() = default;

    virtual size_t size() const = 0;

    // ── 同步传输：返回时数据已就绪 ───────────────────────────────────────────
    virtual void upload(const T* src, size_t n)    = 0;
    virtual void upload(const std::vector<T>& v)   = 0;

    virtual void download(T* dst, size_t n) const  = 0;
    virtual void download(std::vector<T>& v) const = 0;

    // ── 异步传输：立即返回，后台执行 ─────────────────────────────────────────
    // CPU / host-visible Vulkan 中等同于同步版本。
    virtual void uploadAsync(const T* src, size_t n)    = 0;
    virtual void uploadAsync(const std::vector<T>& v)   = 0;

    virtual void downloadAsync(T* dst, size_t n) const  = 0;
    virtual void downloadAsync(std::vector<T>& v) const = 0;

    // ── 同步点：等待该 buffer 上所有异步操作完成 ──────────────────────────────
    // CPU / host-visible Vulkan 中为空操作。
    virtual void sync() const = 0;
};

// ── Primary template（未特化时报编译错误）────────────────────────────────────
template<typename T, typename Backend>
class MemBuffer;

// ── CPU specialization ───────────────────────────────────────────────────────
// 所有传输均为内存拷贝，无真正的异步概念；async 版本直接委托同步版本，sync() 为空操作。
template<typename T>
class MemBuffer<T, CPU> : public IMemBuffer<T>
{
public:
    explicit MemBuffer(size_t count) : data_(count) {}

    size_t   size()  const override { return data_.size(); }
    T*       data()                 { return data_.data(); }
    const T* data()  const          { return data_.data(); }

    // 同步
    void upload(const T* src, size_t n)      override { std::copy(src, src + n, data_.begin()); }
    void upload(const std::vector<T>& v)     override { upload(v.data(), v.size()); }

    void download(T* dst, size_t n)    const override { std::copy(data_.cbegin(), data_.cbegin() + n, dst); }
    void download(std::vector<T>& v)   const override { v.resize(data_.size()); download(v.data(), data_.size()); }

    // 异步（等同同步）
    void uploadAsync(const T* src, size_t n)    override { upload(src, n); }
    void uploadAsync(const std::vector<T>& v)   override { upload(v); }

    void downloadAsync(T* dst, size_t n) const  override { download(dst, n); }
    void downloadAsync(std::vector<T>& v) const override { download(v); }

    void sync() const override {}   // 无异步操作，空实现

private:
    std::vector<T> data_;
};

} // namespace skl

// ── Vulkan specialization ─────────────────────────────────────────────────────
#ifdef SKL_HAS_VULKAN

namespace skl {

// 构造时自动检测显卡类型，选择最优传输策略：
//
// ┌─────────────────┬──────────────────────────────────────────────────────┐
// │ 统一内存         │ 单 buffer：DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT │
// │（集成 / ReBAR）  │ upload/download = map + memcpy；async ≡ sync          │
// ├─────────────────┼──────────────────────────────────────────────────────┤
// │ 离散显卡        │ deviceBuf_（DEVICE_LOCAL）+ stagingBuf_（HOST_VISIBLE）  │
// │                 │ uploadAsync: memcpy→staging, vkCmdCopyBuffer + fence   │
// │                 │ downloadAsync: vkCmdCopyBuffer + fence，sync() memcpy   │
// └─────────────────┴──────────────────────────────────────────────────────┘
//
// vulkanBuffer() 始终返回 deviceBuf_（GPU 端 buffer），供 bindVulkan() 使用。
template<typename T>
class MemBuffer<T, Vulkan> : public IMemBuffer<T>
{
public:
    // ctx 的生命周期必须超过 MemBuffer 自身
    MemBuffer(const VulkanContext& ctx, size_t count)
        : ctx_(&ctx), count_(count)
        , unified_(isUnifiedMemory(ctx.physicalDevice))
    {
        const VkDeviceSize bytes = count * sizeof(T);

        if (unified_) {
            // 统一内存：单 buffer，GPU 和 CPU 均可直接访问
            deviceBuf_ = createVulkanBuffer(ctx, bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        } else {
            // 离散显卡：GPU 专用 buffer + CPU staging buffer
            deviceBuf_ = createVulkanBuffer(ctx, bytes,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT    |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            stagingBuf_ = createVulkanBuffer(ctx, bytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            // 预分配用于数据传输的 command buffer（可复用）
            VkCommandBufferAllocateInfo cmdAI{};
            cmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cmdAI.commandPool        = ctx.commandPool;
            cmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cmdAI.commandBufferCount = 1;
            vkAllocateCommandBuffers(ctx.device, &cmdAI, &transferCmd_);

            // fence 用于追踪 GPU 传输完成（初始 unsignaled）
            VkFenceCreateInfo fenceCI{};
            fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            vkCreateFence(ctx.device, &fenceCI, nullptr, &fence_);
        }
    }

    ~MemBuffer()
    {
        if (!ctx_) return;
        if (!unified_) {
            if (fence_) {
                vkWaitForFences(ctx_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
                vkDestroyFence(ctx_->device, fence_, nullptr);
            }
            if (transferCmd_)
                vkFreeCommandBuffers(ctx_->device, ctx_->commandPool, 1, &transferCmd_);
            destroyVulkanBuffer(ctx_->device, stagingBuf_);
        }
        destroyVulkanBuffer(ctx_->device, deviceBuf_);
    }

    MemBuffer(const MemBuffer&)            = delete;
    MemBuffer& operator=(const MemBuffer&) = delete;
    MemBuffer(MemBuffer&& o) noexcept
        : ctx_(o.ctx_), count_(o.count_), unified_(o.unified_)
        , deviceBuf_(o.deviceBuf_), stagingBuf_(o.stagingBuf_)
        , transferCmd_(o.transferCmd_), fence_(o.fence_)
        , fenceSubmitted_(o.fenceSubmitted_)
        , pendingDst_(o.pendingDst_), pendingN_(o.pendingN_)
    {
        o.ctx_          = nullptr;
        o.deviceBuf_    = {};
        o.stagingBuf_   = {};
        o.transferCmd_  = VK_NULL_HANDLE;
        o.fence_        = VK_NULL_HANDLE;
        o.fenceSubmitted_ = false;
        o.pendingDst_   = nullptr;
    }

    size_t        size()          const override { return count_; }
    // 始终返回 GPU 端 buffer，供 KernelManager::bindVulkan() 使用
    VulkanBuffer* vulkanBuffer()                 { return &deviceBuf_; }

    // ── 同步传输（= async + sync）────────────────────────────────────────────
    void upload(const T* src, size_t n) override
    {
        uploadAsync(src, n);
        sync();
    }
    void upload(const std::vector<T>& v) override { upload(v.data(), v.size()); }

    void download(T* dst, size_t n) const override
    {
        downloadAsync(dst, n);
        sync();
    }
    void download(std::vector<T>& v) const override { v.resize(count_); download(v.data(), count_); }

    // ── 异步传输 ─────────────────────────────────────────────────────────────
    // 统一内存：等同同步（map 本身已即时完成）
    // 离散显卡：CPU 侧 memcpy 到 staging，再提交 vkCmdCopyBuffer + fence，立即返回
    void uploadAsync(const T* src, size_t n) override
    {
        if (unified_) {
            void* p = mapVulkanBuffer(ctx_->device, deviceBuf_);
            memcpy(p, src, n * sizeof(T));
            unmapVulkanBuffer(ctx_->device, deviceBuf_);
        } else {
            waitPrevious();   // 确保 staging / fence 空闲后再用
            void* p = mapVulkanBuffer(ctx_->device, stagingBuf_);
            memcpy(p, src, n * sizeof(T));
            unmapVulkanBuffer(ctx_->device, stagingBuf_);
            submitCopy(stagingBuf_.buffer, deviceBuf_.buffer, n * sizeof(T));
        }
    }
    void uploadAsync(const std::vector<T>& v) override { uploadAsync(v.data(), v.size()); }

    // 统一内存：等同同步
    // 离散显卡：提交 vkCmdCopyBuffer（device→staging）+ fence，记录目标指针，立即返回；
    //           真正的 memcpy（staging→dst）延迟到 sync() 完成
    void downloadAsync(T* dst, size_t n) const override
    {
        if (unified_) {
            void* p = mapVulkanBuffer(ctx_->device, deviceBuf_);
            memcpy(dst, p, n * sizeof(T));
            unmapVulkanBuffer(ctx_->device, deviceBuf_);
        } else {
            waitPrevious();
            pendingDst_ = dst;
            pendingN_   = n;
            submitCopy(deviceBuf_.buffer, stagingBuf_.buffer, n * sizeof(T));
        }
    }
    void downloadAsync(std::vector<T>& v) const override
    {
        v.resize(count_);
        downloadAsync(v.data(), count_);
    }

    // ── 同步点 ───────────────────────────────────────────────────────────────
    // 等待 fence 信号，若有 pending download 则将 staging buffer 内容拷贝到目标
    void sync() const override
    {
        if (unified_ || !fenceSubmitted_) return;
        vkWaitForFences(ctx_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx_->device, 1, &fence_);
        fenceSubmitted_ = false;
        if (pendingDst_) {
            void* p = mapVulkanBuffer(ctx_->device, stagingBuf_);
            memcpy(pendingDst_, p, pendingN_ * sizeof(T));
            unmapVulkanBuffer(ctx_->device, stagingBuf_);
            pendingDst_ = nullptr;
        }
    }

private:
    // 若上次提交尚未完成，等待后重置（防止 staging / fence 被并发占用）
    void waitPrevious() const
    {
        if (!fenceSubmitted_) return;
        vkWaitForFences(ctx_->device, 1, &fence_, VK_TRUE, UINT64_MAX);
        vkResetFences(ctx_->device, 1, &fence_);
        fenceSubmitted_ = false;
        pendingDst_ = nullptr;   // 取消未完成的 download（与新 upload 冲突）
    }

    // 录制并提交一次 buffer 拷贝命令，attach fence
    void submitCopy(VkBuffer src, VkBuffer dst, VkDeviceSize bytes) const
    {
        vkResetCommandBuffer(transferCmd_, 0);

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(transferCmd_, &bi);

        VkBufferCopy region{ 0, 0, bytes };
        vkCmdCopyBuffer(transferCmd_, src, dst, 1, &region);

        vkEndCommandBuffer(transferCmd_);

        VkSubmitInfo si{};
        si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers    = &transferCmd_;
        vkQueueSubmit(ctx_->computeQueue, 1, &si, fence_);
        fenceSubmitted_ = true;
    }

    const VulkanContext* ctx_    = nullptr;
    size_t               count_  = 0;
    bool                 unified_ = true;
    VulkanBuffer         deviceBuf_{};
    VulkanBuffer         stagingBuf_{};              // 离散显卡专用

    mutable VkCommandBuffer transferCmd_    = VK_NULL_HANDLE;  // 离散显卡专用
    mutable VkFence         fence_          = VK_NULL_HANDLE;  // 离散显卡专用
    mutable bool            fenceSubmitted_ = false;
    mutable T*              pendingDst_     = nullptr;         // download 目标
    mutable size_t          pendingN_       = 0;
};

} // namespace skl

#endif // SKL_HAS_VULKAN

// ── CUDA specialization ───────────────────────────────────────────────────────
#ifdef SKL_HAS_CUDA_DRIVER

namespace skl {

// 内部持有一个专用 CUstream：
//   uploadAsync / downloadAsync  → cuMemcpyHtoDAsync / cuMemcpyDtoHAsync（真正异步）
//   upload / download            → async + sync()（等效同步）
//   sync()                       → cuStreamSynchronize
template<typename T>
class MemBuffer<T, CUDA> : public IMemBuffer<T>
{
public:
    explicit MemBuffer(size_t count) : count_(count)
    {
        cuMemAlloc(&ptr_, count * sizeof(T));
        cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING);
    }

    ~MemBuffer()
    {
        if (stream_) { cuStreamSynchronize(stream_); cuStreamDestroy(stream_); }
        if (ptr_)    cuMemFree(ptr_);
    }

    MemBuffer(const MemBuffer&)            = delete;
    MemBuffer& operator=(const MemBuffer&) = delete;
    MemBuffer(MemBuffer&& o) noexcept
        : ptr_(o.ptr_), count_(o.count_), stream_(o.stream_)
    { o.ptr_ = 0; o.stream_ = nullptr; }

    size_t size()      const override { return count_; }
    // 返回 void* 用于 KernelManager::bindCUDA({ buf.devicePtr(), ... })
    void*  devicePtr()               { return reinterpret_cast<void*>(ptr_); }

    // 异步（真正异步，立即返回）
    void uploadAsync(const T* src, size_t n) override
    {
        cuMemcpyHtoDAsync(ptr_, src, n * sizeof(T), stream_);
    }
    void uploadAsync(const std::vector<T>& v) override { uploadAsync(v.data(), v.size()); }

    void downloadAsync(T* dst, size_t n) const override
    {
        cuMemcpyDtoHAsync(dst, ptr_, n * sizeof(T), stream_);
    }
    void downloadAsync(std::vector<T>& v) const override
    {
        v.resize(count_);
        downloadAsync(v.data(), count_);
    }

    // 同步点
    void sync() const override { cuStreamSynchronize(stream_); }

    // 同步传输（= async + sync）
    void upload(const T* src, size_t n) override
    {
        uploadAsync(src, n);
        sync();
    }
    void upload(const std::vector<T>& v)   override { upload(v.data(), v.size()); }

    void download(T* dst, size_t n) const override
    {
        downloadAsync(dst, n);
        sync();
    }
    void download(std::vector<T>& v) const override { v.resize(count_); download(v.data(), count_); }

private:
    CUdeviceptr ptr_    = 0;
    size_t      count_  = 0;
    CUstream    stream_ = nullptr;
};

} // namespace skl

#endif // SKL_HAS_CUDA_DRIVER
