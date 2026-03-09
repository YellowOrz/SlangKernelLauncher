#pragma once

#ifdef SKL_HAS_VULKAN

#include <vulkan/vulkan.h>
#include <cstdint>
#include <vector>

namespace skl {

// ── Context ───────────────────────────────────────────────────────────────────
// 持有一个完整的 Vulkan compute 环境。
// 可由 createVulkanContext() 自动创建，或由调用方手动填充（嵌入已有引擎时）。
// 生命周期必须长于所有 VulkanFunction / VulkanBuffer 对象。
struct VulkanContext
{
    VkInstance       instance           = VK_NULL_HANDLE;  // 由 createVulkanContext 创建
    VkPhysicalDevice physicalDevice     = VK_NULL_HANDLE;
    VkDevice         device             = VK_NULL_HANDLE;
    VkQueue          computeQueue       = VK_NULL_HANDLE;
    uint32_t         computeQueueFamily = 0;
    VkCommandPool    commandPool        = VK_NULL_HANDLE;
    // 预分配的 dispatch command buffer，跨调用复用（reset 后重录制），零分配开销
    VkCommandBuffer  dispatchCmd        = VK_NULL_HANDLE;
};

// 自动创建一个只含 compute queue 的最小 Vulkan Context。
// preferDiscreteGpu: 优先选择独立显卡（false 则使用第一个可用设备）
// 失败时返回 .device == VK_NULL_HANDLE 的零值结构体
VulkanContext createVulkanContext(bool preferDiscreteGpu = true);

// 销毁 VulkanContext 持有的所有 Vulkan 对象（instance/device/commandPool）
void destroyVulkanContext(VulkanContext& ctx);

// ── 已编译的 Vulkan 计算内核 ─────────────────────────────────────────────────
// 由 loadVulkanFunction() 填充；pipeline == VK_NULL_HANDLE 表示加载失败。
struct VulkanFunction
{
    VkShaderModule        shaderModule        = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout      = VK_NULL_HANDLE;
    VkPipeline            pipeline            = VK_NULL_HANDLE;
    uint32_t              bindingCount        = 0;
};

// ── GPU 存储缓冲区 ────────────────────────────────────────────────────────────
struct VulkanBuffer
{
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

// ── 加载 / 卸载内核 ───────────────────────────────────────────────────────────

// 从 SPIR-V 字节码加载计算内核，创建 shader module、descriptor set layout、pipeline。
// bindingCount: 着色器使用的 storage buffer 数量（set=0, binding=0,1,...,N-1）
// entryPoint:   SPIR-V 入口函数名（Slang 编译为 SPIR-V 时固定为 "main"）
// 失败时返回 .pipeline == VK_NULL_HANDLE 的零值结构体
VulkanFunction loadVulkanFunction(const std::vector<uint8_t>& spirv,
                                   const VulkanContext&         ctx,
                                   uint32_t                     bindingCount,
                                   const char*                  entryPoint = "main");

// 销毁 VulkanFunction 持有的所有 Vulkan 对象，并将各字段清零
void unloadVulkanFunction(VulkanFunction& fn, VkDevice device);

// ── 缓冲区管理 ────────────────────────────────────────────────────────────────

// 检测物理设备是否支持统一内存（DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT 同时存在）。
// 集成显卡 / Apple Silicon 返回 true；纯离散显卡（无 ReBAR）返回 false。
bool isUnifiedMemory(VkPhysicalDevice physDev);

// 创建 buffer
// usage:    buffer 用途位掩码，默认 STORAGE_BUFFER
// memProps: 内存属性，默认 HOST_VISIBLE | HOST_COHERENT
// 失败时返回 .buffer == VK_NULL_HANDLE 的零值结构体
VulkanBuffer createVulkanBuffer(
    const VulkanContext&  ctx,
    VkDeviceSize          size,
    VkBufferUsageFlags    usage    = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VkMemoryPropertyFlags memProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                   | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

void destroyVulkanBuffer(VkDevice device, VulkanBuffer& buf);

// 将缓冲区内存映射到 CPU 地址空间（仅 HOST_VISIBLE 缓冲区可用）
// 返回 nullptr 表示映射失败
void* mapVulkanBuffer  (VkDevice device, const VulkanBuffer& buf);
void  unmapVulkanBuffer(VkDevice device, const VulkanBuffer& buf);

// ── Dispatch ─────────────────────────────────────────────────────────────────

// ① 一次性绑定：创建持久 descriptor pool + descriptor set，写入 buffer 信息。
//    outPool / outSet 由调用方持有，析构时需 vkDestroyDescriptorPool(device, outPool, nullptr)。
//    失败时 outPool/outSet 保持 VK_NULL_HANDLE，返回 false。
bool bindVulkanBuffers(const VulkanFunction&             fn,
                       const VulkanContext&              ctx,
                       const std::vector<VulkanBuffer*>& buffers,
                       VkDescriptorPool&                 outPool,
                       VkDescriptorSet&                  outSet);

// ② 多次 dispatch：使用已绑定的 descriptor set，不重新分配资源。
//    适合在同一组 buffer 上反复调用（仅录制命令 → 提交 → 等待）。
bool dispatchVulkanBound(const VulkanContext&  ctx,
                          const VulkanFunction& fn,
                          VkDescriptorSet       descriptorSet,
                          uint32_t              gx,
                          uint32_t              gy = 1,
                          uint32_t              gz = 1);

// ③ 一次性 dispatch（bind + dispatch + 释放）：便于偶尔调用、不在意资源复用的场景。
bool dispatchVulkan(const VulkanContext&              ctx,
                    const VulkanFunction&             fn,
                    const std::vector<VulkanBuffer*>& buffers,
                    uint32_t                          groupCountX,
                    uint32_t                          groupCountY = 1,
                    uint32_t                          groupCountZ = 1);

} // namespace skl

#endif // SKL_HAS_VULKAN
