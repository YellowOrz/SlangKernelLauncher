#include "VulkanBackend.h"

#ifdef SKL_HAS_VULKAN

#include <cstdio>
#include <cstring>
#include <vector>

namespace skl {

// ── 内部辅助 ──────────────────────────────────────────────────────────────────

static uint32_t findMemoryType(VkPhysicalDevice       physDev,
                                uint32_t               typeBits,
                                VkMemoryPropertyFlags  props)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props)
            return i;
    return UINT32_MAX;
}

static uint32_t findComputeQueueFamily(VkPhysicalDevice phys)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());
    for (uint32_t i = 0; i < count; ++i)
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            return i;
    return UINT32_MAX;
}

// ── Context 创建 / 销毁 ───────────────────────────────────────────────────────

VulkanContext createVulkanContext(bool preferDiscreteGpu)
{
    VulkanContext ctx{};

    // 1. Instance
    VkApplicationInfo appInfo{};
    appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "SlangKernelLauncher";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instCI{};
    instCI.sType            = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instCI, nullptr, &ctx.instance) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateInstance failed\n");
        return ctx;
    }

    // 2. Physical device
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, nullptr);
    if (devCount == 0)
    {
        fprintf(stderr, "[SKL] No Vulkan physical devices found\n");
        vkDestroyInstance(ctx.instance, nullptr);
        ctx.instance = VK_NULL_HANDLE;
        return ctx;
    }
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(ctx.instance, &devCount, devs.data());

    if (preferDiscreteGpu)
        for (auto d : devs)
        {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(d, &p);
            if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
                { ctx.physicalDevice = d; break; }
        }
    if (ctx.physicalDevice == VK_NULL_HANDLE)
        ctx.physicalDevice = devs[0];

    {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(ctx.physicalDevice, &p);
        fprintf(stdout, "[SKL] Vulkan device: %s\n", p.deviceName);
    }

    // 3. Compute queue family
    ctx.computeQueueFamily = findComputeQueueFamily(ctx.physicalDevice);
    if (ctx.computeQueueFamily == UINT32_MAX)
    {
        fprintf(stderr, "[SKL] No compute queue family found\n");
        vkDestroyInstance(ctx.instance, nullptr);
        ctx = {};
        return ctx;
    }

    // 4. Logical device
    float queuePrio = 1.0f;
    VkDeviceQueueCreateInfo qCI{};
    qCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCI.queueFamilyIndex = ctx.computeQueueFamily;
    qCI.queueCount       = 1;
    qCI.pQueuePriorities = &queuePrio;

    VkDeviceCreateInfo devCI{};
    devCI.sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos    = &qCI;

    if (vkCreateDevice(ctx.physicalDevice, &devCI, nullptr, &ctx.device) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateDevice failed\n");
        vkDestroyInstance(ctx.instance, nullptr);
        ctx = {};
        return ctx;
    }

    vkGetDeviceQueue(ctx.device, ctx.computeQueueFamily, 0, &ctx.computeQueue);

    // 5. Command pool
    VkCommandPoolCreateInfo cpCI{};
    cpCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.queueFamilyIndex = ctx.computeQueueFamily;
    cpCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(ctx.device, &cpCI, nullptr, &ctx.commandPool) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateCommandPool failed\n");
        vkDestroyDevice(ctx.device, nullptr);
        vkDestroyInstance(ctx.instance, nullptr);
        ctx = {};
        return ctx;
    }

    // 6. 预分配 dispatch command buffer（跨 dispatch 调用复用，避免每次 alloc/free）
    VkCommandBufferAllocateInfo dispatchCmdAI{};
    dispatchCmdAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    dispatchCmdAI.commandPool        = ctx.commandPool;
    dispatchCmdAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    dispatchCmdAI.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(ctx.device, &dispatchCmdAI, &ctx.dispatchCmd) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkAllocateCommandBuffers (dispatchCmd) failed\n");
        vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr);
        vkDestroyDevice(ctx.device, nullptr);
        vkDestroyInstance(ctx.instance, nullptr);
        ctx = {};
        return ctx;
    }

    return ctx;
}

void destroyVulkanContext(VulkanContext& ctx)
{
    // dispatchCmd 随 commandPool 一起释放（vkDestroyCommandPool 隐式释放所有子 buffer），
    // 此处仅清零指针，无需单独 vkFreeCommandBuffers。
    ctx.dispatchCmd = VK_NULL_HANDLE;
    if (ctx.commandPool) { vkDestroyCommandPool(ctx.device, ctx.commandPool, nullptr); ctx.commandPool = VK_NULL_HANDLE; }
    if (ctx.device)      { vkDestroyDevice(ctx.device, nullptr);                       ctx.device      = VK_NULL_HANDLE; }
    if (ctx.instance)    { vkDestroyInstance(ctx.instance, nullptr);                   ctx.instance    = VK_NULL_HANDLE; }
    ctx = {};
}


// ── loadVulkanFunction ────────────────────────────────────────────────────────

VulkanFunction loadVulkanFunction(const std::vector<uint8_t>& spirv,
                                   const VulkanContext&         ctx,
                                   uint32_t                     bindingCount,
                                   const char*                  entryPoint)
{
    VulkanFunction fn{};

    // 1. Shader module
    VkShaderModuleCreateInfo smCI{};
    smCI.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smCI.codeSize = spirv.size();
    smCI.pCode    = reinterpret_cast<const uint32_t*>(spirv.data());

    if (vkCreateShaderModule(ctx.device, &smCI, nullptr, &fn.shaderModule) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateShaderModule failed\n");
        return fn;
    }

    // 2. Descriptor set layout（所有 binding 均为 STORAGE_BUFFER，set=0）
    std::vector<VkDescriptorSetLayoutBinding> bindings(bindingCount);
    for (uint32_t i = 0; i < bindingCount; ++i)
    {
        bindings[i].binding            = i;
        bindings[i].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount    = 1;
        bindings[i].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo dslCI{};
    dslCI.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = bindingCount;
    dslCI.pBindings    = bindings.empty() ? nullptr : bindings.data();

    if (vkCreateDescriptorSetLayout(ctx.device, &dslCI, nullptr, &fn.descriptorSetLayout)
        != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateDescriptorSetLayout failed\n");
        vkDestroyShaderModule(ctx.device, fn.shaderModule, nullptr);
        fn.shaderModule = VK_NULL_HANDLE;
        return fn;
    }

    // 3. Pipeline layout
    VkPipelineLayoutCreateInfo plCI{};
    plCI.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1;
    plCI.pSetLayouts    = &fn.descriptorSetLayout;

    if (vkCreatePipelineLayout(ctx.device, &plCI, nullptr, &fn.pipelineLayout) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreatePipelineLayout failed\n");
        vkDestroyDescriptorSetLayout(ctx.device, fn.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(ctx.device, fn.shaderModule, nullptr);
        fn.descriptorSetLayout = VK_NULL_HANDLE;
        fn.shaderModule        = VK_NULL_HANDLE;
        return fn;
    }

    // 4. Compute pipeline
    VkComputePipelineCreateInfo pipelineCI{};
    pipelineCI.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCI.stage.sType        = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCI.stage.stage        = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCI.stage.module       = fn.shaderModule;
    pipelineCI.stage.pName        = entryPoint;
    pipelineCI.layout             = fn.pipelineLayout;

    if (vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipelineCI,
                                  nullptr, &fn.pipeline) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateComputePipelines failed\n");
        vkDestroyPipelineLayout(ctx.device, fn.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(ctx.device, fn.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(ctx.device, fn.shaderModule, nullptr);
        fn.pipelineLayout      = VK_NULL_HANDLE;
        fn.descriptorSetLayout = VK_NULL_HANDLE;
        fn.shaderModule        = VK_NULL_HANDLE;
        return fn;
    }

    fn.bindingCount = bindingCount;
    return fn;
}

void unloadVulkanFunction(VulkanFunction& fn, VkDevice device)
{
    if (fn.pipeline)            { vkDestroyPipeline(device, fn.pipeline, nullptr);              fn.pipeline            = VK_NULL_HANDLE; }
    if (fn.pipelineLayout)      { vkDestroyPipelineLayout(device, fn.pipelineLayout, nullptr);  fn.pipelineLayout      = VK_NULL_HANDLE; }
    if (fn.descriptorSetLayout) { vkDestroyDescriptorSetLayout(device, fn.descriptorSetLayout, nullptr); fn.descriptorSetLayout = VK_NULL_HANDLE; }
    if (fn.shaderModule)        { vkDestroyShaderModule(device, fn.shaderModule, nullptr);      fn.shaderModule        = VK_NULL_HANDLE; }
    fn.bindingCount = 0;
}

// ── Buffer ────────────────────────────────────────────────────────────────────

bool isUnifiedMemory(VkPhysicalDevice physDev)
{
    constexpr VkMemoryPropertyFlags kUnified =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT  |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT  |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physDev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((mp.memoryTypes[i].propertyFlags & kUnified) == kUnified)
            return true;
    return false;
}

VulkanBuffer createVulkanBuffer(const VulkanContext&  ctx,
                                 VkDeviceSize          size,
                                 VkBufferUsageFlags    usage,
                                 VkMemoryPropertyFlags memProps)
{
    VulkanBuffer buf{};
    buf.size = size;

    VkBufferCreateInfo bCI{};
    bCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bCI.size        = size;
    bCI.usage       = usage;
    bCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(ctx.device, &bCI, nullptr, &buf.buffer) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateBuffer failed\n");
        return buf;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(ctx.device, buf.buffer, &memReq);

    uint32_t memTypeIdx = findMemoryType(ctx.physicalDevice, memReq.memoryTypeBits, memProps);
    if (memTypeIdx == UINT32_MAX)
    {
        fprintf(stderr, "[SKL] No suitable memory type for VulkanBuffer\n");
        vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        return buf;
    }

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize  = memReq.size;
    allocInfo.memoryTypeIndex = memTypeIdx;

    if (vkAllocateMemory(ctx.device, &allocInfo, nullptr, &buf.memory) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkAllocateMemory failed\n");
        vkDestroyBuffer(ctx.device, buf.buffer, nullptr);
        buf.buffer = VK_NULL_HANDLE;
        return buf;
    }

    vkBindBufferMemory(ctx.device, buf.buffer, buf.memory, 0);
    return buf;
}

void destroyVulkanBuffer(VkDevice device, VulkanBuffer& buf)
{
    if (buf.memory) { vkFreeMemory(device, buf.memory, nullptr);   buf.memory = VK_NULL_HANDLE; }
    if (buf.buffer) { vkDestroyBuffer(device, buf.buffer, nullptr); buf.buffer = VK_NULL_HANDLE; }
    buf.size = 0;
}

void* mapVulkanBuffer(VkDevice device, const VulkanBuffer& buf)
{
    void* ptr = nullptr;
    if (vkMapMemory(device, buf.memory, 0, buf.size, 0, &ptr) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkMapMemory failed\n");
        return nullptr;
    }
    return ptr;
}

void unmapVulkanBuffer(VkDevice device, const VulkanBuffer& buf)
{
    vkUnmapMemory(device, buf.memory);
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

bool bindVulkanBuffers(const VulkanFunction&             fn,
                       const VulkanContext&              ctx,
                       const std::vector<VulkanBuffer*>& buffers,
                       VkDescriptorPool&                 outPool,
                       VkDescriptorSet&                  outSet)
{
    if (fn.pipeline == VK_NULL_HANDLE)
    {
        fprintf(stderr, "[SKL] bindVulkanBuffers: invalid VulkanFunction\n");
        return false;
    }
    if (buffers.size() != fn.bindingCount)
    {
        fprintf(stderr, "[SKL] bindVulkanBuffers: expected %u buffers, got %zu\n",
                fn.bindingCount, buffers.size());
        return false;
    }

    // Descriptor pool（生命周期由调用方管理，可跨多次 dispatch 复用）
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = fn.bindingCount ? fn.bindingCount : 1;

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;

    if (vkCreateDescriptorPool(ctx.device, &poolCI, nullptr, &outPool) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkCreateDescriptorPool failed\n");
        return false;
    }

    // Allocate + update descriptor set
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = outPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &fn.descriptorSetLayout;

    if (vkAllocateDescriptorSets(ctx.device, &allocInfo, &outSet) != VK_SUCCESS)
    {
        fprintf(stderr, "[SKL] vkAllocateDescriptorSets failed\n");
        vkDestroyDescriptorPool(ctx.device, outPool, nullptr);
        outPool = VK_NULL_HANDLE;
        return false;
    }

    std::vector<VkDescriptorBufferInfo> bufInfos(fn.bindingCount);
    std::vector<VkWriteDescriptorSet>   writes(fn.bindingCount);
    for (uint32_t i = 0; i < fn.bindingCount; ++i)
    {
        bufInfos[i].buffer = buffers[i]->buffer;
        bufInfos[i].offset = 0;
        bufInfos[i].range  = buffers[i]->size;

        writes[i]                 = {};
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet          = outSet;
        writes[i].dstBinding      = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &bufInfos[i];
    }
    vkUpdateDescriptorSets(ctx.device, fn.bindingCount, writes.data(), 0, nullptr);

    return true;
}

bool dispatchVulkanBound(const VulkanContext&  ctx,
                          const VulkanFunction& fn,
                          VkDescriptorSet       descriptorSet,
                          uint32_t              gx,
                          uint32_t              gy,
                          uint32_t              gz)
{
    // 复用 ctx.dispatchCmd：reset 后重新录制，无 alloc/free 开销
    vkResetCommandBuffer(ctx.dispatchCmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(ctx.dispatchCmd, &beginInfo);

    vkCmdBindPipeline(ctx.dispatchCmd, VK_PIPELINE_BIND_POINT_COMPUTE, fn.pipeline);
    vkCmdBindDescriptorSets(ctx.dispatchCmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             fn.pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(ctx.dispatchCmd, gx, gy, gz);

    vkEndCommandBuffer(ctx.dispatchCmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &ctx.dispatchCmd;

    vkQueueSubmit(ctx.computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(ctx.computeQueue);

    return true;
}

bool dispatchVulkanAsync(const VulkanContext&  ctx,
                          const VulkanFunction& fn,
                          VkDescriptorSet       descriptorSet,
                          VkCommandBuffer       cmd,
                          VkFence               fence,
                          uint32_t              gx,
                          uint32_t              gy,
                          uint32_t              gz)
{
    // 录制 — 与同步版本相同，但 submit 时挂 fence，不等待
    vkResetCommandBuffer(cmd, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fn.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                             fn.pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, gx, gy, gz);

    vkEndCommandBuffer(cmd);

    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;

    // fence 标记 GPU 执行完成，调用方通过 vkWaitForFences 获取结果
    vkQueueSubmit(ctx.computeQueue, 1, &si, fence);
    return true;
}

bool dispatchVulkan(const VulkanContext&              ctx,
                    const VulkanFunction&             fn,
                    const std::vector<VulkanBuffer*>& buffers,
                    uint32_t                          gx,
                    uint32_t                          gy,
                    uint32_t                          gz)
{
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet  set  = VK_NULL_HANDLE;
    if (!bindVulkanBuffers(fn, ctx, buffers, pool, set))
        return false;
    bool ok = dispatchVulkanBound(ctx, fn, set, gx, gy, gz);
    vkDestroyDescriptorPool(ctx.device, pool, nullptr);
    return ok;
}

} // namespace skl

#endif // SKL_HAS_VULKAN
