#pragma once

#ifdef SKL_HAS_METAL

#include <cstdint>
#include <vector>

namespace skl {

// ── Context ───────────────────────────────────────────────────────────────────
// 持有一个完整的 Metal compute 环境。
// 可由 createMetalContext() 自动创建，或由调用方手动填充（嵌入已有引擎时）。
// 生命周期必须长于所有 MetalFunction / MetalBuffer 对象。
// 所有字段均为 void*，内部持有 __bridge_retained 的 ObjC 对象，
// 以便在纯 C++ 头文件中使用而无需引入 Metal/Metal.h。
struct MetalContext
{
    void* device       = nullptr;  // id<MTLDevice>        (__bridge_retained)
    void* commandQueue = nullptr;  // id<MTLCommandQueue>  (__bridge_retained)
};

// 自动创建一个含 compute command queue 的最小 Metal Context。
// 失败时返回 .device == nullptr 的零值结构体。
MetalContext createMetalContext();

// 销毁 MetalContext 持有的所有 Metal 对象。
void destroyMetalContext(MetalContext& ctx);

// ── 已编译的 Metal 计算内核 ───────────────────────────────────────────────────
// 由 loadMetalFunction() 填充；pipelineState == nullptr 表示加载失败。
// threadGroupX/Y/Z：Metal dispatch 时需要显式传入 threadsPerThreadgroup，
// 存储于此以便 dispatch 函数统一使用（对应 Slang [numthreads(...)] 注解）。
struct MetalFunction
{
    void*    pipelineState = nullptr;  // id<MTLComputePipelineState> (__bridge_retained)
    uint32_t bindingCount  = 0;
    uint32_t threadGroupX  = 64;
    uint32_t threadGroupY  = 1;
    uint32_t threadGroupZ  = 1;
};

// ── GPU 存储缓冲区 ────────────────────────────────────────────────────────────
// Apple Silicon 始终为统一内存（Shared），可直接 CPU 读写 buffer.contents。
struct MetalBuffer
{
    void*    buffer = nullptr;  // id<MTLBuffer> (__bridge_retained)
    uint64_t size   = 0;
};

// ── 加载 / 卸载内核 ───────────────────────────────────────────────────────────

// 从 MSL 源代码（UTF-8 字节数组）编译并加载计算内核。
// msl:          Slang 编译器输出的 MSL 源文本（vector<uint8_t> 包含 UTF-8 文本）
// bindingCount: 着色器使用的 storage buffer 数量（binding 顺序与 MSL [[buffer(i)]] 一致）
// entryPoint:   MSL 入口函数名（Slang 编译 Metal 时保留原始函数名，非 "main"）
// threadGroupX/Y/Z: kernel [numthreads(X,Y,Z)] 的值，dispatch 时传给 threadsPerThreadgroup
// 失败时返回 .pipelineState == nullptr 的零值结构体。
MetalFunction loadMetalFunction(const std::vector<uint8_t>& msl,
                                 const MetalContext&          ctx,
                                 uint32_t                     bindingCount,
                                 const char*                  entryPoint,
                                 uint32_t                     threadGroupX = 64,
                                 uint32_t                     threadGroupY = 1,
                                 uint32_t                     threadGroupZ = 1);

// 销毁 MetalFunction 持有的 pipeline state，并将各字段清零。
void unloadMetalFunction(MetalFunction& fn);

// ── 缓冲区管理 ────────────────────────────────────────────────────────────────

// 创建 MTLBuffer（MTLResourceStorageModeShared）。
// Apple Silicon 上此缓冲区对 CPU 和 GPU 均可见，无需显式同步。
// 失败时返回 .buffer == nullptr 的零值结构体。
MetalBuffer createMetalBuffer(const MetalContext& ctx, uint64_t size);

// 销毁 MetalBuffer，并将各字段清零。
void destroyMetalBuffer(MetalBuffer& buf);

// 返回 MTLBuffer.contents 指针（CPU 可直接读写，无需 map/unmap）。
// 仅对 MTLResourceStorageModeShared 缓冲区有效。
void* mapMetalBuffer(const MetalBuffer& buf);

// Apple Silicon 上统一内存始终 coherent，unmap 为空操作，仅为接口对称而存在。
void  unmapMetalBuffer(const MetalBuffer& buf);

// ── Dispatch ──────────────────────────────────────────────────────────────────

// 同步 dispatch：录制 → commit → waitUntilCompleted，返回时 GPU 已完成。
// buffers 顺序与 MSL 着色器 [[buffer(0)]], [[buffer(1)]], ... 一致。
bool dispatchMetal(const MetalContext&              ctx,
                   const MetalFunction&             fn,
                   const std::vector<MetalBuffer*>& buffers,
                   uint32_t                         groupCountX,
                   uint32_t                         groupCountY = 1,
                   uint32_t                         groupCountZ = 1);

// 异步 dispatch：录制 → commit，立即返回，GPU 在后台执行。
// outCmdBuf：输出一个 __bridge_retained 的 MTLCommandBuffer 指针，
//            由调用方用 waitMetalCommandBuffer + releaseMetalCommandBuffer 管理。
bool dispatchMetalAsync(const MetalContext&              ctx,
                         const MetalFunction&             fn,
                         const std::vector<MetalBuffer*>& buffers,
                         void**                           outCmdBuf,
                         uint32_t                         groupCountX,
                         uint32_t                         groupCountY = 1,
                         uint32_t                         groupCountZ = 1);

// 等待 dispatchMetalAsync 提交的命令缓冲区完成（waitUntilCompleted）。
bool waitMetalCommandBuffer(void* cmdBuf);

// 释放 dispatchMetalAsync 输出的 MTLCommandBuffer 引用（CFRelease）。
void releaseMetalCommandBuffer(void* cmdBuf);

} // namespace skl

#endif // SKL_HAS_METAL
