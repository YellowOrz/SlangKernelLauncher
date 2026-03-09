// MetalBackend.mm — Metal compute backend implementation
// Compiled with -fobjc-arc (automatic reference counting).
// ObjC objects are stored as void* via __bridge_retained / CFRelease
// so that the public header (MetalBackend.h) remains pure C++.

#include "MetalBackend.h"

#ifdef SKL_HAS_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// ── Helper ───────────────────────────────────────────────────────────────────

// Encode a compute dispatch into a NEW command buffer and return it (retained).
// Caller is responsible for committing / waiting / releasing.
static id<MTLCommandBuffer> encodeDispatch(const skl::MetalContext&              ctx,
                                            const skl::MetalFunction&             fn,
                                            const std::vector<skl::MetalBuffer*>& buffers,
                                            uint32_t gx, uint32_t gy, uint32_t gz)
{
    id<MTLCommandQueue>          queue    = (__bridge id<MTLCommandQueue>)ctx.commandQueue;
    id<MTLComputePipelineState>  pipeline = (__bridge id<MTLComputePipelineState>)fn.pipelineState;
    if (!queue || !pipeline) return nil;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    if (!cmdBuf) return nil;

    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    if (!encoder) return nil;

    [encoder setComputePipelineState:pipeline];

    for (uint32_t i = 0; i < (uint32_t)buffers.size(); ++i)
    {
        if (!buffers[i] || !buffers[i]->buffer) continue;
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffers[i]->buffer;
        [encoder setBuffer:buf offset:0 atIndex:i];
    }

    MTLSize threadsPerGroup = MTLSizeMake(fn.threadGroupX, fn.threadGroupY, fn.threadGroupZ);
    MTLSize groupCount      = MTLSizeMake(gx, gy, gz);
    [encoder dispatchThreadgroups:groupCount threadsPerThreadgroup:threadsPerGroup];

    [encoder endEncoding];
    return cmdBuf;
}

// ── Context ───────────────────────────────────────────────────────────────────

namespace skl {

MetalContext createMetalContext()
{
    MetalContext ctx{};

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev)
    {
        fprintf(stderr, "[SKL] Metal: no device available\n");
        return ctx;
    }
    fprintf(stdout, "[SKL] Metal device: %s\n", dev.name.UTF8String);

    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (!queue)
    {
        fprintf(stderr, "[SKL] Metal: newCommandQueue failed\n");
        return ctx;
    }

    ctx.device       = (__bridge_retained void*)dev;
    ctx.commandQueue = (__bridge_retained void*)queue;
    return ctx;
}

void destroyMetalContext(MetalContext& ctx)
{
    if (ctx.commandQueue) { CFRelease(ctx.commandQueue); ctx.commandQueue = nullptr; }
    if (ctx.device)       { CFRelease(ctx.device);       ctx.device       = nullptr; }
}

// ── Kernel loading ────────────────────────────────────────────────────────────

// MSL 保留关键字集合（不能作为 [[kernel]] 函数名）
static bool isMslReservedName(const char* name)
{
    static const char* kReserved[] = {
        "kernel", "vertex", "fragment", "device", "constant",
        "threadgroup", "threadgroup_imageblock", "object", "mesh", nullptr
    };
    for (int i = 0; kReserved[i]; ++i)
        if (strcmp(name, kReserved[i]) == 0) return true;
    return false;
}

MetalFunction loadMetalFunction(const std::vector<uint8_t>& msl,
                                 const MetalContext&          ctx,
                                 uint32_t                     bindingCount,
                                 const char*                  entryPoint,
                                 uint32_t                     threadGroupX,
                                 uint32_t                     threadGroupY,
                                 uint32_t                     threadGroupZ)
{
    MetalFunction fn{};
    fn.bindingCount  = bindingCount;
    fn.threadGroupX  = threadGroupX;
    fn.threadGroupY  = threadGroupY;
    fn.threadGroupZ  = threadGroupZ;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx.device;
    if (!dev) return fn;

    // MSL source is stored as UTF-8 bytes
    NSString* source = [[NSString alloc] initWithBytes:msl.data()
                                                length:msl.size()
                                              encoding:NSUTF8StringEncoding];
    if (!source)
    {
        fprintf(stderr, "[SKL] Metal: MSL source decode failed\n");
        return fn;
    }

    // 若入口函数名是 MSL 保留关键字（如 "kernel"），在源码中将
    // "void {name}(" 替换为 "void {name}_entry("，避免 Metal 编译器报错。
    NSString* epName = [NSString stringWithUTF8String:entryPoint];
    if (isMslReservedName(entryPoint))
    {
        NSString* renamed = [epName stringByAppendingString:@"_entry"];
        NSString* oldDecl = [NSString stringWithFormat:@"void %@(", epName];
        NSString* newDecl = [NSString stringWithFormat:@"void %@(", renamed];
        source = [source stringByReplacingOccurrencesOfString:oldDecl withString:newDecl];
        epName = renamed;
    }

    NSError* err = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    id<MTLLibrary> lib = [dev newLibraryWithSource:source options:opts error:&err];
    if (!lib)
    {
        fprintf(stderr, "[SKL] Metal: newLibraryWithSource failed: %s\n",
                err.localizedDescription.UTF8String);
        return fn;
    }

    id<MTLFunction> mtlFn = [lib newFunctionWithName:epName];
    if (!mtlFn)
    {
        fprintf(stderr, "[SKL] Metal: function '%s' not found in library\n", entryPoint);
        return fn;
    }

    id<MTLComputePipelineState> pipeline =
        [dev newComputePipelineStateWithFunction:mtlFn error:&err];
    if (!pipeline)
    {
        fprintf(stderr, "[SKL] Metal: newComputePipelineState failed: %s\n",
                err.localizedDescription.UTF8String);
        return fn;
    }

    fn.pipelineState = (__bridge_retained void*)pipeline;
    return fn;
}

void unloadMetalFunction(MetalFunction& fn)
{
    if (fn.pipelineState) { CFRelease(fn.pipelineState); fn.pipelineState = nullptr; }
    fn.bindingCount = 0;
    fn.threadGroupX = fn.threadGroupY = fn.threadGroupZ = 0;
}

// ── Buffer management ─────────────────────────────────────────────────────────

MetalBuffer createMetalBuffer(const MetalContext& ctx, uint64_t size)
{
    MetalBuffer buf{};
    buf.size = size;

    id<MTLDevice> dev = (__bridge id<MTLDevice>)ctx.device;
    if (!dev || size == 0) return buf;

    id<MTLBuffer> b = [dev newBufferWithLength:size
                                       options:MTLResourceStorageModeShared];
    if (!b)
    {
        fprintf(stderr, "[SKL] Metal: newBufferWithLength(%llu) failed\n",
                (unsigned long long)size);
        buf.size = 0;
        return buf;
    }

    buf.buffer = (__bridge_retained void*)b;
    return buf;
}

void destroyMetalBuffer(MetalBuffer& buf)
{
    if (buf.buffer) { CFRelease(buf.buffer); buf.buffer = nullptr; }
    buf.size = 0;
}

void* mapMetalBuffer(const MetalBuffer& buf)
{
    if (!buf.buffer) return nullptr;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf.buffer;
    return b.contents;
}

void unmapMetalBuffer(const MetalBuffer& /*buf*/)
{
    // Apple Silicon: MTLResourceStorageModeShared 始终 coherent，无需显式操作
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

bool dispatchMetal(const MetalContext&              ctx,
                   const MetalFunction&             fn,
                   const std::vector<MetalBuffer*>& buffers,
                   uint32_t gx, uint32_t gy, uint32_t gz)
{
    id<MTLCommandBuffer> cmdBuf = encodeDispatch(ctx, fn, buffers, gx, gy, gz);
    if (!cmdBuf) return false;

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return cmdBuf.status == MTLCommandBufferStatusCompleted;
}

bool dispatchMetalAsync(const MetalContext&              ctx,
                         const MetalFunction&             fn,
                         const std::vector<MetalBuffer*>& buffers,
                         void**                           outCmdBuf,
                         uint32_t gx, uint32_t gy, uint32_t gz)
{
    id<MTLCommandBuffer> cmdBuf = encodeDispatch(ctx, fn, buffers, gx, gy, gz);
    if (!cmdBuf) { *outCmdBuf = nullptr; return false; }

    [cmdBuf commit];
    // 保留一个引用返回给调用方（由 releaseMetalCommandBuffer 释放）
    *outCmdBuf = (__bridge_retained void*)cmdBuf;
    return true;
}

bool waitMetalCommandBuffer(void* cmdBuf)
{
    if (!cmdBuf) return false;
    id<MTLCommandBuffer> cb = (__bridge id<MTLCommandBuffer>)cmdBuf;
    [cb waitUntilCompleted];
    return cb.status == MTLCommandBufferStatusCompleted;
}

void releaseMetalCommandBuffer(void* cmdBuf)
{
    if (cmdBuf) CFRelease(cmdBuf);
}

} // namespace skl

#endif // SKL_HAS_METAL
