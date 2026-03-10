// arrayReduce.slang — 数组规约求和 kernel
// 使用共享内存并行归约 + 原子操作

RWStructuredBuffer<float> inputBuffer;   // 输入数组
RWStructuredBuffer<float> resultBuffer;  // 最终结果（单值）
RWStructuredBuffer<int> countBuffer;     // 数组元素个数
RWStructuredBuffer<int> numWorkGroupsBuffer;  // dispatch 的工作组数量

static const uint THREADS_PER_GROUP = 64;

#if defined(__VULKAN__) || defined(__SPIRV__) || defined(__CUDA__)
groupshared float sums[THREADS_PER_GROUP];  // 共享内存，GPU only
#endif

[shader("compute")]
[numthreads(THREADS_PER_GROUP, 1, 1)]
void arrayReduce(uint3 groupThreadId : SV_GroupThreadID, uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint tid = groupThreadId.x;
    uint id  = dispatchThreadId.x;
    int count = countBuffer[0];
    int numWorkGroups = numWorkGroupsBuffer[0];

#if defined(__VULKAN__) || defined(__SPIRV__) || defined(__CUDA__)
    // GPU 路径：并行归约
    // 每个线程累积多个元素（跨越所有线程组）
    float sum_tmp = 0.0f;
    uint stride = THREADS_PER_GROUP * numWorkGroups;
    for (uint i = id; i < count; i += stride)
        sum_tmp += inputBuffer[i];
    sums[tid] = sum_tmp;

    GroupMemoryBarrierWithGroupSync();

    // 组内并行归约
    for (stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            sums[tid] += sums[tid + stride];
        GroupMemoryBarrierWithGroupSync();
    }

    // 只有线程组第一个线程执行原子加
    if (tid == 0)
        InterlockedAdd(resultBuffer[0], sums[0]);
#else
    // CPU 路径：由 tid == 0 的线程执行全部求和
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < count; i++)
            sum += inputBuffer[i];
        resultBuffer[0] = sum;
    }
#endif
}