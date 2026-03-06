// kernel.slang — 示例 compute kernel
// 每个线程将 input[i] * 2 写入 output[i]

RWStructuredBuffer<float> inputBuffer;
RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void kernel(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    outputBuffer[idx] = inputBuffer[idx] * 2.0f;
}
