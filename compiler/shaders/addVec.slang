// addVec.slang — element-wise addition of two equal-length buffers
// output[i] = a[i] + b[i]

RWStructuredBuffer<float> aBuffer;
RWStructuredBuffer<float> bBuffer;
RWStructuredBuffer<float> outputBuffer;

[shader("compute")]
[numthreads(64, 1, 1)]
void addVec(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint idx = dispatchThreadID.x;
    outputBuffer[idx] = aBuffer[idx] + bBuffer[idx];
}
