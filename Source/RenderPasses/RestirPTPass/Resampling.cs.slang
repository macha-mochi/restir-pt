#include "RestirTypes.slang"

//Inputs (will also modify reservoirs)
RWStructuredBuffer<Reservoir> reservoirs;
// Outputs
RWTexture2D<float4> gOutputColor;

/**
* For now, just puts color to screen
*/
[numthreads(16, 16, 1)] // TODO 16 was just chosen arbitrarily since that's what RTXDI uses
void spatiotemporalResampling(uint3 threadID: SV_DispatchThreadID) {
    uint2 pixel = threadID.xy;
    uint2 dims;
    gOutputColor.GetDimensions(dims.x, dims.y);

    if (any(pixel >= dims)) return; // return if oob

    uint bufferInd = pixel.y * dims.x + pixel.x;
    float3 f_X = reservoirs[bufferInd].p_hat_X;
    float W_X = reservoirs[bufferInd].W_X;

    //temporal

    //spatial

    gOutputColor[pixel] = float4(f_X * W_X, 1.f);
}
