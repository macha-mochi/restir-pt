#include "RestirTypes.slang" //TODO: check if the filepath works here (ie are we starting from Falcor root? or what idk)

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
    float3 color = reservoirs[bufferInd].radiance;

    gOutputColor[pixel] = float4(color, 1.f);
}
