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
    float3 color = reservoirs[bufferInd].p_hat_X;
    /*if (dot(color, color) == 0) {
        color = float3(1, 0, 0);
    } else {
        color = float3(0, 1, 0);
    }*/

    gOutputColor[pixel] = float4(color * 10, 1.f);
}
/*void updateReservoir(Reservoir r, Sample sample, float3 radiance_new, float w_i_new, inout SampleGenerator sg) {
    r.w_sum += w_i_new;
    if (sampleNext1D(sg) < w_i_new / r.w_sum) {
        // replace current sample with new one
        r.X = sample;
        r.radiance = radiance_new;
        r.w_i = w_i_new;
        r.M++;
    }
}*/
