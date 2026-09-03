import Utils.Sampling.SampleGenerator;
import RestirReservoir;
import PathState;

StructuredBuffer<Reservoir> gReservoirBuffer; // Buffer of reservoirs to visualize
RWTexture2D<float4> gOutputColor; // Output color

cbuffer CB
{
    uint gFrameCount;
    uint2 gOutputDimensions;

}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    uint2 pixel = threadID.xy;
    uint pixelInd = pixel.y * gOutputDimensions.x + pixel.x;

    PathState path = gReservoirBuffer[pixelInd].X.path;
    uint k = gReservoirBuffer[pixelInd].X.k;
    uint seed = gReservoirBuffer[pixelInd].X.seed;
    uint len = path.getVertexIndex();

    /*uint r = ((seed >> 16) & 0xFF);
    uint g = ((seed >> 8) & 0xFF);
    uint b = ((seed) & 0xFF);
    float4 color = float4(r / 256.f, g / 256.f, b / 256.f, 1);*/

    float4 color = float4(0, 0, 0, 1);
    switch (len) {
    case 0:
        color = float4(1, 0, 1, 1);
        break;
    case 2:
        color = float4(1, 0, 0, 1);
        break;
    case 3:
        color = float4(1, 1, 0, 1);
        break;
    case 4:
        color = float4(0, 1, 0, 1);
        break;
    case 5:
        color = float4(0, 0, 1, 1); 
        break;
    default:
        color = float4(0, 0, 0, 1);
        break;
    }
    gOutputColor[pixel] = color;
}