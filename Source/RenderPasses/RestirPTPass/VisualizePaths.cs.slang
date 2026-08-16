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
    uint len = path.getVertexIndex();

    float4 color;
    switch (len) {
    case 0:
        color = float4(1, 0, 1, 1);
        break;
    case 2:
        color = float4(1, 0, 0, 1);
        break;
    case 3:
        color = float4(1, 1, 0, 1);
        if (path.isNEESampled()) {
            color = float4(1, 1, 1, 1);
        }
        break;
    case 4:
        color = float4(0, 0.5, 0, 1);
        break;
    case 5:
        //color = float4(0, 0, 1, 1); 
        break;
    default:
        color = float4(0, 0, 0, 1);
        break;
    }
    gOutputColor[pixel] = color;
}