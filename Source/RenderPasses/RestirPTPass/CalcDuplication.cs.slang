import Utils.Debug.PixelDebug;

// Inputs
StructuredBuffer<uint2> gSeedBuffer; //this will be k and seed data from the end of the frame, seed is 2nd uint

//Outputs
RWStructuredBuffer<float> gDuplicationScoreBuffer;

cbuffer CB
{
    uint2 gOutputDimensions;
}

bool withinBounds(int2 pos)
{
    return all(int2(0, 0) <= pos) && all(pos < gOutputDimensions);
}

uint ind(uint2 pixel)
{
    return pixel.y * gOutputDimensions.x + pixel.x;
}

uint getSeed(uint2 pixel){
    return gSeedBuffer[ind(pixel)].y;
}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID)
{
    uint2 pixel = threadID.xy;
    uint seed = getSeed(pixel);

    float d = 0.f;
    for(int i = -8; i <= 8; i++){
        for(int j = -8; j <= 8; j++){
            int2 pos = pixel + int2(i, j);
            if(withinBounds(pos) && !(i == 0 && j == 0))
            {
                if(getSeed(pos) == seed)
                {
                    d += 1.f;
                }
            }
        }
    }
    d /= 288.f;

    gDuplicationScoreBuffer[ind(pixel)] = d;
}