import Utils.Sampling.SampleGenerator;
import Utils.Math.MathHelpers;

static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
static const uint kSpatialNeighborRadius = SPATIAL_NEIGHBOR_RADIUS;

//Outputs
RWStructuredBuffer<int2> spatialOffsetBuffer;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint2 gOutputDimensions;
}

/**
 Returns int2 random neighbor offset of a certain radius.
*/
int2 createSpatialOffset(uint radius, inout SampleGenerator sg) {
    return (int2)(radius * sample_disk(sampleNext2D(sg)));
}

// Populates the buffer for this pixel
void generateSpatialOffsets(uint2 pixel, inout SampleGenerator sg) {
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
        uint idx = (pixel.y * gOutputDimensions.x + pixel.x) * kNumSpatialNeighbors + i;
        spatialOffsetBuffer[idx] = createSpatialOffset(kSpatialNeighborRadius, sg);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID) {
    uint2 pixel = threadID.xy;

    if(any(pixel >= gOutputDimensions)) return;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount); // TODO prolly need to change this seed
    generateSpatialOffsets(pixel, sg);
}
