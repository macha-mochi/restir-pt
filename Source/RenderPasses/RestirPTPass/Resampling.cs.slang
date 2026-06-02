import RestirReservoir;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
}

//Inputs (will also modify reservoirs)
RWStructuredBuffer<Reservoir> reservoirs;
RWStructuredBuffer<float3> di_bgBuffer; // direct illumination / env map bg buffer, to be added to f_X * W_X at the end
// Outputs
RWTexture2D<float4> gOutputColor;

RWStructuredBuffer<DebugData> debugBuffer;

static const uint kNumSpatialNeighbors = 3;
static const uint kSpatialNeighborRadius = 20;
static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

struct DebugData
{
    int2 neighbor1;
    int2 neighbor2;
    int2 neighbor3;
};

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
    float3 f_X = reservoirs[bufferInd].X.p_hat_X;
    float W_X = reservoirs[bufferInd].W_X;
    float3 di_bg = di_bgBuffer[bufferInd];

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    //temporal

    // spatial
    int2 neighbor;
    DebugData d = {};
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
        if (getSpatialNeighbor(pixel, dims, kSpatialNeighborRadius, sg, neighbor)) {
            if (i == 0) {
                d.neighbor1 = neighbor;
                gOutputColor[neighbor] = float4(1, 0, 0, 1);
            } else if (i == 1) {
                d.neighbor2 = neighbor;
                gOutputColor[neighbor] = float4(0, 1, 0, 1);
            } else {
                d.neighbor3 = neighbor;
                gOutputColor[neighbor] = float4(0, 0, 1, 1);
            }

            Sample X = reservoirs[neighbor.y * dims.x + neighbor.x].X;
            Sample Y;
            float w_i = 0.f;
            bool shiftSucceeded = false;
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                shiftSucceeded = performReconnectionShift(X, Y);
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
                // do hybrid mapping
            }

            if (shiftSucceeded) {
                // evaluate resampling mis weight

                // evaluate w_i = mis * p hat * w * jacobian
            }

            //update reservoir streaming (in a new buffer)
        }
    }
    debugBuffer[bufferInd] = d;

    //gOutputColor[pixel] = float4(f_X * W_X + di_bg, 1.f);
}

// Returns a pixel location randomly selected from within a certain radius of the current pixel in [neighbor]. Returns false if it was oob
bool getSpatialNeighbor(uint2 pixel, uint2 dims, uint radius, inout SampleGenerator sg, out int2 neighbor) {
    int2 offset = (int2)(radius * sample_disk(sampleNext2D(sg)));
    neighbor = pixel + offset;
    if(neighbor.x < 0 || neighbor.x >= dims.x || neighbor.y < 0 || neighbor.y >= dims.y) {
        return false;
    }
    return true;
}

// Shifts sample X into the current pixel's domain using reconnection shift
bool performReconnectionShift(Sample X, out Sample Y) {
    return false;
}
