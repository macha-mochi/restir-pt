import RestirReservoir;
import RestirHelpers;
import ShiftMapping;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import Utils.Debug.PixelDebug;

struct DebugData
{
    float dummy;

    __init(){
        dummy = 1.f;
    }
};

static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

cbuffer CB
{
    uint gFrameCount;        // Frame count since scene was loaded.
    uint gPRNGDimension;     // First available PRNG dimension.
    uint2 gOutputDimensions;

    SpatialReusePass gResamplePass;
}

struct SpatialReusePass {
    // Inputs
    Texture2D<PackedHitInfo> vBuffer;
    Texture2D<float4> viewW; // Originally optional but we include it
    StructuredBuffer<Reservoir> inputBuffer;
    StructuredBuffer<Reservoir> temporalBuffer;
    StructuredBuffer<float3> di_bgBuffer;
    StructuredBuffer<int2> spatialOffsetBuffer; // stores the offsets from a pixel, which could take on negative values
    StructuredBuffer<RandomReplayOutput> replayedDataBuffer;
    // Outputs
    RWStructuredBuffer<RandomReplayInput> replayedInputBuffer;
    RWStructuredBuffer<Reservoir> outputBuffer;
    RWTexture2D<float4> outputColor;

    // Debug
    RWStructuredBuffer<DebugData> debugBuffer;

    uint ind(uint2 pixel) {
        return pixel.y * gOutputDimensions.x + pixel.x;
    }

    // Returns the index of the [neighborInd]th offset of [pixel]. FOr exmaple, if i wanted the spatial offset of pixel (x, y) in the first spatial pass, parameters are (x, y) and 0
    // This is 0 indexed!!
    uint getOffsetBufferIndex(uint2 pixel, uint neighborInd) {
        return getSpatialOffsetBufferIndex(pixel, gOutputDimensions, kNumSpatialNeighbors, neighborInd);
    }

    // Returns the pixel coordinates of the [neighborInd]th neighbor of [pixel] in [neighbor]. 1st spatial pass = 0th neighbor. Returns false if it was oob
    bool getSpatialNeighbor(uint2 pixel, uint neighborInd, out int2 neighbor) {
        int2 offset = spatialOffsetBuffer[getOffsetBufferIndex(pixel, neighborInd)];
        neighbor = pixel + offset;
        if (neighbor.x < 0 || neighbor.x >= gOutputDimensions.x || neighbor.y < 0 || neighbor.y >= gOutputDimensions.y) {
            return false;
        }
        return true;
    }

    /*
    Spatial reuse pseudocode with pairwise MIS
    for(i = 0, i < num spatial neighbors){
      shift neighbor i into current, get p from i, calc mis weight
    }
    calc mis weight for canonical by shifting it back into each domain, put that into reservoir last

    Uses defensive pairwise MIS by default
    */
    void execute(uint2 pixel) {
        if (any(pixel >= gOutputDimensions)) return; // return if oob

        Reservoir r = {};
        DebugData debugData = {};

        SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 4 + 3);
        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        uint replayDataBufferInd = getReplayOutputIndex_S(pixel, gOutputDimensions, kNumSpatialNeighbors);
        uint c_c = inputBuffer[ind(pixel)].c;
        uint non_canonical_c_sum = 0;
        for (int i = 0; i < kNumSpatialNeighbors; i++) {
            int2 neighbor = int2(-1, -1);
            if (getSpatialNeighbor(pixel, i, neighbor)) {
                non_canonical_c_sum += inputBuffer[ind(neighbor)].c;
            }
        }

        // Spatial passes
        for(int i = 0; i < kNumSpatialNeighbors; i++){
            int2 neighbor = int2(0);
            if (!getSpatialNeighbor(pixel, i, neighbor)) {
                replayDataBufferInd++;
                continue;
            }

            Sample neighbor_curDomain = {};
            float w_i = 0.f;
            float jacobian = 0.f; // jacobian of shifting from neighbor into [pixel]'s domain, |T(X)|
            bool shiftSucceeded = false;
            ShiftDebugData d = {};
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                HitInfo dstHit = HitInfo(vBuffer[pixel]);
                ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[pixel].xyz);
                shiftSucceeded = computeShiftedSample(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                    y1_SD, // vertex right before reconnection vert on the dst domain's path
                    inputBuffer[ind(neighbor)].X,    // base path aka neighbor's sample
                    float3(1.f),                          // previous thp = 1
                    neighbor_curDomain, jacobian, sg, d); // out or inout variables
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
                HitInfo dstHit = HitInfo(replayedDataBuffer[replayDataBufferInd].x_kMinus1_hitInfo);
                float3 rayOrigin = replayedDataBuffer[replayDataBufferInd].x_kMinus2_pos;
                float3 rayDir = normalize(gScene.getVertexData(dstHit.getTriangleHit()).posW - rayOrigin);
                ShadingData y_kMinus1_SD = loadShadingData(dstHit, rayOrigin, rayDir);
                shiftSucceeded = computeShiftedSample( ShiftMappingType::Hybrid, false, false, // settings: hybrid, no temporal domains
                    y_kMinus1_SD,
                    inputBuffer[ind(neighbor)] .X, // base path is neigbor sample, which we are trying to shift into current pixel's domain
                    replayedDataBuffer[replayDataBufferInd].replayedThp,
                    neighbor_curDomain, jacobian, sg, d); // out or inout variables
            }

            if (shiftSucceeded) {
                uint c_i = inputBuffer[ind(neighbor)].c;
                float p_from_i = luminance(inputBuffer[ind(neighbor)].X.p_hat) / jacobian; // i is the domain that [preshiftNeighbor] came from
                float num = c_i * p_from_i;
                float den = non_canonical_c_sum * p_from_i + c_c * luminance(neighbor_curDomain.p_hat);
                float m_i = ((float)non_canonical_c_sum / (non_canonical_c_sum + c_c)) * num / den; //ok to only calc MIS weight if succeed bc if fail, mis weight should be 0.

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = m_i * luminance(neighbor_curDomain.p_hat) * inputBuffer[ind(neighbor)].W_X * jacobian;
                if (isnan(w_i)) w_i = 0.f;
            }

            neighbor_curDomain.w_i = w_i;
            r.updateReservoir(neighbor_curDomain, w_i, inputBuffer[ind(neighbor)].c, sg);

            replayDataBufferInd++;
        }

        // Treat input buffer sample as canonical.
        float p_hat_c = luminance(inputBuffer[ind(pixel)].X.p_hat);
        float m_c = (float)c_c / (c_c + non_canonical_c_sum);
        for (int j = 0; j < kNumSpatialNeighbors; j++) { // want to shift the canonical sample to spatial neighbors' domains
            int2 neighbor = int2(0);
            if (getSpatialNeighbor(pixel, j, neighbor)) {
                Sample Y_neighborDomain = {};
                float inverse_jacobian = 0.f;
                bool success = false;
                ShiftDebugData d = {};
                if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                {
                    HitInfo dstHit = HitInfo( vBuffer[neighbor]); // get hit info for the vertex right before rc shift vertex on this pixel
                    ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[neighbor].xyz);
                    success = computeShiftedSample(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                                             y1_SD, // vertex right before reconnection vert on the dst domain's path
                                             inputBuffer[ind(pixel)].X, // base path aka canonical sample
                                             float3(1.f),               // previous thp = 1
                                             Y_neighborDomain, inverse_jacobian, sg, d); // out or inout variables
                } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                    HitInfo dstHit = HitInfo(replayedDataBuffer[replayDataBufferInd].x_kMinus1_hitInfo);
                    float3 rayOrigin = replayedDataBuffer[replayDataBufferInd] .x_kMinus2_pos; // this was calc'd as ray origin in Retrace pass
                    float3 rayDir = normalize(gScene.getVertexData(dstHit.getTriangleHit()).posW - rayOrigin);
                    ShadingData y_kMinus1_SD = loadShadingData(dstHit, rayOrigin, rayDir);
                    success = computeShiftedSample( ShiftMappingType::Hybrid, false, false, // settings for shift: hybrid, no temporal
                        y_kMinus1_SD,
                        inputBuffer[ind(pixel)] .X, // base path is canonical, while dst hit is replayed from neighbor's domain
                        replayedDataBuffer[replayDataBufferInd].replayedThp,
                        Y_neighborDomain, inverse_jacobian, sg, d); // out or inouts
                }
                float c_j = inputBuffer[ind(neighbor)].c;
                float p_from_j = 0.f; 
                if (success) {
                    p_from_j = luminance(Y_neighborDomain.p_hat) * inverse_jacobian; // we just did the inverse shift mapping, jacobian = |T(-1)(y)|  
                }
                float num = c_c * p_hat_c;
                float den = non_canonical_c_sum * p_from_j + num;
                m_c += ((float)c_j / (c_c + non_canonical_c_sum)) * num / den;
            }

            replayDataBufferInd++;
        }
        float w_i = m_c * luminance(inputBuffer[ind(pixel)].X.p_hat) * inputBuffer[ind(pixel)].W_X; // jacobian is 1 bc alr in right domain
        if (isnan(w_i)) w_i = 0.f;
        r.updateReservoir(inputBuffer[ind(pixel)].X, w_i, inputBuffer[ind(pixel)].c, sg);

        outputBuffer[ind(pixel)] = r;
        debugBuffer[ind(pixel)] = debugData;
        replayedInputBuffer[ind(pixel)].k = r.X.k;
        replayedInputBuffer[ind(pixel)].seed = r.X.seed;

        outputColor[pixel] = float4(r.X.p_hat * r.W_X + di_bgBuffer[ind(pixel)], 1);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID) {
    gResamplePass.execute(threadID.xy);
}
