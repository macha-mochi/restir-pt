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

    void execute(uint2 pixel) {
        if (any(pixel >= gOutputDimensions)) return; // return if oob

        outputColor[pixel] = float4(0, 0, 0, 1);

        Reservoir r = {};
        DebugData debugData = {};

        SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 4 + 3);
        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        uint replayDataBufferInd = getReplayOutputIndex_S(pixel, gOutputDimensions, kNumSpatialNeighbors);
        uint ogInd = replayDataBufferInd;
        // Treat input buffer sample as canonical.
        float m_num = inputBuffer[ind(pixel)].c * luminance(inputBuffer[ind(pixel)].X.p_hat); // alr in the correct domain
        float m_den = m_num;
        for (int i = 0; i < kNumSpatialNeighbors; ++i) { // want to shift the canonical sample to spatial neighbors' domains
            int2 neighbor = int2(0);
            bool validNeighbor = getSpatialNeighbor(pixel, i, neighbor);
            if (validNeighbor) {
                Sample Y_neighborDomain = {};
                float inverse_jacobian = inputBuffer[ind(pixel)].X.J; // initialize to srcG, src is this pixel
                bool success = false;
                ShiftDebugData d = {};
                if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                {
                    HitInfo dstHit = HitInfo(vBuffer[neighbor]); // get hit info for the vertex right before rc shift vertex on this pixel
                    ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[neighbor].xyz);
                    success = computeShiftedSample(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                        y1_SD, // vertex right before reconnection vert on the dst domain's path
                        inputBuffer[ind(pixel)].X,      // base path aka canonical sample
                        float3(1.f),                         // previous thp = 1
                        Y_neighborDomain, inverse_jacobian, sg, d); // out or inout variables
                } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                    HitInfo dstHit = HitInfo(replayedDataBuffer[replayDataBufferInd].x_kMinus1_hitInfo);
                    float3 rayOrigin = replayedDataBuffer[replayDataBufferInd].x_kMinus2_pos; //this was calc'd as ray origin in Retrace pass
                    float3 rayDir = normalize(gScene.getVertexData(dstHit.getTriangleHit()).posW - rayOrigin);
                    ShadingData y_kMinus1_SD = loadShadingData(dstHit, rayOrigin, rayDir);
                    success = computeShiftedSample(ShiftMappingType::Hybrid, false, false, // settings for shift: hybrid, no temporal
                        y_kMinus1_SD,
                        inputBuffer[ind(pixel)].X, // base path is canonical, while dst hit is replayed from neighbor's domain
                        replayedDataBuffer[replayDataBufferInd].replayedThp,
                        Y_neighborDomain, inverse_jacobian, sg, d); // out or inouts
                }
                if (success) {
                    m_den += inputBuffer[ind(neighbor)].c * luminance(Y_neighborDomain.p_hat) * inverse_jacobian; // we just did the inverse shift mapping, jacobian = |T(-1)(y)|
                }
            }

            replayDataBufferInd++;
        }
        float w_i = (m_num / m_den) * luminance(inputBuffer[ind(pixel)].X.p_hat) * inputBuffer[ind(pixel)].W_X; // jacobian is 1 bc alr in right domain
        if (isnan(w_i)) w_i = 0.f;
        r.updateReservoir(inputBuffer[ind(pixel)].X, w_i, inputBuffer[ind(pixel)].c, sg);

        // Spatial passes
        for(int i = 0; i < kNumSpatialNeighbors; i++){
            int2 neighbor = int2(0);
            if (!getSpatialNeighbor(pixel, i, neighbor)) {
                replayDataBufferInd += kNumSpatialNeighbors;
                continue;
            }

            Sample neighbor_curDomain = {};
            float w_i = 0.f;
            float jacobian = inputBuffer[ind(neighbor)].X.J; // denom of jacobian of shifting from neighbor into [pixel]'s domain, |T(X)|
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

                replayDataBufferInd++; //incremenet this no matter success or failure.
            }

            if(i == 0){
                if (!shiftSucceeded) {

                    float4 color = float4(0, 0, 0, 1);
                    if(d.failReason == 1){
                        //rc vertex invalid (no)
                        color = float4(1, 1, 0, 1);
                    } else if(d.failReason == 2){
                        // other hit type (no)
                        color = float4(0, 1, 1, 1);
                    } else if(d.failReason == 3){
                        // shadow ray k-1 to k not visible
                        color = float4(1, 0, 1, 1);
                    } else if(d.failReason == 4){
                        // jacobian 0 or p hat 0 (no)
                        // color = float4(0, 0, 1, 1);
                    } else {
                        if (inputBuffer[ind(neighbor)].X.k != inputBuffer[ind(neighbor)].X.path.getVertexIndex()) {
                            color = float4(1, 0, 0, 1);
                        }
                        
                    }
                    outputColor[pixel] = color;
                }
            }

            float newJ = 0.f;
            if (shiftSucceeded) {
                newJ = jacobian * inputBuffer[ind(neighbor)].X.J;
                float m_num = inputBuffer[ind(neighbor)].c * luminance(inputBuffer[ind(neighbor)].X.p_hat) / jacobian; // i is the domain that [preshiftNeighbor] came from
                float m_den = m_num;
                // process canonical candidate for m_den
                m_den += inputBuffer[ind(pixel)].c * luminance(neighbor_curDomain.p_hat); // jacobian is 1 since it came from this domain

                // add terms to m_den for what if this sample came from the other spatial domains
                Sample shifted = {};
                float inverse_jacobian = 0.f;
                bool success = false;
                ShiftDebugData d = {};
                for (int j = 0; j < kNumSpatialNeighbors; j++) {
                    if (j != i) {
                        shifted = {};
                        inverse_jacobian = newJ; // jacobian denom of shifting samples into this spatial neighbor's domain, |T(-1)(y)|, is jacobianNum of shifting from neighbor to [pixel]
                        success = false;
                        d = {};

                        int2 otherNeighbor = int2(0);
                        if (!getSpatialNeighbor(pixel, j, otherNeighbor)) {
                            replayDataBufferInd++; //even if no neighbor here, need to incr buffer ind
                            continue;
                        }

                        if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                        {
                            HitInfo dstHit = HitInfo(vBuffer[otherNeighbor]); // get hit info for the vertex right before rc shift vertex on this pixel
                            ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[otherNeighbor].xyz);
                            success = computeShiftedSample(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                                y1_SD, // vertex right before reconnection vert on the dst domain's path
                                neighbor_curDomain,   // information from the base path, which is in [pixel]'s domain
                                float3(1.f), // previous thp = 1
                                shifted, inverse_jacobian, sg, d); // out or inout variables
                        } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
                        {
                            HitInfo dstHit = HitInfo(replayedDataBuffer[replayDataBufferInd].x_kMinus1_hitInfo);
                            float3 rayOrigin = replayedDataBuffer[replayDataBufferInd].x_kMinus2_pos;
                            float3 rayDir = normalize(gScene.getVertexData(dstHit.getTriangleHit()).posW - rayOrigin);
                            ShadingData y_kMinus1_SD = loadShadingData(dstHit, rayOrigin, rayDir);
                            success = computeShiftedSample(ShiftMappingType::Hybrid, false, false, //settings: hybrid, no temporal
                                y_kMinus1_SD,
                                neighbor_curDomain, //base path sample is neighbor shifted to cur domain
                                replayedDataBuffer[replayDataBufferInd].replayedThp,
                                shifted, inverse_jacobian, sg, d); //out or inout

                            replayDataBufferInd++; //regardless of success or fail
                        }
                        if (success) {
                            m_den += inputBuffer[ind(otherNeighbor)].c * luminance(shifted.p_hat) * inverse_jacobian;
                        }
                    }
                }

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = (m_num / m_den) * luminance(neighbor_curDomain.p_hat) * inputBuffer[ind(neighbor)].W_X * jacobian;
                if (isnan(w_i)) w_i = 0.f;
            } else {
                replayDataBufferInd += (kNumSpatialNeighbors - 1); //even if shift failed, need to move replay buffer ind forward
            }

            neighbor_curDomain.w_i = w_i;
            r.updateReservoir(neighbor_curDomain, w_i, inputBuffer[ind(neighbor)].c, sg);
        }

        outputBuffer[ind(pixel)] = r;
        debugBuffer[ind(pixel)] = debugData;

        outputColor[pixel] = float4(r.X.p_hat * r.W_X + di_bgBuffer[ind(pixel)], 1);
    }
}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID) {
    gResamplePass.execute(threadID.xy);
}
