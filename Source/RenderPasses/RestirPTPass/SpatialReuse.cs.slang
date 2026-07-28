import RestirReservoir;
import RestirHelpers;
import ShiftMapping;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;

struct DebugData
{
    float J_pixelToNeighbor;
    float J_neighborToPixel;
    float3 orig_p_hat;
    float3 pixelToNeighbor_p_hat;
    float3 neighborToPixel_p_hat;

    __init(){
        J_pixelToNeighbor = 0;
        J_neighborToPixel = 0;
        orig_p_hat = float3(0);
        pixelToNeighbor_p_hat = float3(0);
        neighborToPixel_p_hat = float3(0);
    }
};

static const bool kPerformSpatialReuse = USE_SPATIAL;
static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
static const uint kSpatialNeighborRadius = 20;
static const bool kPerformTemporalReuse = USE_TEMPORAL;
static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

cbuffer CB
{
    uint gFrameCount;        // Frame count since scene was loaded.
    uint gPRNGDimension;     // First available PRNG dimension.
    uint2 gOutputDimensions;

    uint gSpatialPassIndex; /// 0 indexed

    SpatialReusePass gResamplePass;
}

struct SpatialReusePass {
    // Inputs
    Texture2D<PackedHitInfo> vBuffer;
    Texture2D<PackedHitInfo> temporalVBuffer;
    Texture2D<float4> viewW; // Originally optional but we include it
    Texture2D<float4> temporalViewW;
    Texture2D<float2> motionVectors;
    StructuredBuffer<Reservoir> inputBuffer;
    StructuredBuffer<Reservoir> temporalBuffer;
    StructuredBuffer<float3> di_bgBuffer;
    // Outputs
    RWStructuredBuffer<int2> spatialOffsetBuffer; //stores the offsets from a pixel, which could take on negative values
    RWStructuredBuffer<Reservoir> outputBuffer;
    RWTexture2D<float4> outputColor;

    // Debug
    RWStructuredBuffer<DebugData> debugBuffer;

    // Finds the pixel position corresponding to the triangle hit by this frame's [pixel], last frame. Returning false if it would be out of bounds
    // intended usage: use [result] to sample a Texture2D holding data from LAST FRAME. ex: sampling last frame's vbuffer for where our CURRENT frame's pixel would have been in the last frame.
    bool calcLastFramePixelPos(uint2 pixel, out int2 result) {
        float2 inter = (float2)pixel + float2(0.5f);
        inter += gResamplePass.motionVectors[pixel] * gOutputDimensions;
        result = (int2)inter;
        return all(result >= 0) && all(result < gOutputDimensions);
    }

    uint ind(uint2 pixel) {
        return pixel.y * gOutputDimensions.x + pixel.x;
    }

    int2 createSpatialOffset(uint radius, inout SampleGenerator sg) {
        return (int2)(radius * sample_disk(sampleNext2D(sg)));
    }

    // Returns the index of the [neighborInd]th offset of [pixel]. FOr exmaple, if i wanted the spatial offset of pixel (x, y) in the first spatial pass, parameters are (x, y) and 0
    // This is 0 indexed!!
    uint getOffsetBufferIndex(uint2 pixel, uint neighborInd) {
        return ind(pixel) * kNumSpatialNeighbors + neighborInd;
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

    void generateSpatialOffsets(uint2 pixel, inout SampleGenerator sg) {
        for (int i = 0; i < kNumSpatialNeighbors; i++) {
            uint idx = getOffsetBufferIndex(pixel, i);
            spatialOffsetBuffer[idx] = createSpatialOffset(kSpatialNeighborRadius, sg);
        }
    }

    void execute(uint2 pixel) {
        if (any(pixel >= gOutputDimensions)) return; // return if oob

        uint pixelInd = ind(pixel);
        Reservoir r = inputBuffer[pixelInd]; //copy the input reservoir

        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        DebugData debugData = {};

        // If this is the 1st spatial pass, choose spatial neighbors and put them in their buffer
        if (gSpatialPassIndex == 0) {
            generateSpatialOffsets(pixel, sg);
        }
        // Prep temporal domain data. TODO: this calculation will be the same for every pixel, potential optimization also calc it once and store in buffer?
        int2 pixelLastFrame = int2(0, 0);
        bool pixelHasValidTemporalPos = calcLastFramePixelPos(pixel, pixelLastFrame);

        #if IS_FIRST_PASS
            // If this is the first resampling pass, we need to fix the w_i for the canonical (new) sample generated to include the MIS weights.
            float m_num = r.c * luminance(r.X.p_hat); // alr in the correct domain
            float m_den = m_num;
            for (int i = 0; i < kNumSpatialNeighbors; i++) { // want to shift the canonical sample to spatial neighbors' domains
                int2 neighbor = int2(0);
                bool validNeighbor = getSpatialNeighbor(pixel, i, neighbor);
                if (validNeighbor) {
                    Sample Y_neighborDomain = {};
                    float j_inverse = inputBuffer[ind(pixel)].J; //initialize to srcG, src is this pixel
                    bool success = false;
                    ShiftDebugData d = {};
                    if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                    {
                        HitInfo dstHit = HitInfo(vBuffer[neighbor]); //get hit info for the vertex right before rc shift vertex on this pixel
                        ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[neighbor].xyz);
                        success = performReconnectionShift(ShiftMappingType::Reconnection, false, false, //settings for the shift: reconnection, no temporal src or dst
                                                            y1_SD, // vertex right before reconnection vert on the dst domain's path
                                                            inputBuffer[ind(pixel)].X.rcVertex, 2, inputBuffer[ind(pixel)].X.path, // information from the base path aka canonical sample
                                                            1.f, // previous thp = 1
                                                            Y_neighborDomain, j_inverse, sg, d); //out or inout variables
                    } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                        
                    }
                    if (success) {
                        m_den += inputBuffer[ind(neighbor)].c * luminance(Y_neighborDomain.p_hat) * j_inverse; // we just did the inverse shift mapping, jacobian = |T(-1)(y)|
                    }
                }
            }
            if (kPerformTemporalReuse) {
                if (pixelHasValidTemporalPos) { //try to shift Y into last frame
                    Sample Y_lastFrame = {};
                    float j_inverse = inputBuffer[ind(pixel)].J; // initialize to srcG, src is this pixel
                    bool success = false;
                    ShiftDebugData d = {};
                    if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                    {
                        HitInfo dstHit = HitInfo(temporalVBuffer[pixelLastFrame]);
                        ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.data.prevPosW, -temporalViewW[pixelLastFrame].xyz);
                        success = performReconnectionShift(ShiftMappingType::Reconnection, false, true, // settings for the shift: reconnection, src is this frame and dst is last frame
                            y1_SD, // vertex right before reconnection vert on the dst domain's path
                            inputBuffer[ind(pixel)].X.rcVertex, 2, inputBuffer[ind(pixel)].X.path,   // information from the base path aka canonical sample
                            1.f,                              // previous thp = 1
                            Y_lastFrame, j_inverse, sg, d); // out or inout variables
                    } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                    }
                    if (success) {
                        m_den += temporalBuffer[ind(pixelLastFrame)].c * luminance(Y_lastFrame.p_hat) * j_inverse;
                    }
                }
            }
            float w_i = (m_num / m_den) * luminance(r.X.p_hat) * r.W_X; // jacobian is 1 bc alr in right domain
            if (!isnan(w_i)) {
                r.X.w_i = w_i;
                r.w_sum = w_i;
                r.W_X *= (m_num / m_den); // W_X gets set to w_sum / luminance(X) when updating reservoir, terms cancel
            }
        #endif

        /*
        outputColor[pixel] = float4(0, 0, 0, 1);
        // test reconnection shift by shifting pixel to neighbor and then back to pixel, also test neighbor to pixel and then back to neighbor
        int2 neighbor = int2(0);
        bool validNeighbor = getSpatialNeighbor(pixel, 0, neighbor);
        bool success = false;
        Sample Y_neighborDomain = {};
        float jacobian = inputBuffer[ind(pixel)].J; //J is the jacobian denom rn, so the portion with data from the src
        if (validNeighbor) {
            //shift from pixel to neighbor
            ShiftDebugData d = {};
            HitInfo dstHit = HitInfo(vBuffer[neighbor]); // get hit info for the vertex right before rc shift vertex on this pixel
            ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[neighbor].xyz);
            success = performReconnectionShift(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                                         y1_SD, // vertex right before reconnection vert on the dst domain's path
                                         inputBuffer[ind(pixel)].X.rcVertex, 2, inputBuffer[ind(pixel)].X.path,     // information from the base path aka pixel
                                         1.f,                                // previous thp = 1
                                         Y_neighborDomain, jacobian, sg, d); // out or inout variables
            debugData.J_pixelToNeighbor = jacobian;
            if (success) {
                //try shifting back to pixel
                dstHit = HitInfo(vBuffer[pixel]);
                y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[pixel].xyz);
                Sample Y_shiftBack = {};
                float backJacobian = jacobian * inputBuffer[ind(pixel)].J; //this should the numerator of jacobian of pixel to neighbor
                bool backSuccess = performReconnectionShift(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                                         y1_SD, // vertex right before reconnection vert on the dst domain's path
                                         Y_neighborDomain.rcVertex, 2, Y_neighborDomain.path,     // information from the base path aka neighbpr
                                         1.f,                                // previous thp = 1
                                         Y_shiftBack, backJacobian, sg, d); // out or inout variables
                if(backSuccess){
                    // need: p hat of shifted back needs to be abt the same as original
                    // jacobians to cancel to 1
                    debugData.J_neighborToPixel = backJacobian;
                    Sample og = inputBuffer[ind(pixel)].X;
                    debugData.orig_p_hat = og.p_hat;
                    debugData.pixelToNeighbor_p_hat = Y_neighborDomain.p_hat;
                    debugData.neighborToPixel_p_hat = Y_shiftBack.p_hat;

                    float3 diff = abs(og.p_hat - Y_shiftBack.p_hat);
                    float4 color = (1, 1, 1, 1);
                    if (dot(diff, diff) > 1e-3) {
                        color.x = 0;
                    }

                    float err = abs(jacobian * backJacobian - 1.f);
                    if (err > 1e-3) {
                        color.y = 0;
                    }
                    outputColor[pixel] = color;
                } else {
                    outputColor[pixel] = float4(1, 0, 0, 1);
                }
            }
        }*/
        

        // Spatial: do one iteration of spatial reuse
        int2 neighbor = int2(0);
        bool validNeighbor = getSpatialNeighbor(pixel, gSpatialPassIndex, neighbor);
        if (validNeighbor) {
            Sample neighbor_curDomain = {};
            float w_i = 0.f;
            float jacobian = inputBuffer[ind(neighbor)].J; // denom of jacobian of shifting X from neighbor into [pixel]'s domain, |T(X)|
            bool shiftSucceeded = false;
            ShiftDebugData d = {};
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                HitInfo dstHit = HitInfo(vBuffer[pixel]);
                ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[pixel].xyz);
                shiftSucceeded = performReconnectionShift(ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                    y1_SD, // vertex right before reconnection vert on the dst domain's path
                    inputBuffer[ind(neighbor)].X.rcVertex, 2, inputBuffer[ind(neighbor)].X.path,   // information from the base path aka neighbor's sample
                    1.f,                              // previous thp = 1
                    neighbor_curDomain, jacobian, sg, d); // out or inout variables
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
               
            }

            float newJ = 0.f;
            if (shiftSucceeded) {
                newJ = jacobian * inputBuffer[ind(neighbor)].J;
                float m_num = inputBuffer[ind(neighbor)].c * luminance(inputBuffer[ind(neighbor)].X.p_hat) / jacobian; // i is the domain that [preshiftNeighbor] came from
                float m_den = m_num;
                // process new/canonical candidate for m_den
                m_den += inputBuffer[ind(pixel)].c * luminance(neighbor_curDomain.p_hat); // jacobian is 1 since it came from this domain
                // add terms to m_den for what if this sample came from the other spatial domains
                Sample shifted = {};
                float j_inverse = 0.f;
                bool success = false;
                ShiftDebugData d = {};
                for (int j = 0; j < kNumSpatialNeighbors; j++) {
                    if (j != gSpatialPassIndex) {
                        shifted = {};
                        d = {};
                        j_inverse = newJ; // jacobian denom of shifting samples into this spatial neighbor's domain, |T(-1)(y)|, is jacobianNum of shifting from neighbor to [pixel]
                        int2 otherNeighbor = int2(0);
                        bool validOtherNeighbor = getSpatialNeighbor(pixel, j, otherNeighbor);
                        if (!validOtherNeighbor) continue;

                        if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                        {
                            HitInfo dstHit = HitInfo(vBuffer[otherNeighbor]); // get hit info for the vertex right before rc shift vertex on this pixel
                            ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[otherNeighbor].xyz);
                            success = performReconnectionShift( ShiftMappingType::Reconnection, false, false, // settings for the shift: reconnection, no temporal src or dst
                                y1_SD, // vertex right before reconnection vert on the dst domain's path
                                neighbor_curDomain.rcVertex, 2, neighbor_curDomain.path, // information from the base path, which is in [pixel]'s domain
                                1.f,                     // previous thp = 1
                                shifted, j_inverse, sg, d); // out or inout variables
                        } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
                        {

                        }
                        if (success) {
                            m_den += inputBuffer[ind(otherNeighbor)].c * luminance(shifted.p_hat) * j_inverse;
                        }
                    }
                }
                if (kPerformTemporalReuse) { // add to mis denom for temporal, shift this sample into what it would have looked like last frame
                    j_inverse = newJ;
                    if (pixelHasValidTemporalPos) {
                        Sample shifted = {};
                        success = false;
                        d = {};
                        if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                        {
                            HitInfo dstHit = HitInfo(temporalVBuffer[pixelLastFrame]);
                            ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.data.prevPosW, -temporalViewW[pixelLastFrame].xyz);
                            success = performReconnectionShift( ShiftMappingType::Reconnection, false, true,  // settings for the shift: reconnection, src is this frame and dst is last frame
                                y1_SD, // vertex right before reconnection vert on the dst domain's path
                                neighbor_curDomain.rcVertex, 2, neighbor_curDomain.path, // information from the base path aka canonical sample
                                1.f,                            // previous thp = 1
                                shifted, j_inverse, sg, d);    // out or inout variables
                        } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
                        {
                        }
                        
                        if (success) {
                            m_den += temporalBuffer[ind(pixelLastFrame)].c * luminance(shifted.p_hat) * j_inverse;
                        }
                    }
                }

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = (m_num / m_den) * luminance(neighbor_curDomain.p_hat) * inputBuffer[ind(neighbor)].W_X * jacobian;
                if (isnan(w_i)) w_i = 0.f;
            }
            // update reservoir streaming (in new buffer)
            neighbor_curDomain.w_i = w_i;
            if(r.updateReservoir(neighbor_curDomain, w_i, inputBuffer[ind(neighbor)].c, sg))
            {
                r.J = newJ;
            }
        }
        outputBuffer[ind(pixel)] = r;
        debugBuffer[pixelInd] = debugData;

        #if IS_LAST_PASS
            // If this is the last pass, should calculate the output color
            outputColor[pixel] = float4(r.X.p_hat * r.W_X + di_bgBuffer[pixelInd], 1);
        #endif
    }
}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID) {
    gResamplePass.execute(threadID.xy);
}
