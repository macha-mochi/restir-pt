import RestirReservoir;
import ShiftMapping;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    uint2 gOutputDimensions; // dimensions of gOutputColor
}

// Inputs (will also modify reservoirs)
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Originally optional but we include it
Texture2D<float2> gMotionVectors;
Texture2D<PackedHitInfo> gTemporalVBuffer;
Texture2D<float4> gTemporalViewW;

StructuredBuffer<Reservoir> gCandidateBuffer;
StructuredBuffer<Reservoir> gTemporalBuffer;
RWStructuredBuffer<Reservoir> gResampledBuffer;
RWStructuredBuffer<float3> gDI_BGBuffer; // direct illumination / env map bg buffer, to be added to f_X * W_X at the end
// Outputs
RWTexture2D<float4> gOutputColor;

//Debug
RWStructuredBuffer<DebugData> debugBuffer;
RWStructuredBuffer<Path> gPathDebugBuffer;

static const bool kPerformSpatialReuse = USE_SPATIAL;
static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
static const uint kSpatialNeighborRadius = 20;
static const bool kPerformTemporalReuse = USE_TEMPORAL;
static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

struct DebugData
{
    int2 tPos;
    bool updatedReservoir[2];
    float mi_num[2];
    float mi_den[2];
    float wi[2];

    __init() {
        tPos = int2(0);
        updatedReservoir = bool[2](false);
        mi_num = float[2](0);
        mi_den = float[2](0);
        wi = float[2](0);
    }
};

uint ind(uint2 pixel) {
    return pixel.y * gOutputDimensions.x + pixel.x;
}

// Finds the pixel position corresponding to the triangle hit by this frame's [pixel], last frame. Returning false if it would be out of bounds
// intended usage: use [result] to sample a Texture2D holding data from LAST FRAME. ex: sampling last frame's vbuffer for where our CURRENT frame's pixel would have been in the last frame.
bool calcLastFramePixelPos(uint2 pixel, out int2 result) {
    float2 inter = (float2)pixel + float2(0.5f);
    inter += gMotionVectors[pixel] * gOutputDimensions;
    result = (int2)inter;
    return all(result >= 0) && all(result < gOutputDimensions);
}

//finds screen space position of posW last frame.
int2 worldToScreenTemporal(float3 posW, uint2 dims) {
    float4 clipSpace = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(posW, 1.f));
    float3 ndc = clipSpace.xyz / clipSpace.w; // in range [-1, 1] if on screen
    float2 coord = float2((ndc.x + 1) / 2, (1 - ndc.y) / 2) * float2(dims);
    return (int2)coord;
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    let rayFlags = RAY_FLAG_NONE; // gets OR'd with the flag passed into rayquery generic

    q.TraceRayInline(gScene.rtAccel, rayFlags, 0xff /* instance inclusion mask */, ray);

    q.Proceed();
    if (q.CommittedStatus() == COMMITTED_NOTHING) {
        return true;
    } else {
        /*if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            TriangleHit triangleHit;
            triangleHit.instanceID = GeometryInstanceID(q.CommittedInstanceID(), q.CommittedGeometryIndex());
            triangleHit.primitiveIndex = q.CommittedPrimitiveIndex();
            triangleHit.barycentrics = q.CommittedTriangleBarycentrics();
            hitPos = gScene.getVertexData(triangleHit).posW;
        }*/ // use for debugging if you ever need to find triangle hit data
        return false;
    }
}

// Returns a pixel location randomly selected from within a certain radius of the current pixel in [neighbor]. Returns false if it was oob
bool getSpatialNeighbor(uint2 pixel, uint radius, inout SampleGenerator sg, out int2 neighbor) {
    int2 offset = (int2)(radius * sample_disk(sampleNext2D(sg)));
    neighbor = pixel + offset;
    if (neighbor.x < 0 || neighbor.x >= gOutputDimensions.x || neighbor.y < 0 || neighbor.y >= gOutputDimensions.y) {
        return false;
    }
    return true;
}

/**
 * Does restir
 */
[numthreads(16, 16, 1)]
void spatiotemporalResampling(uint3 threadID: SV_DispatchThreadID) {
    uint2 pixel = threadID.xy;

    if (any(pixel >= gOutputDimensions)) return; // return if oob

    uint pixelInd = ind(pixel);
    Reservoir r;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    DebugData d = {};

    // Prep spatial domain data
    bool validSpatialNeighbor[kNumSpatialNeighbors];
    int2 spatialNeighbors[kNumSpatialNeighbors];
    if (kPerformSpatialReuse) {
        for (int i = 0; i < kNumSpatialNeighbors; i++) {
            int2 neighbor;
            validSpatialNeighbor[i] = getSpatialNeighbor(pixel, kSpatialNeighborRadius, sg, neighbor);
            if (validSpatialNeighbor[i]) {
                spatialNeighbors[i] = neighbor;
            } else {
                spatialNeighbors[i] = int2(-1, -1);
            }
        }
    }
    int2 pixelLastFrame = int2(0, 0);
    bool pixelHasValidTemporalPos = calcLastFramePixelPos(pixel, pixelLastFrame);
    d.tPos = pixelLastFrame;

    // New candidate generated this frame
    float m_num = gCandidateBuffer[pixelInd].c * luminance(gCandidateBuffer[pixelInd].X.p_hat); // alr in the correct domain
    float m_den = m_num;
    if (kPerformSpatialReuse) {
        for (int i = 0; i < kNumSpatialNeighbors; i++) {
            if (validSpatialNeighbor[i]) {
                Sample Y_neighborDomain = Sample::createNull();
                float j_inverse = 0.f;
                // shift Y into others' domain: what if it came from spatial neighbors' domain?
                bool success = false;
                if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                {
                    success = performReconnectionShift(pixel, false, spatialNeighbors[i], false, gCandidateBuffer[pixelInd].X, Y_neighborDomain, j_inverse, sg, d);
                } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                }
                if (success) {
                    m_den += gCandidateBuffer[ind(spatialNeighbors[i])].c * luminance(Y_neighborDomain.p_hat) * j_inverse; // we just did the inverse shift mapping, jacobian = |T(-1)(y)|
                }
            }
        }
    }
    if (kPerformTemporalReuse) {
        if(pixelHasValidTemporalPos){
            Sample Y_lastFrame = Sample::createNull();
            float j_inverse = 0.f; // this is the jacobian of shifting Y into last frame's domain, so it's an inverse shift mapping
            // what if Y came from last frame's domain?
            bool success = performReconnectionShift(pixel, false, pixelLastFrame, true, gCandidateBuffer[pixelInd].X, Y_lastFrame, j_inverse, sg, d);
            if (success) {
                m_den += gTemporalBuffer[ind(pixelLastFrame)].c * luminance(Y_lastFrame.p_hat) * j_inverse;
            }
        }
    }
    float w_i = (m_num / m_den) * luminance(gCandidateBuffer[pixelInd].X.p_hat) * gCandidateBuffer[pixelInd].W_X; // jacobian is 1
    if (isnan(w_i)) w_i = 0.f;
    d.updatedReservoir[0] = r.updateReservoir(gCandidateBuffer[pixelInd].X, w_i, gCandidateBuffer[pixelInd].c, sg);
    d.mi_num[0] = m_num;
    d.mi_den[0] = m_den;
    d.wi[0] = w_i;

    // Temporal
    if (kPerformTemporalReuse) {
        if (pixelHasValidTemporalPos) {
            Sample X_thisFrame = Sample::createNull();
            float w_i = 0.f;
            float jacobian = 0.f; // this is the jacobian of shifting last frame's sample into this domain, so this is the forward jacobian
            bool success = performReconnectionShift(pixelLastFrame, true, pixel, false, gTemporalBuffer[ind(pixelLastFrame)].X, X_thisFrame, jacobian, sg, d);

            if (success) {
                float m_num = gTemporalBuffer[ind(pixelLastFrame)].c * luminance(gTemporalBuffer[ind(pixelLastFrame)].X.p_hat) / jacobian;
                float m_den = m_num;
                // what if this came from the new candidate's domain?
                m_den += gCandidateBuffer[pixelInd].c * luminance(X_thisFrame.p_hat); // jacobian is 1 since X_thisframe alr came from this domain
                // what if this came from spatial neighbors' domains?
                if (kPerformSpatialReuse) {
                    Sample shifted = Sample::createNull();
                    float j_inverse = 0.f; // jacobian of shifting pixel into neighbors' domain, |T(-1)(y)|
                    bool success = false;
                    for (int j = 0; j < kNumSpatialNeighbors; j++) {
                        if (validSpatialNeighbor[j]) {
                            success = performReconnectionShift(pixel, false, spatialNeighbors[j], false, X_thisFrame, shifted, j_inverse, sg, d);
                            if (success) {
                                m_den += gCandidateBuffer[ind(spatialNeighbors[j])].c * luminance(shifted.p_hat) * j_inverse;
                            }
                        }
                    }
                }

                w_i = (m_num / m_den) * luminance(X_thisFrame.p_hat) * gTemporalBuffer[ind(pixelLastFrame)].W_X * jacobian;
                d.mi_num[1] = m_num;
                d.mi_den[1] = m_den;
                if (isnan(w_i)) w_i = 0.f;
            }
            d.wi[1] = w_i;
            X_thisFrame.w_i = w_i;
            d.updatedReservoir[1] = r.updateReservoir(X_thisFrame, w_i, gTemporalBuffer[ind(pixelLastFrame)].c, sg);
        }
    }

    // Spatial
    if (kPerformSpatialReuse) {
        for (int i = 0; i < kNumSpatialNeighbors; i++) {
            if (validSpatialNeighbor[i]) {
                uint neighborInd = ind(spatialNeighbors[i]);
                Sample neighbor_curDomain = Sample::createNull();
                float w_i = 0.f;
                float jacobian = 0.f; // jacobian of shifting X into [pixel]'s domain, |T(X)|
                bool shiftSucceeded = false;
                if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
                {
                    shiftSucceeded = performReconnectionShift(spatialNeighbors[i], false, pixel, false, gCandidateBuffer[neighborInd].X, neighbor_curDomain, jacobian, sg, d);
                } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
                {
                    // do hybrid mapping
                }

                // d.mi[i + 1] = 0;
                // d.wi[i + 1] = 0;
                if (shiftSucceeded) {
                    float m_num = gCandidateBuffer[neighborInd].c * luminance(gCandidateBuffer[neighborInd].X.p_hat) / jacobian; // i is the domain that [preshiftNeighbor] came from
                    float m_den = m_num;
                    // process new candidate for m_den
                    m_den += gCandidateBuffer[pixelInd].c * luminance(neighbor_curDomain.p_hat); // jacobian is 1 since it came from this domain
                    // process other spatial ones
                    Sample shifted = Sample::createNull();
                    float j_inverse = 0.f; // jacobian of shifting samples into this spatial neighbor's domain, |T(-1)(y)|
                    bool success = false;
                    for (int j = 0; j < kNumSpatialNeighbors; j++) {
                        if (j != i && validSpatialNeighbor[j]) {
                            success = performReconnectionShift(pixel, false, spatialNeighbors[j], false, neighbor_curDomain, shifted, j_inverse, sg, d);
                            if (success) {
                                m_den += gCandidateBuffer[ind(spatialNeighbors[j])].c * luminance(shifted.p_hat) * j_inverse;
                            }
                        }
                    }
                    if (kPerformTemporalReuse) { // add to mis denom for temporal, shift this sample into what it would have looked like last frame
                        j_inverse = 0.f;
                        if (pixelHasValidTemporalPos) {
                            Sample shifted = Sample::createNull();
                            bool success = performReconnectionShift(pixel, false, pixelLastFrame, true, neighbor_curDomain, shifted, j_inverse, sg, d);
                            if (success) {
                                m_den += gTemporalBuffer[ind(pixelLastFrame)].c * luminance(shifted.p_hat) * j_inverse;
                            }
                        }
                    }

                    // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                    w_i = (m_num / m_den) * luminance(neighbor_curDomain.p_hat) * gCandidateBuffer[neighborInd].W_X * jacobian;
                    if (isnan(w_i)) w_i = 0.f;
                    // d.mi[i + 1] = m_num / m_den;
                    // d.wi[i + 1] = w_i;
                }
                // update reservoir streaming (in new buffer)
                neighbor_curDomain.w_i = w_i;
                r.updateReservoir(neighbor_curDomain, w_i, gCandidateBuffer[neighborInd].c, sg);
            }
        }
    }
    
    gResampledBuffer[pixelInd] = r;
    debugBuffer[pixelInd] = d;
    gPathDebugBuffer[pixelInd].rcIndex = 2;

    float3 f_X = r.X.p_hat;
    float W_X = r.W_X;
    float3 di_bg = gDI_BGBuffer[pixelInd];
    gOutputColor[pixel] = float4(f_X * W_X + di_bg, 1.f);
}
