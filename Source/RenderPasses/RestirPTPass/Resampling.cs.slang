import RestirReservoir;
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
StructuredBuffer<Reservoir> gCandidateBuffer;
RWStructuredBuffer<Reservoir> gResampledBuffer;
RWStructuredBuffer<float3> gDI_BGBuffer; // direct illumination / env map bg buffer, to be added to f_X * W_X at the end
// Outputs
RWTexture2D<float4> gOutputColor;

//Debug
RWStructuredBuffer<DebugData> debugBuffer;
RWStructuredBuffer<Path> gPathDebugBuffer;

static const uint kNumSpatialNeighbors = 3;
static const uint kSpatialNeighborRadius = 20;
static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

struct DebugData
{
    bool nanFlag;
    float3 shifted_p_hat;
    float jacobian;
    bool updatedReservoir[4];
    float mi[4];
    float wi[4];
    int neighborInds[3];

    __init() {
        nanFlag = false;
        shifted_p_hat = float3(0.f);
        jacobian = 0.f;
        updatedReservoir = bool[4](false);
        mi = float[4](0);
        wi = float[4](0);
        neighborInds = int[3](-1);
    }
};

float genBalanceHeuristic(float c_i, float p_from_i_y, float c_j, float p_from_j_y) {
    return c_i * p_from_i_y / (c_i * p_from_i_y + c_j * p_from_j_y);
}

uint ind(uint2 pixel) {
    return pixel.y * gOutputDimensions.x + pixel.x;
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
bool getSpatialNeighbor(uint2 pixel, uint2 dims, uint radius, inout SampleGenerator sg, out int2 neighbor) {
    int2 offset = (int2)(radius * sample_disk(sampleNext2D(sg)));
    neighbor = pixel + offset;
    if(neighbor.x < 0 || neighbor.x >= dims.x || neighbor.y < 0 || neighbor.y >= dims.y) {
        return false;
    }
    return true;
}

// Shifts sample X into the current pixel's domain using reconnection shift. srcPixel = pixel to shift from, dstPixel = pixel to shift to
bool performReconnectionShift(uint2 srcPixel, uint2 dstPixel, in Sample X, out Sample Y, out float jacobian, inout SampleGenerator sg, inout DebugData d) {
    uint2 dims;
    gOutputColor.GetDimensions(dims.x, dims.y);

    const float3 primaryRayOrigin = gScene.camera.getPosition();

    uint srcBufferInd = srcPixel.y * dims.x + srcPixel.x;

    const HitInfo srcHit = HitInfo(gVBuffer[srcPixel]);
    const HitInfo dstHit = HitInfo(gVBuffer[dstPixel]);
    if (srcHit.isValid() && dstHit.isValid())
    {
        //Both pixels were valid primary hits. Find modified thp

        ///>> Find bsdf * cos * bsdf * cos for offset path, so you can multiply by incomingRadiance of base path.

        // Load shading data.
        ShadingData dstSD = loadShadingData(dstHit, primaryRayOrigin, -gViewW[dstPixel].xyz);
       
        // Compute ray origin for new rays spawned from the G-buffer.
        float3 dstRayOrigin = dstSD.computeRayOrigin();

        // Create material instance at shading point.
        let lod1 = ExplicitLodTextureSampler(0.f);
        let dstMI1 = gScene.materials.getMaterialInstance(dstSD, lod1);

        RCVertex rcVertex = X.rcVertex; //gCandidateBuffer[srcBufferInd].X.rcVertex;
        HitInfo rcHitInfo = HitInfo(rcVertex.packedHitInfo);
        if (!rcHitInfo.isValid()) { // TODO: should have another condition here. used to be X.thp_denom == 0 as in path ended before x2 but lk incoming radiance would be good check
            return false; // can't do shift if x2 isn't valid
        }

        // Since we have a valid x2 vertex, first find geometry term for base path to use later in jacobian
        ShadingData srcSD = loadShadingData(srcHit, primaryRayOrigin, -gViewW[srcPixel].xyz);
        float3 srcRayOrigin = srcSD.computeRayOrigin(); //this is the place where the ray from x1 to x2 originated
        //Find rc position
        float3 rcPos;
        if (rcHitInfo.getType() == HitType::Triangle) {
            rcPos = gScene.getVertexData(rcHitInfo.getTriangleHit()).posW;
        } else {
            // oither types not supported for now
            return false;
        }
        float3 srcX1_to_X2 = rcPos - srcRayOrigin;
        float3 dirX1_to_X2 = normalize(srcX1_to_X2);
        srcSD = loadShadingData(rcHitInfo, srcRayOrigin, dirX1_to_X2); // set srcSD to be shading data at x2 on the base path
        float srcG = abs(dot(srcSD.faceN, -dirX1_to_X2)) / dot(srcX1_to_X2, srcX1_to_X2);

        //Proceed with re-eval bsdf at y1
        float3 toRCVert = rcPos - dstRayOrigin; //use sd.pos- rayOrigin pattern
        float3 rcIncomingDir = normalize(toRCVert);
        float3 hitPos = float3(0.f);
        float epsilon = 1e-5f; //shorten shadow ray distance to avoid reporting rcPos as intersection
        if (!traceShadowRay(dstRayOrigin, rcIncomingDir, length(toRCVert) - epsilon)) {
            return false; //this shift is not valid if x2 is occluded from y1
        }
        float3 new_thp = dstMI1.evalBSDF(dstSD, rcIncomingDir, rcVertex.prevSampledLobeType, sg);
        if (any(isnan(new_thp))) new_thp = float3(0, 0, 0);

        // Now re-eval bsdf at x2
        dstSD = loadShadingData(rcHitInfo, dstRayOrigin, rcIncomingDir); // sd for y2 = x2

        jacobian = abs(dot(dstSD.faceN, -rcIncomingDir)) / (dot(toRCVert, toRCVert) * srcG); //jacobian: ratio of cos * dist^2, for y1->x2 / x1->x2

        let lod2 = ExplicitLodTextureSampler(0.f);
        let mi2 = gScene.materials.getMaterialInstance(dstSD, lod2);                              // mi for y2 = x2
        float3 weight2 = mi2.evalBSDF(dstSD, rcVertex.outgoingDir, rcVertex.sampledLobeType, sg); // bsdf * cos for y2
        if (any(isnan(weight2))) weight2 = float3(0, 0, 0);
        new_thp *= weight2;

        // Find p_hat for shifted X
        Y.p_hat = new_thp * rcVertex.incomingRadiance;
        // you will find shifted w_i afneter this method since we need to eval mis weight
        Y.rcVertex = rcVertex;

        return jacobian != 0.f && luminance(Y.p_hat) != 0.f; //if jacobian = 0, or p hat is 0, w_i will be 0 so it'll be as if the shift failed
    }
    return false;
}

/**
 * Does restir
 */
[numthreads(16, 16, 1)] // TODO 16 was just chosen arbitrarily since that's what RTXDI uses
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
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
        int2 neighbor;
        validSpatialNeighbor[i] = getSpatialNeighbor(pixel, gOutputDimensions, kSpatialNeighborRadius, sg, neighbor);
        if (validSpatialNeighbor[i]) {
            spatialNeighbors[i] = neighbor;
        } else {
            spatialNeighbors[i] = int2(-1, -1);
        }
        //d.neighborInds[i] = ind(spatialNeighbors[i]);
    }

    // New candidate generated this frame
    float m_num = gCandidateBuffer[pixelInd].c * luminance(gCandidateBuffer[pixelInd].X.p_hat); // alr in the correct domain
    float m_den = m_num;
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
        if (validSpatialNeighbor[i]) {
            uint neighborInd = ind(spatialNeighbors[i]);
            Sample Y_neighborDomain = Sample::createNull();
            float jacobian = 0.f;
            // shift Y into others' domain
            bool success = false;
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                success = performReconnectionShift(pixel, spatialNeighbors[i], gCandidateBuffer[pixelInd].X, Y_neighborDomain, jacobian, sg, d);
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {

            }
            if (success) {
                m_den += gCandidateBuffer[pixelInd].c * luminance(Y_neighborDomain.p_hat) * jacobian; // we just did the inverse shift mapping, jacobian = |T(-1)(y)|
            }
        }
    }
    float w_i = (m_num == 0.f || m_den == 0.f) ? 0.f : (m_num / m_den) * luminance(gCandidateBuffer[pixelInd].X.p_hat) * gCandidateBuffer[pixelInd].W_X; // jacobian is 1
    d.updatedReservoir[0] = r.updateReservoir(gCandidateBuffer[pixelInd].X, w_i, gCandidateBuffer[pixelInd].c, sg);
    d.mi[0] = m_num / m_den;
    d.wi[0] = w_i;

    // Temporal

    // Spatial
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
        if (validSpatialNeighbor[i]) {
            uint neighborInd = ind(spatialNeighbors[i]);
            Sample neighbor_curDomain = Sample::createNull();
            float w_i = 0.f;
            float jacobian = 0.f; // jacobian of shifting X into [pixel]'s domain, |T(X)|
            bool shiftSucceeded = false;
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                shiftSucceeded = performReconnectionShift(spatialNeighbors[i], pixel, gCandidateBuffer[neighborInd].X, neighbor_curDomain, jacobian, sg, d);
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
                // do hybrid mapping
            }

            //d.mi[i + 1] = 0;
            //d.wi[i + 1] = 0;
            if (shiftSucceeded) {
                float m_num = gCandidateBuffer[neighborInd].c * luminance(gCandidateBuffer[neighborInd].X.p_hat) / jacobian; // i is the domain that [preshiftNeighbor] came from
                float m_den = m_num;
                // process new candidate for m_den
                m_den += gCandidateBuffer[pixelInd].c * luminance(gCandidateBuffer[pixelInd].X.p_hat); //jacobian is 1 since it came from this domain
                // process other spatial ones
                Sample shifted = Sample::createNull();
                float j_inverse = 0.f; // jacobian of shifting samples into this spatial neighbor's domain, |T(-1)(y)|
                bool success = false;
                for (int j = 0; j < kNumSpatialNeighbors; j++) {
                    if (j != i && validSpatialNeighbor[j]) {
                        success = performReconnectionShift(pixel, spatialNeighbors[j], neighbor_curDomain, shifted, j_inverse, sg, d);
                        if (success) {
                            m_den += gCandidateBuffer[ind(spatialNeighbors[j])].c * luminance(shifted.p_hat) * j_inverse;
                        }
                    }
                }

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = (m_num / m_den) * luminance(neighbor_curDomain.p_hat) * gCandidateBuffer[neighborInd].W_X * jacobian;
                d.shifted_p_hat = neighbor_curDomain.p_hat;
                d.jacobian = jacobian;
                //d.mi[i + 1] = m_num / m_den;
                //d.wi[i + 1] = w_i;
            }
            // update reservoir streaming (in new buffer)
            neighbor_curDomain.w_i = w_i;
            r.updateReservoir(neighbor_curDomain, w_i, gCandidateBuffer[neighborInd].c, sg);
            if (isnan(w_i)) {
                gOutputColor[pixel] = float4(1, 0, 1, 1);
                gResampledBuffer[pixelInd] = r;
                debugBuffer[pixelInd] = d;
                debugBuffer[0].nanFlag = true;
                gPathDebugBuffer[pixelInd].rcIndex = 2;
                return;
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
