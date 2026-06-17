import RestirReservoir;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
}

// Inputs (will also modify reservoirs)
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Originally optional but we include it
RWStructuredBuffer<Reservoir> gCandidateBuffer;
RWStructuredBuffer<Reservoir> gResampledBuffer;
RWStructuredBuffer<float3> gDI_BGBuffer; // direct illumination / env map bg buffer, to be added to f_X * W_X at the end
// Outputs
RWTexture2D<float4> gOutputColor;

RWStructuredBuffer<DebugData> debugBuffer;

static const uint kNumSpatialNeighbors = 1;
static const uint kSpatialNeighborRadius = 20;
static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

struct DebugData
{
    uint neighborInd;
    bool shiftSucceeded;

    bool hitInfoInvalid;
    bool occluded;

    float3 rcPos;
    float3 toRCVert;
    float3 bsdfcos1;
    float3 bsdfcos2;
    float srcG;
    float dstG;
    float jacobian;
    float3 shifted_p_hat;

    __init() {
        shifted_p_hat = { -1, -1, -1 };
        jacobian = -1.f;
        shiftSucceeded = false;
    }
};

float genBalanceHeuristic(float c_i, float p_from_i_y, float c_j, float p_from_j_y) {
    return c_i * p_from_i_y / (c_i * p_from_i_y + c_j * p_from_j_y);
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
bool performReconnectionShift(uint2 srcPixel, uint2 dstPixel, out Sample Y, out float jacobian, inout SampleGenerator sg, inout DebugData d) {
    uint2 dims;
    gOutputColor.GetDimensions(dims.x, dims.y);

    const float3 primaryRayOrigin = gScene.camera.getPosition();

    uint srcBufferInd = srcPixel.y * dims.x + srcPixel.x;
    uint dstBufferInd = dstPixel.y * dims.x + dstPixel.x;

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

        RCVertex rcVertex = gResampledBuffer[srcBufferInd].X.rcVertex;
        HitInfo rcHitInfo = HitInfo(rcVertex.packedHitInfo);
        if (!rcHitInfo.isValid()) { // TODO: should have another condition here. used to be X.thp_denom == 0 as in path ended before x2 but lk incoming radiance would be good check
            d.hitInfoInvalid = true;
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
        d.srcG = srcG;

        //Proceed with re-eval bsdf at y1
        float3 toRCVert = rcPos - dstRayOrigin; //use sd.pos- rayOrigin pattern
        float3 rcIncomingDir = normalize(toRCVert);
        d.rcPos = rcPos;
        d.toRCVert = toRCVert;
        if (!traceShadowRay(dstRayOrigin, rcIncomingDir, length(toRCVert))) {
            d.occluded = true;
            return false; //this shift is not valid if x2 is occluded from y1
        }
        float3 new_thp = dstMI1.evalBSDF(dstSD, rcIncomingDir, rcVertex.prevSampledLobeType, sg); // THIS WILL RETURN 0 BC NOT IMPLEMENTED
        d.bsdfcos1 = new_thp;

        // Now re-eval bsdf at x2
        dstSD = loadShadingData(rcHitInfo, dstRayOrigin, rcIncomingDir); // sd for y2 = x2

        d.dstG = abs(dot(dstSD.faceN, -rcIncomingDir)) / dot(toRCVert, toRCVert);
        jacobian = abs(dot(dstSD.faceN, -rcIncomingDir)) / (dot(toRCVert, toRCVert) * srcG); //jacobian: ratio of cos * dist^2, for y1->x2 / x1->x2

        let lod2 = ExplicitLodTextureSampler(0.f);
        let mi2 = gScene.materials.getMaterialInstance(dstSD, lod2);                        // mi for y2 = x2
        new_thp *= mi2.evalBSDF(dstSD, rcVertex.outgoingDir, rcVertex.sampledLobeType, sg); // bsdf * cos for y2
        d.bsdfcos2 = mi2.evalBSDF(dstSD, rcVertex.outgoingDir, rcVertex.sampledLobeType, sg);

        // Find p_hat for shifted X
        Y.p_hat = new_thp * rcVertex.incomingRadiance;
        // you will find shifted w_i afneter this method since we need to eval mis weight
        Y.rcVertex = rcVertex;

        return jacobian != 0.f; //if jacobian = 0, w_i will be 0 so it'll be as if the shift failed
    }
    return false;
}

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
    Reservoir r = gCandidateBuffer[bufferInd];

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    // temporal

    // spatial
    int2 neighbor;
    DebugData d = {};
    /*for (int i = 0; i < kNumSpatialNeighbors; i++) {
        if (getSpatialNeighbor(pixel, dims, kSpatialNeighborRadius, sg, neighbor)) {
            uint neighborInd = neighbor.y * dims.x + neighbor.x;
            d.neighborInd = neighborInd;
            Sample neighbor = gCandidateBuffer[neighborInd].X;
            // d.preShiftNeighbor = neighbor;
            Sample shiftedNeighbor = Sample::createNull();
            float w_i = 0.f;
            float jacobian = 0.f; // jacobian of shifting X into Y's domain
            bool shiftSucceeded = false;
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                shiftSucceeded = performReconnectionShift(neighbor, pixel, neighborInd, shiftedNeighbor, jacobian, sg, d);
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
                // do hybrid mapping
            }
            d.shiftSucceeded = shiftSucceeded;

            if (shiftSucceeded) {
                // evaluate resampling mis weight
                d.jacobian = jacobian;
                float p_from_i_y = luminance(neighbor.p_hat_X) / jacobian; // i is the domain that [neigbor] came from
                float p_from_j_y = luminance(r.X.p_hat_X); // j is the domain of the current pixel so we use identity shift, jacobian = 1
                float mi = genBalanceHeuristic(gCandidateBuffer[neighborInd].c, p_from_i_y, r.c, p_from_j_y);

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = mi * luminance(shiftedNeighbor.p_hat_X) * r.W_X * jacobian;

                d.shifted_p_hat = shiftedNeighbor.p_hat_X;
                d.mi = mi;
                d.wi = w_i;
            }
            // update reservoir streaming (in new buffer)
            r.updateReservoir(shiftedNeighbor, w_i, gCandidateBuffer[neighborInd].c, sg);
        }
    }*/
    Sample shifted = Sample::createNull();
    float jacobian = 0.f;
    bool shiftSucceeded = performReconnectionShift(pixel, pixel, shifted, jacobian, sg, d);
    d.shiftSucceeded = shiftSucceeded;
    d.jacobian = jacobian;
    d.shifted_p_hat = shifted.p_hat;

    if (d.hitInfoInvalid) {
        gOutputColor[pixel] = float4(0, 0.5, 0, 1); //short path, ok
    } else if (d.occluded) {
        gOutputColor[pixel] = float4(1, 0, 0, 1);
    } else if (!shiftSucceeded) {
        gOutputColor[pixel] = float4(1, 0, 1, 1);
    } else {
        gOutputColor[pixel] = float4(1, 1, 1, 1);
    }
    gResampledBuffer[bufferInd] = r;
    debugBuffer[bufferInd] = d;
    return;

    float3 f_X = r.X.p_hat;
    float W_X = r.W_X;
    float3 di_bg = gDI_BGBuffer[bufferInd];
    gOutputColor[pixel] = float4(f_X * W_X + di_bg, 1.f);
}
