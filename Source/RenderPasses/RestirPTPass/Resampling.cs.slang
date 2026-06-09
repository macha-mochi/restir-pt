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
RWStructuredBuffer<ShiftMappingInfo> gShiftMappingInfoBuffer; // shift mapping info buffer
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
    float jacobian;
    float3 bsdfcos1;
    float3 g1;
    float3 bsdfcos2;
    float3 toX2;
    bool specular2nd;
    float3 jacobian_num;
    float3 jacobian_denom;
    float mi;
    float wi;
    float3 shifted_p_hat;

    __init() {
        shifted_p_hat = { -1, -1, -1 };
        jacobian = -1.f;
        shiftSucceeded = false;
        mi = -1.f;
        wi = -1.f;
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

// Shifts sample X into the current pixel's domain using reconnection shift. X = sample to shift, Y = sample for current pixel.
bool performReconnectionShift(Sample X, uint2 pixelY, uint bufferIndX, out Sample Y, out float jacobian, inout SampleGenerator sg, inout DebugData d) {
    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = -gViewW[pixelY].xyz;

    const HitInfo hit = HitInfo(gVBuffer[pixelY]);
    if (hit.isValid())
    {
        // Pixel represents a valid primary hit. Re evaluate its bsdf

        const bool isCurveHit = hit.getType() == HitType::Curve;

        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin;
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            rayOrigin = sd.posW - sd.curveRadius * sd.frame.N;
        }
        else
        {
            rayOrigin = sd.computeRayOrigin();
        }

        // Create material instance at shading point.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi1 = gScene.materials.getMaterialInstance(sd, lod);

        HitInfo x2HitInfo = HitInfo(gShiftMappingInfoBuffer[bufferIndX].packedHitInfo);
        if (!x2HitInfo.isValid() || all(X.jacobian_denom == 0.f)) {
            return false; // can't do shift if x2 isn't valid or if jacobian denominator is 0
        }

        //Since we have a valid x2 vertex, proceed with re-eval bsdf at x1.
        float3 x2Pos = gShiftMappingInfoBuffer[bufferIndX].x2;
        float3 toX2 = x2Pos - rayOrigin;
        float3 wo = normalize(toX2);
        if(!traceShadowRay(rayOrigin, wo, length(toX2))){
            return false; //this shift is not valid if x2 is occluded from y1
        }
        float3 jacobian_num = mi1.eval(sd, wo, sg);
        d.bsdfcos1 = mi1.eval(sd, wo, sg);
        d.toX2 = toX2;

        // Update values for x2.
        sd = loadShadingData(x2HitInfo, rayOrigin, wo); // sd for x2;
        jacobian_num *= abs(dot(sd.faceN, -wo)) / dot(toX2, toX2); // geometry term: cos(angle between normal and ray dir) / distance^2
        d.g1 = abs(dot(sd.faceN, -wo)) / dot(toX2, toX2);
        let mi2 = gScene.materials.getMaterialInstance(sd, lod); // mi for x2
        wo = X.reconnection_vert_wo;
        jacobian_num *= mi2.eval(sd, wo, sg); // bsdf * cos for x2
        d.bsdfcos2 = mi2.eval(sd, wo, sg);
        d.jacobian_num = jacobian_num;
        d.jacobian_denom = X.jacobian_denom;
        d.specular2nd = ((mi2.getLobeTypes(sd)) & LobeType::Specular) != 0;

        // Find p_hat for shifted X
        Y.p_hat_X = X.p_hat_X * jacobian_num / X.jacobian_denom;
        // you will find shifted w_i after this method since we need to eval mis weight
        Y.reconnection_vert_wo = X.reconnection_vert_wo;
        Y.jacobian_denom = jacobian_num;

        jacobian = luminance(jacobian_num / X.jacobian_denom);

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
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
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
                float p_from_j_y = luminance(r.X.p_hat_X); // j is the domain of the current pixel so we use identity shift
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
    }
    gResampledBuffer[bufferInd] = r;
    debugBuffer[bufferInd] = d;

    float3 f_X = r.X.p_hat_X;
    float W_X = r.W_X;
    float3 di_bg = gDI_BGBuffer[bufferInd];
    gOutputColor[pixel] = float4(f_X * W_X + di_bg, 1.f);
}
