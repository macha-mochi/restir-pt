/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * Slang file copied from MinimalPathTracer.rt.slang but modified to generate the initial
 * candidates and store them in a reservoir for restir.
 *
 * The host sets the following defines:
 *
 * MAX_BOUNCES             Maximum number of indirect bounces (0 means no indirect).
 * COMPUTE_DIRECT          Nonzero if direct illumination should be included.
 * USE_IMPORTANCE_SAMPLING Nonzero if importance sampling should be used for sampling materials.
 * USE_ANALYTIC_LIGHTS     Nonzero if Falcor's analytic lights should be used.
 * USE_EMISSIVE_LIGHTS     Nonzero if emissive geometry should be used as lights.
 * USE_ENV_LIGHT           Nonzero if env map is available and should be used as light source.
 * USE_ENV_BACKGROUND      Nonzero if env map is available and should be used as background.
 * is_valid_<name>         1 if optional I/O buffer with this name should be used.
 */

#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import RestirReservoir;
import RestirHelpers;
import PathState;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
}

ParameterBlock<EnvMapSampler> envMapSampler; ///< Environment map sampler. Only valid when kUseEnvLight == true.
ParameterBlock<EmissiveLightSampler> emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional
Texture2D<float2> gMotionVectors; // Not used in this shader

// Outputs (not an actual output channel but actually just the reservoir buffer)
RWStructuredBuffer<Reservoir> gReservoirBuffer;
RWStructuredBuffer<float3> gDI_BGBuffer;
RWStructuredBuffer<RandomReplayInput> gReplayInputBuffer; // contains data that will be used for random replay later

RWTexture2D<float4> outputColor;

//Debug buffers
RWStructuredBuffer<DebugData> debugBuffer;
RWStructuredBuffer<PathViewerPath> gPathViewerPathBuffer;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;
static const float kShiftMappingType = SHIFT_MAPPING_TYPE;
static const float kHybridShiftRoughnessThreshold = 0.2f; //got this value from the OG paper

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

struct DebugData {
    float3 preRcThp;

    __init() {
        preRcThp = float3(1.f);
    }
}

/**
ShiftContext is used for things that we capture early on in the path but then stay the same, it lives in tracePath but not ray payload.
RayData is for stuff that changes at every vertex.
*/
struct ShiftContext{
    uint rcVertexInd;
    float3 thp_upToIncl_xk;
    float primaryRayFootprint; //used for hybrid only
    float jacobian_denom; // only set once rcvertexind is set!
    uint x_kPlus1_seed;
    float prevRayPdf; // during the loop, if vertex = i, this holds the pdf of the ray from vertex i-1 to vertex i

    __init(){
        rcVertexInd = 0;
        thp_upToIncl_xk = float3(0.f);
        primaryRayFootprint = 0.f;
        jacobian_denom = 0.f;
        x_kPlus1_seed = 0;
        prevRayPdf = 0.f;
    }
}

struct SampleData{
    float3 bsdf_p_hat;
    float bsdf_w_i;
    float3 nee_p_hat;
    float nee_w_i;
    float j_denom;
    float g; //cos / dist for x(i-1) to x_i

    // RC vertex info for nee case
    float3 nee_wo;
    float3 nee_Li;
    uint4 rcPackedHitInfo;

    // hybrid shift jacobian / rc vertex criteria info
    float nee_pdf; // needed for jacobian denom calculation
    float nee_g; // also needed for jacobian denom calc
    float dualRayFootprint; // this is for x_k, updated every hit and set to 0 if conditions not met
    uint neeSeed;           // State of random number generator right before we called generateNEESample

    __init(){
        bsdf_p_hat = {0.f, 0.f, 0.f};
        bsdf_w_i = 0.f;
        nee_p_hat = {0.f, 0.f, 0.f};
        nee_w_i = 0.f;
        j_denom = 0.f;
        g = 0.f;

        nee_Li = { 0.f, 0.f, 0.f };
        rcPackedHitInfo = {0, 0, 0, 0};
        nee_wo = { 0.f, 0.f, 0.f };
        neeSeed = 0;

        nee_pdf = 0.f;
        nee_g = 0.f;
        dualRayFootprint = 0.f;
    }
}

/**
 * Payload for scatter ray
 */
struct ScatterRayData
{
    PathState path; //1 uint of backing data, includes various flags and vertex index. ///< 0 at origin, 1 at primary hit, 2 at first secondary hit, etc.). Max is what fits in 10 bits

    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.
    float3 normal;    ///< Shading normal at origin. Used for light sampling and MIS weight calculations.
    float pdf; ///< Pdf of last path segment (the one leading to the current vertex). This is needed for MIS weight calculations at the current vertex for the bsdf
    float roughness;  ///< Roughness at origin, used for thresholding for hybrid shift

    float3 p_hat_thp; // Throughput for the p_hat calculations. doesn't include pdf, only bsdf * geo term
    float path_pdf;   ///< Current path pdf (just multiplying) in terms of solid angle (changed from area)
    uint prevSampledLobeType;
    uint sampledLobeType;

    SampleData sampleData; ///< Data that is only valid at one vertex and gets written to at each vertex

    SampleGenerator sg;
    SampleGenerator sgScatterRays; ///< Per-ray state for the sample generator (up to 16B).

    DebugData d;

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg, SampleGenerator sgScatterRays)
    {
        path = {};
        path.setActive();
        path.setVertexIndex(1); // start at 1 bc we only make this struct if it was a alid primary hit

        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.normal = float3(0, 0, 0);
        this.pdf = 0.f;
        this.roughness = 0.f;

        this.p_hat_thp = float3(1, 1, 1);
        this.path_pdf = 1.f;
        this.prevSampledLobeType = 0;
        this.sampledLobeType = 0;

        this.sampleData = {};

        this.sg = sg;
        this.sgScatterRays = sgScatterRays;
    }
};

/** Describes a path vertex.
 */
struct PathVertex
{
    uint index;        ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
    float3 pos;        ///< Vertex position.
    float3 faceNormal; ///< Geometry normal at the vertex (zero if not on a surface).
    bool frontFacing;  ///< True if path vertex is on the front-facing side (if on a surface).

    /** Initializes a path vertex.
        \param[in] index Vertex index.
        \param[in] pos Vertex position.
        \param[in] faceNormal Geometry normal.
        \param[in] frontFacing Front-facing flag.
    */
    __init(uint index, float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
    {
        this.index = index;
        this.pos = pos;
        this.faceNormal = faceNormal;
        this.frontFacing = frontFacing;
    }
    static PathVertex NULL = { -1, float3(0.f), float3(0.f), false };

    /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
        for visibility rays.
        \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
        \return Returns the offseted position.
    */
    float3 getRayOrigin(float3 rayDir)
    {
        return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
    }

    /** Returns the oriented face normal.
        \return Face normal flipped to the same side as the view vector.
    */
    float3 getOrientedFaceNormal()
    {
        return frontFacing ? faceNormal : -faceNormal;
    }
};

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] isCurveHit True if on curve hit.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData)
{
    rayData.prevSampledLobeType = rayData.sampledLobeType;

    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sgScatterRays, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.normal = sd.getOrientedFaceNormal();
        rayData.pdf = bsdfSample.pdf;
        rayData.path.setTransmission(bsdfSample.isLobe(LobeType::Transmission));
        rayData.path.setSpecular(bsdfSample.isLobe(LobeType::Specular));
        rayData.path.setDelta(bsdfSample.isLobe(LobeType::Delta));
        rayData.direction = bsdfSample.wo;
        rayData.roughness = mi.getProperties(sd).roughness;

        // bsdfsample.weight = f * dot(wo, n) / pdf
        // wo is normalized in world space
        rayData.p_hat_thp *= bsdfSample.weight;
        if (rayData.path.nonDelta()) {
            rayData.p_hat_thp *= bsdfSample.pdf;
            rayData.path_pdf *= bsdfSample.pdf; // pdf is in solid angle
        }
        rayData.sampledLobeType = bsdfSample.lobeType;

        return any(rayData.p_hat_thp > 0.f);
    }
    rayData.sampledLobeType = (uint)LobeType::None;

    return false;
}

// LIGHT SAMPLING / NEE FUNCTIONS

/** Generates a light sample on the environment map.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    if (!kUseEnvLight) return false;

    // Sample environment map.
    EnvMapSample lightSample;
    if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

    // Setup returned sample.
    ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    ls.pdf = lightSample.pdf;
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    ls.distance = kRayTMax;
    ls.dir = lightSample.dir;
    ls.g = 1.f; //since env map stays in solid angle

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the emissive geometry.
    \param[in] vertex Path vertex.
    \param[in] upperHemisphere True if only upper hemisphere should be considered.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.
    if (!kUseEmissiveLights) return false;

    TriangleLightSample tls;
    if (!emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls))
    {
        return false;
    }

    // Setup returned sample.
    ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
    ls.pdf = tls.pdf;
    // Offset shading and light position to avoid self-intersection.
    float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
    ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
    float3 toLight = lightPos - ls.origin;
    ls.distance = length(toLight);
    ls.dir = normalize(toLight);

    //Calculate geometry term. info needed: cos at light and distance to light which we have
    ls.g = abs(dot(tls.normalW, -ls.dir)) / (ls.distance * ls.distance);

    return any(ls.Li > 0.f);
}

/** Generates a light sample on the analytic lights.
    \param[in] vertex Path vertex.
    \param[in,out] sg Sample generator.
    \param[out] ls Struct describing valid samples.
    \return True if the sample is valid and has nonzero contribution, false otherwise.
*/
bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
{
    ls = {}; // Default initialization to avoid divergence at returns.

    uint lightCount = gScene.getLightCount();
    if (!kUseAnalyticLights || lightCount == 0) return false;

    // Sample analytic light source selected uniformly from the light list.
    // TODO: Sample based on estimated contributions as pdf.
    uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

    // Sample local light source.
    AnalyticLightSample lightSample;
    LightData ld = gScene.getLight(lightIndex);
    if (!sampleLight(vertex.pos, ld, sg, lightSample)) return false;

    // Setup returned sample.
    ls.pdf = (lightSample.pdf == 0 ? 1.f : lightSample.pdf) / lightCount; // handle the Point and Directional pdfs being 0
    ls.Li = lightSample.Li * lightCount;
    // Offset shading position to avoid self-intersection.
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    // Analytic lights do not currently have a geometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;
    ls.g = (ls.distance == kMaxLightDistance || lightSample.pdf == 0) ? 1.f : (abs(dot(-ls.dir, lightSample.normalW)) / (ls.distance * ls.distance));
    //if dist squared is infinity you cant convert to area

    return any(ls.Li > 0.f);
}

/** Return the probabilities for selecting different light types.
    \param[out] p Probabilities.
*/
void getLightTypeSelectionProbabilities(out float p[3])
{
    // Set relative probabilities of the different sampling techniques.
    // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
    p[0] = kUseEnvLight ? 1.f : 0.f;
    p[1] = kUseEmissiveLights ? 1.f : 0.f;
    p[2] = kUseAnalyticLights ? 1.f : 0.f;

    // Normalize probabilities. Early out if zero.
    float sum = p[0] + p[1] + p[2];
    if (sum == 0.f) return;

    float invSum = 1.f / sum;
    p[0] *= invSum;
    p[1] *= invSum;
    p[2] *= invSum;
}

float getEnvMapSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
float getAnalyicSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

/** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
{
    float p[3];
    getLightTypeSelectionProbabilities(p);

    float u = sampleNext1D(sg);

    [unroll]
    for (lightType = 0; lightType < 3; ++lightType)
    {
        if (u < p[lightType])
        {
            pdf = p[lightType];
            return true;
        }
        u -= p[lightType];
    }

    lightType = {};
    pdf = {};

    return false;
}

/** Samples a light source in the scene.
       This function first stochastically selects a type of light source to sample,
       and then calls that the sampling function for the chosen light type.
       \param[in] vertex Path vertex.
       \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
       \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
       \param[in,out] sg Sample generator.
       \param[out] ls Struct describing valid samples.
       \return True if the sample is valid and has nonzero contribution, false otherwise.
   */
bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg)) return false;

    bool valid = false;
    if (kUseEnvLight && lightType == (uint)NEELightType::EnvMap)
    {
        valid = generateEnvMapSample(vertex, sg, ls);
    }
    if (kUseEmissiveLights && lightType == (uint)NEELightType::Emissive)
    {
        // Emissive light samplers have an option to exclusively sample the upper hemisphere.
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
    }
    if (kUseAnalyticLights && lightType == (uint)NEELightType::Analytic)
    {
        valid = generateAnalyticLightSample(vertex, sg, ls);
    }
    if (!valid) return false;

    // Reject samples in non-requested hemispheres.
    float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
    if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
    {
        return false;
    }

    // Account for light type selection.
    ls.lightType = lightType;
    ls.pdf *= selectionPdf;
    ls.Li /= selectionPdf;

    return true;
}

// HELPER FUNCTIONS

/** Calcs balance heuristic for mis
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
float evalMIS(float n0, float p0, float n1, float p1)
{
    // Balance heuristic
    float q0 = n0 * p0;
    float q1 = n1 * p1;
    return q0 / (q0 + q1);
}

/**
 * Process a hit.
 * Loads the shading data, samples analytic lights and samples a new scatter ray.
 * Terminates the path if maximum number of bounces is reached.
 * @param[in] hit Hit info.
 * @param[in,out] rayData Ray payload.
 *
 */
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    rayData.path.incrementVertexIndex();
    rayData.sampleData.rcPackedHitInfo = hit.pack();

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Add emitted light.
    float misWeight = 1.f;
    float3 bsdf_p_hat = float3(0.f, 0.f, 0.f);
    if (kUseEmissiveLights && (kComputeDirect || rayData.path.getVertexIndex() > 2))
    {
        float3 Le = mi.getProperties(sd).emission;
        bsdf_p_hat = rayData.p_hat_thp * Le;

        if (any(Le > 0.f) && rayData.path.nonDelta()) { // if emission was non zero, and the last bsdf hit wasn't delta, this is where you calculate mis weight for bsdf sampling
            //assume that we hit an emissive triangle and this is not the primary hit

            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleLightHit lightHit;
            TriangleHit triangleHit = hit.getTriangleHit();
            lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            lightHit.posW = sd.posW;
            lightHit.normalW = sd.getOrientedFaceNormal();

            // Evaluate PDF at the hit, had it been generated with light sampling.
            bool upperHemisphere = rayData.path.isLightSampledUpper() && !rayData.path.isLightSampledLower(); // Emissive light samplers have an option to exclusively sample the upper hemisphere
            float lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(rayData.origin, rayData.normal, upperHemisphere, lightHit);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(1, rayData.pdf, 1, lightPdf); //both of these r in solid angle
        }
    }
    // update curBSDFSample, which alr contains rc info from genScatterRay
    rayData.sampleData.bsdf_p_hat = bsdf_p_hat;
    rayData.sampleData.bsdf_w_i = misWeight * luminance(bsdf_p_hat) / rayData.path_pdf;

    // Check whether to terminate based on max depth.
    if (rayData.path.getVertexIndex() - 2 >= kMaxBounces)
    {
        rayData.path.terminate();
        return;
    }

    // Update things for NEE, or if we don't end up doing it bc our path ends
    rayData.path.setLightSampled(false, false);
    // This should be where you add NEE. but if we're processing kth hit rn, this NEE is for the (k+1)th vertex to be on a light.
    bool neeSampled = sampleNEE(sd, mi, rayData); // populates nextNEESample. bc the thp of the ray changes in generateScatterRay, we have to do this before even if the ray might fail

    // Save info for hybrid shift mapping
    float3 x_kMinus1 = rayData.origin;
    float kMinus1_to_k_pdf = rayData.pdf;
    float3 normalAt_kMinus1 = rayData.normal;
    float kMinus1_roughness = rayData.roughness;
    // Calculate g for jacobian calcs later
    float3 toX_k = sd.posW - x_kMinus1;
    float3 dirToX_k = normalize(toX_k);
    rayData.sampleData.g = abs(dot(sd.getOrientedFaceNormal(), -dirToX_k) / dot(toX_k, toX_k));

    if (kShiftMappingType == (uint)ShiftMappingType::Reconnection && rayData.path.getVertexIndex() == 2){
        // jacobian is a ratio of just the geometry terms
        // in this case, toX_k is x1_to_x2
        rayData.sampleData.j_denom = rayData.sampleData.g; //abs(dot(sd.getOrientedFaceNormal(), -rayData.direction)) / dot(x1_to_x2, x1_to_x2);
    }

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin = sd.computeRayOrigin();

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, false, rayOrigin, rayData))
    {
        rayData.path.terminate();
        return;
    }

    //this code tests if the vert we are currently at could be the rc vertex. smallest possible index is 2
    if (kShiftMappingType == (uint)ShiftMappingType::Hybrid && kMinus1_roughness >= kHybridShiftRoughnessThreshold) { //single vertex roughness threshold from restir pt enhanced
        float kMinus1_k = kMinus1_to_k_pdf * rayData.sampleData.g;
        float k_kMinus1 = rayData.pdf * abs(dot(normalAt_kMinus1, dirToX_k) / dot(toX_k, toX_k));
        if (kMinus1_k == 0.f || k_kMinus1 == 0.f) {
            rayData.sampleData.dualRayFootprint = 0.f;
        } else {
            rayData.sampleData.dualRayFootprint = min(1 / kMinus1_k, 1 / k_kMinus1);
        }

        // also record data for the jacobian denom of the hybrid shift
        rayData.sampleData.j_denom = kMinus1_k; // p of going from x_(k-1) to x_k * geometry term of doing that.
        //pdf of going from x_k to x_(k+1) will be multiplied in main path loop since rayDAta.pdf is easily accessible there + we need to handle NEE differently
    } else {
        rayData.sampleData.dualRayFootprint = 0.f;
    }
}

bool sampleNEE(const ShadingData sd, const IMaterialInstance mi, inout ScatterRayData rayData) {
    // Determine if BSDF has non-delta lobes.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

    if (hasNonDeltaLobes)
    {
        LightSample ls = {};
        bool validSample = false;

        // Setup path vertex.
        // the vertex index isn't even used in the light sampling functions. used to be rayData.pathLength which is rayData.curVertex - 2 (minus 1, if it's the DI case since curVertex would still be 1 then)
        PathVertex vertex = PathVertex(rayData.path.getVertexIndex(), sd.posW, sd.faceN, sd.frontFacing);

        // Determine if upper/lower hemispheres need to be sampled.
        bool sampleUpperHemisphere = ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
        bool sampleLowerHemisphere = ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);

        // Sample a light.
        rayData.sampleData.neeSeed = rayData.sg.rng.state; // store state of rng before calling generateLightSample
        validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, rayData.sg, ls);
        rayData.path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);
        if (validSample)
        {
            // Apply MIS weight.
            float misWeight = 1.f;
            if (ls.lightType != (uint)NEELightType::Analytic)
            {
                float scatterPdf = mi.evalPdf(sd, ls.dir, true); // true because we are using BSDF sampling
                misWeight = evalMIS(1, ls.pdf, 1, scatterPdf);
            }

            float3 weight = mi.eval(sd, ls.dir, rayData.sg); // f * cos
            float3 Lr = weight * ls.Li;
            if (any(Lr > 0.f))
            {
                bool visible = traceShadowRay(ls.origin, ls.dir, ls.distance);
                if (visible)
                {
                    float3 nee_p_hat = rayData.p_hat_thp * Lr * ls.pdf; // since Li was already divided by pdf, we undivide to get the realization of f for this NEE sample
                    float change_in_pdf = ls.pdf;
                    /*if (ls.lightType == (uint)NEELightType::Emissive || ls.lightType == (uint)NEELightType::Analytic) //this code is for parametrizing path in area measure
                    {
                        // for emissive, need to convert pdf from solid angle to area by multiplying by g. also multiply nee_p_hat by g.
                        //for analytic, only certain types will need to be converted to solid angle. for the others, g will be set to 1
                        //nee_p_hat *= ls.g;
                        //change_in_pdf *= ls.g;
                    } // for env map, don't need to convert since it stays in solid angle */

                    rayData.sampleData.nee_p_hat = nee_p_hat;
                    rayData.sampleData.nee_w_i = misWeight * luminance(nee_p_hat) / (rayData.path_pdf * change_in_pdf);
                    //store the following data for the main path tracing loop
                    rayData.sampleData.nee_wo = ls.dir;
                    rayData.sampleData.nee_Li = ls.Li * ls.pdf; // bc Li was predivided by pdf
                    // packedHitInfo and prevSampledLobeType alr initialized, curLobeType will be initialized later
                    rayData.sampleData.nee_pdf = ls.pdf;
                    rayData.sampleData.nee_g = ls.g;

                    switch (ls.lightType) {
                        case (uint)NEELightType::Analytic:
                            rayData.path.setSampleType(PathFlags::neeSampled, PathFlags::sampledAnalytic);
                            break;
                        case (uint)NEELightType::Emissive:
                            rayData.path.setSampleType(PathFlags::neeSampled, PathFlags::sampledEmissive);
                            break;
                        case (uint)NEELightType::EnvMap:
                            rayData.path.setSampleType(PathFlags::neeSampled, PathFlags::sampledEnvMap);
                            break;
                        default:
                            break;
                    }
                        
                    return true;
                }
            }
        }
    }
    return false;
}

/**
 * This is the main entry point for the minimal path tracer.
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.
 *
 * Built-in light sources (point, directional) are sampled explicitly at each
 * path vertex. The contributions from area lights (env map and mesh lights)
 * are explicitly added by the scatter ray hit/miss shaders.
 *
 * @param[in] pixel Pixel to trace a path for.
 * @param[in] frameDim Dimension of the frame in pixels.
 */
Reservoir tracePath(const uint2 pixel, const uint2 frameDim)
{
    uint bufferInd = pixel.y * frameDim.x + pixel.x;
    Reservoir r = {};
    PathViewerPath pathViewerPath = {};

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid())
    {
        // Load shading data and Create material instance at shading point.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generators.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount * 4);
        SampleGenerator sgScatterRays = SampleGenerator(pixel, gFrameCount * 4 + 1);
        uint sgScatterRaysState = sgScatterRays.rng.state; //TODO: if this was default uniform sg store as uint4

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        /*for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);*/ // Commenting out bc im pretty sure this loop doesnt run since I don't use available dimensions on this render pass

        float3 direct_illum = { 0.f, 0.f, 0.f };
        if (kComputeDirect) //this code adds direct lighting, that is, emission from anything that the primary camera ray hit. TODO this is why ur version doesn't have DI!'
        {
            // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
            direct_illum += mi.getProperties(sd).emission;

            //analytic lights used to be here but now they are accounted for by the first sampleNEE.
        }
        gDI_BGBuffer[bufferInd] = direct_illum; // add DI to f * W_X, since its Le + (integral)f, and f * W_X estimates the integral

        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData(sg, sgScatterRays);

        // Initialize data carriers and information needed for shift mappings
        Sample curBSDFSample = {};
        Sample curNEESample = {};
        RCVertex rcVertex = {};
        ShiftContext shiftContext = {};
        if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
            float3 x0_to_x1 = sd.posW - primaryRayOrigin;
            float temp = dot(x0_to_x1, x0_to_x1) / abs(dot(sd.getOrientedFaceNormal(), -primaryRayDir));
            temp *= 0.02f * 4 * M_PI / 100;
            shiftContext.primaryRayFootprint = temp;
        } else {
            shiftContext.rcVertexInd = 2;
        }

        // do nee for first bounce bc when that scatter ray gets traced we (might) need NEESample to represent the length 2 paths
        // this must be done before the thp changes in generateScatterRay
        if (sampleNEE(sd, mi, rayData)) {
            RCVertex nee_x2 = { float3(0.f), 0.f, float3(0.f), uint4(0), (uint)LobeType::All, (uint)LobeType::All };
            curNEESample = Sample(rayData.sampleData.nee_p_hat, rayData.sampleData.nee_w_i, nee_x2, rayData.path);
            curNEESample.path.setVertexIndex(2); // sampleNEE already sets the NEE related flags of rayData, but we need to update vertex ind here
            curNEESample.seed = sgScatterRaysState;    // this won't be used in replay, but set anyway
            curNEESample.postRCSeed = rayData.sampleData.neeSeed; // neeSeed was set in sampleNEE (if sampleNEE returned true)
            curNEESample.k = 2;
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection) {
                // jacobian is ratio of cos / distance (aka g term)
                curNEESample.J = rayData.sampleData.nee_g;
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid){
                // in k = d case, jacobian is ratio of pdf_nee (in solid angle) * g
                curNEESample.J = rayData.sampleData.nee_pdf * rayData.sampleData.nee_g;
            }
            r.updateReservoir(curNEESample, curNEESample.w_i, 1, sg);
        }

        // Compute ray origin for new rays spawned from the G-buffer.
        float3 rayOrigin = sd.computeRayOrigin();
        // Add primary hit to debug path
        pathViewerPath.vertices[0] = rayOrigin;
        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, false, rayOrigin, rayData))
            rayData.path.terminate();

        // Follow path into the scene and compute its total contribution.
        // Note: if we reached this point then we've already generated a valid ray starting from the first hit point in the scene.
        for (uint depth = 0; depth <= kMaxBounces && !rayData.path.isTerminated(); depth++)
        {
            // Process NEE samples with RIS
            if (depth != 0) { // since we alr updated reservoir with the first nee sample (light vertex = x2)
                if (rayData.path.getVertexIndex() == shiftContext.rcVertexInd) {
                    // Case where RC vertex is k and k+1 is a NEE light vertex needs special handling because we can't get wo or incomingRadiance from standard rayData fields like we can otherwise
                    RCVertex neeX_kMinus1 = rcVertex;
                    neeX_kMinus1.outgoingDir = rayData.sampleData.nee_wo;
                    neeX_kMinus1.outgoingPdf = rayData.sampleData.nee_pdf;
                    neeX_kMinus1.incomingRadiance = rayData.sampleData.nee_Li;
                    //we get prevSampledLobeType from rcVertex, which should have been populated alr since rc index has been found
                    neeX_kMinus1.sampledLobeType = (uint)LobeType::All;
                    curNEESample = Sample(rayData.sampleData.nee_p_hat, rayData.sampleData.nee_w_i, neeX_kMinus1, rayData.path);
                    curNEESample.path .incrementVertexIndex(); // rayData.path includes data abt type of light sampled, but its vertexIndex was only incremented at the hit that spawned this nee
                    // path seed (from beginning of path) gets set at the end
                    curNEESample.postRCSeed = rayData.sampleData.neeSeed; // this is the seed we'd want to replay x(k+1)
                } else {
                    if(shiftContext.rcVertexInd == 0){ //we must be in hybrid mode
                        // k = d case
                        RCVertex nee_kEqualsD = { float3(0.f), 0.f, float3(0.f), uint4(0), (uint)LobeType::All, (uint)LobeType::All };
                        curNEESample = Sample(rayData.sampleData.nee_p_hat, rayData.sampleData.nee_w_i, nee_kEqualsD, rayData.path);
                        curNEESample.path.incrementVertexIndex(); // rayData.path includes data abt type of light sampled, but its vertexIndex was only incremented at the hit that spawned this nee
                        // path seed (from beginning of path) gets set at the end
                        curNEESample.postRCSeed = rayData.sampleData.neeSeed; // this is the seed we'd want to replay x(k) with a diff incoming dir
                    } else {
                        //normal case, rcVertex has alr been set but we just need to update incomingRadiance bc you only know that after youre done tracing
                        curNEESample = Sample(rayData.sampleData.nee_p_hat, rayData.sampleData.nee_w_i, rcVertex, rayData.path);
                        curNEESample.rcVertex.incomingRadiance = curNEESample.p_hat / shiftContext.thp_upToIncl_xk;
                        curNEESample.path.incrementVertexIndex(); // rayData.path includes data abt type of light sampled, but its vertexIndex was only incremented at the hit that spawned this nee
                        // path seed (from beginning of path) gets set at the end
                        curNEESample.postRCSeed = shiftContext.x_kPlus1_seed; //this is the seed to replay generate scatter ray originating at x_(k+1)
                    }
                }
                
                if (r.updateReservoir(curNEESample, curNEESample.w_i, 1, sg)) {
                    if (kShiftMappingType == (uint)ShiftMappingType::Reconnection) {
                        r.X.k = 2;
                        r.X.J = shiftContext.jacobian_denom;
                    } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                        if (shiftContext.rcVertexInd == 0) { // k = d case
                            r.X.k = curNEESample.path.getVertexIndex();
                            r.X.J = rayData.sampleData.nee_pdf * rayData.sampleData.nee_g; // this is handled the way the restir enhanced paper says to. p of x_(k-1) to k is just pdf of choosing x_k light vertex, we don't include p of x_k to x_(k+1) bc x_(k+1) doesn't exist
                        } else {
                            r.X.k = shiftContext.rcVertexInd;
                            r.X.J = shiftContext.jacobian_denom * r.X.rcVertex.outgoingPdf;
                            // in k = d-1 case: rcVertex.outgoingPdf is just nee_pd in solid angle for p(k to k+1). and in normal k < d case, rcVertex.outgoingPdf is the normal bsdf pdf
                        }
                    }

                    pathViewerPath.numNonCameraVertices = depth + 1; // technucally, the last sample is has index depth + 1 + 1, but were not including it bc it might not be a point in space
                    pathViewerPath.rcIndex = r.X.k;
                }
            }
            // Clear sample data in preparation for next vertex's data
            rayData.sampleData = {};
            rayData.path.clearSampleTechniqueAndLightTypes();

            shiftContext.prevRayPdf = rayData.pdf; // also save previous ray pdf before new scatter ray generated
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(rayData);

            if (!rayData.path.isTerminated()) {
                // check if this vertex can be the rc vertex. it must not be terminated because dualRayFootprint gets set at the very end of handleHit, after we check num bounces and generateScatterRay
                if (shiftContext.rcVertexInd == 0 && rayData.sampleData.dualRayFootprint >= shiftContext.primaryRayFootprint) {
                    // we have not yet selected a rc vertex, implicitly this means we are in HYBRID mode.
                    // if the ray missed, or if x(k-1) did not meet roughness threshold, dualRayFootprint is 0.f
                    shiftContext.rcVertexInd = rayData.path.getVertexIndex();
                }

                //it can't be that rc vertex got set to k and we missed at k
                if (rayData.path.getVertexIndex() == shiftContext.rcVertexInd) {
                    shiftContext.thp_upToIncl_xk = rayData.p_hat_thp;
                    shiftContext.jacobian_denom = rayData.sampleData.j_denom; // this "capture" of jacobian denom should work for hybrid and RC shifts, if we found valid vertex
                    shiftContext.x_kPlus1_seed = rayData.sgScatterRays.rng.state; // the next time this random number will be used is for scatter from k+1

                    // if kmaxbounces was 0 or we hit env map or scatter ray failed, path would be terminated and we wouldn't reach here
                    // otherwise, we hit smth in the scene and the scatter ray succeeded so it's a valid path
                    // if gen scatter ray succeeded, rayData.direction is dir from xk to x(k+1)
                    rcVertex = RCVertex(rayData.direction, rayData.pdf, { 0.f, 0.f, 0.f }, rayData.sampleData.rcPackedHitInfo,
                    rayData.prevSampledLobeType, rayData.sampledLobeType);
                }
            }

            curBSDFSample = Sample(rayData.sampleData.bsdf_p_hat, rayData.sampleData.bsdf_w_i, rcVertex, rayData.path);
            if (shiftContext.rcVertexInd == 0) { // we havent found a rc vert yet so set k = d
                // rc vertex outgoing dir and pdf will be 0
                float3 xk_radiance = float3(0.f);
                // rayData lobe types r accurate for emissive. but one off for env map
                uint kMinus1_lobeType = rayData.prevSampledLobeType;
                uint k_lobeType = rayData.sampledLobeType;
                if(rayData.path.isTerminated()){
                    xk_radiance = curBSDFSample.p_hat / rayData.p_hat_thp; //hit env map. but hit info at xk must be invalid.
                    kMinus1_lobeType = rayData.sampledLobeType; // this info is still for x(k-1) bc handleHit wasn't called when we missed
                    k_lobeType = (uint)LobeType::All; // following NEE conventions: if its a light source use lobetype::all
                } //meanwhile if we didn't hit env map, incoming radiance will be 0 but the hit info will be valid

                // rc vertex outgoing dir and pdf will be 0 bc k = d
                RCVertex bsdf_kEqualsD = { float3(0),        0,         xk_radiance, rayData.sampleData.rcPackedHitInfo, kMinus1_lobeType, k_lobeType };
                curBSDFSample.rcVertex = bsdf_kEqualsD;
                // path seed (from beginning of path) gets set at the end
                curBSDFSample.postRCSeed = 0; // invalid bc you can't replay if k = d
            } else {
                curBSDFSample.rcVertex.incomingRadiance = curBSDFSample.p_hat / shiftContext.thp_upToIncl_xk;
                // rc vertex all set. path seed (from beginning of path) gets set at the end
                curBSDFSample.postRCSeed = shiftContext.x_kPlus1_seed; // seed of gen scatter ray from x_(k+1)
            }
            
            if (r.updateReservoir(curBSDFSample, curBSDFSample.w_i, 1, sg)) {
                if (kShiftMappingType == (uint)ShiftMappingType::Reconnection) {
                    r.X.k = 2;
                    r.X.J = shiftContext.jacobian_denom;
                } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                    if (shiftContext.rcVertexInd == 0) { // we didn't find an rc vertex
                        r.X.k = rayData.path.getVertexIndex(); //k = d case
                        if (rayData.path.sampledEnvMap()) {
                            r.X.J = rayData.pdf; // might be wrong? but we dont have a better way rn bc if its the env map, that doesnst have x(k+1) or a G term
                        } else { // but if we hit a triangle, need g term. also can't use rayData.pdf bc thats for ray coming out of xk
                            r.X.J = shiftContext.prevRayPdf * rayData.sampleData.g;
                        }
                    } else {
                        r.X.k = shiftContext.rcVertexInd;
                        r.X.J = shiftContext.jacobian_denom * r.X.rcVertex.outgoingPdf; //j_denom alr calculated
                    }
                }

                if (!rayData.path.sampledEnvMap()) {
                    r.X.path.setSampleType(PathFlags::bsdfSampled, PathFlags::sampledEmissive);
                } //if we did hit the env map, path flags are alr set

                pathViewerPath.numNonCameraVertices = depth + 1;
                pathViewerPath.rcIndex = r.X.k;
            }

            if (!rayData.path.isTerminated()) {
                pathViewerPath.vertices[depth + 1] = rayData.origin;
            }
        }
        r.X.seed = sgScatterRaysState;

        // Set info that might be needed by random replay later
        gReplayInputBuffer[bufferInd].k = r.X.k;
        gReplayInputBuffer[bufferInd].seed = sgScatterRaysState;

        rayData.d.preRcThp = shiftContext.thp_upToIncl_xk; //DEBUGGING FOR THE RANDOM REPLAY
        debugBuffer[bufferInd] = rayData.d;
    }
    else
    {
        // Background pixel. We don't hit anything in the scene so this is completely independent from NEE and you don't even need restir for this
        gDI_BGBuffer[bufferInd] = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor; //this will be added onto a 0 reservoir value

        debugBuffer[bufferInd] = {};
    }

    gPathViewerPathBuffer[bufferInd] = pathViewerPath;

    return r;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.path.terminate();
    rayData.path.incrementVertexIndex(); // incr vertex ind even if we missed.
    rayData.path.setSampleType(PathFlags::bsdfSampled, PathFlags::sampledEnvMap);
    rayData.sampleData.dualRayFootprint = 0.f;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || rayData.path.getVertexIndex() > 1)) //we didn't add 1 to curVertex here like in handleHit
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        float3 bsdf_p_hat = rayData.p_hat_thp * Le;

        float misWeight = 1.f;
        if (any(Le > 0.f) && rayData.path.nonDelta()) { // if emission was non zero, and the last bsdf hit wasn't delta, this is where you calculate mis weight for bsdf sampling
            float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(rayData.direction);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume rayData.pdf > 0.f since we shouldn't have got here otherwise. it holds the pdf that the bsdf sampler chose this dir
            misWeight = evalMIS(1, rayData.pdf, 1, lightPdf); // both of these r in solid angle
        }
        rayData.sampleData.bsdf_p_hat = bsdf_p_hat;
        rayData.sampleData.bsdf_w_i = misWeight * luminance(bsdf_p_hat) / rayData.path_pdf;
    } else {
        rayData.sampleData.bsdf_p_hat = { 0.f, 0.f, 0.f };
        rayData.sampleData.bsdf_w_i = 0.f;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

//
// Shader entry point for ray generation shader.

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    uint pixelInd = pixel.y * frameDim.x + pixel.x;
    gReplayInputBuffer[pixelInd] = {}; //clear it before it gets set in tracePath

    Reservoir r = tracePath(pixel, frameDim);
    gReservoirBuffer[pixelInd] = r;

    #if IS_LAST_PASS
        // we calculate the final color here
        outputColor[pixel] = float4(r.X.p_hat * r.W_X + gDI_BGBuffer[pixelInd], 1);
    #endif
}

/**
Restir Gen Candidates Pass

What it needs to do:
- make fresh samples every frame
- record data abt those samples that is needed for resampling
- populate a buffer of path viewer data so we can visualize the paths
- record data needed to replay these paths later

Notes
- r.X.k will be 0 if you missed the scene or the reservoir just never got a sample (wi stayed 0) 
*/
