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

#include "RestirTypes.slang"
#include "Scene/SceneDefines.slangh"
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

// Outputs (not an actual output channel but actually just the reservoir buffer)
RWStructuredBuffer<Reservoir> reservoirs;
RWStructuredBuffer<DebugData> debugSpareBuffer;
RWStructuredBuffer<float3> di_bgBuffer;

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

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

struct DebugData {
    uint path_length_from_cam;

    // bsdf sample info
    float3 bsdf_p_hat;                  
    float bsdfMISWeight;

    // nee sample info
    float3 nee_p_hat;
    float neeMISWeight;
    uint lightType; // type of nee sampled light

    //next nee sample info
    float3 nextNEE_p_hat;
    float nextNEEMISWeight;
    uint nextLightType;
}

/**
 * Payload for scatter ray
 */
struct ScatterRayData
{
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    bool terminated;  ///< Set to true when path is terminated.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    uint pathLength;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 normal;    ///< Shading normal at origin. Used for light sampling and MIS weight calculations.
    float pdf; ///< Pdf of last path segment (the one leading to the current vertex). This is needed for MIS weight calculations at the current vertex for the bsdf
    float3 direction; ///< Next path segment direction.

    // added for restir
    Sample BSDFSample;
    Sample NEESample;
    Sample nextNEESample;
    bool lightSampledUpper; ///< (true if) Last path vertex sampled lights using NEE (in upper hemisphere).
    bool lightSampledLower; ///< (true if) Last path vertex sampled lights using NEE (in lower hemisphere).
    float3 p_hat_thp;                   // Throughput for the p_hat calculations. doesn't include pdf, only bsdf * geo term
    float path_pdf_area;                        ///< Current path pdf (just multiplying) in terms of area
    float g;                    // geometry term for the path segment leading to this vertex

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    DebugData d;

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.sg = sg;

        this.BSDFSample = Sample::createNull();
        this.NEESample = Sample::createNull();
        this.nextNEESample = Sample::createNull();
        this.p_hat_thp = float3(1, 1, 1);
        this.path_pdf_area = 1.f;
        this.g = 1.f;
        this.lightSampledUpper = false;
        this.lightSampledLower = false;
        this.normal = float3(0, 0, 0);

        this.d = {};
    }
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
    }
#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
}

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
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return float3(0.f);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls))
        return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, ls.dir, ls.distance);
    if (!V)
        return float3(0.f);

    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
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
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        rayData.normal = sd.getOrientedFaceNormal();
        rayData.pdf = bsdfSample.pdf;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.thp *= bsdfSample.weight;

        // additional info needed for restir
        // bsdfsample.weight = f * dot(wo, n) / pdf
        // wo is normalized in world space
        float3 f = bsdfSample.weight * bsdfSample.pdf;
        rayData.p_hat_thp *= f; //geometry term will be multiplied on the next bounce
        rayData.path_pdf_area *= bsdfSample.pdf; //pdf is in solid angle, converts to area upon a hit. TODO if this ray doesn't end up hitting anything then pdf might be wrong

        return any(rayData.thp > 0.f);
    }

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
bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls, inout DebugData d)
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
    if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

    // Setup returned sample.
    ls.pdf = (lightSample.pdf == 0 ? 1.f : lightSample.pdf) / lightCount; // handle the Point and Directional pdfs being 0
    ls.Li = lightSample.Li * lightCount;
    // Offset shading position to avoid self-intersection.
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    // Analytic lights do not currently have a geometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;
    ls.g = ls.distance == kMaxLightDistance ? 1.f : (abs(dot(-ls.dir, lightSample.normalW)) / (ls.distance * ls.distance));
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
bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg,
                                inout DebugData d, out LightSample ls)
{
    ls = {};

    uint lightType;
    float selectionPdf;
    if (!selectLightType(lightType, selectionPdf, sg)) return false;

    bool valid = false;
    if (kUseEnvLight && lightType == (uint)LightType::EnvMap)
    {
        valid = generateEnvMapSample(vertex, sg, ls);
    }
    if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
    {
        // Emissive light samplers have an option to exclusively sample the upper hemisphere.
        bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
        valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls, d);
    }
    if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
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
    const bool isCurveHit = hit.getType() == HitType::Curve;

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Multiply path_pdf_area by pdf and geometry term for the path segment that led to this hit
    float cos = abs(dot(-rayData.direction, sd.faceN)); // sd.faceN is face normal in world space on the front facing side. is probably unit?
    float3 dist = sd.posW - rayData.origin;
    rayData.g = cos / dot(dist, dist); //(hit point) - (origin of ray that hit this point)
    rayData.p_hat_thp *= rayData.g;
    rayData.path_pdf_area *= rayData.g;        // convert pdf from solid angle to area for the segment that ends at this vertex

    // Add emitted light.
    float misWeight = 1.f;
    float3 bsdf_p_hat = float3(0.f, 0.f, 0.f);
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        float3 Le = mi.getProperties(sd).emission;
        rayData.radiance += rayData.thp * Le;

        bsdf_p_hat = rayData.p_hat_thp * Le;
        if (any(Le > 0.f)) { // if emission was non zero, this is where you calculate mis weight for bsdf sampling
            //assume that we hit an emissive triangle and this is not the primary hit

            // Prepare hit point struct with data needed for emissive light PDF evaluation.
            TriangleLightHit lightHit;
            TriangleHit triangleHit = hit.getTriangleHit();
            lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
            lightHit.posW = sd.posW;
            lightHit.normalW = sd.getOrientedFaceNormal();

            // Evaluate PDF at the hit, had it been generated with light sampling.
            bool upperHemisphere = rayData.lightSampledUpper && !rayData.lightSampledLower; // Emissive light samplers have an option to exclusively sample the upper hemisphere
            float lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(rayData.origin, rayData.normal, upperHemisphere, lightHit);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
            misWeight = evalMIS(1, rayData.pdf, 1, lightPdf); //both of these r in solid angle
        }
    }
    rayData.d.bsdf_p_hat = bsdf_p_hat;
    rayData.d.bsdfMISWeight = misWeight;
    // store bsdf sample data into rayData TODO might need to add some info when it comes time to do shift mappings.
    rayData.BSDFSample.p_hat_X = bsdf_p_hat;
    rayData.BSDFSample.w_i = misWeight * luminance(bsdf_p_hat) / rayData.path_pdf_area;

    //Update things for NEE, or if we don't end up doing it bc our path ends
    rayData.lightSampledUpper = false;
    rayData.lightSampledLower = false;
    rayData.NEESample = rayData.nextNEESample;
    rayData.nextNEESample = Sample::createNull();
    rayData.d.nee_p_hat = rayData.NEESample.p_hat_X;
    rayData.d.neeMISWeight = rayData.d.nextNEEMISWeight;
    rayData.d.lightType = rayData.d.nextLightType;
    rayData.d.nextNEE_p_hat = float3(0.f, 0.f, 0.f);
    rayData.d.nextNEEMISWeight = 0.f;
    rayData.d.nextLightType = 10; // invalid

    // Check whether to terminate based on max depth.
    if (rayData.pathLength >= kMaxBounces)
    {
        rayData.terminated = true;
        return;
    }

    rayData.pathLength++;

    // Compute ray origin for new rays spawned from the hit.
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

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }

    // This should be where you add NEE. but if we're processing kth hit rn, this NEE is for the (k+1)th vertex to be on a light.
    bool neeSampled = sampleNEE(sd, mi, rayData); // populates nextNEESample. we don't need to do this if our ray terminates above
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
        // we use rayData.pathLength() as the vertex index
        PathVertex vertex = PathVertex(rayData.pathLength, sd.posW, sd.faceN, sd.frontFacing);

        // Determine if upper/lower hemispheres need to be sampled.
        bool sampleUpperHemisphere = ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
        bool sampleLowerHemisphere = ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);

        // Sample a light.
        validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, rayData.sg, rayData.d, ls);
        rayData.lightSampledUpper = sampleUpperHemisphere;
        rayData.lightSampledLower = sampleLowerHemisphere;
        //debugData attribtues for nextNEESample were already reset before this was called so we can just overwrite them
        if (validSample)
        {
            // Apply MIS weight.
            float misWeight = 1.f;
            if (ls.lightType != (uint)LightType::Analytic)
            {
                float scatterPdf = mi.evalPdf(sd, ls.dir, true); // true because we are using BSDF sampling
                misWeight = evalMIS(1, ls.pdf, 1, scatterPdf);
            }

            float3 weight = mi.eval(sd, ls.dir, rayData.sg); //f * cos
            float3 Lr = weight * ls.Li;
            rayData.d.nextLightType = ls.lightType;
            rayData.d.nextNEEMISWeight = misWeight;
            if (any(Lr > 0.f))
            {
                bool visible = traceShadowRay(ls.origin, ls.dir, ls.distance);
                if (visible)
                {
                    float3 nee_p_hat = rayData.p_hat_thp * Lr * ls.pdf; // since Li was already divided by pdf, we undivide to get the realization of f for this NEE sample
                    float change_in_pdf = ls.pdf;
                    if (ls.lightType == (uint)LightType::Emissive || ls.lightType == (uint)LightType::Analytic)
                    {
                        // for emissive, need to convert pdf from solid angle to area by multiplying by g. also multiply nee_p_hat by g.
                        //for analytic, only certain types will need to be converted to solid angle. for the others, g will be set to 1
                        nee_p_hat *= ls.g;
                        change_in_pdf *= ls.g;
                    } // for env map, don't need to convert since it stays in solid angle
                    rayData.d.nextNEE_p_hat = nee_p_hat;
                    rayData.nextNEESample.p_hat_X = nee_p_hat;
                    rayData.nextNEESample.w_i = misWeight * luminance(nee_p_hat) / (rayData.path_pdf_area * change_in_pdf);
                }
            }
            return true; //if Lr = 0, then nextNEESample is just going to stay null
        }
        return false;
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
    Reservoir r;
    float3 outColor = float3(0.f);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid())
    {
        // Pixel represents a valid primary hit. Compute its contribution.

        const bool isCurveHit = hit.getType() == HitType::Curve;

        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);

        // Create material instance at shading point.
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

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

        float3 direct_illum = { 0.f, 0.f, 0.f };
        if (kComputeDirect) //this code adds direct lighting, that is, emission from anything that the primary camera ray hit. TODO this is why ur version doesn't have DI!'
        {
            // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
            float3 temp = mi.getProperties(sd).emission;
            outColor += temp;
            direct_illum += temp;

            // Add contribution of direct light from analytic lights.
            // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
            temp = kUseAnalyticLights ? evalDirectAnalytic(sd, mi, sg) : float3(0.f);
            outColor += temp;
            direct_illum += temp;
        }
        di_bgBuffer[bufferInd] = direct_illum; // add DI after you calculated W_X, the estimate of 1/pdf, using just the f of the actual integral
        // since its Le + (integral)f

        // Prepare ray payload.
        ScatterRayData rayData = ScatterRayData(sg);

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
            rayData.terminated = true;

        if(!rayData.terminated) {
            // do nee for first bounce bc when that scatter ray gets traced we (might) need nextNEESample to represent the length 2 paths
            sampleNEE(sd, mi, rayData);
        }

        // Follow path into the scene and compute its total contribution.
        // Note: if we reached this point then we've already generated a valid ray starting from the first hit point in the scene. so path length > 0 by now
        for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
        {
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            traceScatterRay(rayData);

            // perform RIS, resampling between current sample, bsdf sample, and nee sample for path of this length
            // updateReservoir(r, bsdf sample info)
            // updateReservoir(r, nee sample info)

            // temporaily put nee samples into the reservoir, check if good
            r.X = rayData.BSDFSample;
            if (r.X.w_i == 0.f) {
                r.W_X = 0.f;
            } else {
                r.W_X = r.X.w_i / luminance(r.X.p_hat_X);
            }

            rayData.d.path_length_from_cam = rayData.pathLength + 2;

            debugSpareBuffer[bufferInd] = rayData.d;
        }

        // Store contribution from scatter ray.
        outColor += rayData.radiance;
    }
    else
    {
        // Background pixel.
        outColor = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;

        // we don't hit anything in the scene so this is completely independent from NEE and you don't even need restir for this
        di_bgBuffer[bufferInd] = outColor; //this will be added onto a 0 reservoir value
    }

    return r;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.thp * Le;

        float3 bsdf_p_hat = rayData.p_hat_thp * Le;
        float misWeight = 1.f;
        if (any(Le > 0.f)) { // if emission was non zero, this is where you calculate mis weight for bsdf sampling
            float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(rayData.direction);

            // Compute MIS weight by combining this with BSDF sampling.
            // Note we can assume rayData.pdf > 0.f since we shouldn't have got here otherwise. it holds the pdf that the bsdf sampler chose this dir
            misWeight = evalMIS(1, rayData.pdf, 1, lightPdf); // both of these r in solid angle
        }
        rayData.d.bsdfMISWeight = misWeight;
        rayData.BSDFSample.p_hat_X = bsdf_p_hat;
        rayData.BSDFSample.w_i = misWeight * luminance(bsdf_p_hat) / rayData.path_pdf_area;
        //atp, path_pdf_area is the product of all the pdfs except the one for last vertex in area AND pdf of last vertex in solid angle
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

    Reservoir r = tracePath(pixel, frameDim);

    uint bufferInd = pixel.y * frameDim.x + pixel.x;
    reservoirs[bufferInd] = r;
}
