#include "Utils/Math/MathConstants.slangh"

import RestirReservoir;
import PathState;
import RestirHelpers;
import Scene.Raytracing;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Utils.Geometry.GeometryHelpers;

static const float kRayTMax = FLT_MAX;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;

ParameterBlock<EnvMapSampler> envMapSampler; ///< Environment map sampler. Only valid when kUseEnvLight == true.
ParameterBlock<EmissiveLightSampler> emissiveSampler; ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

struct ShiftDebugData {
    uint failReason;

    float3 rcThp;
    float3 incomingRadiance;
    float misWeight;
    float3 fOverP1;
    float3 eval1;
    float3 fOverP2;

    __init() {
        failReason = 0;

        misWeight = 0;
        rcThp = float3(0.f);
        incomingRadiance = float3(0.f);
        fOverP1 = float3(0.f);
        eval1 = float3(0.f);
        fOverP2 = float3(0.f);
    }
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
    ls.g = 1.f; // since env map stays in solid angle

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

    // Calculate geometry term. info needed: cos at light and distance to light which we have
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
    ls.pdf = (lightSample.pdf > 0.f ? lightSample.pdf : 1.f) / lightCount; // handle the Point and Directional pdfs being 0
    ls.Li = lightSample.Li * lightCount;
    // Offset shading position to avoid self-intersection.
    ls.origin = vertex.getRayOrigin(lightSample.dir);
    // Analytic lights do not currently have a geometric representation in the scene.
    // Do not worry about adjusting the ray length to avoid self-intersections at the light.
    ls.distance = lightSample.distance;
    ls.dir = lightSample.dir;
    ls.g = (ls.distance < kMaxLightDistance && lightSample.pdf > 0.f) ? (abs(dot(-ls.dir, lightSample.normalW)) / (ls.distance * ls.distance)) : 1.f;
    // if dist squared is infinity you cant convert to area

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

/**
 * Traces a shadow ray, checks occlusion.
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

    while(q.Proceed());
    if (q.CommittedStatus() == COMMITTED_NOTHING) {
        return true;
    } else {
        return false;
    }
}

/**
 * Trace a ray and get hit info about the next triangle it hits.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if we hit smth (and [packed] is valid triangle), false otherwise
 */
bool traceRayClosestHit(float3 origin, float3 dir, float distance, out PackedHitInfo packed) {
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    RayQuery<RAY_FLAG_NONE> q; //we want closest hit so no special flags
    let rayFlags = RAY_FLAG_NONE; // gets OR'd with the flag passed into rayquery generic

    q.TraceRayInline(gScene.rtAccel, rayFlags, 0xff /* instance inclusion mask */, ray);

    while(q.Proceed()); //returns true if candidate that needs further eval, false if done searching bvh. you can't be sure that CommittedStatus() is correct until you've searched entire bvh
    if (q.CommittedStatus() == COMMITTED_NOTHING) {
        return false;
    } else {
        if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            TriangleHit triangleHit;
            triangleHit.instanceID = GeometryInstanceID(q.CommittedInstanceID(), q.CommittedGeometryIndex());
            triangleHit.primitiveIndex = q.CommittedPrimitiveIndex();
            triangleHit.barycentrics = q.CommittedTriangleBarycentrics();
            packed = triangleHit.pack();
            return true;
        }
    }
    return false;
}

/**
how computeShiftedSample should work for k = d

k = d
NEE: replay gen light sample using srcX's seed. then re-eval y_(k-1) just like in a normal NEE function (remember to do MIS)
BSDF: if emissive tri, re calculate the emission at xk and link up y_(k-1)
      if env map, eval y_(k-1) in the direction of the env map and then multiply by Le
*/

/** Shifts a path from domain [src] to domain [dst].

This just does the reconnection step of reconnecting y_(k-1) in domain [dst] to x_k in domain [src].

Preconditions if hybrid shift:
User must provide a valid previous throughput for the path up to y_(k-1)
Function assumes x(k-1), xk, y(k-1) are rough enough, and both paths have the same k, and lobe k-1 is present on both paths

Procedure:
Find bsdf * cos * bsdf * cos for offset path (y_(k-1) to x_k and then x_k to x_(k+1))
Multiply by incomingRadiance of base path and potential replayed thp of previous vertices if using hybrid shift.

Notes:
[y_sd] is originally y_(k-1)
[prevThp] should be 1 for the normal reconnection shift, and be drawn from the result of random replayed paths if hybrid
incomingRadiance might need to be re evaluated if k = d or temporal, TODO

*/
bool computeShiftedSample(ShiftMappingType shiftType, bool temporalSrc, bool temporalDst, ShadingData y_sd, Sample srcX, float3 prevThp, out Sample Y, out float jacobian, inout SampleGenerator sg, out ShiftDebugData d) {

    // Create material instance at shading point.
    let lod1 = ExplicitLodTextureSampler(0.f);
    let y_kMinus1_mi = gScene.materials.getMaterialInstance(y_sd, lod1);

    // Create variables we will use later
    float misWeight = 1.f;
    float3 rcThp = float3(1.f); //includes new thp at y_(k-1) and y_k if needed
    float3 incomingRadiance = float3(0.f); // depending on case, could be radiance coming into x_k from x_(k+1), or from x_k into x_(k-1)
    float jacobian_denom = srcX.J;
    float jacobian_num = 1.f;

    // Handle k = d NEE case and BSDF+envmap case, bc these dont need rcVertex
    if(srcX.k == srcX.path.getVertexIndex()){
        if (srcX.path.isNEESampled()) {
            PathVertex vertex_kMinus1 = PathVertex(srcX.path.getVertexIndex() - 1, y_sd.posW, y_sd.faceN, y_sd.frontFacing);
            SampleGenerator neeSG = SampleGenerator(uint2(0), 0);
            neeSG.rng.state = srcX.postRCSeed;
            LightSample ls = {};

            bool validSample = generateLightSample(vertex_kMinus1, srcX.path.isLightSampledUpper(), srcX.path.isLightSampledLower(), neeSG, ls);
            if (!validSample) {
                return false; // the base path was valid so if this isnt valid return false
            }

            // trace shadow ray
            bool visible = traceShadowRay(ls.origin, ls.dir, ls.distance);
            if (!visible) {
                return false;
            }

            // calc mis weight
            if (ls.lightType != (uint)NEELightType::Analytic) { 
                float bsdfPdf = y_kMinus1_mi.evalPdf(y_sd, ls.dir, true);
                misWeight = ls.pdf / (bsdfPdf + ls.pdf);
            }

            rcThp = y_kMinus1_mi.eval(y_sd, ls.dir, neeSG); // this is still same random number as used in gen candidates so its ok
            incomingRadiance = ls.Li * ls.pdf;                           // ls.Li is predivided by pdf
            if (shiftType == ShiftMappingType::Reconnection) {
                jacobian_num = ls.g; // since we used the same random numbers, we should sample the same light type so g being invalid (i set it to 1.f) shouldnt cause problems
            } else if (shiftType == ShiftMappingType::Hybrid) {
                rcThp /= ls.pdf; //for PSS
                jacobian_num = ls.pdf * ls.g;
            }
            d.fOverP1 = rcThp;
        } else if(srcX.path.isBSDFSampled()){
            if (srcX.path.sampledEnvMap()) {
                // trace shadow ray
                float3 origin = y_sd.computeRayOrigin();
                bool visible = traceShadowRay(origin, srcX.rcVertex.outgoingDir, kRayTMax);
                if (!visible) {
                    return false;
                }

                //calc mis weight
                float bsdfPdf = y_kMinus1_mi.evalPdf(y_sd, srcX.rcVertex.outgoingDir, true); // actually the pdf of incoming to x_k
                float lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(srcX.rcVertex.outgoingDir); // actually incoming dir to x_k
                misWeight = bsdfPdf / (bsdfPdf + lightPdf);

                incomingRadiance = srcX.rcVertex.incomingRadiance;
                if (shiftType == ShiftMappingType::Reconnection) {
                    rcThp = y_kMinus1_mi.eval(y_sd, srcX.rcVertex.outgoingDir, sg);
                    jacobian_num = 1.f; //CAUTION: not sure if this is right (see comment where r.J was assigned in GenCandidates) but we will probably use hybrid most of the time anyway
                } else if (shiftType == ShiftMappingType::Hybrid) {
                    rcThp = y_kMinus1_mi.evalLobe(y_sd, srcX.rcVertex.outgoingDir, srcX.rcVertex.prevSampledLobeType, sg);
                    rcThp /= y_kMinus1_mi.evalPdf(y_sd, srcX.rcVertex.outgoingDir, true); // we divide contribution by normal (not lobe) p
                    jacobian_num = y_kMinus1_mi.evalLobePdf(y_sd, srcX.rcVertex.outgoingDir, srcX.rcVertex.prevSampledLobeType); //also not including g bc env map has no valid g
                }
                d.fOverP1 = rcThp;
            } // leave emissive for rcVertex is valid case
        } else {
            return false;
        }
    }

    // Get info abt y_(k-1) and rcPos
    HitInfo rcHitInfo = HitInfo(srcX.rcVertex.packedHitInfo);
    if (!rcHitInfo.isValid() && srcX.k < srcX.path.getVertexIndex()) {
        d.failReason = 1;
        return false; // can't do shift if x_k isn't valid and k < d, since we've yet to handle k = d bsdf emissive case
    }
    float3 rcPos;
    if (rcHitInfo.isValid() && rcHitInfo.getType() == HitType::Triangle) {
        // getPrevPosW: where was this triangle last frame? if the mesh itself didn't move then the world space pos should be the same
        rcPos = temporalSrc ? gScene.getPrevPosW(rcHitInfo) : gScene.getVertexData(rcHitInfo.getTriangleHit()).posW;
    } else if (rcHitInfo.isValid()) {
        // other types not supported
        d.failReason = 2;
        return false;
    }
    bool isTransmission = (srcX.rcVertex.prevSampledLobeType & (uint)LobeType::Transmission) != 0;
    float3 y_kMinus1_rayOrigin = y_sd.computeRayOrigin(!isTransmission);
    float3 y_kMinus1_normal = y_sd.getOrientedFaceNormal(); 
    float3 toRCVert = rcPos - y_kMinus1_rayOrigin; // use sd.pos- rayOrigin pattern because that is what's used in other parts of falcor. stay consistent! this also prevents self intersections when we trace shadow ray
    float3 rcIncomingDir = normalize(toRCVert);

    //covers: k < d case, and k = d bsdf emissive
    if (srcX.k < srcX.path.getVertexIndex() || srcX.path.isBSDFSampled() && srcX.path.sampledEmissive()) {
        float epsilon = 1e-2f; // shorten shadow ray distance to avoid reporting rcPos as intersection
        if (!traceShadowRay(y_kMinus1_rayOrigin, rcIncomingDir, length(toRCVert) - epsilon)) { // TODO / CAUTION: this is traced against current BVH / accel structure. in a dynamic scene this will cause bias.
            d.failReason = 3;
            return false; // this shift is not valid if x_k is occluded from y_(k-1)
        }

        if (shiftType == ShiftMappingType::Reconnection) {
            rcThp = y_kMinus1_mi.evalLobe(y_sd, rcIncomingDir, srcX.rcVertex.prevSampledLobeType, sg);
            d.fOverP1 = rcThp;
        } else if (shiftType == ShiftMappingType::Hybrid) {
            float3 weight1 = y_kMinus1_mi.evalLobe(y_sd, rcIncomingDir, srcX.rcVertex.prevSampledLobeType, sg);
            rcThp = weight1;
            rcThp /= y_kMinus1_mi.evalLobePdf(y_sd, rcIncomingDir, srcX.rcVertex.prevSampledLobeType);
            d.fOverP1 = rcThp;
            d.eval1 = weight1;

            // jacobian numerator: solid angle pdf of y_(k-1) to x_k * G * solid angle pdf of x_k to x_(k+1)
            // we get the first p here for specific lobe
            jacobian_num = y_kMinus1_mi.evalLobePdf(y_sd, rcIncomingDir, srcX.rcVertex.prevSampledLobeType);
        }

        // Now re-eval bsdf at x_k. start by getting sd
        if (temporalDst) {
            y_sd = loadTemporalShadingData(rcHitInfo, y_kMinus1_rayOrigin, rcIncomingDir);
        } else {
            y_sd = loadShadingData(rcHitInfo, y_kMinus1_rayOrigin, rcIncomingDir);
        }

        // Calculate jacobians
        float dstG = abs(dot(y_sd.faceN, -rcIncomingDir)) / dot(toRCVert, toRCVert);
        if (shiftType == ShiftMappingType::Reconnection) {
            jacobian_num = dstG; // jacobian: ratio of cos * dist^2, for y1->x2 / x1->x2
        } else if (shiftType == ShiftMappingType::Hybrid) {
            // we now have the first p * G
            jacobian_num *= dstG;
        }

        // Continue to evaluate bsdf at x_k
        let lod2 = ExplicitLodTextureSampler(0.f);
        let x_k_mi = gScene.materials.getMaterialInstance(y_sd, lod2); // mi for y_k = x)k

        //Branch: handle k = d bsdf emissive
        if (srcX.k == srcX.path.getVertexIndex() && srcX.path.isBSDFSampled() && srcX.path.sampledEmissive()) {
            incomingRadiance = x_k_mi.getProperties(y_sd).emission;
            //jacobian numerator, rcThp are both done in this case bc no x_(k+1)

            // Calc mis weight if non delta and hybrid
            if (shiftType == ShiftMappingType::Hybrid && (srcX.rcVertex.prevSampledLobeType & LobeType::NonDelta) != 0) {
                float bsdfPdf = y_kMinus1_mi.evalPdf(y_sd, rcIncomingDir, true);
                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleLightHit lightHit;
                lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(rcHitInfo.getInstanceID(), rcHitInfo.getPrimitiveIndex());
                lightHit.posW = y_sd.posW;
                lightHit.normalW = y_sd.getOrientedFaceNormal();

                // Evaluate PDF at the hit, had it been generated with light sampling.
                bool upperHemisphere = srcX.path.isLightSampledUpper() && !srcX.path.isLightSampledLower(); // Emissive light samplers have an option to exclusively sample the upper hemisphere
                // first 2 args to evalPdf are for the vertex before the light
                float lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(y_kMinus1_rayOrigin, y_kMinus1_normal, upperHemisphere, lightHit);
                misWeight = bsdfPdf / (bsdfPdf + lightPdf);
            }
        } else {
            incomingRadiance = srcX.rcVertex.incomingRadiance;
            if (shiftType == ShiftMappingType::Reconnection) {
                float3 weight2 = x_k_mi.evalLobe(y_sd, srcX.rcVertex.outgoingDir, srcX.rcVertex.sampledLobeType, sg);
                rcThp *= weight2;
                d.fOverP2 = weight2;
            } else if (shiftType == ShiftMappingType::Hybrid) {
                float k_kPlus1_lobe_pdf = x_k_mi.evalLobePdf(y_sd, srcX.rcVertex.outgoingDir, srcX.rcVertex.sampledLobeType);
                // Find bsdf/nee mis weight if this is k = d-1 path and x_k was not delta.
                // if delta, k_kPlus1_lobe_pdf will get set to p_bsdf. this is correct for hybrid shift bc you shouldnt have a delta x_k and nee sampled k+1 anyway
                if (srcX.k == srcX.path.getVertexIndex() - 1 && (srcX.rcVertex.sampledLobeType & LobeType::NonDelta) != 0) {
                    float bsdfPdf = x_k_mi.evalPdf(y_sd, srcX.rcVertex.outgoingDir, true);
                    float lightPdf = 0.f;
                    if (srcX.path.sampledEnvMap()) {
                        lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(srcX.rcVertex.outgoingDir);
                    } else if (srcX.path.sampledEmissive()) {
                        float3 xk_rayOrigin = y_sd.computeRayOrigin();
                        uint4 packed = uint4(0);
                        if (traceRayClosestHit(xk_rayOrigin, srcX.rcVertex.outgoingDir, kRayTMax, packed)) { //TODO/CAUTION: always traces in cur frame bvh
                            HitInfo x_kPlus1 = HitInfo(packed);
                            ShadingData x_kPlus1_sd = temporalDst ? loadTemporalShadingData(x_kPlus1, xk_rayOrigin, srcX.rcVertex.outgoingDir) : 
                                                                    loadShadingData(x_kPlus1, xk_rayOrigin, srcX.rcVertex.outgoingDir);

                            TriangleLightHit lightHit;
                            lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(x_kPlus1.getInstanceID(), x_kPlus1.getPrimitiveIndex());
                            if (lightHit.triangleIndex == LightCollection::kInvalidIndex) {
                                return false; // we didnt hit a triangle
                            }
                            lightHit.posW = x_kPlus1_sd.posW;
                            lightHit.normalW = x_kPlus1_sd.getOrientedFaceNormal();

                            bool upperHemisphere = srcX.path.isLightSampledUpper() && !srcX.path.isLightSampledLower(); // when we did nee for this originally we would have set the path flags
                            lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(xk_rayOrigin, y_sd.getOrientedFaceNormal(), upperHemisphere, lightHit);
                        } else {
                            // otherwise, we failed to find light in this dir which means non invertible
                            return false;
                        }
                    } else if(srcX.path.sampledAnalytic()) {
                        // mis weight should be 1, p of x_k to x_k+1 is the same as it was for original sample so it should cancel to 1 in jacobian
                        lightPdf = srcX.rcVertex.outgoingPdf; // it cant be analytic and bsdf so if it was analytic then outgoing pdf will be nee pdf
                    }

                    //Finally, calc mis weight
                    misWeight = 1.f;
                    if(srcX.path.isBSDFSampled()){
                        misWeight = bsdfPdf / (bsdfPdf + lightPdf); // impossible to have sampled analytic here
                        // k_kPlus1_lobe_pdf alr set
                        jacobian_num *= k_kPlus1_lobe_pdf;
                    } else if (srcX.path.isNEESampled()) {
                        misWeight = srcX.path.sampledAnalytic() ? 1.f : (lightPdf / (lightPdf + bsdfPdf));
                        k_kPlus1_lobe_pdf = lightPdf;
                        jacobian_num *= lightPdf;
                    }
                } else {
                    // k < d - 1
                    jacobian_num *= k_kPlus1_lobe_pdf;
                }

                // jacobian_num has been calculated alr
                float3 weight2 = x_k_mi.evalLobe(y_sd, srcX.rcVertex.outgoingDir, srcX.rcVertex.sampledLobeType, sg);
                rcThp *= weight2; // bsdf * cos for y2. if next vert was nee, evalLobe will basically call eval which is good
                rcThp /= k_kPlus1_lobe_pdf; // for PSS
                d.fOverP2 = weight2 / k_kPlus1_lobe_pdf;
            }
            
        }
    }

    if(any(isnan(rcThp))) rcThp = float3(0.f);
    // Find p_hat for shifted X
    if (shiftType == ShiftMappingType::Reconnection) {
        Y.p_hat = prevThp * rcThp * incomingRadiance;
    } else if(shiftType == ShiftMappingType::Hybrid){
        Y.p_hat = prevThp * misWeight * rcThp * incomingRadiance;
    }
    // CAUTION / TODO: we assume that the rest of the path didnt become occluded temporally, for incomingRadiance
    d.misWeight = misWeight;
    d.rcThp = rcThp;
    d.incomingRadiance = incomingRadiance;
    // you will find shifted w_i afneter this method since we need to eval mis weight
    Y.rcVertex = srcX.rcVertex;
    Y.path = srcX.path;
    Y.seed = srcX.seed;
    Y.postRCSeed = srcX.postRCSeed;
    Y.k = srcX.k;
    Y.J = jacobian_num; // jacobian numerator will be jacobian denominator for paths where the shifted path becomes the base path
    jacobian = jacobian_num / jacobian_denom;

    bool success = jacobian != 0.f && luminance(Y.p_hat) != 0.f; // if jacobian = 0, or p hat is 0, w_i will be 0 so it'll be as if the shift failed
    if (!success) {
        d.failReason = 4;
    }
    return success;
}
