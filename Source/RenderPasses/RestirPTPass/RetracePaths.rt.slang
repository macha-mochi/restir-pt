#include "RestirTypes.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import PathState;
import RestirHelpers;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
}

// Inputs
Texture2D<PackedHitInfo> vBuffer;
Texture2D<float4> viewW; // Optional

// Outputs
RWStructuredBuffer<ReconnectionData> gReconnectionData;
RWStructuredBuffer<PathViewerPath> replayedPathBuffer;
RWTexture2D<float4> outputColor;

static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const float kRayTMax = FLT_MAX;
static const bool kPerformTemporalReuse = USE_TEMPORAL;
static const bool kPerformSpatialReuse = USE_SPATIAL;
static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;

struct ScatterRayData
{
    bool terminated;
    float3 thp;       ///< The replayed thp
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    __init(SampleGenerator sg) {
        terminated = false;
        thp = float3(1.f);
        origin = float3(0.f);
        direction = float3(0.f);

        this.sg = sg;
    }
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
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;

        // bsdfsample.weight = f * dot(wo, n) / pdf
        // wo is normalized in world space
        rayData.thp *= bsdfSample.weight;
        if (!bsdfSample.isLobe(LobeType::Delta)) { //nondelta
            rayData.thp *= bsdfSample.pdf;
        }
        return any(rayData.thp > 0.f);
    }

    return false;
}

void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin = sd.computeRayOrigin();

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, rayOrigin, rayData))
    {
        rayData.terminated = true;
    }
}

/**
 * This is the main entry point for the path retrace (prolly only supports tracing in current frame for now, unless I bind last frame's BVH. TODO might get to it)
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.

 Goal: to find replayed throughput from x1 up to but excluding x_(k-1).

 * @param[in] pixel Pixel to trace a path for (could be a neighbor)
 * @param[in] frameDim Dimension of the frame in pixels.
 * @param[in] seed The random seed to use
 */
ReconnectionData tracePath(const uint2 pixel, const uint2 frameDim, uint seed)
{
    uint pixelInd = pixel.y * frameDim.x + pixel.x;
    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = -viewW[pixel].xyz;

    ReconnectionData rd = gReconnectionData[pixelInd];

    uint stopVertex = gReconnectionData[pixelInd].k;
    const HitInfo hit = HitInfo(vBuffer[pixel]);
    if (hit.isValid() && stopVertex > 2)
    {
        SampleGenerator sg = SampleGenerator(0);
        sg.rng.state = seed;

        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        float3 rayOrigin = sd.computeRayOrigin();

        //Prepare payload
        ScatterRayData rayData = ScatterRayData(sg);

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, rayOrigin, rayData))
            rayData.terminated = true;

        // Follow path into the scene and compute its total contribution, up to the RC vertex.
        // Note: if we reached this point then we've already generated a valid ray starting from the first hit point in the scene.
        for (uint vertex = 2; vertex <= stopVertex && !rayData.terminated; vertex++)
        {
            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            // when this call finishes thp should be updated for x_[vertex]
            traceScatterRay(rayData);
            if(rayData.terminated){
                // ray missed before it hit the kth vertex, so invalidate the thp
                rayData.thp = float3(0.f);
            }
        }

        rd.replayedThp = rayData.thp;
    }
    else
    {
        // Missed primary hit or stopVertex <= 2 in which case we dont need replayedThp
        rd.replayedThp = float3(1.f);
    }

    return rd;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;
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

uint ind(uint2 pixel, uint2 frameDim){
    return pixel.y * frameDim.x + pixel.x;
}

//
// Shader entry point for ray generation shader.

[shader("raygeneration")]
void rayGenTemporal()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    uint pixelInd = ind(pixel, frameDim);

    ReconnectionData rd = tracePath(pixel, frameDim, gReconnectionData[pixelInd].seed);
    gReconnectionData[pixelInd] = rd;
}

[shader("raygeneration")]
void rayGenSpatial()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    uint pixelInd = ind(pixel, frameDim);

    ReconnectionData rd = tracePath(pixel, frameDim, gReconnectionData[pixelInd].seed);
    gReconnectionData[pixelInd] = rd;

    //outputColor[pixel] = float4(1, 0, 0, 1);
}

/*
Testing plan for random replay

- first test if retracing same pixel with same seed gets same thp (using up to and including x_k bc thats easier to get from candidate gen pass)
- get the reference thp from the candidate debug pass
- check neighbor: do a small offset from pixel but use pixel's seed and add vertices to another buffer that you can feed into the debug path viewer
(will compare to the original path)
*/
