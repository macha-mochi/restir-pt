#include "RestirTypes.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import PathState;
import RestirHelpers;
import Utils.Debug.PixelDebug;

cbuffer CB
{
    uint gFrameCount; // Frame count since scene was loaded.
    uint2 gOutputDimensions;
}

// Inputs
Texture2D<PackedHitInfo> vBuffer;
Texture2D<float4> viewW;
StructuredBuffer<RandomReplayInput> inputBuffer;
Texture2D<PackedHitInfo> temporalVBuffer;
Texture2D<float4> temporalViewW;
StructuredBuffer<RandomReplayInput> temporalInputBuffer;
Texture2D<float2> motionVectors;

// Outputs
RWStructuredBuffer<int2> spatialOffsetBuffer;
RWStructuredBuffer<RandomReplayOutput> outputBuffer;
RWStructuredBuffer<PathViewerPath> replayedPathBuffer;
RWTexture2D<float4> outputColor;

static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const float kRayTMax = FLT_MAX;
static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
static const float kHybridShiftRoughnessThreshold = 0.2f; // got this value from the OG paper

struct ScatterRayData
{
    uint2 pixel;

    PathState path;
    float3 thp;       ///< The replayed thp, in PSS parametrization (product of "f/p" terms)
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.
    float3 normal;    ///< Shading normal at origin. Used for light sampling and MIS weight calculations.
    float pdf;        ///< Pdf of last path segment (the one leading to the current vertex).
    float roughness; ///< Roughness at [origin]
    bool shouldScatter; ///< Should the path continue with another scatter ray

    float dualRayFootprint;
    uint4 x_kMinus1_hitInfo;

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    __init(SampleGenerator sg) {
        pixel = uint2(0);

        thp = float3(1.f);
        origin = float3(0.f);
        direction = float3(0.f);
        shouldScatter = true;
        path = {};
        path.setActive();
        path.setVertexIndex(1); // since we start at x1

        dualRayFootprint = 0.f;
        x_kMinus1_hitInfo = uint4(0);

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
        rayData.normal = sd.getOrientedFaceNormal();
        rayData.pdf = bsdfSample.pdf;
        rayData.roughness = mi.getProperties(sd).roughness;

        // bsdfsample.weight = f * dot(wo, n) / pdf
        // wo is normalized in world space
        rayData.thp *= bsdfSample.weight;

        return any(rayData.thp > 0.f);
    }

    return false;
}

void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    rayData.path.incrementVertexIndex();

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Save info for hybrid shift mapping
    float3 x_kMinus1 = rayData.origin;
    float kMinus1_to_k_pdf = rayData.pdf;
    float3 normalAt_kMinus1 = rayData.normal;
    float kMinus1_roughness = rayData.roughness;

    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin = sd.computeRayOrigin();

    if (!rayData.shouldScatter) {
        // means we just hit x_k-1
        rayData.x_kMinus1_hitInfo = hit.pack();
    } else {
        // Generate scatter ray for the next path segment.
        // The raygen shader will continue the path based on the returned payload.
        if (!generateScatterRay(sd, mi, rayOrigin, rayData))
        {
            rayData.path.terminate();
        }
    }

    // Still test if this vertex could be the rc vertex
    if (kMinus1_roughness >= kHybridShiftRoughnessThreshold) { // single vertex roughness threshold from restir pt enhanced
        float3 toX_k = sd.posW - x_kMinus1;
        float3 dirToX_k = normalize(toX_k);
        float kMinus1_k = kMinus1_to_k_pdf * abs(dot(sd.getOrientedFaceNormal(), -dirToX_k) / dot(toX_k, toX_k));
        float k_kMinus1 = rayData.pdf * abs(dot(normalAt_kMinus1, dirToX_k) / dot(toX_k, toX_k));
        if (kMinus1_k == 0.f || k_kMinus1 == 0.f) {
            rayData.dualRayFootprint = 0.f;
        } else {
            rayData.dualRayFootprint = min(1 / kMinus1_k, 1 / k_kMinus1);
        }
    } else {
        rayData.dualRayFootprint = 0.f;
    }
}

/**
 * This is the main entry point for the path retrace (prolly only supports tracing in current frame for now, unless I bind last frame's BVH. TODO might get to it)
 * If [inLastFrame] true, function assumes [pixel] is valid for indexing into resources related to last frame
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.

 Goal: to find replayed throughput from x1 up to but excluding x_(k-1).

 * @param[in] pixel Pixel to start the path at, aka location of x1
 * @param[in] frameDim Dimension of the frame in pixels.
 * @param[in] seed The random seed to use
 */
RandomReplayOutput tracePath(const uint2 pixel, const uint2 frameDim, RandomReplayInput replayParams, inout PathViewerPath path, const bool inLastFrame = false)
{
    RandomReplayOutput op = {};

    const float3 primaryRayOrigin = inLastFrame ? gScene.camera.data.prevPosW : gScene.camera.getPosition();
    const float3 primaryRayDir = inLastFrame ? -temporalViewW[pixel].xyz : -viewW[pixel].xyz;
    const HitInfo hit = HitInfo(inLastFrame ? temporalVBuffer[pixel] : vBuffer[pixel]);
    const uint stopVertex = replayParams.k;
    if (hit.isValid() && stopVertex > 2)
    {
        SampleGenerator sg = SampleGenerator(0);
        sg.rng.state = replayParams.seed;

        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir);
        let lod = ExplicitLodTextureSampler(0.f);
        let mi = gScene.materials.getMaterialInstance(sd, lod);
        float3 rayOrigin = sd.computeRayOrigin();

        //Prepare payload
        ScatterRayData rayData = ScatterRayData(sg);
        rayData.pixel = pixel;

        // Calc primary ray footprint
        float3 x0_to_x1 = sd.posW - primaryRayOrigin;
        float primaryRayFootprint = dot(x0_to_x1, x0_to_x1) / abs(dot(sd.getOrientedFaceNormal(), -primaryRayDir));
        primaryRayFootprint *= 0.02f * 4 * M_PI / 100;

        // Generate scatter ray.
        if (!generateScatterRay(sd, mi, rayOrigin, rayData))
            rayData.path.terminate();
        path.numNonCameraVertices = 1;
        path.vertices[0] = rayData.origin;
        path.rcIndex = stopVertex;

        // Follow path into the scene and compute its total contribution, up to the RC vertex.
        // Note: if we reached this point then we've already generated a valid ray starting from the first hit point in the scene.
        for (uint vertex = 2; vertex <= stopVertex - 1 && !rayData.path.isTerminated(); vertex++)
        {
            // Vertex dependent things
            if (vertex == stopVertex - 1) { //we are about to hit x_k-1 
                op.replayedThp = rayData.thp; // thp right now is what we want, having just hit x_k-2. Store the info carried by rayData
                op.x_kMinus2_pos = rayData.origin;
                rayData.shouldScatter = false; // we don't want to generate a scatter ray from x_(k-1) bc it's unneeded work
            }

            // Trace scatter ray. If it hits geometry, the closest hit shader samples
            // direct illumination and generates the next scatter ray.
            // when this call finishes thp should be updated for x_[vertex]
            traceScatterRay(rayData);
            
            if(rayData.path.isTerminated()){
                // ray missed at or before the (k-1)th vertex, so invalidate the thp
                rayData.thp = float3(0.f);
            } else {
                path.numNonCameraVertices++;
                path.vertices[vertex - 1] = rayData.origin;

                // check if this vertex can be the rc vertex. ray must not be terminated because dualRayFootprint gets set at the very end of handleHit, after we check num bounces and generateScatterRay
                //if the ray missed, or if x(k-1) did not meet roughness threshold, dualRayFootprint is 0.f
                if (rayData.dualRayFootprint >= primaryRayFootprint) {
                    op = {}; // if we found a k' < k on the replayed path, this path isn't shiftable bc non invertible, so clear struct and break out
                    path.rcIndex = vertex;
                    break;
                }

                // Vertex dependent things
                if (vertex == stopVertex - 1) {
                    op.x_kMinus1_hitInfo = rayData.x_kMinus1_hitInfo; // if ray didn't end up hitting anything for x_k-1 theres no valid hit info. but if it did, store that
                }
            }
        }
    }
    else if(hit.isValid() && stopVertex == 2){
        //is just reconnection shift case
        op.replayedThp = float3(1.f);
        op.x_kMinus2_pos = primaryRayOrigin;
        op.x_kMinus1_hitInfo = hit.pack();
    } else
    {
        // Missed primary hit or stopVertex < 2 in which case we dont need replayedThp
        op.replayedThp = float3(0.f);
    }

    return op;
}

//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated but increment vertex index anyway
    rayData.path.incrementVertexIndex();
    rayData.path.terminate();
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

uint ind(uint2 pixel){
    return pixel.y * gOutputDimensions.x + pixel.x;
}

// Finds the pixel position corresponding to the triangle hit by this frame's [pixel], last frame. Returning false if it would be out of bounds
// intended usage: use [result] to sample a Texture2D holding data from LAST FRAME. ex: sampling last frame's vbuffer for where our CURRENT frame's pixel would have been in the last frame.
bool calcLastFramePixelPos(uint2 pixel, out int2 result) {
    float2 inter = (float2)pixel + float2(0.5f);
    inter += motionVectors[pixel] * gOutputDimensions;
    result = (int2)inter;
    return all(result >= 0) && all(result < gOutputDimensions);
}

//
// Shader entry point for ray generation shader.

[shader("raygeneration")]
void rayGenTemporal()
{
    uint2 basePixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    if (any(basePixel >= frameDim)) return;

    uint basePixelInd = ind(basePixel);
    uint outputBufferInd = getReplayOutputIndex_T(basePixel, frameDim);

    int2 pixelLastFrame;
    bool pixelValidLastFrame = calcLastFramePixelPos(basePixel, pixelLastFrame);

    // prev thps needed:
    // canonical -> last frame for mis (NOTE: this is just tracing in current frame's bvh with current vbuffer. is wrong for scenes w motion)
    // last frame -> this frame for reservoir update

    PathViewerPath path = {};
    RandomReplayOutput op = {};
    if (pixelValidLastFrame) {
        op = tracePath(pixelLastFrame, frameDim, inputBuffer[basePixelInd], path, true); // replay should occur in last frame, but use this frame's k and seed
    }
    op.sourcePixel = basePixel;
    outputBuffer[outputBufferInd] = op;

    path = {};
    op = {};
    if(pixelValidLastFrame) {
        op = tracePath(basePixel, frameDim, temporalInputBuffer[ind(pixelLastFrame)], path); // replay should occur in this frame, but use last frame's k and seed
    }
    op.sourcePixel = basePixel;
    outputBuffer[outputBufferInd + 1] = op;
    replayedPathBuffer[basePixelInd] = path;
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

[shader("raygeneration")]
void rayGenSpatial()
{
    uint2 basePixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    uint basePixelInd = ind(basePixel);
    uint outputBufferInd = getReplayOutputIndex_S(basePixel, frameDim, kNumSpatialNeighbors);
    //^^ this will be updated as we get our spatial thps bc there are so many of them.

    // prev thps needed:
    // canonical -> each spatial neighbor for mis
    // for each neighbor:
    //    neighbor -> this pixel for reservoir update
    //    neighbor -> each neighbor that is not itself for mis

    PathViewerPath path = {};
    RandomReplayOutput op = {};

    // First shift each spatial neighbor to canonical domain.
    for (int i = 0; i < kNumSpatialNeighbors; i++) {
        path = {};
        op = {};
        int2 neighborPixel = int2(-1);
        if (getSpatialNeighbor(basePixel, i, neighborPixel)) {
            op = tracePath(basePixel, frameDim, inputBuffer[ind(neighborPixel)], path); // using neighbor's seed, from canonical.
            op.sourcePixel = basePixel;
            outputBuffer[outputBufferInd] = op;
        }
        outputBufferInd++;
    }

    // Replay outputs from here will be used for when we shift canonical sample into neighbors' domains, for mis_denom of canonical sample.
    for(int i = 0; i < kNumSpatialNeighbors; i++)
    {
        path = {};
        op = {};
        int2 neighborPixel = int2(-1);
        if(getSpatialNeighbor(basePixel, i, neighborPixel)) {
            op = tracePath(neighborPixel, frameDim, inputBuffer[basePixelInd], path); //using canonical's seed, from each neighbor.
        }
        op.sourcePixel = neighborPixel;
        outputBuffer[outputBufferInd] = op;
        outputBufferInd++;
    }
}

/*
Testing plan for random replay

- first test if retracing same pixel with same seed gets same thp (using up to and including x_k bc thats easier to get from candidate gen pass)
- get the reference thp from the candidate debug pass
- check neighbor: do a small offset from pixel but use pixel's seed and add vertices to another buffer that you can feed into the debug path viewer
(will compare to the original path)
*/
