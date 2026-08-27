#include "Scene/SceneDefines.slangh"
import Utils.Geometry.GeometryHelpers;
import Scene.Raytracing;

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
 * Setup ShadingData based on loaded vertex/material attributes for a hit point, LAST FRAME.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin (of the ray that produced this hit last frame).
 * @param[in] rayDir Normalized ray direction (of the ray that produced this hit last frame).
 * @return ShadingData struct.
 */
ShadingData loadTemporalShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getTemporalVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)

#error "getTemporalVertexData not defined for HitType::DisplacedTriangle!"

#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)

#error "getTemporalVertexData not defined for HitType::Curve!"

#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)

#error "getTemporalVertexData not defined for HitType::SDFGrid!"

#endif

    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
}

/**
 Gets the index of the first of the two RandomReplayOutput structs in ReplayOutputBuffer that are needed in shifts in TemporalReuse pass.

 [pixel]: the pixel whose sample is being shifted in/out of last frame, aka the one we're doing restir at

 The same buffer is used for the random replay outputs for temporal and spatial.
 This index is for indexing into the temporal scheme of organizing data which is:
- canonical -> last frame for mis
- last frame -> this frame for reservoir update
*/
uint getReplayOutputIndex_T(uint2 pixel, uint2 frameDim) {
    return (pixel.y * frameDim.x + pixel.x) * 2;
}

/**
 Gets the index of the first of the RandomReplayOutput structs in ReplayOutputBuffer that are needed in shifts in SpatialReuse pass.

 [pixel]: the "current pixel" we are doing Restir at

 The same buffer is used for the random replay outputs for temporal and spatial, but spatial needs more memory.
 This index is for indexing into the spatial scheme of organizing data which is:
    // for each neighbor:
    //    neighbor -> this pixel for reservoir update
    // canonical -> each spatial neighbor for mis
 For a total of 2 * kNumSpatialNeighbors shifts.

  Example with 3 spatial neighbors:
  | 0 -> canon | 1 -> canon | 2 -> canon || canon -> 0 | canon -> 1 | canon -> 2
  This will return the index of |0 -> canon| and the user is expected to add offsets for the rest.
*/
uint getReplayOutputIndex_S(uint2 pixel, uint2 frameDim, uint numNeighbors) {
    return (pixel.y * frameDim.x + pixel.x) * (numNeighbors * 2);
}

// Returns the index of the [neighborInd]th offset of [pixel], given frameDim and neighbors per pixel
// FOr exmaple, if i wanted the spatial offset of pixel (x, y) in the first spatial pass, parameters are (x, y) and 0
// This is 0 indexed!!
uint getSpatialOffsetBufferIndex(uint2 pixel, uint2 frameDim, uint neighborsPerPixel, uint neighborInd) {
    return (pixel.y * frameDim.x + pixel.x) * neighborsPerPixel + neighborInd;
}
