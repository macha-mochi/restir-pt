#include "RestirTypes.slangh"
#include "Scene/SceneDefines.slangh"

__exported import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Scene.Raytracing;

struct Sample {
    float3 p_hat; // aka f, the realized contribution of this sample (no pdfs multiplied yet)
    float w_i; // w_i = (resampling MIS weight m_i) * (target function evaluated for X_i) / (pdf of sampling X_i), where reciprocal of pdf of X_i is just W_Xi
    RCVertex rcVertex; // struct with reconnection vertex data

    static Sample createNull() {
        Sample s;
        s.p_hat = { 0.f, 0.f, 0.f };
        s.w_i = 0.f;
        s.rcVertex = {};
        return s;
    }

    __init(float3 p_hat, float w_i, RCVertex rcv) {
        this.p_hat = p_hat;
        this.w_i = w_i;
        this.rcVertex = rcv;
    }

    static bool isNull(Sample X) {
        return all(X.p_hat == 0.f) || X.w_i == 0.f;
    }
};
struct Reservoir
{
    // info needed for reservoir: the sample X, its weight W, the sum of weights W_sum, and the sample count M (capped)
    Sample X = Sample::createNull(); // the current sample

    float W_X = 1.f;                              // unbiased contribution weight of sample X
    float w_sum = 0; //sum of all lowercase w seen so far, for use in resampling
    uint c = 0; //cap at 20

    [mutating] //Returns true if sample was replaced, false ow
    bool updateReservoir(Sample sample, float w_i_new, uint c_i, SampleGenerator sg) {
        w_sum += w_i_new;
        c += c_i;
        if (w_i_new == 0) { // no chance to be chosen and no need to update W_X since w_sum doesn't change
            return false;
        }
        bool updateSample = false;
        if (sampleNext1D(sg) < w_i_new / w_sum) {
            // replace current sample with new one, if non null
            X = sample;
            updateSample = true;
        }
        W_X = w_sum / luminance(X.p_hat); // if we got to this point, w_i_new is non zero meaning mi, p_hat, and W_X were all nonzero. so not a null sample
        c = min(c, 20); //setting a c cap of 20

        return updateSample;
    }
};
float luminance(float3 vec) {
    return dot(vec, float3(0.2126, 0.7152, 0.0722));
}

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
