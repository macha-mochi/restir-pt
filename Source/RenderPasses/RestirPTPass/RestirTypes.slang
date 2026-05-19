__exported import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;

struct Reservoir
{
    // info needed for reservoir: the sample X, its weight W, the sum of weights W_sum, and the sample count M (capped)
    Sample X = Sample.NULL; // the current sample

    float W_X = 0;                              // unbiased contribution weight of sample X
    float w_sum = 0; //sum of all lowercase w seen so far, for use in resampling
    uint M = 1;
    uint c = 0; //cap at 20
};
struct Sample {
    static Sample NULL = { float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), 0 };

    float3 x0; // first vertex
    float3 p_hat_X; // aka f, the realized contribution of this sample (no pdfs multiplied yet)
    float w_i;    // w_i = (resampling MIS weight m_i) * (target function evaluated for X_i) / (pdf of sampling X_i), where reciprocal of pdf of X_i is just W_Xi
}
bool operator ==(const Sample s) {
    return p_hat_X == s.p_hat_X && w_i == s.w_i;
}
float luminance(float3 vec) {
    return dot(vec, float3(0.2126, 0.7152, 0.0722));
}
void updateReservoir(Reservoir r, Sample sample, float w_i_new, uint c_i, SampleGenerator sg) {
    r.w_sum += w_i_new;
    r.c += c_i;
    if (sampleNext1D(sg) < w_i_new / r.w_sum) {
        // replace current sample with new one, if non null
        r.X = sample;
        if (!(sample == Sample.NULL)) {
            r.W_X = r.w_sum / luminance(sample.p_hat_X);
        } else {
            r.W_X = 0;
        }
        r.M++;
    }
}
/** Types of samplable lights.
 */
enum class LightType
{
    EnvMap,
    Emissive,
    Analytic
};

/** Describes a light sample.
 */
struct LightSample
{
    float3 Li;      ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
    float pdf;      ///< Pdf with respect to solid angle at the shading point.
    float3 origin;  ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
    float distance; ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
    float3 dir;     ///< Ray direction for visibility evaluation (normalized).
    uint lightType; ///< Light type this sample comes from (LightType casted to uint).
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
