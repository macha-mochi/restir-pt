import RestirReservoir;
import PathState;
import RestirHelpers;
import Scene.Raytracing;
import Utils.Sampling.SampleGenerator;

struct ShiftDebugData {
    uint reasonFail;

    __init() {
        reasonFail = 0;
    }
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
        /*if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
            TriangleHit triangleHit;
            triangleHit.instanceID = GeometryInstanceID(q.CommittedInstanceID(), q.CommittedGeometryIndex());
            triangleHit.primitiveIndex = q.CommittedPrimitiveIndex();
            triangleHit.barycentrics = q.CommittedTriangleBarycentrics();
            hitPos = gScene.getVertexData(triangleHit).posW;
        }*/ // use for debugging if you ever need to find triangle hit data
        return false;
    }
}

/** Shifts a path from domain [src] to domain [dst] by reconnecting y_(k-1) in domain [dst] to x_k in domain [src].

Procedure:
Find bsdf * cos * bsdf * cos for offset path (y_(k-1) to x_k and then x_k to x_(k+1))
Multiply by incomingRadiance of base path and potential replayed thp of previous vertices if using hybrid shift

Notes:
[y_sd] is originally y_(k-1)
[jacobian] initially holds the denominator of the jacobian for the RC step of the shift
[prevThp] should be 1 for the normal reconnection shift

*/
bool performReconnectionShift(ShiftMappingType shiftType, bool temporalSrc, bool temporalDst, ShadingData y_sd, RCVertex x_k_rcVertex, uint rcVertexIndex, PathState x_path, float3 prevThp, out Sample Y, inout float jacobian, inout SampleGenerator sg, inout ShiftDebugData d) {

    // Create material instance at shading point.
    let lod1 = ExplicitLodTextureSampler(0.f);
    let y_kMinus1_mi = gScene.materials.getMaterialInstance(y_sd, lod1);

    HitInfo rcHitInfo = HitInfo(x_k_rcVertex.packedHitInfo);
    if (!rcHitInfo.isValid() && luminance(x_k_rcVertex.incomingRadiance) == 0.f) {
        d.reasonFail = 1;
        return false; // can't do shift if x_k isn't valid. if incoming radiance is 0 the shifted path will have 0 contribution so we can fail it
        //this step will also exclude the k = d cases
    }

    // Proceed with re-eval bsdf at y_(k-1)
    float3 rcPos;
    if (rcHitInfo.getType() == HitType::Triangle) {
        // getPrevPosW: where was this triangle last frame? if the mesh itself didn't move then the world space pos should be the same
        rcPos = temporalSrc ? gScene.getPrevPosW(rcHitInfo) : gScene.getVertexData(rcHitInfo.getTriangleHit()).posW;
    } else {
        // oither types not supported for now
        d.reasonFail = 2;
        return false;
    }
    float3 y_kMinus1_rayOrigin = y_sd.computeRayOrigin();
    float3 toRCVert = rcPos - y_kMinus1_rayOrigin; // use sd.pos- rayOrigin pattern because that is what's used in other parts of falcor. stay consistent! this also prevents self intersections when we trace shadow ray
    float3 rcIncomingDir = normalize(toRCVert);
    float epsilon = 1e-5f; // shorten shadow ray distance to avoid reporting rcPos as intersection
    if (!traceShadowRay(y_kMinus1_rayOrigin, rcIncomingDir, length(toRCVert) - epsilon)) { // TODO / CAUTION: this is traced against current BVH / accel structure. in a dynamic scene this will cause bias.
        d.reasonFail = 3;
        return false; // this shift is not valid if x_k is occluded from y_(k-1)
    }
    float3 rcThp = y_kMinus1_mi.evalBSDF(y_sd, rcIncomingDir, x_k_rcVertex.prevSampledLobeType, sg);
    if (shiftType == ShiftMappingType::Hybrid) {
        // jacobian numerator: solid angle pdf of y_(k-1) to x_k * G * solid angle pdf of x_k to x_(k+1)
        // we first get the first pdf / j_denom here
        float j_denom = jacobian;
        float p_ykMinus1_xk = y_kMinus1_mi.evalPdf(y_sd, rcIncomingDir, true);
        jacobian = p_ykMinus1_xk / j_denom;
    }
    if (any(isnan(rcThp))) rcThp = float3(0, 0, 0);

    // Now re-eval bsdf at x_k. start by getting sd
    if (temporalDst) {
        y_sd = loadTemporalShadingData(rcHitInfo, y_kMinus1_rayOrigin, rcIncomingDir);
    } else {
        y_sd = loadShadingData(rcHitInfo, y_kMinus1_rayOrigin, rcIncomingDir);
    }

    // Calculate jacobians
    float dstG = abs(dot(y_sd.faceN, -rcIncomingDir)) / dot(toRCVert, toRCVert);
    if (shiftType == ShiftMappingType::Reconnection) {
        float srcG = jacobian;
        jacobian = dstG / srcG; // jacobian: ratio of cos * dist^2, for y1->x2 / x1->x2
    } else if(shiftType == ShiftMappingType::Hybrid) {
        // we now have the first p * G / j_denom
        jacobian *= dstG;
    }

    //Continue to evaluate bsdf at x_k
    let lod2 = ExplicitLodTextureSampler(0.f);
    let x_k_mi = gScene.materials.getMaterialInstance(y_sd, lod2); // mi for y_k = x)k
    float3 weight2 = x_k_mi.evalBSDF(y_sd, x_k_rcVertex.outgoingDir, x_k_rcVertex.sampledLobeType, sg); // bsdf * cos for y2
    if (any(isnan(weight2))) weight2 = float3(0, 0, 0);
    rcThp *= weight2;
    if(shiftType == ShiftMappingType::Hybrid){
        // finish calculating jacobian for hybrid shift (form: ratio of pdf * G * pdf)

        // k = d-1 case (x_(k+1) is light vertex):
        //  thp = prevThp * mis weight * re eval at y_k-1 * re eval at x_k * Li
        // light sampling terms should cancel in the jacobian, leaving it just p of y_k-1 to x_k * G
        //  but you need the nee pdf anyway for the mis weight. bsdf pdf is easily calc-able
        float p_xk_xkPlus1 = x_k_mi.evalPdf(y_sd, x_k_rcVertex.outgoingDir, true);
        float misWeight = 1.f;
        if (rcVertexIndex == x_path.getVertexIndex() - 1) { // getVertexIndex will return index of the last vertex on the path x
            float p_bsdf = p_xk_xkPlus1;
            float p_nee = x_k_rcVertex.outgoingPdf;
            if (x_path.isBSDFSampled()) {
                misWeight = p_bsdf / (p_bsdf + p_nee);
            } else if (x_path.isNEESampled()) {
                misWeight = p_nee / (p_bsdf + p_nee);
                p_xk_xkPlus1 = p_nee;
            }
        }
        jacobian *= p_xk_xkPlus1;
        //k < d-1 case: x_(k+1) is not light vertex
        // mis weight in thp = 1
        rcThp *= misWeight;
    }

    // Find p_hat for shifted X
    Y.p_hat = prevThp * rcThp * x_k_rcVertex.incomingRadiance; // CAUTION / TODO: we assume that incomingRadiance stays the same (that is, k < d and the rest of the path didnt become occluded temporally)
    // you will find shifted w_i afneter this method since we need to eval mis weight
    Y.rcVertex = x_k_rcVertex;
    Y.path = x_path;

    bool success = jacobian != 0.f && luminance(Y.p_hat) != 0.f; // if jacobian = 0, or p hat is 0, w_i will be 0 so it'll be as if the shift failed
    if (!success) d.reasonFail = 4;
    return success;
}
