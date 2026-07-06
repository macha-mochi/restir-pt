import RestirReservoir;
import RestirHelpers;
import Scene.Raytracing;

// Shifts sample X into the current pixel's domain using reconnection shift
// srcPixel = pixel to shift from, dstPixel = pixel to shift to. if either is into last frame's domain, assumes we already did temporal projection.
bool performReconnectionShift(uint2 srcPixel, bool temporalSrc, uint2 dstPixel, bool temporalDst, in Sample X, out Sample Y, out float jacobian, inout SampleGenerator sg, inout DebugData d) {
    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 temporalPrimaryRayOrigin = gScene.camera.data.prevPosW;

    const HitInfo srcHit = temporalSrc ? HitInfo(gTemporalVBuffer[srcPixel]) : HitInfo(gVBuffer[srcPixel]); //TODO: can probably replace this by just passing in the info for x(k-1) if x(k) is rc vert
    const HitInfo dstHit = temporalDst ? HitInfo(gTemporalVBuffer[dstPixel]) : HitInfo(gVBuffer[dstPixel]);
    if (srcHit.isValid() && dstHit.isValid())
    {
        //Both pixels were valid primary hits. Find modified thp

        ///>> Find bsdf * cos * bsdf * cos for offset path, so you can multiply by incomingRadiance of base path.

        // Load shading data.
        ShadingData dstSD = {};
        if (temporalDst) {
            dstSD = loadTemporalShadingData(dstHit, temporalPrimaryRayOrigin, -gTemporalViewW[dstPixel].xyz);
        } else {
            dstSD = loadShadingData(dstHit, primaryRayOrigin, -gViewW[dstPixel].xyz);
        }
        // Compute ray origin for new rays spawned from the G-buffer.
        float3 dstRayOrigin = dstSD.computeRayOrigin();

        // Create material instance at shading point.
        let lod1 = ExplicitLodTextureSampler(0.f);
        let dstMI1 = gScene.materials.getMaterialInstance(dstSD, lod1);

        RCVertex rcVertex = X.rcVertex;
        HitInfo rcHitInfo = HitInfo(rcVertex.packedHitInfo);
        if (!rcHitInfo.isValid()) { // TODO: should have another condition here. used to be X.thp_denom == 0 as in path ended before x2 but lk incoming radiance would be good check
            return false; // can't do shift if x2 isn't valid
        }

        // Since we have a valid x2 vertex, first find geometry term for base path to use later in jacobian
        ShadingData srcSD = {};
        if (temporalSrc) {
            srcSD = loadTemporalShadingData(srcHit, temporalPrimaryRayOrigin, -gTemporalViewW[srcPixel].xyz);
        } else {
            srcSD = loadShadingData(srcHit, primaryRayOrigin, -gViewW[srcPixel].xyz);
        }
        float3 srcRayOrigin = srcSD.computeRayOrigin(); //this is the place where the ray from x1 to x2 originated
        float3 x2;
        if (rcHitInfo.getType() == HitType::Triangle) {
            //getPrevPosW: where was this triangle last frame? if the mesh itself didn't move then the world space pos should be the same
            x2 = temporalSrc ? gScene.getPrevPosW(rcHitInfo) : gScene.getVertexData(rcHitInfo.getTriangleHit()).posW;
        } else {
            // oither types not supported for now
            return false;
        }
        float3 srcX1_to_X2 = x2 - srcRayOrigin;
        float3 dirX1_to_X2 = normalize(srcX1_to_X2);
        if (temporalSrc) { // set srcSD to be shading data at x2 on the base path
            srcSD = loadTemporalShadingData(rcHitInfo, srcRayOrigin, dirX1_to_X2);
        } else {
            srcSD = loadShadingData(rcHitInfo, srcRayOrigin, dirX1_to_X2);
        }
        float srcG = abs(dot(srcSD.faceN, -dirX1_to_X2)) / dot(srcX1_to_X2, srcX1_to_X2);

        // Proceed with re-eval bsdf at y1
        float3 rcPos = x2;
        float3 toRCVert = rcPos - dstRayOrigin; //use sd.pos- rayOrigin pattern
        float3 rcIncomingDir = normalize(toRCVert);
        float epsilon = 1e-5f; //shorten shadow ray distance to avoid reporting rcPos as intersection
        if (!traceShadowRay(dstRayOrigin, rcIncomingDir, length(toRCVert) - epsilon)) { //TODO / CAUTION: this is traced against current BVH / accel structure. in a dynamic scene this will cause bias.
            return false; //this shift is not valid if x2 is occluded from y1
        }
        float3 new_thp = dstMI1.evalBSDF(dstSD, rcIncomingDir, rcVertex.prevSampledLobeType, sg);
        if (any(isnan(new_thp))) new_thp = float3(0, 0, 0);

        // Now re-eval bsdf at x2. start by getting sd for y2 = x2
        if (temporalDst) {
            dstSD = loadTemporalShadingData(rcHitInfo, dstRayOrigin, rcIncomingDir);
        } else {
            dstSD = loadShadingData(rcHitInfo, dstRayOrigin, rcIncomingDir);
        }
        jacobian = abs(dot(dstSD.faceN, -rcIncomingDir)) / (dot(toRCVert, toRCVert) * srcG); //jacobian: ratio of cos * dist^2, for y1->x2 / x1->x2

        let lod2 = ExplicitLodTextureSampler(0.f);
        let mi2 = gScene.materials.getMaterialInstance(dstSD, lod2);                              // mi for y2 = x2
        float3 weight2 = mi2.evalBSDF(dstSD, rcVertex.outgoingDir, rcVertex.sampledLobeType, sg); // bsdf * cos for y2
        if (any(isnan(weight2))) weight2 = float3(0, 0, 0);
        new_thp *= weight2;

        // Find p_hat for shifted X
        Y.p_hat = new_thp * rcVertex.incomingRadiance; //CAUTION / TODO: we assume that incomingRadiance stays the same (that is, the rest of the path didnt become occluded temporally)
        // you will find shifted w_i afneter this method since we need to eval mis weight
        Y.rcVertex = rcVertex;

        return jacobian != 0.f && luminance(Y.p_hat) != 0.f; //if jacobian = 0, or p hat is 0, w_i will be 0 so it'll be as if the shift failed
    }
    return false;
}
