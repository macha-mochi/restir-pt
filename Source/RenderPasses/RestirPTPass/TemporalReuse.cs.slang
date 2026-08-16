import RestirReservoir;
import RestirHelpers;
import ShiftMapping;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import PathState;

struct DebugData
{
    uint dummy;
    float3 replayThp;

    __init() {
        replayThp = 0.f;
        dummy = 0;
    }
};

static const uint kShiftMappingType = SHIFT_MAPPING_TYPE;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    uint2 gOutputDimensions; // dimensions of gOutputColor

    TemporalReusePass gResamplePass;
}

struct TemporalReusePass {
    // Inputs
    Texture2D<PackedHitInfo> vBuffer;
    Texture2D<PackedHitInfo> temporalVBuffer;
    Texture2D<float4> viewW; // Originally optional but we include it
    Texture2D<float4> temporalViewW;
    Texture2D<float2> motionVectors;
    StructuredBuffer<Reservoir> inputBuffer;
    StructuredBuffer<Reservoir> temporalBuffer;
    StructuredBuffer<float3> di_bgBuffer;
    StructuredBuffer<RandomReplayOutput> replayedDataBuffer;
    // Outputs
    RWStructuredBuffer<Reservoir> outputBuffer;
    RWTexture2D<float4> outputColor;

    // Debug
    RWStructuredBuffer<DebugData> debugBuffer;

    // Finds the pixel position corresponding to the triangle hit by this frame's [pixel], last frame. Returning false if it would be out of bounds
    // intended usage: use [result] to sample a Texture2D holding data from LAST FRAME. ex: sampling last frame's vbuffer for where our CURRENT frame's pixel would have been in the last frame.
    bool calcLastFramePixelPos(uint2 pixel, out int2 result) {
        float2 inter = (float2)pixel + float2(0.5f);
        inter += gResamplePass.motionVectors[pixel] * gOutputDimensions;
        result = (int2)inter;
        return all(result >= 0) && all(result < gOutputDimensions);
    }

    uint ind(uint2 pixel) {
        return pixel.y * gOutputDimensions.x + pixel.x;
    }

    void execute(uint2 pixel) {
        if (any(pixel >= gOutputDimensions)) return; // return if oob

        outputColor[pixel] = float4(0, 0, 0, 1);

        Reservoir r = {}; // start w blank reservoir

        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        DebugData debugData = {};

        // Prep temporal domain data. TODO: this calculation will be the same for every pixel, potential optimization also calc it once and store in buffer?
        int2 pixelLastFrame = int2(0, 0);
        bool pixelHasValidTemporalPos = calcLastFramePixelPos(pixel, pixelLastFrame);
        uint replayInd = getReplayOutputIndex_T(pixel, gOutputDimensions); //used to key into random replay buffer. only foor hybrid shift

        // Add sample from input buffer to the reservoir
        float m_num = inputBuffer[ind(pixel)].c * luminance(inputBuffer[ind(pixel)].X.p_hat); // normally we'd need to shift this to the right domain but it alr is
        float m_den = m_num;
        if (pixelHasValidTemporalPos) { // try to shift input buffer sample into last frame
            Sample Y_lastFrame = {};
            float inverse_jacobian = inputBuffer[ind(pixel)].X.J; // initialize to srcG, src is this pixel
            bool success = false;
            ShiftDebugData d = {};
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                HitInfo dstHit = HitInfo(temporalVBuffer[pixelLastFrame]);
                ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.data.prevPosW, -temporalViewW[pixelLastFrame].xyz);
                success = computeShiftedSample(ShiftMappingType::Reconnection, false, true,  // settings for the shift: reconnection, src is this frame and dst is last frame
                    y1_SD, // vertex right before reconnection vert on the dst domain's path
                    inputBuffer[ind(pixel)].X,        // base path aka canonical sample
                    float3(1.f),                           // previous thp = 1
                    Y_lastFrame, inverse_jacobian, sg, d);  // out or inout variables
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
                // info needed: x_(k-1) hit info, x(k-2) position, x(k-2) to x(k-1) direction (Which can be recovered from x(k-2) and the hit info)
                //remember the data format of the replay output buffer is interleaved, this shift is 1st in the pair

                //CAUTION / TODO: to avoid issue of binding last frame's bvh, the input buffer sample is replayed in the current frame instead.
                //CAUTION / TODO: this will cause errors in a scene with animation!

                HitInfo dstHit = HitInfo(replayedDataBuffer[replayInd].x_kMinus1_hitInfo);
                float3 rayOrigin = replayedDataBuffer[replayInd].x_kMinus2_pos; //epsilon alr applied since this was a rayData.origin
                float3 rayDir = normalize(gScene.getVertexData(dstHit.getTriangleHit()).posW - rayOrigin);
                ShadingData y_kMinus1_SD = loadShadingData(dstHit, rayOrigin, rayDir);
                success = computeShiftedSample(ShiftMappingType::Hybrid, false, true,  // settings for the shift: hybrid, src is this frame and dst is last frame
                    y_kMinus1_SD, // vertex right before reconnection vert on the dst domain's path
                    inputBuffer[ind(pixel)].X,             // base path aka canonical sample
                    replayedDataBuffer[replayInd].replayedThp, // previous thp
                    Y_lastFrame, inverse_jacobian, sg, d); // out or inout variables
            }
            if (success) {
                m_den += temporalBuffer[ind(pixelLastFrame)].c * luminance(Y_lastFrame.p_hat) * inverse_jacobian;
            }
        }
        float w_i = (m_num / m_den) * luminance(inputBuffer[ind(pixel)].X.p_hat) * inputBuffer[ind(pixel)].W_X; // jacobian is 1 bc alr in right domain
        if (isnan(w_i)) {
            w_i = 0.f;
        }
        r.updateReservoir(inputBuffer[ind(pixel)].X, w_i, inputBuffer[ind(pixel)].c, sg);

        // Temporal
        if (pixelHasValidTemporalPos) {
            Sample X_thisFrame = {};
            float jacobian = temporalBuffer[ind(pixelLastFrame)].X.J; // this is the denom of jacobian of shifting last frame's sample into this domain, so this is the forward jacobian
            bool shiftSucceeded = false;
            ShiftDebugData d = {};
            d.rcThp = float3(-1.f);
            d.incomingRadiance = float3(-1.f);
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                HitInfo dstHit = HitInfo(vBuffer[pixel]);
                ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[pixel].xyz);
                shiftSucceeded = computeShiftedSample(ShiftMappingType::Reconnection, true, false, // settings for the shift: reconnection, temporal src not dst
                                             y1_SD, // vertex right before reconnection vert on the dst domain's path
                                             temporalBuffer[ind(pixelLastFrame)].X, // base path aka last frame's sample
                                             float3(1.f), // previous thp = 1
                                             X_thisFrame, jacobian, sg, d); // out or inout variables
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
                // info needed: x_(k-1) hit info, x(k-2) position, x(k-2) to x(k-1) direction (Which can be recovered from x(k-2) and the hit info)
                // remember the data format of the replay output buffer is interleaved, this shift is 2nd in the pair
                replayInd++;

                HitInfo dstHit = HitInfo(replayedDataBuffer[replayInd].x_kMinus1_hitInfo); //this hit info shouldve been played with last frame's random seed
                float3 rayOrigin = replayedDataBuffer[replayInd].x_kMinus2_pos; // epsilon alr applied since this was a rayData.origin
                float3 rayDir = normalize(gScene.getVertexData(dstHit.getTriangleHit()).posW - rayOrigin);
                ShadingData y_kMinus1_SD = loadShadingData(dstHit, rayOrigin, rayDir);
                shiftSucceeded = computeShiftedSample(ShiftMappingType::Hybrid, true, false, // settings for the shift: hybrid, temporal src not dst
                                         y_kMinus1_SD, // vertex right before reconnection vert on the dst domain's path
                                         temporalBuffer[ind(pixelLastFrame)].X, // base path aka last frame's sample
                                         replayedDataBuffer[replayInd].replayedThp, // previous thp
                                         X_thisFrame, jacobian, sg, d);         // out or inout variables

                debugData.replayThp = replayedDataBuffer[replayInd].replayedThp;
            }

            float w_i = 0.f;
            if (shiftSucceeded) {
                float m_num = temporalBuffer[ind(pixelLastFrame)].c * luminance(temporalBuffer[ind(pixelLastFrame)].X.p_hat) / jacobian;
                float m_den = m_num;
                // process new/canonical domain for m_den
                m_den += inputBuffer[ind(pixel)].c * luminance(X_thisFrame.p_hat); // jacobian is 1 since X_thisframe alr came from this domain

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = (m_num / m_den) * luminance(X_thisFrame.p_hat) * temporalBuffer[ind(pixelLastFrame)].W_X * jacobian;
            }
            r.updateReservoir(X_thisFrame, w_i, temporalBuffer[ind(pixelLastFrame)].c, sg);
        }
        outputBuffer[ind(pixel)] = r;
        debugBuffer[ind(pixel)] = debugData;

        #if IS_LAST_PASS
        // If this is the last pass, should calculate the output color
            outputColor[pixel] = float4(r.X.p_hat * r.W_X + di_bgBuffer[ind(pixel)], 1);
        #endif

    }
}

[numthreads(16, 16, 1)]
void main(uint3 threadID: SV_DispatchThreadID) {
    gResamplePass.execute(threadID.xy);
}

//finds screen space position of posW last frame.
int2 worldToScreenTemporal(float3 posW, uint2 dims) {
    float4 clipSpace = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(posW, 1.f));
    float3 ndc = clipSpace.xyz / clipSpace.w; // in range [-1, 1] if on screen
    float2 coord = float2((ndc.x + 1) / 2, (1 - ndc.y) / 2) * float2(dims);
    return (int2)coord;
}
