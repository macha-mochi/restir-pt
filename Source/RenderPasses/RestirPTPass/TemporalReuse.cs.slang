import RestirReservoir;
import RestirHelpers;
import ShiftMapping;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import Scene.Raytracing;
import PathState;

struct DebugData
{
    ShiftDebugData d;

    //shifting canonical sample
    float m_num;
    float m_den;
    float jacobian;

    //these are all for the temporal sample
    float m_num_t;
    float m_den_t;
    float jacobian_t;

    uint pathState;
    uint vertexIndex;
    uint neeSampled;

    __init() {

        d = {};

        m_num = 0;
        m_den = 0;
        jacobian = 0;

        m_num_t = 0;
        m_den_t = 0;
        jacobian_t = 0;
        pathState = 0;
        vertexIndex = 0;
        neeSampled = 0;
    }
};

static const bool kPerformSpatialReuse = USE_SPATIAL;
static const uint kNumSpatialNeighbors = NUM_SPATIAL_NEIGHBORS;
static const uint kSpatialNeighborRadius = 20;
static const bool kPerformTemporalReuse = USE_TEMPORAL;
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

        // Add sample from input buffer to the reservoir
        float m_num = inputBuffer[ind(pixel)].c * luminance(inputBuffer[ind(pixel)].X.p_hat); // normally we'd need to shift this to the right domain but it alr is
        float m_den = m_num;
        if (pixelHasValidTemporalPos) { // try to shift input buffer sample into last frame
            Sample Y_lastFrame = {};
            float inverse_jacobian = inputBuffer[ind(pixel)].J; // initialize to srcG, src is this pixel
            bool success = false;
            ShiftDebugData d = {};
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                HitInfo dstHit = HitInfo(temporalVBuffer[pixelLastFrame]);
                ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.data.prevPosW, -temporalViewW[pixelLastFrame].xyz);
                success = performReconnectionShift( ShiftMappingType::Reconnection, false, true,  // settings for the shift: reconnection, src is this frame and dst is last frame
                    y1_SD, // vertex right before reconnection vert on the dst domain's path
                    inputBuffer[ind(pixel)].X.rcVertex, 2, inputBuffer[ind(pixel)].X.path,        // information from the base path aka canonical sample
                    float3(1.f),                           // previous thp = 1
                    Y_lastFrame, inverse_jacobian, sg, d);  // out or inout variables
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid) {
            }
            if (success) {
                m_den += temporalBuffer[ind(pixelLastFrame)].c * luminance(Y_lastFrame.p_hat) * inverse_jacobian;
            }
        }
        debugData.m_num = m_num;
        debugData.m_den = m_den;
        float w_i = (m_num / m_den) * luminance(inputBuffer[ind(pixel)].X.p_hat) * inputBuffer[ind(pixel)].W_X; // jacobian is 1 bc alr in right domain
        if (isnan(w_i)) w_i = 0.f;
        if (r.updateReservoir(inputBuffer[ind(pixel)].X, w_i, inputBuffer[ind(pixel)].c, sg)) {
            r.copySampleGenerationInfo(inputBuffer[ind(pixel)]);
        }
            
        // Temporal
        if (pixelHasValidTemporalPos) {
            Sample X_thisFrame = {};
            float jacobian = temporalBuffer[ind(pixelLastFrame)].J; // this is the denom of jacobian of shifting last frame's sample into this domain, so this is the forward jacobian
            bool shiftSucceeded = false;
            ShiftDebugData d = {};
            if (kShiftMappingType == (uint)ShiftMappingType::Reconnection)
            {
                HitInfo dstHit = HitInfo(vBuffer[pixel]);
                ShadingData y1_SD = loadShadingData(dstHit, gScene.camera.getPosition(), -viewW[pixel].xyz);
                shiftSucceeded = performReconnectionShift(ShiftMappingType::Reconnection, true, false, // settings for the shift: reconnection, no temporal src or dst
                                             y1_SD, // vertex right before reconnection vert on the dst domain's path
                                             temporalBuffer[ind(pixelLastFrame)].X.rcVertex, 2, temporalBuffer[ind(pixelLastFrame)].X.path, // information from the base path aka last frame's sample
                                             float3(1.f), // previous thp = 1
                                             X_thisFrame, jacobian, sg, d); // out or inout variables
            } else if (kShiftMappingType == (uint)ShiftMappingType::Hybrid)
            {
            }
            debugData.d = d;
            PathState p = temporalBuffer[ind(pixelLastFrame)].X.path;
            debugData.pathState = p.flagsAndVertexIndex;
            debugData.vertexIndex = p.getVertexIndex();
            debugData.neeSampled = p.isNEESampled();

            float w_i = 0.f;
            float newJ = 0.f;
            if (shiftSucceeded) {
                newJ = jacobian * temporalBuffer[ind(pixelLastFrame)].J;
                
                float m_num = temporalBuffer[ind(pixelLastFrame)].c * luminance(temporalBuffer[ind(pixelLastFrame)].X.p_hat) / jacobian;
                float m_den = m_num;
                // process new/canonical domain for m_den
                m_den += inputBuffer[ind(pixel)].c * luminance(X_thisFrame.p_hat); // jacobian is 1 since X_thisframe alr came from this domain

                // evaluate Y's 'w_i = mis * p hat * W_X * jacobian
                w_i = (m_num / m_den) * luminance(X_thisFrame.p_hat) * temporalBuffer[ind(pixelLastFrame)].W_X * jacobian;
                if (isnan(w_i)) w_i = 0.f;

                debugData.m_num_t = m_num;
                debugData.m_den_t = m_den;
            }
            // update reservoir streaming (in new buffer)
            X_thisFrame.w_i = w_i;
            if (r.updateReservoir(X_thisFrame, w_i, temporalBuffer[ind(pixelLastFrame)].c, sg))
            {
                r.copySampleGenerationInfo(temporalBuffer[ind(pixelLastFrame)]);
                r.J = newJ; //overwrite old jacobian info tho
            }
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
