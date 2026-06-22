#include "RestirTypes.slangh"

import Utils.Math.MathHelpers;
import Scene.Camera.CameraData;

// Inputs
StructuredBuffer<Path> gPathDataBuffer;
uniform uint2 gMousePixelPos;
uniform float4x4 gViewProjMat;
uniform float3 testPoint;
// Outputs
RWTexture2D<float4> gOutputColor;

RWStructuredBuffer<DebugData> debugBuffer;

struct DebugData {
    float A;
    float B;
    float C;
    float dist;

    __init() {
        A = B = C = dist = 0.f;
    }
}

/**
 converts posW to a uint2 in pixel coordinates. [0, 0] is top left, (1920, 1080) is bottom right.
 this only converts to within 1 pixel of precision, do not use if you need pixel perfect :(
*/
uint2 worldToScreen(float3 posW, uint2 dims) {
    float4 clipSpace = mul(gViewProjMat, float4(posW, 1.f));
    float3 ndc = clipSpace.xyz / clipSpace.w; // in range [-1, 1]
    float2 coord = float2((ndc.x + 1) / 2, (1 - ndc.y) / 2) * float2(dims);
    return (uint2)coord;
}

// Finds the distance from a pixel pos on the screen to a line segment stretching between two pixel space positions.
// If there was a line between v0 and v1 and it extended infinitely, and the closest point to pos on that line was not between v0 and v1,
// then return arbitrary large value.
float distanceToSegment(int2 pos, int2 v0, int2 v1) {
    // find standard form of the line segment, Ax + By + C = 0. p0 and p1 are both points on this line segment

    float2 fPos = float2(pos.x + 0.5f, pos.y + 0.5f);
    float2 p0 = float2(v0.x + 0.5f, v0.y + 0.5f);
    float2 p1 = float2(v1.x + 0.5f, v1.y + 0.5f);
    float A = p1.y - p0.y;
    float B = p0.x - p1.x;
    float C = p1.x * p0.y - p0.x * p1.y;

    if (dot(pos - v0, v1 - v0) > 0.f && dot(pos - v1, v0 - v1) > 0.f) {
        return abs(A * fPos.x + B * fPos.y + C) / sqrt(A * A + B * B);
    }

    return 3e38f; //pretty close to FLT_MAX
}

[numthreads(16, 16, 1)]
void drawPaths(uint3 threadID: SV_DispatchThreadID) {
    float lineWidth = 2.f;

    uint2 pixel = threadID.xy;
    uint2 dims;
    gOutputColor.GetDimensions(dims.x, dims.y);

    if (any(pixel >= dims)) return; // return if oob

    // Overall goal: if this pixel is part of one of the line segments on this path, color it

    uint mouseBufferInd = gMousePixelPos.y * dims.x + gMousePixelPos.x;
    uint bufferInd = pixel.y * dims.x + pixel.x;

    DebugData d = {};
    for (int i = 0; i < gPathDataBuffer[mouseBufferInd].length - 1; i++) {
        uint2 p0 = worldToScreen(gPathDataBuffer[mouseBufferInd].vertices[i], dims);
        uint2 p1 = worldToScreen(gPathDataBuffer[mouseBufferInd].vertices[i + 1], dims);
        float dist = distanceToSegment(pixel, p0, p1);

        if (dist < lineWidth) //distanceToSegment(gMousePixelPos,
                            //worldToScreen(gPathDataBuffer[mouseBufferInd].vertices[i], dims),
                            //worldToScreen(gPathDataBuffer[mouseBufferInd].vertices[i+1], dims)) < lineWidth)
        {
            if (i + 1 < gPathDataBuffer[mouseBufferInd].rcIndex) //pre reconnection
            {
                gOutputColor[pixel] = float4(1, 0, 0, 1);
            } else // post reconnection
            {
                gOutputColor[pixel] = float4(0, 1, 0, 1);
            }
        }
    }
}
