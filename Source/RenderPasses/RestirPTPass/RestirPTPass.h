/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/EmissiveLightSampler.h"

#include "RestirTypes.slangh"

using namespace Falcor;

/**
 * Pathtracer using ReSTIR PT.
 *
 * This pass implements a real time pathtracer using ReSTIR PT.
 */
class RestirPTPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(RestirPTPass, "RestirPTPass", "Restir PT pass.");

    static ref<RestirPTPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<RestirPTPass>(pDevice, props);
    }

    RestirPTPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override;

private:
    void parseProperties(const Properties& props);
    void prepareVars();
    void resetLighting();
    bool prepareLighting(RenderContext* pRenderContext);

    // Internal state

    ref<Scene> mpScene; /// Current scene.
    ref<SampleGenerator> mpSampleGenerator; /// GPU sample generator.
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;          ///< Environment map sampler or nullptr if not used.
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler; ///< Emissive light sampler or nullptr if not used.

    // Configuration

    /// Max number of indirect bounces (0 = none).
    uint mMaxBounces = 3;
    /// Compute direct illumination (otherwise indirect only).
    bool mComputeDirect = true;
    /// Use importance sampling for materials.
    bool mUseImportanceSampling = true;
    /// Type of shift mapping used
    ShiftMappingType mShiftMappingType = ShiftMappingType::Reconnection;
    /// Whether to do spatial reuse
    bool mUseSpatialReuse = true;
    /// Number of spatial neighbors per pixel
    uint mNumSpatialNeighbors = 3;
    /// Whether to do temporal reuse
    bool mUseTemporalReuse = true;

    // Runtime data

    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    bool mOptionsChanged = false;

    // Raytracing program for generating new candidates. The compute passes later on don't need programs (see compute passes made by createComputePass in RTXDI.cpp). 
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;

    //Resources
    ref<Buffer> mpReservoirBuffers[3]; ///< Length 3 array containing buffers of path reservoirs: last frame's final samples, this frame's new samples, and this frame's final samples
    ///< these will be 0, 1, or 2
    uint mTemporalReservoirID; //temporal = last frame
    uint mCandidateReservoirID; 
    uint mOutputReservoirID;
    ref<Texture> temporalVBuffer; ///< Last frame's vbuffer of packedhitinfos
    ref<Texture> temporalViewDir; ///< Last frame's view direction texture

    ref<Buffer> mpDiBgBuffer; ///< Buffer storing direct illumination samples (or env map samples if camera ray missed) for each pixel

    //Compute passes
    //do you need one for testing candidate visibility??
    ref<ComputePass> mpSpatiotemporalResamplingPass; /// Spatiotemporal resampling.

    //Debugging resources
    ref<Buffer> mpCandidateGenDebugBuffer;
    ref<Buffer> mpResamplingDebugBuffer;

    bool mUsePathViewer = false; // if true, the renderer is paused so you can click around pixels on that frame
    uint2 mMousePixelPos;
    ref<Buffer> mpDebugPathBuffer;
    ref<ComputePass> mpPathViewerPass;

    ref<Buffer> tempBuffer;
};
