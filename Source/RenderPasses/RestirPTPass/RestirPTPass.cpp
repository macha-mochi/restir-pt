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
#include "RestirPTPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, RestirPTPass>();
}

namespace
{
const char kReflectTypesShaderFile[] = "RenderPasses/RestirPTPass/ReflectTypes.cs.slang";
const char kCandidateGenShaderFile[] = "RenderPasses/RestirPTPass/RestirGenCandidates.rt.slang";
const char kGenSpatialOffsetShaderFile[] = "RenderPasses/RestirPTPass/GenerateSpatialNeighbors.cs.slang";
const char kSpatialComputeShaderFile[] = "RenderPasses/RestirPTPass/SpatialReuse.cs.slang";
const char kTemporalComputeShaderFile[] = "RenderPasses/RestirPTPass/TemporalReuse.cs.slang";
const char kRetracePathsShaderFile[] = "RenderPasses/RestirPTPass/RetracePaths.rt.slang";
const char kPathViewerShaderFile[] = "RenderPasses/RestirPTPass/PathViewer.cs.slang";
const char kVisualizePathsShaderFile[] = "RenderPasses/RestirPTPass/VisualizePaths.cs.slang";

//CANDIDATE GENERATION SETTINGS
// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 350u;
const uint32_t kMaxRetracePayloadSize = 128u;
const uint32_t kMaxRecursionDepth = 2u;

const std::string kInputVBuffer = "vbuffer";
const std::string kInputViewDir = "viewW";
const std::string kInputMotionVectors = "mvec";
const std::string kOutputColor = "color";

const ChannelList kInputChannels = {
    // clang-format off
    { kInputVBuffer,        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { kOutputColor,          "outputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
const char kShiftMappingType[] = "shiftMappingType";

} // namespace

RestirPTPass::RestirPTPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice) {
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);

    mOutputReservoirID = 0;
    mInputReservoirID = 1;
    mTemporalReservoirID = 2;

    mReplayInputID = 0;
    mTemporalReplayInputID = 1;

    mpPixelDebug = std::make_unique<PixelDebug>(pDevice);
}

void RestirPTPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mPathParams.maxSurfaceBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else if (key == kShiftMappingType)
            mShiftMappingType = value;
        else
            logWarning("Unknown property '{}' in RestirPTPass properties.", key);
    }
}

Properties RestirPTPass::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mPathParams.maxSurfaceBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    props[kShiftMappingType] = mShiftMappingType;
    return props;
}

RenderPassReflection RestirPTPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void RestirPTPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Get dimensions of ray dispatch / compute dispatch.
    targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    // PATH VIEWER DEBUG COMPUTE PASS
    if (mUsePathViewer)
    {
        bool viewReplayPaths = mShiftMappingType == ShiftMappingType::Hybrid;
        PathViewerPass(pRenderContext, renderData, viewReplayPaths);

        return;
    }

    mpPixelDebug->beginFrame(pRenderContext, targetDim);

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    bool lightingChanged = prepareLighting(pRenderContext);

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Prepare resources like buffers, textures etc (not including debug buffers). we need to know the reflect types for this
    if (!mpReflectTypes)
    {
        DefineList defines;
        defines.add(mpSampleGenerator->getDefines());
        ProgramDesc desc;
        desc.addShaderLibrary(kReflectTypesShaderFile);
        desc.csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines);
    }
    prepareResources(pRenderContext, renderData);

    GenerateInitialCandidates(pRenderContext, renderData);
    
    //Update reseroir IDs for upcoming reuse
    std::swap(mInputReservoirID, mOutputReservoirID);

    //SPATIOTEMPORAL RESAMPLING
    if (mUseTemporalReuse && mFrameCount > 0) //no temporal history yet on frame 0
    {
        // RANDOM REPLAY PATH RETRACE (hybrid only)
        if (mShiftMappingType == ShiftMappingType::Hybrid)
        {
            PathRetracePass(pRenderContext, renderData, true);
        }
    
        PathReusePass(pRenderContext, renderData, true);
        std::swap(mInputReservoirID, mOutputReservoirID);
    }
    if (mUseSpatialReuse)
    {
        GenSpatialOffsetsPass(pRenderContext, renderData);
        // RANDOM REPLAY PATH RETRACE (hybrid only)
        if (mShiftMappingType == ShiftMappingType::Hybrid)
        {
            PathRetracePass(pRenderContext, renderData, false);
        }
        PathReusePass(pRenderContext, renderData, false);
        std::swap(mInputReservoirID, mOutputReservoirID);
    }

    //Update data for next frame
    pRenderContext->copyResource(mpTemporalVBuffer.get(), renderData.getTexture(kInputVBuffer).get());
    pRenderContext->copyResource(mpTemporalViewDir.get(), renderData.getTexture(kInputViewDir).get());
    //the buffer that input reservoir id points to was actually the one that just got output written to it it just got switched in prep for next round
    if (mVisualizePathInfo)
    {
        VisualizePathsPass(pRenderContext, renderData, mInputReservoirID);
    }
    std::swap(mInputReservoirID, mTemporalReservoirID);
    std::swap(mReplayInputID, mTemporalReplayInputID);

    mpPixelDebug->endFrame(pRenderContext);

    mFrameCount++;
}

void RestirPTPass::GenerateInitialCandidates(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mPathParams.maxSurfaceBounces));
    mTracer.pProgram->addDefine("MAX_DIFFUSE_BOUNCES", std::to_string(mPathParams.maxDiffuseBounces));
    mTracer.pProgram->addDefine("MAX_SPECULAR_BOUNCES", std::to_string(mPathParams.maxSpecularBounces));
    mTracer.pProgram->addDefine("MAX_TRANSMISSION_BOUNCES", std::to_string(mPathParams.maxTransmissionBounces));

    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler)
    {
        mTracer.pProgram->addDefines(mpEmissiveSampler->getDefines());
    }
    mTracer.pProgram->addDefine("SHIFT_MAPPING_TYPE", std::to_string((uint32_t)mShiftMappingType));
    mTracer.pProgram->addDefine("IS_LAST_PASS", !mUseSpatialReuse && !mUseTemporalReuse ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData)); // it doesn't need this bc the tracer is not the one
                                                                                        // outputting final color

    // Prepare program vars (for mTracer). This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    auto& dict = renderData.getDictionary();
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;

    if (mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
    }
    if (mpEnvMapSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEnvMapSampler->bindShaderData(var["envMapSampler"]);
    }

    auto bind = [&](const ShaderVar& var, const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    for (auto channel : kInputChannels)
        bind(var, channel);
    for (auto channel : kOutputChannels)
        bind(var, channel);

    var["gReservoirBuffer"] = mpReservoirBuffers[mOutputReservoirID];
    mpReservoirBuffers[mOutputReservoirID]->setName("Restir Candidate Output Buffer");
    var["gDI_BGBuffer"] = mpDiBgBuffer;
    var["gReplayInputBuffer"] = mpReplayInputBuffers[mReplayInputID]; // this will be written to in this pass, and read from in replay
    var["gPathViewerPathBuffer"] = mpPathDataBuffer;

    uint elementCount = targetDim.x * targetDim.y;
    if (!mpCandidateGenDebugBuffer || mpCandidateGenDebugBuffer->getElementCount() < elementCount)
    {
        mpCandidateGenDebugBuffer = mpDevice->createStructuredBuffer(var["debugBuffer"], elementCount);
        mpCandidateGenDebugBuffer->setName("Restir Candidate Debug Buffer");
        var["debugBuffer"] = mpCandidateGenDebugBuffer;
    }

    mpScene->bindShaderData(var["gScene"]); // binds the Scene parameter block (as seen in Scene.slang w all the vertex and geometry buffers

    mpPixelDebug->prepareProgram(mTracer.pProgram, var);
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
}

/** Last param: index of the buffer whose samples you want to view debug colors for
*/
void RestirPTPass::VisualizePathsPass(RenderContext* pRenderContext, const RenderData& renderData, uint bufferInd)
{
    if (!mpVisualizePathsPass)
    {
        DefineList defines;
        if (mpSampleGenerator)
        {
            defines.add(mpSampleGenerator->getDefines());
        }

        ProgramDesc desc;
        desc.addShaderLibrary(kVisualizePathsShaderFile);
        desc.csEntry("main");
        mpVisualizePathsPass = ComputePass::create(mpDevice, desc, defines);
    }

    auto var = mpVisualizePathsPass->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gOutputDimensions"] = targetDim;
    var["gReservoirBuffer"] = mpReservoirBuffers[bufferInd];
    var["gOutputColor"] = renderData.getTexture(kOutputColor);

    mpVisualizePathsPass->execute(pRenderContext, targetDim.x, targetDim.y);
}

void RestirPTPass::GenSpatialOffsetsPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!mpGenSpatialOffsetsPass) // create the compute pass if it doesn't exist yet
    {
        DefineList defines;
        defines.add(mpSampleGenerator->getDefines());
        defines.add("NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors));
        defines.add("SPATIAL_NEIGHBOR_RADIUS", std::to_string(mSpatialNeighborRadius));

        ProgramDesc desc;
        desc.addShaderLibrary(kGenSpatialOffsetShaderFile);
        desc.csEntry("main");
        mpGenSpatialOffsetsPass = ComputePass::create(mpDevice, desc, defines);
    }

    // Defines
    auto program = mpGenSpatialOffsetsPass->getProgram();
    program->addDefine("NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors));
    program->addDefine("SPATIAL_NEIGHBOR_RADIUS", std::to_string(mSpatialNeighborRadius));

    auto var = mpGenSpatialOffsetsPass->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gOutputDimensions"] = targetDim;
    var["spatialOffsetBuffer"] = mpSpatialOffsetBuffer;

    mpGenSpatialOffsetsPass->execute(pRenderContext, targetDim.x, targetDim.y);
}

// Adds defines, prepares vars, binds resources and executes for path retrace passes
void RestirPTPass::PathRetracePass(RenderContext* pRenderContext, const RenderData& renderData, bool isTemporal) {
    //Defines
    mReplayTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mReplayTracer.pProgram->addDefine("NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors));
    mReplayTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData)); // you only need vbuffer and vieww

    //Prepare vars (basically copy of mTracer prepareVars() method)
    if (isTemporal && !mReplayTracer.pVarsTemporal || !isTemporal && !mReplayTracer.pVarsSpatial)
    {
        // Configure program.
        mReplayTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
        mReplayTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

        if (isTemporal && !mReplayTracer.pVarsTemporal)
        {
            // Create program variables for the current program.
            // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
            mReplayTracer.pVarsTemporal = RtProgramVars::create(mpDevice, mReplayTracer.pProgram, mReplayTracer.pBindingTableTemporal);

            // Bind utility classes into shared data.
            auto varT = mReplayTracer.pVarsTemporal->getRootVar();
            mpSampleGenerator->bindShaderData(varT);
        }
        if (!isTemporal && !mReplayTracer.pVarsSpatial)
        {
            mReplayTracer.pVarsSpatial = RtProgramVars::create(mpDevice, mReplayTracer.pProgram, mReplayTracer.pBindingTableSpatial);

            auto varS = mReplayTracer.pVarsSpatial->getRootVar();
            mpSampleGenerator->bindShaderData(varS);
        }
    }

    ref<RtProgramVars> pVars = isTemporal ? mReplayTracer.pVarsTemporal : mReplayTracer.pVarsSpatial;
    auto var = pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gOutputDimensions"] = targetDim;
    var["vBuffer"] = renderData.getTexture(kInputVBuffer);
    var["viewW"] = renderData.getTexture(kInputViewDir);
    var["inputBuffer"] = mpReplayInputBuffers[mReplayInputID];
    mpReplayInputBuffers[mReplayInputID]->setName("Restir Replay Input Buffer");
    if (isTemporal)
    {
        var["temporalInputBuffer"] = mpReplayInputBuffers[mTemporalReplayInputID];
        mpReplayInputBuffers[mTemporalReplayInputID]->setName("Restir Replay Input Buffer - Temporal");
        if (mpTemporalVBuffer)
        {
            var["temporalVBuffer"] = mpTemporalVBuffer;
        }
        if (mpTemporalViewDir)
        {
            var["temporalViewW"] = mpTemporalViewDir;
        }
        var["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    }
    var["outputBuffer"] = mpReplayOutputBuffer;
    var["outputColor"] = renderData.getTexture(kOutputColor);
    var["spatialOffsetBuffer"] = mpSpatialOffsetBuffer;
    var["replayedPathBuffer"] = mpReplayPathDataBuffer; //fill w replayed vertices for the path viewer

    mpPixelDebug->prepareProgram(mReplayTracer.pProgram, var);
    mpScene->raytrace(pRenderContext, mReplayTracer.pProgram.get(), pVars, uint3(targetDim, 1));
}

// Adds defines, binds resources, and executes path reuse passes
void RestirPTPass::PathReusePass(RenderContext* pRenderContext, const RenderData& renderData, bool isTemporal)
{
    // Set shader vars, and bind i/o buffers. don't add new defines here bc the compute pass is already set after you created it when you
    // set the scene
    ref<ComputePass> pPass = isTemporal ? mpTemporalReusePass : mpSpatialReusePass;
    auto program = pPass->getProgram();
    program->addDefine("SHIFT_MAPPING_TYPE", std::to_string((uint32_t)mShiftMappingType));
    if (!isTemporal)
    {
        program->addDefine("NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors));
    }
    bool lastPass = (isTemporal && !mUseSpatialReuse);
    program->addDefine("IS_LAST_PASS", lastPass ? "1" : "0");
    if (mpEmissiveSampler)
    {
        program->addDefines(mpEmissiveSampler->getDefines());
    }
    program->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    program->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    program->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");

    auto var = pPass->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    auto& dict = renderData.getDictionary();
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gOutputDimensions"] = targetDim;

    //Bind textures and buffers to the resample pass specifically
    auto passVar = pPass->getRootVar()["CB"]["gResamplePass"];
    passVar["vBuffer"] = renderData.getTexture(kInputVBuffer);
    passVar["viewW"] = renderData.getTexture(kInputViewDir);
    if (lastPass || !isTemporal)
    {
        passVar["outputColor"] = renderData.getTexture(kOutputColor);
    }
    if (isTemporal)
    {
        passVar["motionVectors"] = renderData.getTexture(kInputMotionVectors);
        if (mpTemporalVBuffer)
        {
            passVar["temporalVBuffer"] = mpTemporalVBuffer;
        }
        if (mpTemporalViewDir)
        {
            passVar["temporalViewW"] = mpTemporalViewDir;
        }
    }
    else
    {
        passVar["spatialOffsetBuffer"] = mpSpatialOffsetBuffer;
    }
    // We assume that the reservoir IDs have already been set to correct values before this func was called
    passVar["inputBuffer"] = mpReservoirBuffers[mInputReservoirID];
    passVar["outputBuffer"] = mpReservoirBuffers[mOutputReservoirID];
    mpReservoirBuffers[mInputReservoirID]->setName("Restir Resampled Input Buffer");
    mpReservoirBuffers[mOutputReservoirID]->setName("Restir Resampled Output Buffer");
    passVar["temporalBuffer"] = mpReservoirBuffers[mTemporalReservoirID];
    mpReservoirBuffers[mTemporalReservoirID]->setName("Restir Resampled Temporal Buffer");
    passVar["di_bgBuffer"] = mpDiBgBuffer;
    if (mShiftMappingType == ShiftMappingType::Hybrid)
    {
        passVar["replayedInputBuffer"] = mpReplayInputBuffers[mReplayInputID];
        passVar["replayedDataBuffer"] = mpReplayOutputBuffer;
    }

    uint elementCount = targetDim.x * targetDim.y;
    if (isTemporal)
    {
        if (!mpTemporalDebugBuffer || mpTemporalDebugBuffer->getElementCount() < elementCount)
        {
            mpTemporalDebugBuffer = mpDevice->createStructuredBuffer(passVar["debugBuffer"], elementCount);
            mpTemporalDebugBuffer->setName("Restir Debug Buffer - Temporal");
        }
        passVar["debugBuffer"] = mpTemporalDebugBuffer;
    }
    else
    {
        if (!mpSpatialDebugBuffer || mpSpatialDebugBuffer->getElementCount() < elementCount)
        {
            mpSpatialDebugBuffer = mpDevice->createStructuredBuffer(passVar["debugBuffer"], elementCount);
            mpSpatialDebugBuffer->setName("Restir Debug Buffer - Spatial");
        }
        passVar["debugBuffer"] = mpSpatialDebugBuffer;
    }

    // Samplers bind to root var because they are needed in ShiftMapping which we import
    if (mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
    }
    if (mpEnvMapSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEnvMapSampler->bindShaderData(var["envMapSampler"]);
    }
    mpScene->bindShaderData(var["gScene"]); // binds the Scene parameter block (as seen in Scene.slang w all the vertex and geometry buffers

    mpPixelDebug->prepareProgram(program, var);
    pPass->execute(pRenderContext, targetDim.x, targetDim.y);
}

void RestirPTPass::PathViewerPass(RenderContext* pRenderContext, const RenderData& renderData, bool alsoViewReplayPaths = false)
{
    if (!mpPathViewerPass) // create the compute pass for path viewer if it doesn't exist yet
    {
        DefineList defines;
        ProgramDesc desc;
        desc.addShaderLibrary(kPathViewerShaderFile);
        desc.csEntry("drawPaths");
        mpPathViewerPass = ComputePass::create(mpDevice, desc, defines);
    }
    
    if (!mpPathDataBuffer)
    {
        std::cout << "Path Viewer: failed, path data buffer not created" << std::endl;
        return;
    }

    if (!mpScene)
    {
        std::cout << "Path Viewer: failed, scene not set so could not bind camera" << std::endl;
        return;
    }

    auto var = mpPathViewerPass->getRootVar();
    var["gIsReplay"] = 0;
    var["gPathDataBuffer"] = mpPathDataBuffer;
    var["gMousePixelPos"] = mMousePixelPos;
    var["gOutputColor"] = renderData.getTexture(kOutputColor);
    var["gViewProjMatNoJitter"] = mpScene->getCamera()->getViewProjMatrixNoJitter();

    uint32_t elementCount = targetDim.x * targetDim.y;
    if (!mpPathViewerDebugBuffer || mpPathViewerDebugBuffer->getElementCount() < elementCount)
    {
        mpPathViewerDebugBuffer = mpDevice->createStructuredBuffer(var["debugBuffer"], elementCount);
        mpPathViewerDebugBuffer->setName("Restir Path Viewer Debug Buffer");
        var["debugBuffer"] = mpPathViewerDebugBuffer;
    }

    mpPathViewerPass->execute(pRenderContext, targetDim.x, targetDim.y);

    if (alsoViewReplayPaths)
    {
        if (!mpReplayPathDataBuffer)
        {
            std::cout << "Path Viewer: failed, replay path data buffer not created" << std::endl;
            return;
        }
        std::cout << "view replay" << std::endl;
        var["gIsReplay"] = 1;
        var["gPathDataBuffer"] = mpReplayPathDataBuffer;
        mpPathViewerPass->execute(pRenderContext, targetDim.x, targetDim.y);
    }
}

void RestirPTPass::renderUI(Gui::Widgets& widget) {
    bool dirty = false;

    if (widget.var("Max surface bounces", mPathParams.maxSurfaceBounces, 0u, MAX_BOUNCES))
    {
        // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
        mPathParams.maxDiffuseBounces = std::min(mPathParams.maxDiffuseBounces, mPathParams.maxSurfaceBounces);
        mPathParams.maxSpecularBounces = std::min(mPathParams.maxSpecularBounces, mPathParams.maxSurfaceBounces);
        mPathParams.maxTransmissionBounces = std::min(mPathParams.maxTransmissionBounces, mPathParams.maxSurfaceBounces);
        dirty = true;
    }
    widget.tooltip(
        "Maximum number of surface bounces (diffuse + specular + transmission).\n"
        "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold (0.2f) are also classified "
        "as diffuse events.\n"
        "0 = direct only, 1 = one indirect bounce etc"
    );

    dirty |= widget.var("Max diffuse bounces", mPathParams.maxDiffuseBounces, 0u, MAX_BOUNCES);
    widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max specular bounces", mPathParams.maxSpecularBounces, 0u, MAX_BOUNCES);
    widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max transmission bounces", mPathParams.maxTransmissionBounces, 0u, MAX_BOUNCES);
    widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.checkbox("Temporal reuse", mUseTemporalReuse);
    widget.tooltip("Use temporal reuse for ReSTIR", true);

    dirty |= widget.checkbox("Spatial reuse", mUseSpatialReuse);
    widget.tooltip("Use spatial reuse for ReSTIR", true);

    if (mUseSpatialReuse)
    {
        dirty |= widget.var("Num spatial neighbors", mNumSpatialNeighbors, 1u, 10u);
        widget.tooltip("Number of spatial neighbors to resample from. Capped at 10 currently, recommended 3", true);
    }

    dirty |= widget.dropdown("Shift mapping type", mShiftMappingType);
    widget.tooltip("What type of shift mapping to use for spatial/temporal reuse.", true);

    // Debugging UI

    if (widget.checkbox("Pause renderer and use path viewer", mUsePathViewer))
    {
        dirty = true;
        dirty |= widget.var("Max vertices to store", mNumPathViewerVertices, 1u, 5u);
        widget.tooltip("Number of vertices (starting x1) that path viewer will store. Capped at 5 currently.", true);
    }
    widget.tooltip("Whether we should pause the renderer and allow user to click a pixel to display its final path in space", true);

    dirty |= widget.checkbox("Visualize path info", mVisualizePathInfo);
    widget.tooltip("Whether information about the final path at each pixel is shown using debug colors", true);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }

    mpPixelDebug->renderUI(widget);
}

bool RestirPTPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    mpPixelDebug->onMouseEvent(mouseEvent);

    if (mouseEvent.type == MouseEvent::Type::ButtonDown)
    {
        float2 mousePos = (mouseEvent.pos) * float2(1920, 1080);
        mMousePixelPos = (uint2)mousePos;
        std::cout << mMousePixelPos.x << " " << mMousePixelPos.y << std::endl;

        return true;
    }
    if (mUsePathViewer)
    {
        return true; //don't want the camera to move, but i think this is broken TODO
    }
    
    return false;
}

bool RestirPTPass::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mUsePathViewer)
    {
        return true; // don't want the camera to move
    }
    return false;
}

void RestirPTPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mFrameCount = 0;
    resetLighting();
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    for (int i = 0; i < 3; i++)
    {
        mpReservoirBuffers[i] = nullptr;
    }
    mpTemporalVBuffer = nullptr;
    mpTemporalViewDir = nullptr;
    mpDiBgBuffer = nullptr;
    for (int i = 0; i < 2; i++)
    {
       mpReplayInputBuffers[i] = nullptr;
    }
    mpReplayOutputBuffer = nullptr;
    mReplayTracer.pProgram = nullptr;
    mReplayTracer.pBindingTableTemporal = nullptr;
    mReplayTracer.pVarsTemporal = nullptr;
    mReplayTracer.pBindingTableSpatial = nullptr;
    mReplayTracer.pVarsSpatial = nullptr;

    mUsePathViewer = false;
    mpPathDataBuffer = nullptr;
    mpReplayPathDataBuffer = nullptr;
    mMousePixelPos = uint2(0, 0);

    //Set the scene to the new one
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("RestirPTPass: This render pass does not support custom primitives.");
        }
        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            logWarning("RestirPTPass: This render pass does not support curves");
        }
        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            logWarning("RestirPTPass: This render pass does not support sdfs");
        }

        // Create ray tracing program for candidate gen
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kCandidateGenShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());

        // Create ray tracing programs for path retrace.
        ProgramDesc rDesc;
        rDesc.addShaderModules(mpScene->getShaderModules());
        rDesc.addShaderLibrary(kRetracePathsShaderFile);
        rDesc.setMaxPayloadSize(kMaxRetracePayloadSize);
        rDesc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        rDesc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mReplayTracer.pBindingTableTemporal = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& tsbt = mReplayTracer.pBindingTableTemporal;
        tsbt->setRayGen(rDesc.addRayGen("rayGenTemporal"));
        auto scatterMissShader = rDesc.addMiss("scatterMiss");
        tsbt->setMiss(0, scatterMissShader);
        mReplayTracer.pBindingTableSpatial = RtBindingTable::create(1, 1, mpScene->getGeometryCount());
        auto& ssbt = mReplayTracer.pBindingTableSpatial;
        ssbt->setRayGen(rDesc.addRayGen("rayGenSpatial"));
        ssbt->setMiss(0, scatterMissShader);

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            auto hitGroup = rDesc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit");
            tsbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                hitGroup
            );
            ssbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                hitGroup
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            auto hitGroup = rDesc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection");
            tsbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                hitGroup
            );
            ssbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                hitGroup
            );
        }

        mReplayTracer.pProgram = Program::create(mpDevice, rDesc, mpScene->getSceneDefines());

        // Create compute passes here since we needed mpScene
        // Helper for creating compute passes.
        auto createComputePass = [&](const std::string& file, const std::string& entryPoint,
                                     const std::vector<std::pair<std::string, std::string>>& customDefines = {})
        {
            DefineList defines;
            mpScene->getShaderDefines(defines);
            if (mpEmissiveSampler)
            {
                defines.add(mpEmissiveSampler->getDefines());
            }
            defines.add(mpSampleGenerator->getDefines());
            for (std::pair<std::string, std::string> def : customDefines)
            {
                defines.add(def.first, def.second);
            }

            ProgramDesc desc;
            mpScene->getShaderModules(desc.shaderModules);
            desc.addShaderLibrary(file);
            desc.csEntry(entryPoint);
            mpScene->getTypeConformances(desc.typeConformances);
            ref<ComputePass> pPass = ComputePass::create(mpDevice, desc, defines);
            return pPass;
        };
        mpSpatialReusePass = createComputePass(
            kSpatialComputeShaderFile, "main",
            {{"SHIFT_MAPPING_TYPE", std::to_string((uint32_t)mShiftMappingType)},
             {"NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors)},
             {"USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0"},
            {"USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0"},
            {"USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0"}}
        );
        mpTemporalReusePass = createComputePass(
            kTemporalComputeShaderFile, "main",
            {{"SHIFT_MAPPING_TYPE", std::to_string((uint32_t)mShiftMappingType)},
             {"IS_LAST_PASS", "0"},
             {"USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0"},
            {"USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0"},
             {"USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0"}}
        );
    }
}

// does it for mTracer only!
void RestirPTPass::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}

void RestirPTPass::resetLighting()
{
    // Retain the options for the emissive sampler. TODO: uncomment later when you add options for light bvh sampler
    /* if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }*/

    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
}

bool RestirPTPass::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    /* if (is_set(mUpdateFlags, IScene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mUpdateFlags, IScene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mUpdateFlags, IScene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }*/

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getILightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getILightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            /* switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler =
                    std::make_unique<LightBVHSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext), mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            default:
                FALCOR_THROW("Unknown emissive light sampler type");
            } */ //commented out bc we only have one type of emissive sampler for now, uniform (TODO can change later)
            mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                
            lightingChanged = true;
            //mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            /* if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }*/

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            //mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        /* auto defines = mpEmissiveSampler->getDefines();
        if (mpTracePass && mpTracePass->pProgram->addDefines(defines))
            mRecompile = true; */ //commented out bc we dont use trace pass
    }

    return lightingChanged;
}

/**
 Allocate resources like buffers, textures, etc. We only initialize resources here that are used in actual restir functionality (ie no debug data).
*/
void RestirPTPass::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint elementCount = targetDim.x * targetDim.y;
    auto var = mpReflectTypes->getRootVar();

    if (!mpPathDataBuffer || mpPathDataBuffer->getElementCount() < elementCount)
    {
        mpPathDataBuffer = mpDevice->createStructuredBuffer(var["pathViewerPath"], elementCount);
        mpPathDataBuffer->setName("Restir Debug Path Data Buffer");
    }
    if (mShiftMappingType == ShiftMappingType::Hybrid &&
        (!mpReplayPathDataBuffer || mpReplayPathDataBuffer->getElementCount() < elementCount))
    {
        mpReplayPathDataBuffer = mpDevice->createStructuredBuffer(var["pathViewerPath"], elementCount);
        mpReplayPathDataBuffer->setName("Restir Debug Path Data Buffer - Replayed");
    }

    ref<Buffer> pBuffer;
    for (int i = 0; i < 3; i++)
    {
        pBuffer = mpReservoirBuffers[i];
        if (!pBuffer || pBuffer->getElementCount() < elementCount)
        {
            pBuffer = mpDevice->createStructuredBuffer(var["reservoir"], elementCount);
            mpReservoirBuffers[i] = pBuffer;
        }
    }
    if (!mpDiBgBuffer || mpDiBgBuffer->getElementCount() < elementCount)
    {
        mpDiBgBuffer = mpDevice->createStructuredBuffer(var["color"], elementCount);
        mpDiBgBuffer->setName("Restir DI_BG_Buffer");
    }
    if (mShiftMappingType == ShiftMappingType::Hybrid)
    {
        for (int i = 0; i < 2; i++)
        {
            pBuffer = mpReplayInputBuffers[i];
            if (!pBuffer || pBuffer->getElementCount() < elementCount)
            {
                pBuffer = mpDevice->createStructuredBuffer(var["randomReplayInput"], elementCount);
                mpReplayInputBuffers[i] = pBuffer;
            }
        }
        uint maxReplaysPerPixel = std::max(2u, mNumSpatialNeighbors * 2); // because we are using pairwise mis for spatial
        if (!mpReplayOutputBuffer || mpReplayOutputBuffer->getElementCount() < maxReplaysPerPixel * elementCount)
        {
            mpReplayOutputBuffer = mpDevice->createStructuredBuffer(var["randomReplayOutput"], maxReplaysPerPixel * elementCount);
            mpReplayOutputBuffer->setName("Restir Replay Output Buffer");
        }
    }
    if (!mpSpatialOffsetBuffer || mpSpatialOffsetBuffer->getElementCount() < elementCount * mNumSpatialNeighbors)
    {
        mpSpatialOffsetBuffer = mpDevice->createStructuredBuffer(var["spatialOffset"], elementCount * mNumSpatialNeighbors);
        mpSpatialOffsetBuffer->setName("Restir Spatial Offset Buffer");
    }

    if (!mpTemporalVBuffer || (mpTemporalVBuffer->getWidth() < targetDim.x || mpTemporalVBuffer->getHeight() < targetDim.y))
    {
        mpTemporalVBuffer = mpDevice->createTexture2D(
            targetDim.x, targetDim.y,
            ResourceFormat::RGBA32Uint,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    }

    if (!mpTemporalViewDir || (mpTemporalViewDir->getWidth() < targetDim.x || mpTemporalViewDir->getHeight() < targetDim.y))
    {
        mpTemporalViewDir = mpDevice->createTexture2D(
            targetDim.x, targetDim.y,
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    }
}
