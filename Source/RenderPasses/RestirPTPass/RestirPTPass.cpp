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
const char kSpatialComputeShaderFile[] = "RenderPasses/RestirPTPass/SpatialReuse.cs.slang";
const char kTemporalComputeShaderFile[] = "RenderPasses/RestirPTPass/TemporalReuse.cs.slang";
const char kPathViewerShaderFile[] = "RenderPasses/RestirPTPass/PathViewer.cs.slang";

//CANDIDATE GENERATION SETTINGS
// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 350u;
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
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);

    mOutputReservoirID = 0;
    mInputReservoirID = 1;
    mTemporalReservoirID = 2;
}

void RestirPTPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
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
    props[kMaxBounces] = mMaxBounces;
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
        PathViewerPass(pRenderContext, renderData);

        return;
    }

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

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
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
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData)); //it doesn't need this bc the tracer is not the one outputting final color

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
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

    uint elementCount = targetDim.x * targetDim.y;
    if (!mpCandidateGenDebugBuffer || mpCandidateGenDebugBuffer->getElementCount() < elementCount)
    {
        mpCandidateGenDebugBuffer = mpDevice->createStructuredBuffer(var["debugBuffer"], elementCount);
        mpCandidateGenDebugBuffer->setName("Restir Candidate Debug Buffer");
        var["debugBuffer"] = mpCandidateGenDebugBuffer;
    }
    if (!mpPathDataBuffer || mpPathDataBuffer->getElementCount() < elementCount)
    {
        mpPathDataBuffer = mpDevice->createStructuredBuffer(var["gPathDebugBuffer"], elementCount);
        mpPathDataBuffer->setName("Restir Debug Path Data Buffer");
        var["gPathDebugBuffer"] = mpPathDataBuffer;
    }

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    
    //Update reseroir IDs for upcoming reuse
    mInputReservoirID = mOutputReservoirID;
    mOutputReservoirID = 1 - mOutputReservoirID;

    //SPATIOTEMPORAL RESAMPLING
    if (mUseTemporalReuse && mFrameCount > 0) //no temporal history yet on frame 0
    {
        PathReusePass(pRenderContext, renderData, true);
        mInputReservoirID = mOutputReservoirID;
        mOutputReservoirID = 1 - mOutputReservoirID;
    }
    if (mUseSpatialReuse)
    {
        PathReusePass(pRenderContext, renderData, false);
        mInputReservoirID = mOutputReservoirID;
        mOutputReservoirID = 1 - mOutputReservoirID;
    }

    //Update data for next frame
    pRenderContext->copyResource(mpTemporalVBuffer.get(), renderData.getTexture(kInputVBuffer).get());
    pRenderContext->copyResource(mpTemporalViewDir.get(), renderData.getTexture(kInputViewDir).get());
    //the buffer that input reservoir id points to was actually the one that just got output written to it it just got switched in prep for next round
    pRenderContext->copyResource(mpReservoirBuffers[mTemporalReservoirID].get(), mpReservoirBuffers[mInputReservoirID].get()); //TODO potentially wasteful, mb make a function to swap the reservoir ids or smth instead

    mFrameCount++;
}

void RestirPTPass::PathReusePass(RenderContext* pRenderContext, const RenderData& renderData, bool isTemporal)
{
    // Set shader vars, and bind i/o buffers. don't add new defines here bc the compute pass is already set after you created it when you
    // set the scene
    ref<ComputePass> pPass = isTemporal ? mpTemporalReusePass : mpSpatialReusePass;
    auto program = pPass->getProgram();
    program->addDefine("SHIFT_MAPPING_TYPE", std::to_string((uint32_t)mShiftMappingType));
    program->addDefine("NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors));
    program->addDefine("USE_SPATIAL", mUseSpatialReuse ? "1" : "0");
    program->addDefine("USE_TEMPORAL", mUseTemporalReuse && mFrameCount > 0 ? "1" : "0");
    bool firstPass = isTemporal || (!mUseTemporalReuse);
    program->addDefine("IS_FIRST_PASS", firstPass ? "1" : "0");
    bool lastPass = (isTemporal && !mUseSpatialReuse);
    program->addDefine("IS_LAST_PASS", lastPass ? "1" : "0");

    auto var = pPass->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    auto& dict = renderData.getDictionary();
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gOutputDimensions"] = targetDim;

    //Bind textures and buffers to the resample pass specifically
    /*
    * // Inputs
    Texture2D<PackedHitInfo> vBuffer;
    Texture2D<PackedHitInfo> temporalVBuffer;
    Texture2D<float4> viewW; // Originally optional but we include it
    Texture2D<float4> temporalViewW;
    Texture2D<float2> motionVectors;
    StructuredBuffer<Reservoir> inputBuffer;
    StructuredBuffer<Reservoir> temporalBuffer;
    StructuredBuffer<float3> di_bgBuffer;
    // Outputs
    RWStructuredBuffer<uint> spatialOffsetBuffer; 
    RWStructuredBuffer<Reservoir> outputBuffer;
    RWTexture2D<float4> outputColor;

    // Debug
    RWStructuredBuffer<DebugData> debugBuffer;
    */
    auto passVar = pPass->getRootVar()["CB"]["gResamplePass"];
    passVar["vBuffer"] = renderData.getTexture(kInputVBuffer);
    passVar["viewW"] = renderData.getTexture(kInputViewDir);
    passVar["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    if (lastPass || !isTemporal)
    {
        passVar["outputColor"] = renderData.getTexture(kOutputColor);
    }
    if (mpTemporalVBuffer)
    {
        passVar["temporalVBuffer"] = mpTemporalVBuffer;
    }
    if (mpTemporalViewDir)
    {
        passVar["temporalViewW"] = mpTemporalViewDir;
    }
    // We assume that the reservoir IDs have already been set to correct values before this func was called
    passVar["inputBuffer"] = mpReservoirBuffers[mInputReservoirID];
    passVar["outputBuffer"] = mpReservoirBuffers[mOutputReservoirID];
    mpReservoirBuffers[mInputReservoirID]->setName("Restir Resampled Input Buffer");
    mpReservoirBuffers[mOutputReservoirID]->setName("Restir Resampled Output Buffer");
    passVar["temporalBuffer"] = mpReservoirBuffers[mTemporalReservoirID];
    mpReservoirBuffers[mTemporalReservoirID]->setName("Restir Resampled Temporal Buffer");
    passVar["di_bgBuffer"] = mpDiBgBuffer;
    passVar["spatialOffsetBuffer"] = mpSpatialOffsetBuffer;

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
    
    mpScene->bindShaderData(var["gScene"]); // binds the Scene parameter block (as seen in Scene.slang w all the vertex and geometry buffers

    pPass->execute(pRenderContext, targetDim.x, targetDim.y);
}

void RestirPTPass::PathViewerPass(RenderContext* pRenderContext, const RenderData& renderData)
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
}

void RestirPTPass::renderUI(Gui::Widgets& widget) {
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.checkbox("Temporal reuse", mUseTemporalReuse);
    widget.tooltip("Use temporal reuse for Restir resampling", true);

    dirty |= widget.checkbox("Spatial reuse", mUseSpatialReuse);
    widget.tooltip("Use spatial reuse for Restir resampling", true);

    if (mUseSpatialReuse)
    {
        dirty |= widget.var("Num spatial neighbors", mNumSpatialNeighbors, 1u, 10u);
        widget.tooltip("Number of spatial neighbors to resample from. Capped at 10 currently, recommended 3", true);
    }

    dirty |= widget.dropdown("Shift mapping type", mShiftMappingType);
    widget.tooltip("What type of shift mapping to use for spatial/temporal reuse.", true);

    dirty |= widget.checkbox("Pause renderer and use path viewer", mUsePathViewer);
    widget.tooltip("Whether we should pause the renderer and allow user to click a pixel to display its final path", true);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

bool RestirPTPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    if (mouseEvent.type == MouseEvent::Type::ButtonDown)
    {
        float2 mousePos = (mouseEvent.pos) * float2(1920, 1080);
        mMousePixelPos = (uint2)mousePos;
        std::cout << mMousePixelPos.x << " " << mMousePixelPos.y << std::endl;

        return true;
    }
    if (mUsePathViewer)
    {
        return true; //don't want the camera to move
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
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;
    mUsePathViewer = false;
    mpPathDataBuffer = nullptr;
    mMousePixelPos = uint2(0, 0);
    resetLighting();
    for (int i = 0; i < 3; i++)
    {
        mpReservoirBuffers[i] = nullptr;
    }
    mpTemporalVBuffer = nullptr;
    mpTemporalViewDir = nullptr;
    mpDiBgBuffer = nullptr;
    

    //Set the scene to the new one
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("RestirPTPass: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kCandidateGenShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            logWarning("RestirPTPass: This render pass does not support curves");
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            logWarning("RestirPTPass: This render pass does not support sdfs");
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());

        // Create the resampling compute passes here since we needed mpScene
        // Helper for creating compute passes.
        auto createComputePass = [&](const std::string& file, const std::string& entryPoint,
                                     const std::vector<std::pair<std::string, std::string>>& customDefines = {})
        {
            DefineList defines;
            mpScene->getShaderDefines(defines);
            
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
             {"USE_SPATIAL", mUseSpatialReuse ? "1" : "0"},
             {"NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors)},
             {"USE_TEMPORAL", mUseTemporalReuse ? "1" : "0"},
             {"IS_FIRST_PASS", "0"},
             {"IS_LAST_PASS", "0"}}
        );
        mpTemporalReusePass = createComputePass(
            kTemporalComputeShaderFile, "main",
            {{"SHIFT_MAPPING_TYPE", std::to_string((uint32_t)mShiftMappingType)},
             {"USE_SPATIAL", mUseSpatialReuse ? "1" : "0"},
             {"NUM_SPATIAL_NEIGHBORS", std::to_string(mNumSpatialNeighbors)},
             {"USE_TEMPORAL", mUseTemporalReuse ? "1" : "0"},
             {"IS_FIRST_PASS", "0"},
             {"IS_LAST_PASS", "0"}}
        );
    }
}

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
