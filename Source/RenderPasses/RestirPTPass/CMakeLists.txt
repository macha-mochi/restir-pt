add_plugin(RestirPTPass)

target_sources(RestirPTPass PRIVATE
    RestirPTPass.cpp
    RestirPTPass.h
    RestirGenCandidates.rt.slang
    RestirTypes.slangh
    RestirReservoir.slang
    Resampling.cs.slang
    PathViewer.cs.slang
)

target_copy_shaders(RestirPTPass RenderPasses/RestirPTPass)

target_source_group(RestirPTPass "RenderPasses")
