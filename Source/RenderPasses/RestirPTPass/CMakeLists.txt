add_plugin(RestirPTPass)

target_sources(RestirPTPass PRIVATE
    RestirPTPass.cpp
    RestirPTPass.h
    RestirGenCandidates.rt.slang
    RestirTypes.slang
    Resampling.cs.slang
)

target_copy_shaders(RestirPTPass RenderPasses/RestirPTPass)

target_source_group(RestirPTPass "RenderPasses")
