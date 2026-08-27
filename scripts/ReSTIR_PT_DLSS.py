from falcor import *

def render_graph_ReSTIR_PT_DLSS():
    g = RenderGraph("ReSTIR_PT_DLSS")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    RestirPTPass = createPass("RestirPTPass", {'maxBounces': 3})
    g.addPass(RestirPTPass, "RestirPTPass")
    DLSSPass = createPass("DLSSPass", {'enabled': True})
    g.addPass(DLSSPass, "DLSSPass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")

    g.addEdge("VBufferRT.vbuffer", "RestirPTPass.vbuffer")
    g.addEdge("VBufferRT.viewW", "RestirPTPass.viewW")
    g.addEdge("VBufferRT.mvec", "RestirPTPass.mvec")

    g.addEdge("RestirPTPass.color", "DLSSPass.color")
    g.addEdge("VBufferRT.depth", "DLSSPass.depth")
    g.addEdge("VBufferRT.mvec", "DLSSPass.mvec")

    g.addEdge("DLSSPass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

ReSTIR_PT_DLSS = render_graph_ReSTIR_PT_DLSS()
try: m.addGraph(ReSTIR_PT_DLSS)
except NameError: None
