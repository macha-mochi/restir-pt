from falcor import *

def render_graph_ReSTIR_PT():
    g = RenderGraph("ReSTIR_PT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    ReSTIR_PTPass = createPass("RestirPTPass")
    g.addPass(ReSTIR_PTPass, "ReSTIR_PTPass")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("VBufferRT.vbuffer", "ReSTIR_PTPass.vbuffer")
    g.addEdge("VBufferRT.viewW", "ReSTIR_PTPass.viewW")
    g.addEdge("VBufferRT.mvec", "ReSTIR_PTPass.mvec")
    g.addEdge("ReSTIR_PTPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

ReSTIR_PT = render_graph_ReSTIR_PT()
try: m.addGraph(ReSTIR_PT)
except NameError: None
