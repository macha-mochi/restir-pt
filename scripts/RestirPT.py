from falcor import *

def render_graph_RestirPT():
    g = RenderGraph("RestirPT")
    AccumulatePass = createPass("AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    RestirPTPass = createPass("RestirPTPass", {'maxBounces': 3})
    g.addPass(RestirPTPass, "RestirPTPass")
    VBufferRT = createPass("VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
    g.addPass(VBufferRT, "VBufferRT")
    g.addEdge("VBufferRT.vbuffer", "RestirPTPass.vbuffer")
    g.addEdge("VBufferRT.viewW", "RestirPTPass.viewW")
    g.addEdge("RestirPTPass.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    return g

RestirPT = render_graph_RestirPT()
try: m.addGraph(RestirPT)
except NameError: None
