#include "RestirTypes.slangh"

import Utils.Sampling.SampleGenerator;

struct Sample {
    float3 p_hat; // aka f, the realized contribution of this sample (no pdfs multiplied yet)
    float w_i; // w_i = (resampling MIS weight m_i) * (target function evaluated for X_i) / (pdf of sampling X_i), where reciprocal of pdf of X_i is just W_Xi
    RCVertex rcVertex; // struct with reconnection vertex data

    static Sample createNull() {
        Sample s;
        s.p_hat = { 0.f, 0.f, 0.f };
        s.w_i = 0.f;
        s.rcVertex = {};
        return s;
    }

    __init(float3 p_hat, float w_i, RCVertex rcv) {
        this.p_hat = p_hat;
        this.w_i = w_i;
        this.rcVertex = rcv;
    }

    static bool isNull(Sample X) {
        return all(X.p_hat == 0.f) || X.w_i == 0.f;
    }
};
struct Reservoir
{
    // Information that is updated every time updateReservoir is called
    Sample X = Sample::createNull(); // the current sample
    float W_X = 1.f;                              // unbiased contribution weight of sample X
    float w_sum = 0; //sum of all lowercase w seen so far, for use in resampling
    uint c = 0; //cap at 20

    // Information that needs to be initialized by caller in path tracing loop
    PRNGSeed seed;
    PRNGSeed postRCSeed;    // only used if we need to replay the path after RC, temporally
    uint rcVertexIndex = 0; //0 is camera, 1 is primary hit, etc. This should be >= 2
    float J; // holds denominator of jacobian calc'd from base path, so we don't have to recompute when resampling 

    [mutating] //Returns true if sample was replaced, false ow
    bool updateReservoir(Sample sample, float w_i_new, uint c_i, SampleGenerator sg) {
        w_sum += w_i_new;
        c += c_i;
        if (w_i_new == 0) { // no chance to be chosen and no need to update W_X since w_sum doesn't change
            return false;
        }
        bool updateSample = false;
        if (sampleNext1D(sg) < w_i_new / w_sum) {
            // replace current sample with new one, if non null
            X = sample;
            updateSample = true;
        }
        W_X = w_sum / luminance(X.p_hat); // if we got to this point, w_i_new is non zero meaning mi, p_hat, and W_X were all nonzero. so not a null sample
        c = min(c, 20); //setting a c cap of 20

        return updateSample;
    }
};
float luminance(float3 vec) {
    return dot(vec, float3(0.2126, 0.7152, 0.0722));
}
