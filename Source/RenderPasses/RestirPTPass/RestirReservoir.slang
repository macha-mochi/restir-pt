#include "RestirTypes.slangh"

__exported import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;

struct Sample {
    float3 x0;      // first vertex
    float3 p_hat_X; // aka f, the realized contribution of this sample (no pdfs multiplied yet)
    float w_i; // w_i = (resampling MIS weight m_i) * (target function evaluated for X_i) / (pdf of sampling X_i), where reciprocal of pdf of X_i is just W_Xi
    float3 jacobian_denom;

    static Sample createNull() {
        Sample s;
        s.x0 = { 0.f, 0.f, 0.f };
        s.p_hat_X = { 0.f, 0.f, 0.f };
        s.w_i = 0.f;
        s.jacobian_denom = { 0.f, 0.f, 0.f };
        return s;
    }
};
struct Reservoir
{
    // info needed for reservoir: the sample X, its weight W, the sum of weights W_sum, and the sample count M (capped)
    Sample X = { { 0.f, 0.f, 0.f }, { 0.f, 0.f, 0.f }, 0.f }; // the current sample

    float W_X = 0;                              // unbiased contribution weight of sample X
    float w_sum = 0; //sum of all lowercase w seen so far, for use in resampling
    uint c = 0; //cap at 20

    [mutating]
    void updateReservoir(Sample sample, float w_i_new, uint c_i, SampleGenerator sg) {
        w_sum += w_i_new;
        c += c_i;
        if (w_i_new == 0) { // no chance to be chosen and no need to update W_X since w_sum doesn't change
            return;
        }
        if (sampleNext1D(sg) < w_i_new / w_sum) {
            // replace current sample with new one, if non null
            X = sample;
        }
        W_X = w_sum / luminance(X.p_hat_X); // if we got to this point, w_i_new is non zero meaning mi, p_hat, and W_X were all nonzero. so not a null sample
    }
};
bool operator ==(const Sample s1, const Sample s2) {
    vector<bool, 3> b = s1.p_hat_X == s2.p_hat_X;
    return b[0] && b[1] && b[2] && s1.w_i == s2.w_i;
}
float luminance(float3 vec) {
    return dot(vec, float3(0.2126, 0.7152, 0.0722));
}
