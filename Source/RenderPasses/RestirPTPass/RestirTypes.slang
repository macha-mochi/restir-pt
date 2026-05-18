struct Reservoir
{
    // info needed for reservoir: the sample X, its weight W, the sum of weights W_sum, and the sample count M (capped)
    Sample X = Sample.NULL; // the current sample

    float W_X = 0;                              // unbiased contribution weight of sample X
    float w_sum = 0; //sum of all lowercase w seen so far, for use in resampling
    uint M = 1;
    uint c = 0; //cap at 20
};
struct Sample {
    static Sample NULL = { float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f), 0 };

    float3 x0; // first vertex
    float3 p_hat_X; // aka f, the realized contribution of this sample (no pdfs multiplied yet)
    float w_i;    // w_i = (resampling MIS weight m_i) * (target function evaluated for X_i) / (pdf of sampling X_i), where reciprocal of pdf of X_i is just W_Xi

}
void updateReservoir(Reservoir r, Sample sample, float w_i_new, uint c_i, SampleGenerator sg) {
    r.w_sum += w_i_new;
    r.c += c_i;
    if (sampleNext1D(sg) < w_i_new / r.w_sum) {
        // replace current sample with new one, if non null
        r.X = sample;
        if (sample != Sample.NULL) {
            r.W_X = r.w_sum / sample.p_hat_X;
        } else {
            r.W_X = 0;
        }
        r.M++;
    }
}
