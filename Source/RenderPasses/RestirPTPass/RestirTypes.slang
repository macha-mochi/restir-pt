struct Reservoir
{
    // info needed for reservoir: the sample X, its weight W, the sum of weights W_sum, and the sample count M (capped)
    //Sample X = Sample.NULL; // the current sample
    float3 p_hat_X = float3(0.0f, 0.0f, 0.0f); //p_hat(X) or f(X), without pdf multiplied
    float W_X = 0;                              // unbiased contribution weight of sample X
    float w_i = 0; // w_i = (resampling MIS weight m_i) * (target function evaluated for X_i) / (pdf of sampling X_i)
    float w_sum = 0; //sum of all lowercase w seen so far, for use in resampling
    uint M = 1;
};
struct Sample { //TODO yeah idk what to put in here yet
    float3 x0; //first vertex

    static Sample NULL = { float3(0.0f, 0.0f, 0.0f) };
}
