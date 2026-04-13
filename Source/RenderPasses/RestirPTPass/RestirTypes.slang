struct Reservoir
{
    // info needed for reservoir: the sample X, its weight W, the sum of weights W_sum, and the sample count M (capped)
    float3 radiance; // the color output of the path TODO: check if any other information is needed
    float W;
    float W_sum;
    uint M;
};
