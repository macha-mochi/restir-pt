/**
 Modified from Falcor's Pathtracer.
*/
enum class PathFlags
{
    active                      = 0x0001,   ///< Path is active/terminated.
    hit                         = 0x0002,   ///< Result of the scatter ray (0 = miss, 1 = hit).

    transmission                = 0x0004,   ///< Scatter ray went through a transmission event.
    specular                    = 0x0008,   ///< Scatter ray went through a specular event.
    delta                       = 0x0010,   ///< Scatter ray went through a delta event.

    lightSampledUpper           = 0x0020,   ///< Last path vertex sampled lights using NEE (in upper hemisphere).
    lightSampledLower           = 0x0040,   ///< Last path vertex sampled lights using NEE (in lower hemisphere).

    bsdfSampled                 = 0x0080,   ///< The final vertex of the path was a bsdf sample.
    neeSampled                  = 0x0100,   ///< The final vertex of the path was a light (NEE) sample.
    sampledAnalytic             = 0x0200,   ///< This path ended in analytic light
    sampledEmissive             = 0x0400,   ///< This path ended in emissive tri
    sampledEnvMap               = 0x0800,   ///< This path ended in env map

    // Other bits unused (leaves space for more path flags).
};

struct PathState {
    uint flags; ///< PathFlags ORd together

    __init() {
        flags = (uint)PathFlags::active;
    }

    __init(uint _f) {
        flags = _f; //assumes _f is a valid combination of OR'd path flags
    }

    bool isTerminated() { return !isActive(); }
    bool isActive() { return hasFlag(PathFlags::active); }
    bool isHit() { return hasFlag(PathFlags::hit); }
    bool isMiss() { return !isHit(); }
    bool isTransmission() { return hasFlag(PathFlags::transmission); }
    bool isSpecular() { return hasFlag(PathFlags::specular); }
    bool isDelta() { return hasFlag(PathFlags::delta); } ///< (true if) bsdf sampled at last path vertex is delta
    bool nonDelta() { return !isDelta(); }

    bool isLightSampledUpper() { return hasFlag(PathFlags::lightSampledUpper); } ///< (true if) Last path vertex sampled lights using NEE (in upper hemisphere).
    bool isLightSampledLower() { return hasFlag(PathFlags::lightSampledLower); } ///< (true if) Last path vertex sampled lights using NEE (in lower hemisphere).

    bool isBSDFSampled() { return hasFlag(PathFlags::bsdfSampled); }
    bool isNEESampled() { return hasFlag(PathFlags::neeSampled); }
    bool sampledAnalytic() { return hasFlag(PathFlags::sampledAnalytic); }
    bool sampledEmissive() { return hasFlag(PathFlags::sampledEmissive); }
    bool sampledEnvMap() { return hasFlag(PathFlags::sampledEnvMap); }

    [mutating] void terminate() { setActive(false); }
    [mutating] void setActive(bool value = true) { setFlag(PathFlags::active, value); }
    [mutating] void setIsHit() { setFlag(PathFlags::hit); }
    [mutating] void setIsMiss() { setFlag(PathFlags::hit, false); }
    [mutating] void setTransmission(bool value = true) { setFlag(PathFlags::transmission, value); }
    [mutating] void setSpecular(bool value = true) { setFlag(PathFlags::specular, value); }
    [mutating] void setDelta(bool value = true) { setFlag(PathFlags::delta, value); }
    [mutating] void setLightSampled(bool upper, bool lower) { setFlag(PathFlags::lightSampledUpper, upper); setFlag(PathFlags::lightSampledLower, lower); }
    [mutating] void setSampleType(PathFlags technique, PathFlags lightType) //sets the corresponding flags to true and all others to false
    {
        const uint bits = uint(technique | lightType);
        clearSampleTechniqueAndLightTypes();
        flags |= bits;
    }
    [mutating] void clearSampleTechniqueAndLightTypes() {
        uint bits = uint(PathFlags::bsdfSampled | PathFlags::neeSampled | PathFlags::sampledAnalytic | PathFlags::sampledEmissive | PathFlags::sampledEnvMap);
        flags &= ~bits;
    }

    bool hasFlag(PathFlags flag)
    {
        const uint bit = uint(flag);
        return (flags & bit) != 0;
    }

    [mutating] void setFlag(PathFlags flag, bool value = true)
    {
        const uint bit = uint(flag);
        if (value) flags |= bit;
        else flags &= ~bit;
    }
}

