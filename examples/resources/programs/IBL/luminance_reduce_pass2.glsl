#version 450 core

// ============================================================================
// PASS 2 — Final reduction to compute mean luminance
// ============================================================================

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input: group sums from pass 1
layout(std430, binding = 0) buffer GroupSums {
    float groupSums[];
};

// Output: single float = mean luminance
layout(std430, binding = 1) buffer Result {
    float meanLuminance;
};

uniform uint numGroups;
uniform uint numPixels;

shared float sharedSum[256];

void main()
{
    uint id = gl_LocalInvocationID.x;
    float sum = 0.0;

    // Each thread processes multiple groups
    for (uint i = id; i < numGroups; i += 256) {
        sum += groupSums[i];
    }

    sharedSum[id] = sum;
    barrier();

    // Parallel reduction
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (id < stride) {
            sharedSum[id] += sharedSum[id + stride];
        }
        barrier();
    }

    // Final write
    if (id == 0) {
        meanLuminance = sharedSum[0] / float(numPixels);
    }
}
