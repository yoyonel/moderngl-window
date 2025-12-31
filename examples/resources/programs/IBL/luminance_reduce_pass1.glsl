#version 450 core

// ============================================================================
// PASS 1 — Compute sum of luminance per workgroup (no atomics)
// ============================================================================

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input HDR texture
layout(binding = 0) uniform sampler2D hdrTexture;

// Output: one float per workgroup
layout(std430, binding = 1) buffer GroupSums {
    float groupSums[];
};

shared float sharedLum[16 * 16];

void main()
{
    ivec2 texSize = textureSize(hdrTexture, 0);
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    uint localIndex = gl_LocalInvocationIndex;

    // Load luminance or 0 if out of bounds
    float lum = 0.0;
    if (coord.x < texSize.x && coord.y < texSize.y) {
        vec3 color = texelFetch(hdrTexture, coord, 0).rgb;
        lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    }

    sharedLum[localIndex] = lum;
    barrier();

    // Parallel reduction (256 → 1)
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (localIndex < stride) {
            sharedLum[localIndex] += sharedLum[localIndex + stride];
        }
        barrier();
    }

    // Write result
    if (localIndex == 0) {
        uint groupIndex =
            gl_WorkGroupID.y * gl_NumWorkGroups.x +
            gl_WorkGroupID.x;

        groupSums[groupIndex] = sharedLum[0];
    }
}
