#version 450 core

/*==============================================================================
    BRDF Integration LUT (Specular IBL – Split Sum Approximation)

    This compute shader precomputes a 2D LUT used to approximate the specular
    part of the image-based lighting equation for GGX microfacet BRDFs.

    Assumptions:
    - GGX NDF
    - Smith geometry term with Schlick-GGX approximation
    - Fresnel-Schlick approximation
    - Normal is fixed to (0, 0, 1)
    - Output LUT stores (scale, bias) for Fresnel term
==============================================================================*/

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

/*==============================================================================
    Constants
==============================================================================*/
const float PI = 3.14159265359;

/*
    Number of Monte Carlo samples.
    1024 provides high-quality results suitable for offline or one-time runtime
    precomputation. Can be lowered for faster builds.
*/
const uint  SAMPLE_COUNT = 1024u;
const float INV_SAMPLE_COUNT = 1.0 / float(SAMPLE_COUNT);

/*
    Small epsilon used to avoid divisions by zero and numerical instabilities.
*/
const float EPSILON = 1e-4;

/*==============================================================================
    Output BRDF LUT
    - RG channels store (scale, bias)
    - 16-bit floats are sufficient and widely used in engines
==============================================================================*/
layout(binding = 0, rg16f) restrict writeonly uniform image2D LUT;

/*==============================================================================
    Low-discrepancy sequence (Hammersley)
    Used to reduce variance compared to pure random sampling.
==============================================================================*/
float radicalInverseVanDerCorput(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);

    // 1 / 2^32
    return float(bits) * 2.3283064365386963e-10;
}

vec2 hammersley(uint i)
{
    return vec2(float(i) * INV_SAMPLE_COUNT,
                radicalInverseVanDerCorput(i));
}

/*==============================================================================
    GGX importance sampling

    We sample the GGX NDF directly in spherical coordinates and transform
    the half-vector from tangent space to world space.
==============================================================================*/
vec3 importanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    /*
        GGX uses alpha = roughness^2.
        IMPORTANT: Do not square alpha again.
    */
    float alpha = roughness * roughness;

    float phi = 2.0 * PI * Xi.x;

    /*
        Inversion of the GGX NDF CDF.
        This produces a distribution of half-vectors matching GGX.
    */
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha * alpha - 1.0) * Xi.y));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    // Half-vector in tangent space
    vec3 Ht = vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );

    /*
        Construct an orthonormal basis around N.
        This avoids numerical issues when N is close to +Z.
    */
    vec3 up = abs(N.z) < 0.999
              ? vec3(0.0, 0.0, 1.0)
              : vec3(1.0, 0.0, 0.0);

    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    // Transform half-vector to world space
    return normalize(T * Ht.x + B * Ht.y + N * Ht.z);
}

/*==============================================================================
    Geometry term (Smith with Schlick-GGX)

    This version works directly with dot products for clarity and efficiency.
==============================================================================*/
float geometrySchlickGGX(float NdotX, float k)
{
    return NdotX / (NdotX * (1.0 - k) + k);
}

float geometrySmith(float NdotV, float NdotL, float roughness)
{
    /*
        k is derived from roughness.
        This formulation is commonly used in real-time engines.
    */
    float r = roughness;
    float k = (r * r) * 0.5;

    return geometrySchlickGGX(NdotV, k) *
           geometrySchlickGGX(NdotL, k);
}

/*==============================================================================
    Main
==============================================================================*/
void main()
{
    ivec2 size  = imageSize(LUT);
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);

    // Prevent out-of-bounds writes for non-multiple work group sizes
    if (coord.x >= size.x || coord.y >= size.y)
        return;

    /*
        LUT parameterization:
        X axis → N·V
        Y axis → roughness

        Using (size - 1) ensures correct coverage of [0, 1] inclusive.
    */
    float NdotV = float(coord.x) / float(size.x - 1);
    float roughness = float(coord.y) / float(size.y - 1);

    /*
        Prevent numerical issues:
        - NdotV must not be zero (division later)
        - roughness should never be exactly zero for GGX
    */
    NdotV    = max(NdotV, EPSILON);
    roughness = max(roughness, 0.001);

    /*
        Fixed normal.
        View vector is reconstructed from N·V assuming symmetry around Z.
    */
    vec3 N = vec3(0.0, 0.0, 1.0);
    vec3 V = vec3(
        sqrt(max(1.0 - NdotV * NdotV, 0.0)),
        0.0,
        NdotV
    );

    float scale = 0.0;
    float bias  = 0.0;

    float invNdotV = 1.0 / NdotV;

    /*--------------------------------------------------------------------------
        Monte Carlo integration of the BRDF
    --------------------------------------------------------------------------*/
    for (uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = hammersley(i);
        vec3 H  = importanceSampleGGX(Xi, N, roughness);

        /*
            Reconstruct light vector from half-vector.
            L = reflect(-V, H)
        */
        vec3 L = normalize(2.0 * dot(V, H) * H - V);

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        // Ignore samples below the horizon
        if (NdotL > 0.0)
        {
            float G = geometrySmith(NdotV, NdotL, roughness);

            /*
                Visibility term with Jacobian compensation from
                half-vector sampling.
            */
            float GVis = (G * VdotH) *
                         (invNdotV / max(NdotH, EPSILON));

            /*
                Fresnel-Schlick factor without F0.
                The split-sum approximation factors F0 later.
            */
            float Fc = pow(1.0 - VdotH, 5.0);

            scale += GVis * (1.0 - Fc);
            bias  += GVis * Fc;
        }
    }

    /*
        Average samples and clamp for FP16 safety.
    */
    vec2 result = clamp(vec2(scale, bias) * INV_SAMPLE_COUNT,
                        0.0, 1.0);

    imageStore(LUT, coord, vec4(result, 0.0, 0.0));
}
