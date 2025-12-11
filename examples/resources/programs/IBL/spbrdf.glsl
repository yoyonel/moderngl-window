#version 450 core

const float PI = 3.141592;
const float TWO_PI = 2 * PI;
const float MIN_EPSILON = 0.001;

const uint SAMPLE_COUNT = 1024u;
const float INV_SAMPLE_COUNT = 1.0 / float(SAMPLE_COUNT);

layout(binding=0, rg16f) restrict writeonly uniform image2D LUT;

// Compute Van der Corput radical inverse
float computeRadicalInverse(uint inputBits)
{
    inputBits = (inputBits << 16u) | (inputBits >> 16u);
    inputBits = ((inputBits & 0x55555555u) << 1u) | ((inputBits & 0xAAAAAAAAu) >> 1u);
    inputBits = ((inputBits & 0x33333333u) << 2u) | ((inputBits & 0xCCCCCCCCu) >> 2u);
    inputBits = ((inputBits & 0x0F0F0F0Fu) << 4u) | ((inputBits & 0xF0F0F0F0u) >> 4u);
    inputBits = ((inputBits & 0x00FF00FFu) << 8u) | ((inputBits & 0xFF00FF00u) >> 8u);
    return float(inputBits) * 2.3283064365386963e-10;
}

vec2 sampleHammersley(uint sampleIndex)
{
    return vec2(sampleIndex * INV_SAMPLE_COUNT, computeRadicalInverse(sampleIndex));
}

vec3 sampleGGXTangentSpace(float randU1, float randU2, float surfaceRoughness)
{
    float alpha = surfaceRoughness * surfaceRoughness;

    float cosTheta = sqrt((1.0 - randU2) / (1.0 + (alpha*alpha - 1.0) * randU2));
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
    float phi = TWO_PI * randU1;

    return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

float schlickGGXApproximationSingleTerm(float cosTheta, float k)
{
    return cosTheta / (cosTheta * (1.0 - k) + k);
}

float schlickGGXApproximationIBL(float incidentLightAngle, float outgoingLightAngle, float surfaceRoughness)
{
    float remappedRoughness = (surfaceRoughness * surfaceRoughness) / 2.0;
    return schlickGGXApproximationSingleTerm(incidentLightAngle, remappedRoughness) * schlickGGXApproximationSingleTerm(outgoingLightAngle, remappedRoughness);
}

// this mirrors the number in binary around the decimal point
// aka return: a0 / 2 + a1 / 4 + a2 / 8 + ...
// where ax is the a'th digit
//
// source: http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html#sec-SourceCode
float radicalInverseVanDerCorput(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

// Hammersley Sequence, which is based on Van der Corput sequence
// This is a "low-discrepancy" sequence, aka quasirandom sequence
// It returns sequence sample i given N total samples
// It corresponds to a point on a unit square [0, 1)
//
// x value is evenly distributed across the unit square
// y is a random value generated with van der corput sequence
vec2 hammersley(uint i, uint N)
{
    return vec2(float(i) / float(N), radicalInverseVanDerCorput(i));
}

vec3 importanceSampleGGX(vec2 unitSquareSample, vec3 N, float roughness) {
    float alpha = roughness * roughness;

    // map the x/y of our unit square sample onto hemisphere
    // using spherical coordinates
    float phi = 2.0 * PI * unitSquareSample.x;
    float cosTheta = sqrt((1.0 - unitSquareSample.y) / (1.0 + (alpha * alpha - 1.0) * unitSquareSample.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // spherical to cartesian
    vec3 H; // halfway vector
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;

    // tangent space to world
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    vec3 sampleVector = tangent * H.x + bitangent * H.y + N * H.z;
    return normalize(sampleVector);
}

// Geometry function
//
//         n * v
//   -------------------
//   (n * v)(1 - k) + k
//
float geometrySchlickGGX(vec3 n, vec3 v, float k) {

    float nDotV = max(dot(n, v), 0.0);

    float numerator = nDotV;
    float denomenator = nDotV * (1.0 - k) + k;

    return numerator / denomenator;
}

// smiths method for taking into account view direction and light direction
float geometrySmith(vec3 n, vec3 v, vec3 l, float roughness) {
    float k = (roughness * roughness) / 2.0;
    return geometrySchlickGGX(n, v, k) * geometrySchlickGGX(n, l, k);
}

layout(local_size_x=32, local_size_y=32, local_size_z=1) in;
void main(void)
{
    float NdotV = gl_GlobalInvocationID.x / float(imageSize(LUT).x);
    float roughness = gl_GlobalInvocationID.y / float(imageSize(LUT).y);

    vec3 N = vec3(0.0, 0.0, 1.0);
    vec3 V;
    V.x = sqrt(1.0 - NdotV * NdotV);
    V.y = 0.0f;
    V.z = NdotV;

    float F0Scale = 0.0;
    float F0Bias = 0.0;

    for (uint i = 0u; i < SAMPLE_COUNT; i++) {
        vec2 unitSquareSample = hammersley(i, SAMPLE_COUNT);
        vec3 H = importanceSampleGGX(unitSquareSample, N, roughness); // halfway
        vec3 L = normalize(2.0 * dot(V, H) * H - V); // light sample direction

        float NdotL = max(L.z, 0.0);
        float NdotH = max(H.z, 0.0);
        float VdotH = max(dot(V, H), 0.0);

        if (NdotL > 0.0) { // light with negative dot product is behind our hemisphere
            float G = geometrySmith(N, V, L, roughness);
            float GVis = (G * VdotH) / (NdotH * NdotV); // not totally sure where this comes from

            float partialFresnel = pow(1.0 - VdotH, 5.0);

            // (I think there is no NDF term here because we already used it for importance sampling)
            F0Scale += GVis * (1.0 - partialFresnel);
            F0Bias += GVis * partialFresnel;
        }
    }

    const vec2 F0 = vec2(F0Scale, F0Bias) * INV_SAMPLE_COUNT;

    imageStore(LUT, ivec2(gl_GlobalInvocationID), vec4(F0, 0, 0));
}