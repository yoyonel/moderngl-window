#version 450 core

#if defined VERTEX_SHADER

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord_0;

out vec2 TexCoords;
out vec3 WorldPos;
out vec3 Normal;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform mat3 normalMatrix;

void main() {
    TexCoords = in_texcoord_0;
    WorldPos = vec3(model * vec4(in_position, 1.0));
    Normal = normalize(normalMatrix * in_normal);

    gl_Position =  projection * view * vec4(WorldPos, 1.0);
}

#elif defined FRAGMENT_SHADER

out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

// material parameters
uniform vec3 albedo;
uniform float metallic;
uniform float roughness;
uniform float ao;

uniform float pbr_exposure;
uniform int debug_mode;

// IBL
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

// lights
//uniform vec3 lightPositions[4];
//uniform vec3 lightColors[4];
vec3 lightPositions[] = {
vec3(-10.0f, 10.0f, 10.0f),
vec3(10.0f, 10.0f, 10.0f),
vec3(-10.0f, -10.0f, 10.0f),
vec3(10.0f, -10.0f, 10.0f),
};
vec3 lightColors[] = {
vec3(300.0f, 300.0f, 300.0f),
vec3(300.0f, 300.0f, 300.0f),
vec3(300.0f, 300.0f, 300.0f),
vec3(300.0f, 300.0f, 300.0f)
};


uniform vec3 camPos;

const float PI = 3.14159265359;
// ----------------------------------------------------------------------------
float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float nom   = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
// ----------------------------------------------------------------------------
vec3 compute_reflectance(in vec3 lightPosition, in vec3 lightColor, in vec3 N, in vec3 V, in vec3 R, in vec3 F0)
{
    // calculate per-light radiance
    vec3 L = normalize(lightPosition - WorldPos);
    vec3 H = normalize(V + L);
    float distance      = length(lightPosition - WorldPos);
    float attenuation   = 1.0 / (distance * distance);
    vec3 radiance       = lightColor * attenuation;

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3 F    = fresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 numerator    = NDF * G * F;
    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;// + 0.0001 to prevent divide by zero
    vec3 specular = numerator / denominator;

    // kS is equal to Fresnel
    vec3 kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    vec3 kD = vec3(1.0) - kS;
    // multiply kD by the inverse metalness such that only non-metals
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0 - metallic;

    // scale light by NdotL
    float NdotL = max(dot(N, L), 0.0);

    // add to outgoing radiance Lo
    return (kD * albedo / PI + specular) * radiance * NdotL;// note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
}

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// ----------------------------------------------------------------------------
void main()
{
    vec3 N = Normal;
    vec3 V = normalize(camPos - WorldPos);
    vec3 R = reflect(-V, N);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // apply reflectance equation for each light
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        Lo += compute_reflectance(lightPositions[i], lightColors[i], N, V, R, F0);
    }

    // ambient lighting (we now use IBL as the ambient term)
    vec3 F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

    vec3 kS = F;
    vec3 kD = 1.0 - kS;
    kD *= 1.0 - metallic;

    vec3 irradiance     = texture(irradianceMap, N).rgb;
    vec3 diffuse        = irradiance * albedo;

    // https://docs.gl/sl4/textureQueryLevels
    // need OpenGL Shading Language Version >= 4.30
    const float MAX_REFLECTION_LOD = float(textureQueryLevels(prefilterMap));
    // or we can precompute and send it with uniform (or like here defined constant)
    //    const float MAX_REFLECTION_LOD = 9.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, roughness * MAX_REFLECTION_LOD).rgb;

    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    // BRDF LUT sampling with half-pixel offset for correct texture addressing
    vec2 brdfUV = vec2(max(dot(N, V), 0.0), roughness);
    vec2 textureSize = vec2(textureSize(brdfLUT, 0));
    brdfUV = brdfUV * (textureSize - 1.0) / textureSize + 0.5 / textureSize;
    vec2 brdf       = texture(brdfLUT, brdfUV).rg;

    // --- High Quality Multiple Scattering Approximation ---
    // Single Scattering term (current)
    vec3 FssEss = F * brdf.x + brdf.y;

    // Multiple Scattering term (Energy Compensation)
    // Approximate Average Fresnel (Favg) for the material
    // For Favg, we can use a simplified fit: F0 + (1-F0)/21
    vec3 Favg = F0 + (1.0 - F0) / 21.0;
    
    // Ess (Energy Single Scattering) approx from LUT (Scale + Bias)
    // Note: brdf.x/y contain the integrated G * F (without F0) terms, but roughly brdf.x + brdf.y is the directional albedo conservation
    float Ess = brdf.x + brdf.y; 

    // Multiple Scattering Factor (Fms)
    // Derivation from Kulla & Conty (Imageworks)
    vec3 Fms = Favg * FssEss / (1.0 - Favg * (1.0 - Ess));
    
    // Combined Specular with Energy Compensation
    // We add the multiple scattering contribution scaled by the energy loss (1 - Ess)
    // Actually Fms is the total term? No. 
    // The approximated formulae is: Specular = SingleScatt + MultiScatt * EnergyLoss
    // But let's use the simplest efficient form:
    // FssEss + (1.0 - Ess) * Fms
    
    vec3 multipleScattering = Fms * (1.0 - Ess);
    vec3 specular = prefilteredColor * (FssEss + multipleScattering);

    // Energy Conservation for Diffuse
    // The energy available for diffuse is what is left after Specular (Single + Multi)
    // kD = 1.0 - (FssEss + multipleScattering);
    kD = 1.0 - (FssEss + multipleScattering);
    kD *= 1.0 - metallic;

    vec3 ambient    = (kD * diffuse + specular) * ao;

    vec3 color = ambient + Lo;
//    color = kD * diffuse;

    color *= pbr_exposure;

    // HDR tonemapping
//    color = color / (color + vec3(1.0));
    color = ACESFilm(color);

    // gamma correct
    color = pow(color, vec3(1.0/2.2));

    // --- Debug: False Color Mode (Luminance Stops) ---
    if (debug_mode == 1) {
        // Recalculate luminance of the FINAL color (after tonemap/gamma? No, usually before or after exposure but before tonemap)
        // Filament doc says: "Visualizing brightness by color coding the stops"
        // Let's visualize the EXPOSED linear color (before ACES/Gamma) to see physical values
        
        vec3 linear_exposed = (ambient + Lo) * pbr_exposure;
        float luma = dot(linear_exposed, vec3(0.2126, 0.7152, 0.0722));
        
        // Middle gray is 0.18.
        // log2(luma / 0.18) gives stops relative to middle gray.
        float stops = log2(luma / 0.18);
        
        vec3 debugColor = vec3(0.0);
        
        if (stops < -4.0) debugColor = vec3(0.0, 0.0, 0.0);       // Very Black
        else if (stops < -3.0) debugColor = vec3(0.2, 0.0, 0.2);  // Purple (-4 to -3)
        else if (stops < -2.0) debugColor = vec3(0.0, 0.0, 0.5);  // Dark Blue (-3 to -2)
        else if (stops < -1.0) debugColor = vec3(0.0, 0.0, 1.0);  // Blue (-2 to -1)
        else if (stops < -0.1) debugColor = vec3(0.0, 0.5, 0.5);  // Cyan/Teal (-1 to 0)
        
        else if (stops < 0.1)  debugColor = vec3(0.0, 1.0, 0.0);  // GREEN = Middle Gray (+/- 0.1 stop) -- Filament uses Cyan for middle? Let's use Green for exact middle match visibility
        
        else if (stops < 1.0)  debugColor = vec3(0.5, 0.5, 0.0);  // Olive (0 to +1)
        else if (stops < 2.0)  debugColor = vec3(1.0, 1.0, 0.0);  // Yellow (+1 to +2)
        else if (stops < 3.0)  debugColor = vec3(1.0, 0.5, 0.0);  // Orange (+2 to +3)
        else if (stops < 4.0)  debugColor = vec3(1.0, 0.0, 0.0);  // Red (+3 to +4)
        else debugColor = vec3(1.0, 0.0, 1.0);                    // Magenta (> +4)
        
        // Filament standard (approx):
        // Cyan = Middle Gray. Blue = Darks. Green/Yellow/Red = Brights.
        // Let's tweak to match description "Cyan is middle gray":
        
        if (stops < -2.0) debugColor = vec3(0.0, 0.0, 1.0); // Blue
        else if (stops < -1.0) debugColor = vec3(0.0, 0.5, 1.0); // Light Blue
        else if (stops < 1.0)  debugColor = vec3(0.0, 1.0, 1.0); // Cyan (Middle +/- 1)
        else if (stops < 2.0)  debugColor = vec3(0.0, 1.0, 0.0); // Green
        else if (stops < 3.0)  debugColor = vec3(1.0, 1.0, 0.0); // Yellow
        else debugColor = vec3(1.0, 0.0, 0.0); // Red
        
        // Let's stick to the rainbow gradients often seen:
        // <-2: Black/Blue
        // -1: Blue
        // 0: Cyan (Middle Gray)
        // +1: Green
        // +2: Yellow
        // +3: Red 
        
        // Final implementation choice:
        if (stops < -2.5) debugColor = vec3(0.0, 0.0, 0.0); // < -2.5 EV
        else if (stops < -1.5) debugColor = vec3(0.0, 0.0, 1.0); // -2 EV (Blue)
        else if (stops < -0.5) debugColor = vec3(0.0, 0.5, 1.0); // -1 EV
        else if (stops < 0.5)  debugColor = vec3(0.0, 1.0, 1.0); // 0 EV (Cyan - Middle Gray)
        else if (stops < 1.5)  debugColor = vec3(0.0, 1.0, 0.0); // +1 EV (Green)
        else if (stops < 2.5)  debugColor = vec3(1.0, 1.0, 0.0); // +2 EV (Yellow)
        else debugColor = vec3(1.0, 0.0, 0.0);                   // +3+ EV (Red)

        FragColor = vec4(debugColor, 1.0);
        return; 
    }

    FragColor = vec4(color, 1.0);
}

#endif
