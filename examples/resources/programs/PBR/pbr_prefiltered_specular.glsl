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
uniform bool use_billboarding;

out vec2 LocalPos;
out float SphereRadius;
out vec3 CenterVS;

void main() {
    TexCoords = in_texcoord_0;
    
    if (use_billboarding) {
        vec3 center_world = vec3(model * vec4(0.0, 0.0, 0.0, 1.0));
        float radius = length(model[0].xyz); // Assume uniform scale
        
        vec3 center_vs = vec3(view * vec4(center_world, 1.0));
        // Increase quad size to 3.0x radius to avoid clipping spherical silhouette in perspective
        vec3 pos_vs = center_vs + vec3(in_position.xy * radius * 3.0, 0.0);
        
        LocalPos = in_position.xy * 3.0;
        SphereRadius = radius;
        CenterVS = center_vs;
        
        // Pass a dummy WorldPos/Normal for now, will be recalculated in FS
        WorldPos = vec3(inverse(view) * vec4(pos_vs, 1.0));
        Normal = vec3(0.0, 0.0, 1.0);
        
        gl_Position = projection * vec4(pos_vs, 1.0);
    } else {
        WorldPos = vec3(model * vec4(in_position, 1.0));
        Normal = normalize(normalMatrix * in_normal);
        gl_Position =  projection * view * vec4(WorldPos, 1.0);
    }
}

#elif defined FRAGMENT_SHADER

out vec4 FragColor;
in vec2 TexCoords;
in vec3 WorldPos;
in vec3 Normal;

in vec2 LocalPos;
in float SphereRadius;
in vec3 CenterVS;

uniform bool use_billboarding;
uniform int light_mode;  // 0=Point, 1=MRP Spherical, 2=LTC Rectangular
uniform float lightRadius;
uniform mat4 projection;
uniform mat4 view;

// LTC
uniform sampler2D ltc_mat;
uniform sampler2D ltc_amp;
uniform float time;

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
// LTC Helper Functions
// ----------------------------------------------------------------------------
float IntegrateEdge(vec3 v1, vec3 v2)
{
    float cosTheta = dot(v1, v2);
    cosTheta = clamp(cosTheta, -0.9999, 0.9999);
    float theta = acos(cosTheta);
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta / sin(theta) : 1.0);
    return res;
}

vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4])
{
    // Construct orthonormal basis around N
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);
    
    // Rotate area light in the tangent frame
    Minv = Minv * transpose(mat3(T1, T2, N));
    
    // Transform polygon vertices
    vec3 L[4];
    for (int i = 0; i < 4; i++)
    {
        L[i] = Minv * (points[i] - P);
        L[i] = normalize(L[i]);
    }
    
    // Integrate
    vec3 dir = points[0] - P;
    vec3 lightNormal = cross(points[1] - points[0], points[3] - points[0]);
    bool behind = (dot(dir, lightNormal) < 0.0);
    
    if(!behind)
    {
        float sum = 0.0;
        sum += IntegrateEdge(L[0], L[1]);
        sum += IntegrateEdge(L[1], L[2]);
        sum += IntegrateEdge(L[2], L[3]);
        sum += IntegrateEdge(L[3], L[0]);
        return vec3(max(0.0, sum));
    }
    
    return vec3(0.0);
}
// ----------------------------------------------------------------------------
vec3 compute_reflectance(in vec3 lightPosition, in vec3 lightColor, in vec3 N, in vec3 V, in vec3 R, in vec3 F0, in vec3 pos)
{
    if (light_mode == 2) {  // LTC Rectangular
        // Define rectangular light oriented towards the surface
        vec3 lightToSurf = normalize(pos - lightPosition);
        
        // Create an orthonormal basis for the rectangle
        vec3 lightRight = normalize(cross(lightToSurf, vec3(0.0, 1.0, 0.0)));
        if (length(lightRight) < 0.001) {
            lightRight = normalize(cross(lightToSurf, vec3(1.0, 0.0, 0.0)));
        }
        vec3 lightUp = normalize(cross(lightRight, lightToSurf));
        
        // Apply time-based rotation around the local Z axis (lightToSurf)
        float angle = time * 0.5;  // Slow rotation
        float cosA = cos(angle);
        float sinA = sin(angle);
        vec3 rotatedRight = lightRight * cosA + lightUp * sinA;
        vec3 rotatedUp = -lightRight * sinA + lightUp * cosA;
        
        // Define the 4 corners (square facing the surface, rotated)
        float hw = lightRadius;
        vec3 points[4];
        points[0] = lightPosition + (-rotatedRight - rotatedUp) * hw;
        points[1] = lightPosition + (rotatedRight - rotatedUp) * hw;
        points[2] = lightPosition + (rotatedRight + rotatedUp) * hw;
        points[3] = lightPosition + (-rotatedRight + rotatedUp) * hw;
        
        // Sample LTC LUTs
        float NdotV = clamp(dot(N, V), 0.0, 1.0);
        float theta = acos(NdotV);
        vec2 uv = vec2(roughness, theta / (0.5 * 3.14159265359));
        
        vec4 t = texture(ltc_mat, uv);
        mat3 Minv = mat3(
            vec3(t.x, 0, t.y),
            vec3(0, 1, 0),
            vec3(t.z, 0, t.w)
        );
        
        vec2 schlick = texture(ltc_amp, uv).xy;
        vec3 spec = LTC_Evaluate(N, V, pos, Minv, points);
        spec *= schlick.x;
        
        // Diffuse term uses identity matrix
        mat3 Mident = mat3(
            vec3(1, 0, 0),
            vec3(0, 1, 0),
            vec3(0, 0, 1)
        );
        vec3 diff = LTC_Evaluate(N, V, pos, Mident, points);
        
        // Normalize energy based on solid angle
        // LTC returns integrated irradiance already, we just need to scale by light intensity
        // The light area affects the result naturally through LTC integration
        float dist = length(lightPosition - pos);
        float atten = 1.0 / max(dist * dist, 1.0);
        
        // Combine spec and diffuse with proper energy balance
        vec3 result = lightColor * atten * (spec * F0 + diff * albedo * (1.0 - metallic) / PI);
        return result;
    }
    
    // Point or MRP modes
    vec3 L = normalize(lightPosition - pos);
    float distance      = length(lightPosition - pos);

    // If MRP (Spherical Area Lights), calculate the Most Representative Point on the light sphere
    if (light_mode == 1 && lightRadius > 0.0) {
        vec3 centerToRay = dot(lightPosition - pos, R) * R - (lightPosition - pos);
        vec3 closestPoint = (lightPosition - pos) + centerToRay * clamp(lightRadius / length(centerToRay), 0.0, 1.0);
        L = normalize(closestPoint);
        // Distance to the closest point for attenuation (approx)
        distance = length(closestPoint);
    }
    
    vec3 H = normalize(V + L);
    float attenuation   = 1.0 / (distance * distance);
    vec3 radiance       = lightColor * attenuation;
    
    // Clamp roughness for analytical lights to ensure highlights are visible 
    // and numerically stable even for mirror-like materials.
    float clampedRoughness = max(roughness, 0.02);
    
    // Adjust roughness for area light size to maintain energy conservation 
    // and avoid highlights smaller than the light itself.
    if (light_mode == 1 && lightRadius > 0.0) {
        float dist = length(lightPosition - pos);
        clampedRoughness = max(clampedRoughness, lightRadius / (2.0 * dist));
    }

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, clampedRoughness);
    float G   = GeometrySmith(N, V, L, clampedRoughness);
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
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

const vec3 debugColors[16] = vec3[](
     vec3(0.0, 0.0, 0.0),         // black
     vec3(0.0, 0.0, 0.1647),      // darkest blue
     vec3(0.0, 0.0, 0.3647),      // darker blue
     vec3(0.0, 0.0, 0.6647),      // dark blue
     vec3(0.0, 0.0, 0.9647),      // blue
     vec3(0.0, 0.9255, 0.9255),   // cyan
     vec3(0.0, 0.5647, 0.0),      // dark green
     vec3(0.0, 0.7843, 0.0),      // green
     vec3(1.0, 1.0, 0.0),         // yellow
     vec3(0.90588, 0.75294, 0.0), // yellow-orange
     vec3(1.0, 0.5647, 0.0),      // orange
     vec3(1.0, 0.0, 0.0),         // bright red
     vec3(0.8392, 0.0, 0.0),      // red
     vec3(1.0, 0.0, 1.0),         // magenta
     vec3(0.6, 0.3333, 0.7882),   // purple
     vec3(1.0, 1.0, 1.0)          // white
);

vec3 Tonemap_DisplayRange(const vec3 x) {
    // The 5th color in the array (cyan) represents middle gray (18%)
    // Every stop above or below middle gray causes a color shift
    float v = log2(luminance(x) / 0.18);
    v = clamp(v + 5.0, 0.0, 15.0);
    int index = int(floor(v));
    return mix(debugColors[index], debugColors[min(15, index + 1)], fract(v));
}

// ----------------------------------------------------------------------------
void raytrace_sphere(out vec3 N, out vec3 V, out vec3 fragWorldPos)
{
    // Ray-Sphere intersection in View Space
    // Ray origin at camera (0,0,0)
    vec3 O = vec3(0.0);
    // Ray direction through current fragment on the billboard quad
    vec3 P = CenterVS + vec3(LocalPos.x, LocalPos.y, 0.0) * SphereRadius;
    vec3 D = normalize(P - O);
    
    // Sphere center and radius
    vec3 C = CenterVS;
    float R = SphereRadius;
    
    // Quadratic: t^2 - 2t(D.C) + C.C - R^2 = 0
    float b = -2.0 * dot(D, C);
    float c = dot(C, C) - R * R;
    float delta = b * b - 4.0 * c;
    
    if (delta < 0.0) discard;
    
    float t = (-b - sqrt(delta)) / 2.0;
    if (t < 0.0) discard;
    
    // Hit point in View Space
    vec3 hit_vs = O + t * D;
    
    // Hit normal in View Space
    vec3 normal_vs = normalize(hit_vs - C);
    
    // Normal in World Space
    mat4 invView = inverse(view);
    N = normalize(mat3(invView) * normal_vs);
    
    // World Position
    fragWorldPos = (invView * vec4(hit_vs, 1.0)).xyz;
    
    // View direction
    V = normalize(camPos - fragWorldPos);
    
    // Update Depth
    vec4 clip_pos = projection * vec4(hit_vs, 1.0);
    gl_FragDepth = (clip_pos.z / clip_pos.w) * 0.5 + 0.5;
}

// ----------------------------------------------------------------------------
void main()
{
    vec3 N;
    vec3 V;
    vec3 fragWorldPos;

    if (use_billboarding) {
        raytrace_sphere(N, V, fragWorldPos);
    } else {
        N = normalize(Normal);
        fragWorldPos = WorldPos;
        V = normalize(camPos - fragWorldPos);
        // Standard depth is automatically written
        gl_FragDepth = gl_FragCoord.z;
    }

    vec3 R = reflect(-V, N);

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // apply reflectance equation for each light
    vec3 Lo = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        Lo += compute_reflectance(lightPositions[i], lightColors[i], N, V, R, F0, fragWorldPos);
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
    const float MAX_REFLECTION_LOD = float(textureQueryLevels(prefilterMap)) - 1.0;
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

    // Exposure
    color *= pbr_exposure;

    // HDR tonemapping
    color = ACESFilm(color);

    // gamma correct
    color = pow(color, vec3(1.0/2.2));

    // --- Debug: False Color Mode (Luminance Stops) ---
    if (debug_mode == 1) {
        vec3 linear_exposed = (ambient + Lo) * pbr_exposure;
        FragColor = vec4(Tonemap_DisplayRange(linear_exposed), 1.0);
        return; 
    }

    FragColor = vec4(color, 1.0);
}

#endif
