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
        float radius = length(model[0].xyz); // uniform scale
        vec3 center_vs = vec3(view * vec4(center_world, 1.0));

        const float QUAD_SCALE = 2.0;
        vec3 pos_vs = center_vs + vec3(in_position.xy * radius * QUAD_SCALE, 0.0);

        LocalPos     = in_position.xy * QUAD_SCALE;
        SphereRadius = radius;
        CenterVS     = center_vs;

        gl_Position = projection * vec4(pos_vs, 1.0);
    } else {
        vec4 worldPos4 = model * vec4(in_position, 1.0);
        WorldPos = worldPos4.xyz;
        Normal = normalMatrix * in_normal; // Normalisation différée au fragment shader
        gl_Position = projection * view * worldPos4;
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
uniform mat4 projection;
uniform mat4 view;
uniform mat4 invView;

// LTC
uniform sampler2D ltc_mat;
uniform sampler2D ltc_amp;
uniform float time;

struct Light {
    float radius;
    float intensity;
};
uniform Light light;

struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
    float ao;
};
uniform Material material;

uniform float pbr_exposure;
uniform int debug_mode;
uniform bool use_analytic_aa;
uniform bool use_comparison_mode; // Split screen comparison
uniform float comparison_split;   // 0.0 to 1.0
uniform vec2 window_size;
uniform float aa_width_multiplier;
uniform float aa_distance_factor;

// IBL
uniform samplerCube irradianceMap;
uniform samplerCube prefilterMap;
uniform sampler2D brdfLUT;

// lights
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
const float INV_PI = 0.31830988618; // 1/PI précalculé
const float MIN_ROUGHNESS = 0.02;   // Roughness minimum pour stabilité
const float EPSILON = 1e-6;         // Pour éviter divisions par zéro

// ----------------------------------------------------------------------------
// Distribution function for GGX (Normal Distribution Function)
// Optimisé: paramètre a2 au lieu de roughness pour éviter double calcul
float DistributionGGX(float NdotH, float a2)
{
    float NdotH2 = NdotH * NdotH;
    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return nom / max(denom, EPSILON);
}

// ----------------------------------------------------------------------------
// Geometry function for Schlick-GGX
// Optimisé: réduction d'une multiplication
float GeometrySchlickGGX(float NdotX, float k)
{
    float denom = NdotX * (1.0 - k) + k;
    return NdotX / max(denom, EPSILON);
}

// ----------------------------------------------------------------------------
// Geometry Smith function combining both view and light
// Optimisé: k calculé une seule fois, passage direct de NdotV/NdotL
float GeometrySmith(float NdotV, float NdotL, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) * 0.125; // /8.0 optimisé
    float ggx1 = GeometrySchlickGGX(NdotL, k);
    float ggx2 = GeometrySchlickGGX(NdotV, k);
    return ggx1 * ggx2;
}

// ----------------------------------------------------------------------------
// Fresnel Schlick approximation
// Optimisé: utilisation de pow5 qui peut être optimisé par le compilateur
vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
    float f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (1.0 - F0) * f;
}

// ----------------------------------------------------------------------------
// Fresnel Schlick with roughness
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness)
{
    float f = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (max(vec3(1.0 - roughness), F0) - F0) * f;
}

// ----------------------------------------------------------------------------
// LTC Helper: Edge integration
float IntegrateEdge(vec3 v1, vec3 v2)
{
    float cosTheta = dot(v1, v2);
    cosTheta = clamp(cosTheta, -0.9999, 0.9999);
    float theta = acos(cosTheta);
    float res = cross(v1, v2).z * ((theta > 0.001) ? theta / sin(theta) : 1.0);
    return res;
}

// ----------------------------------------------------------------------------
// LTC evaluation for rectangular area light
vec3 LTC_Evaluate(vec3 N, vec3 V, vec3 P, mat3 Minv, vec3 points[4])
{
    vec3 T1 = normalize(V - N * dot(V, N));
    vec3 T2 = cross(N, T1);

    Minv = Minv * transpose(mat3(T1, T2, N));

    vec3 L[4];
    L[0] = normalize(Minv * (points[0] - P));
    L[1] = normalize(Minv * (points[1] - P));
    L[2] = normalize(Minv * (points[2] - P));
    L[3] = normalize(Minv * (points[3] - P));

    vec3 dir = points[0] - P;
    vec3 lightNormal = cross(points[1] - points[0], points[3] - points[0]);
    bool behind = (dot(dir, lightNormal) < 0.0);

    if (!behind) {
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
vec3 compute_reflectance(
    in vec3 lightPosition,
    in vec3 lightColor,
    in vec3 N,
    in vec3 V,
    in vec3 R,
    in vec3 F0,
    in vec3 pos,
    in float NdotV  // Optimisation: passé en paramètre pour éviter recalcul
)
{
    // -------------------------------------------------------
    // LTC Rectangular Area Light (light_mode == 2)
    // -------------------------------------------------------
    if (light_mode == 2)
    {
        vec3 lightToSurf = normalize(pos - lightPosition);
        vec3 lightRight = normalize(cross(lightToSurf, vec3(0.0, 1.0, 0.0)));
        if (length(lightRight) < 0.001) {
            lightRight = normalize(cross(lightToSurf, vec3(1.0, 0.0, 0.0)));
        }
        vec3 lightUp = normalize(cross(lightRight, lightToSurf));

        float cosX = cos(time * 0.0);
        float sinX = sin(time * 0.0);
        float cosY = cos(time * 0.0);
        float sinY = sin(time * 0.0);
        float cosZ = cos(time * 0.4);
        float sinZ = sin(time * 0.4);

        vec3 tempUp = lightUp * cosX + lightToSurf * sinX;
        vec3 tempForward = -lightUp * sinX + lightToSurf * cosX;
        vec3 rotatedRight = lightRight * cosY - tempForward * sinY;
        vec3 finalForward = lightRight * sinY + tempForward * cosY;
        vec3 finalRight = rotatedRight * cosZ + tempUp * sinZ;
        vec3 finalUp    = -rotatedRight * sinZ + tempUp * cosZ;

        float hw = light.radius;
        vec3 points[4];
        points[0] = lightPosition + (-finalRight - finalUp) * hw;
        points[1] = lightPosition + ( finalRight - finalUp) * hw;
        points[2] = lightPosition + ( finalRight + finalUp) * hw;
        points[3] = lightPosition + (-finalRight + finalUp) * hw;

        float theta = acos(NdotV);
        vec2 uv = vec2(material.roughness, theta * (2.0 * INV_PI)); // Optimisé: 2/PI

        vec4 t = texture(ltc_mat, uv);
        mat3 Minv = mat3(
            vec3(t.x, 0, t.y),
            vec3(  0, 1,   0),
            vec3(t.z, 0, t.w)
        );

        vec2 schlick = texture(ltc_amp, uv).xy;

        vec3 spec = LTC_Evaluate(N, V, pos, Minv, points) * schlick.x;
        vec3 diff = LTC_Evaluate(N, V, pos, mat3(1.0), points);

        // Intensité physique: lumens / (4π × aire)
        // L'aire de la source rectangulaire est (2×hw)² = 4×hw²
        float lightArea = 4.0 * hw * hw;
        float luminousIntensity = light.intensity / (4.0 * PI * lightArea);

        // Distance attenuation (optionnel pour area lights, mais utile pour cohérence)
        float dist = length(lightPosition - pos);
        float distAtten = 1.0 / max(dist * dist, 1.0);

        return lightColor * luminousIntensity * distAtten *
               (spec * F0 + diff * material.albedo * (1.0 - material.metallic) * INV_PI);
    }

    // -------------------------------------------------------
    // Point Light & MRP Spherical (light_mode 0 ou 1)
    // -------------------------------------------------------
    vec3 Lvec = lightPosition - pos;
    float dist2 = dot(Lvec, Lvec);
    float invDist = inversesqrt(dist2); // Optimisation: évite normalize explicite
    vec3 L = Lvec * invDist;

    float effectiveRoughness = material.roughness;

    if (light_mode == 1 && light.radius > 0.0)
    {
        // Most Representative Point (MRP) pour lumières sphériques
        vec3 proj = dot(Lvec, R) * R;
        vec3 centerToRay = proj - Lvec;
        float len2 = dot(centerToRay, centerToRay);
        float scale = clamp(light.radius * light.radius / max(len2, EPSILON), 0.0, 1.0);
        vec3 LvecMRP = Lvec + centerToRay * scale;

        L = normalize(LvecMRP);
        dist2 = dot(LvecMRP, LvecMRP);

        // Ajustement de roughness basé sur la taille angulaire de la lumière
        effectiveRoughness = max(material.roughness, light.radius * inversesqrt(dist2) * 0.5);
    }

    // Atténuation inverse carré
    vec3 radiance = lightColor / max(dist2, 1.0);

    // Clamp roughness pour stabilité numérique
    effectiveRoughness = max(effectiveRoughness, MIN_ROUGHNESS);
    float a = effectiveRoughness * effectiveRoughness;
    float a2 = a * a; // Précalcul pour DistributionGGX

    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float HdotV = max(dot(H, V), 0.0);

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(NdotH, a2);
    float G = GeometrySmith(NdotV, NdotL, effectiveRoughness);
    vec3 F = fresnelSchlick(HdotV, F0);

    // Specular term
    vec3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL, EPSILON);

    // Diffuse term (energy conservation)
    vec3 kD = (1.0 - F) * (1.0 - material.metallic);

    return (kD * material.albedo * INV_PI + specular) * radiance * NdotL;
}

// ----------------------------------------------------------------------------
// compute_IBL_PBR helper (Image-Based Lighting with multiple scattering)
vec3 compute_IBL_PBR(vec3 N, vec3 V, vec3 R, vec3 F0, float NdotV)
{
    // Fresnel pour IBL
    vec3 F = fresnelSchlickRoughness(NdotV, F0, material.roughness);
    vec3 kS = F;
    vec3 kD = (1.0 - kS) * (1.0 - material.metallic);

    // Diffuse IBL
    vec3 irradiance = texture(irradianceMap, N).rgb;
    vec3 diffuse = irradiance * material.albedo;

    // Specular IBL avec prefiltered environment map
    const float MAX_REFLECTION_LOD = float(textureQueryLevels(prefilterMap)) - 1.0;
    vec3 prefilteredColor = textureLod(prefilterMap, R, material.roughness * MAX_REFLECTION_LOD).rgb;

    // BRDF LUT lookup avec correction de coordonnées pour éviter artifacts aux bords
    vec2 brdfUV = vec2(NdotV, material.roughness);
    vec2 texSize = vec2(textureSize(brdfLUT, 0));
    brdfUV = brdfUV * (texSize - 1.0) / texSize + 0.5 / texSize;
    vec2 brdf = texture(brdfLUT, brdfUV).rg;

    // Split-sum approximation avec compensation de multiple scattering
    vec3 FssEss = F * brdf.x + brdf.y;

    // Average Fresnel pour multiple scattering
    vec3 Favg = F0 + (1.0 - F0) * (1.0 / 21.0);
    float Ess = brdf.x + brdf.y;
    vec3 Fms = Favg * FssEss / max(1.0 - Favg * (1.0 - Ess), EPSILON);
    vec3 multipleScattering = Fms * (1.0 - Ess);

    vec3 specular = prefilteredColor * (FssEss + multipleScattering);

    // Energy conservation finale
    kD = (1.0 - (FssEss + multipleScattering)) * (1.0 - material.metallic);

    vec3 ambient = (kD * diffuse + specular) * material.ao;
    return ambient;
}

// ----------------------------------------------------------------------------
// Raytrace sphere pour billboarding
// Modifié pour supporter l'AA analytique (retourne 'true' pour les frôlements)
bool raytrace_sphere(out vec3 N, out vec3 V, out vec3 fragWorldPos) {
    vec3 O = vec3(0.0);
    vec3 P = CenterVS + vec3(LocalPos, 0.0) * SphereRadius;
    vec3 D = normalize(P - O);

    vec3 C = CenterVS;
    float R = SphereRadius;

    // Ray-sphere intersection: ||O + tD - C||² = R²
    float b = -2.0 * dot(D, C);
    float c = dot(C, C) - R * R;
    float delta = b * b - 4.0 * c;

    // Calcul de la distance d'approche pour l'AA
    float distToCenterSq = dot(cross(D, CenterVS), cross(D, CenterVS));
    bool is_near_miss = (distToCenterSq < R * R * 1.05);// Cache un peu plus large pour l'AA

    if (delta < 0.0 && !is_near_miss) return false;

    float t;
    if (delta >= 0.0) {
        t = (-b - sqrt(delta)) * 0.5;
    } else {
        // Near-miss: projeter sur le plan de la silhouette
        t = -dot(O - C, D);
    }

    if (t < 0.0) return false;

    vec3 hit_vs = O + t * D;
    vec3 normal_vs = normalize(hit_vs - C);

    N = normalize(mat3(invView) * normal_vs);
    fragWorldPos = (invView * vec4(hit_vs, 1.0)).xyz;
    V = normalize(camPos - fragWorldPos);

    vec4 clip_pos = projection * vec4(hit_vs, 1.0);
    gl_FragDepth = clip_pos.z / clip_pos.w * 0.5 + 0.5;

    return true;
}

// ----------------------------------------------------------------------------
// ACES Filmic tonemapping
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
// Luminance helper
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Debug colors for false-color luminance display
const vec3 debugColors[16] = vec3[](
    vec3(0.0, 0.0, 0.0),
    vec3(0.0, 0.0, 0.1647),
    vec3(0.0, 0.0, 0.3647),
    vec3(0.0, 0.0, 0.6647),
    vec3(0.0, 0.0, 0.9647),
    vec3(0.0, 0.9255, 0.9255),
    vec3(0.0, 0.5647, 0.0),
    vec3(0.0, 0.7843, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.90588, 0.75294, 0.0),
    vec3(1.0, 0.5647, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(0.8392, 0.0, 0.0),
    vec3(1.0, 0.0, 1.0),
    vec3(0.6, 0.3333, 0.7882),
    vec3(1.0, 1.0, 1.0)
);

vec3 Tonemap_DisplayRange(const vec3 x)
{
    float v = log2(luminance(x) / 0.18);
    v = clamp(v + 5.0, 0.0, 15.0);
    int index = int(floor(v));
    return mix(debugColors[index], debugColors[min(15, index + 1)], fract(v));
}

// ----------------------------------------------------------------------------
void main()
{
    vec3 N;
    vec3 V;
    vec3 fragWorldPos;

    // Alpha coverage for analytic AA (default opaque)
    float coverage = 1.0;

    // ------------------------------------------------------------------------
    // Geometry evaluation
    // ------------------------------------------------------------------------
    if (use_billboarding) {
        // Analytic ray–sphere intersection
        if (!raytrace_sphere(N, V, fragWorldPos)) {
            discard;
        }

        // --------------------------------------------------------------------
        // Analytic edge anti-aliasing (stable version)
        //
        // LocalPos is in [-2, 2] because QUAD_SCALE = 2.0
        // The projected sphere radius in this space is exactly 2.0
        // --------------------------------------------------------------------
        bool enable_aa = use_analytic_aa;

        // Comparison mode logic: Left side has AA, Right side doesn't (if enabled)
        if (use_comparison_mode) {
            float screen_x = gl_FragCoord.x / window_size.x;
            enable_aa = (screen_x < comparison_split);
        }

        if (enable_aa) {
            const float radius = 1.0;

            // Signed distance to sphere edge in billboard plane
            float edgeDist = radius - length(LocalPos);

            // Screen-space pixel footprint of the implicit edge
            // Réglable via aa_width_multiplier et aa_distance_factor
            float dist = length(CenterVS);
            float edgeWidth = fwidth(edgeDist) * (aa_width_multiplier + dist * aa_distance_factor);

            // Prevent sub-pixel instability at far distance
            edgeWidth = max(edgeWidth, 1.0 / 1024.0);

            // Smooth analytic coverage
            coverage = smoothstep(-edgeWidth, edgeWidth, edgeDist);

            // Hard reject only pixels clearly outside the transition zone
            if (edgeDist < -edgeWidth) {
                discard;
            }
        } else {
            // No AA: Hard edge at radius 1.0 (LocalPos scaling)
            if (length(LocalPos) > 1.0) {
                discard;
            }
            coverage = 1.0;
        }
    } else {
        // Standard mesh path
        N = normalize(Normal);
        fragWorldPos = WorldPos;
        V = normalize(camPos - fragWorldPos);
        gl_FragDepth = gl_FragCoord.z;
    }

    // ------------------------------------------------------------------------
    // Common PBR inputs
    // ------------------------------------------------------------------------
    vec3 R = reflect(-V, N);
    vec3 F0 = mix(vec3(0.04), material.albedo, material.metallic);
    float NdotV = max(dot(N, V), 0.0);

    // ------------------------------------------------------------------------
    // Direct lighting
    // ------------------------------------------------------------------------
    vec3 Lo = vec3(0.0);

    if (light_mode != 3) {
        for (int i = 0; i < 4; ++i) {
            Lo += compute_reflectance(
                lightPositions[i],
                lightColors[i],
                N,
                V,
                R,
                F0,
                fragWorldPos,
                NdotV
            );
        }
    }

    // ------------------------------------------------------------------------
    // Image-Based Lighting (IBL)
    // ------------------------------------------------------------------------
    vec3 ambient = compute_IBL_PBR(N, V, R, F0, NdotV);

    // ------------------------------------------------------------------------
    // Final shading
    // ------------------------------------------------------------------------
    vec3 color = ambient + Lo;
    color *= pbr_exposure;

    // ------------------------------------------------------------------------
    // Debug visualization (false-color luminance)
    // ------------------------------------------------------------------------
    if (debug_mode == 1) {
        FragColor = vec4(Tonemap_DisplayRange(color), coverage);
        return;
    }

    // ------------------------------------------------------------------------
    // Tonemapping & gamma correction
    // ------------------------------------------------------------------------
    color = ACESFilm(color);
    color = pow(color, vec3(1.0 / 2.2));

    FragColor = vec4(color, coverage);
}

#endif