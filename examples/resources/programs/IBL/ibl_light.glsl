#version 330 core

#if defined VERTEX_SHADER

const float Epsilon = 0.00001;

in vec3 in_position;
in vec3 in_normal;
in vec2 in_texcoord_0;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

out vec2 textureCoordinates;
out vec3 worldCoordinates;
out vec3 normal;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);

    worldCoordinates = (m_model * vec4(in_position, 1)).xyz;
    textureCoordinates = in_texcoord_0;

    gl_Position =  m_proj * p;

    mat3 m_normal = transpose(inverse(mat3(m_view)));
    normal = normalize(m_normal * in_normal);
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

uniform vec3 cameraPosition;
// lights
uniform vec3 lightPosition;
uniform vec3 lightColor;

in vec3 normal;
in vec3 worldCoordinates;
in vec2 textureCoordinates;

#define PI 3.1415926535897932384626433832795

// PBR
// IBL precomputed maps
const uint PREFILTERED_ENV_MAP_LOD = 9u; // how many mipmap levels

uniform samplerCube prefilteredEnvMap;
uniform samplerCube diffuseIrradianceMap;
uniform sampler2D brdfConvolutionMap;
// Material
uniform sampler2D textureAlbedo;
uniform sampler2D textureAmbientOcclusion;
uniform sampler2D textureRoughness;
uniform sampler2D textureMetallic;
uniform sampler2D textureNormal;

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

// Fresnel function (Fresnel-Schlick approximation)
//
// F_schlick = f0 + (1 - f0)(1 - (h * v))^5
//
vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(max(1 - cosTheta, 0.0), 5.0);
}

// Fresnel schlick roughness
//
// Same as above except with a roughness term
vec3 fresnelSchlickRoughness(float cosTheta, vec3 f0, float roughness)
{
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// smiths method for taking into account view direction and light direction
float geometrySmith(vec3 n, vec3 v, vec3 l, float roughness) {

    // remapping for direct lighting (doesn't work for IBL)
    float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;

    return geometrySchlickGGX(n, v, k) * geometrySchlickGGX(n, l, k);
}

// Normal distribution function (Trowbridge-Reitz GGX)
//
//                alpha ^ 2
//     ---------------------------------
//      PI((n * h)^2(alpha^2 - 1) + 1)^2
//
float ndfTrowbridgeReitzGGX(vec3 n, vec3 h, float roughness) {

    float alpha = roughness * roughness; // recommended by disney/epic papers
    float alphaSquared = alpha * alpha;

    float nDotH = max(dot(n, h), 0.0);
    float nDotHSquared = nDotH * nDotH;
    float innerTerms = nDotHSquared * (alphaSquared - 1.0) + 1.0;

    float numerator = alphaSquared;
    float denomenator = PI * innerTerms * innerTerms;
    denomenator = max(denomenator, 0.0001); // avoid div by zero

    return numerator / denomenator;
}

//// Tangent space to world
//vec3 calculateNormal(vec3 tangentNormal) {
//    vec3 norm = normalize(tangentNormal * 2.0 - 1.0);
//    mat3 TBN  = mat3(tangent, bitangent, normal);
//    return normalize(TBN * norm); // tangent --> world
//}

void main() {
    // retrieve all the material properties

    // albedo
    vec3 albedo = vec3(92.9, 69.8, 46.8) / 255.0;
    albedo = texture(textureAlbedo, textureCoordinates).rgb;

    // metallic/roughness
    float metallic = 0.0000125;
    metallic = texture(textureMetallic, textureCoordinates).r;

    float roughness = 0.3242425;
    roughness = texture(textureRoughness, textureCoordinates).r;

    // normal
    vec3 n = normal; // interpolated vertex normal
//    n = calculateNormal(texture(textureNormal, textureCoordinates).rgb);

    // ambient occlusion
    float ao = 1.0;
    ao = texture(textureAmbientOcclusion, textureCoordinates).r;

    // emissive
    vec3 emissive = (vec3(243.0, 242.0, 240.0) / 255.0) * 0.0071250;

    vec3 v = normalize(cameraPosition - worldCoordinates); // view vector pointing at camera
    vec3 r = reflect(-v, n); // reflection
//    vec3 r = reflect(v, n); // reflection

    // f0 is the "surface reflection at zero incidence"
    // for PBR-metallic we assume dialectrics all have 0.04
    // for metals the value comes from the albedo map
    vec3 f0 = vec3(0.04);
    f0 = mix(f0, albedo, metallic);

    vec3 Lo = vec3(0.0); // total radiance out

    vec3 l = normalize(lightPosition - worldCoordinates); // light vector
    vec3 h = normalize(v + l);

    float distance = length(lightPosition - worldCoordinates);
    float attenuation = 1.0 / (distance * distance); // inverse square law
    // ❗️probleme d'echelle sur l'attenuation lumineuse
    attenuation = 1.0;
    vec3 radiance = lightColor * attenuation; // aka Li

    {
        // calculate Cook-Torrance specular BRDF term
        //
        //                DFG
        //        --------------------
        //         4(w_0 * n)(w_i * n)
        //
        //

        // Normal Distribution term (D)
        float dTerm = ndfTrowbridgeReitzGGX(n, h, roughness);

        // Fresnel term (F)
        // Determines the ratio of light reflected vs. absorbed
        vec3 fTerm = fresnelSchlick(max(dot(h, v), 0.0), f0);

        // Geometry term (G)
        float gTerm = geometrySmith(n, v, l, roughness);

        vec3 numerator = dTerm * fTerm * gTerm;
        float denominator = 4.0 * max(dot(v, n), 0.0) * max(dot(l, n), 0.0);

        // recall fTerm is the proportion of reflected light, so the result here is the specular
        vec3 specular = numerator / max(denominator, 0.001);

        vec3 kSpecular = fTerm;
        vec3 kDiffuse = vec3(1.0) - kSpecular;
        kDiffuse *= 1.0 - metallic; // metallic materials should have no diffuse component

        // now calculate full Cook-Torrance with both diffuse + specular
        //
        // f_r = kd * f_lambert + ks * f_cook-torrance
        //
        // where f_lambert = c / pi

        vec3 diffuse = kDiffuse * albedo / PI;
        vec3 cookTorranceBrdf = diffuse + specular;
        float nDotL = max(dot(n, l), 0.0);

        // Finally, the rendering equation!
        Lo += cookTorranceBrdf * radiance * nDotL;
    }

    // Indirect lighting (IBL)
    vec3 kSpecular = fresnelSchlickRoughness(max(dot(n, v), 0.0), f0, roughness); // aka F
    vec3 kDiffuse = 1.0 - kSpecular;
    kDiffuse *= 1.0 - metallic; // metallic materials should have no diffuse component

    // diffuse
    vec3 irradiance = texture(diffuseIrradianceMap, n).rgb;
    vec3 diffuse = irradiance * albedo;

    // specular
    vec3 prefilteredEnvMapColor = textureLod(prefilteredEnvMap, r, roughness * PREFILTERED_ENV_MAP_LOD).rgb;
    float NdotV = max(dot(n, v), 0.0);
    vec2 brdf = texture(brdfConvolutionMap, vec2(NdotV, roughness)).xy;
    // ❗️pb avec cette texture LUT !
    brdf = vec2(1.0, 0.0);
    vec3 specular = prefilteredEnvMapColor * (kSpecular * brdf.x + brdf.y);

    vec3 ambient = (kDiffuse * diffuse + specular) * ao; // indirect lighting
//    ambient = specular;

    // Combine emissive + indirect + direct
    vec3 color = emissive + ambient + Lo;

    // DEBUG
//    color = Lo;

    // main color output
    fragColor = vec4(color, 1.0);
}
#endif
