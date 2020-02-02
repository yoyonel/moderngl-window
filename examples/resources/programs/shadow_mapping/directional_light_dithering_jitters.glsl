#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec3 in_normal;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform mat4 m_shadow_bias;
uniform vec3 lightDir;

out vec3 light_dir;
out vec3 normal;
out vec4 ShadowCoord;
out vec3 v_vert;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    v_vert = p.xyz;
    gl_Position =  m_proj * p;
    mat3 m_normal = inverse(transpose(mat3(m_view)));
    normal = m_normal * normalize(in_normal);
    light_dir = (m_view * vec4(lightDir, 0.0)).xyz;
    ShadowCoord = m_shadow_bias * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

uniform vec4 color;
uniform sampler2D shadowMap;

in vec3 light_dir;
in vec3 normal;
in vec4 ShadowCoord;
in vec3 v_vert;

// https://github.com/opengl-tutorials/ogl/blob/master/tutorial16_shadowmaps/ShadowMapping.fragmentshader
const int nb_poissonDisk_samples = 16;
vec2 poissonDisk[nb_poissonDisk_samples ] = vec2[](
    vec2( -0.94201624, -0.39906216 ),
    vec2( 0.94558609, -0.76890725 ),
    vec2( -0.094184101, -0.92938870 ),
    vec2( 0.34495938, 0.29387760 ),
    vec2( -0.91588581, 0.45771432 ),
    vec2( -0.81544232, -0.87912464 ),
    vec2( -0.38277543, 0.27676845 ),
    vec2( 0.97484398, 0.75648379 ),
    vec2( 0.44323325, -0.97511554 ),
    vec2( 0.53742981, -0.47373420 ),
    vec2( -0.26496911, -0.41893023 ),
    vec2( 0.79197514, 0.19090188 ),
    vec2( -0.24188840, 0.99706507 ),
    vec2( -0.81409955, 0.91437590 ),
    vec2( 0.19984126, 0.78641367 ),
    vec2( 0.14383161, -0.14100790 )
);

// Returns a random number based on a vec3 and an int.
float random(vec3 seed, int i){
    vec4 seed4 = vec4(seed,i);
    float dot_product = dot(seed4, vec4(12.9898,78.233,45.164,94.673));
    return fract(sin(dot_product) * 43758.5453);
}

float map(float value, float min1, float max1, float min2, float max2) {
    return min2 + (value - min1) * (max2 - min2) / (max1 - min1);
}

float compute_shadow_with_poisson_filtering(in float cosTheta) {
    float bias = 0.005*tan(acos(cosTheta));
    bias = clamp(bias, 0, 0.015);

    float visibility = 1.0;

    const int nb_samples = 4;

    // Sample the shadow map vnb_samples` times
    for (int i=0;i<nb_samples;i++){
        // use either :
        //  - Always the same samples.
        //    Gives a fixed pattern in the shadow, but no noise
        // int index = i;
        //  - A random sample, based on the pixel's screen location.
        //    No banding, but the shadow moves with the camera, which looks weird.
        // int index = int(16.0*random(gl_FragCoord.xyy, i))%16;
        //  - A random sample, based on the pixel's position in world space.
        //    The position is rounded to the millimeter to avoid too much aliasing
        int index = int(float(nb_poissonDisk_samples)*random(floor(v_vert.xyz*1000.0), i)) % nb_poissonDisk_samples ;

        vec2 ShadowCoord_LightView = ShadowCoord.xy/ShadowCoord.w;
        ShadowCoord_LightView += poissonDisk[index]/800.0;
        float z_from_light = texture(shadowMap, ShadowCoord_LightView).r;
        float z_from_cam = (ShadowCoord.z - bias) / ShadowCoord.w;
        float is_texel_shadowed = (z_from_cam >=  z_from_light ? 1.0: 0.0);
        // retrieve the lighting/shadowing contribution of this texel
        visibility -= 1.0/nb_samples * is_texel_shadowed;
    }
    return visibility;
}

void main() {
    float cosTheta = dot(normalize(light_dir), normalize(normal));
    float visibility = compute_shadow_with_poisson_filtering(cosTheta);
    visibility = map(visibility, 0, 1, 0.5, 1.0);
    float l = max(cosTheta, 0.0);
    fragColor = color * (0.25 + abs(l) * 0.9) * visibility;
}
#endif
