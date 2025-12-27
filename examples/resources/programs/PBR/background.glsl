#version 450 core

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_camera;
uniform mat4 m_proj;

out vec3 WorldPos;

void main() {
    gl_Position =  m_proj * m_camera * vec4(in_position, 1.0);;
    WorldPos = in_position.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 FragColor;
in vec3 WorldPos;

uniform samplerCube environmentMap;
// for visualize LUT BRDF texture
//uniform sampler2D environmentMap;

uniform float blur_lod;

void main()
{
    vec3 envColor = textureLod(environmentMap, normalize(WorldPos), blur_lod).rgb;
    // for visualize LUT BRDF texture
//    vec3 envColor = texture(environmentMap, normalize(WorldPos).xy).rgb;

    // HDR tonemap and gamma correct
    envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0/2.2));

    FragColor = vec4(envColor, 1.0);
}
#endif
