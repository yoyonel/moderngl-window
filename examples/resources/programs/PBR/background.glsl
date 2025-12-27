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

uniform float blur_lod;

vec3 ACESFilm(vec3 x)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 envColor = textureLod(environmentMap, normalize(WorldPos), blur_lod).rgb;

    // HDR tonemapping
    // envColor = envColor / (envColor + vec3(1.0));
    envColor = ACESFilm(envColor);
    // gamma correct
    envColor = pow(envColor, vec3(1.0/2.2));

    FragColor = vec4(envColor, 1.0);
}
#endif
