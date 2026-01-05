#version 450 core

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_inv_view_proj;

out vec3 RayDir;

void main() {
    // Render the quad in clip space (ignore z for depth as test is disabled)
    gl_Position = vec4(in_position.xy, 0.0, 1.0);
    
    // Reconstruct world-space direction using a finite NDC point (z=0.0 is halfway)
    // This avoids division by w=0 when using an infinite projection matrix.
    vec4 pos = m_inv_view_proj * vec4(in_position.xy, 0.0, 1.0);
    RayDir = pos.xyz / pos.w;
}

#elif defined FRAGMENT_SHADER

out vec4 FragColor;
in vec3 RayDir;

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
    vec3 envColor = textureLod(environmentMap, normalize(RayDir), blur_lod).rgb;
    envColor = ACESFilm(envColor);
    envColor = pow(envColor, vec3(1.0/2.2));
    FragColor = vec4(envColor, 1.0);
}
#endif
