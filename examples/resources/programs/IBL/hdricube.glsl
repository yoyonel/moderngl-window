#version 330

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_camera;
uniform mat4 m_proj;

out vec3 pos;

void main() {
    gl_Position =  m_proj * m_camera * vec4(in_position, 1.0);;
    pos = in_position.xyz;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

// equirectangular projection HDRI
uniform sampler2D hdri;

in vec3 pos;

const vec2 inverseAtan = vec2(0.1591, 0.3183);
vec2 sphericalToCartesian(vec3 v)
{
    vec2 xy = vec2(atan(v.z, v.x), asin(v.y));
    xy *= inverseAtan;
    xy *= -1; // flip
    xy += 0.5;
    return xy;
}

void main() {
    vec3 sampleDirection = normalize(pos);
    vec2 uv = sphericalToCartesian(sampleDirection);
    vec3 color = texture(hdri, uv).rgb;

    fragColor = vec4(color, 1.0);
}
#endif
