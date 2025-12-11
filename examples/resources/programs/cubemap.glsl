#version 330

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_camera;
uniform mat4 m_proj;

out vec3 pos;

void main() {
    pos = in_position.xyz;

    gl_Position =  m_proj * m_camera * vec4(in_position, 1.0);
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;

uniform samplerCube texture0;

in vec3 pos;

// https://github.com/nbertoa/BRE12/blob/master/BRE/ToneMappingPass/Shaders/PS.hlsl
vec3 FilmicToneMapping(vec3 color)
{
    const float A = 0.15f; // Shoulder Strength
    const float B = 0.5f; // Linear Strength
    const float C = 0.1f; // Linear Angle
    const float D = 0.2f; // Toe Strength
    const float E = 0.02f; // Toe Numerator
    const float F = 0.3f; // Toe Denominator
    vec3 linearWhite = vec3(11.2f, 11.2f, 11.2f);

    color = ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - (E / F);
    linearWhite = ((linearWhite * (A * linearWhite + C * B) + D * E) / (linearWhite * (A * linearWhite + B) + D * F)) - (E / F);
    return color / linearWhite;
}

vec3 HDRToneMapWithGammaCorrect(vec3 color)
{
    // HDR tonemap and gamma correct
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0/2.2));
    return color;
}

void main() {
    vec4 envColor = textureLod(texture0, normalize(pos), 0);

//    fragColor = vec4(FilmicToneMapping(envColor.rgb), envColor.a);
//    fragColor = vec4(HDRToneMapWithGammaCorrect(envColor.rgb), envColor.a);

    fragColor = envColor;
}
#endif
