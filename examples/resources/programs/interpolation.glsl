#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;
out vec2 uv0;

void main() {
    gl_Position = vec4(in_position, 1);
    uv0 = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D texture0;
in vec2 uv0;

vec3 filter_fetch(sampler2D tex, vec2 uv, int layer) {
    ivec2 textureResolution = textureSize(texture0, layer);
    uv = uv*textureResolution + 0.5;
    vec2 iuv = floor(uv);
    vec2 fuv = fract(uv);
    uv = iuv + fuv * fuv * (3.0 - 2.0 * fuv);
    uv = (uv - 0.5)/textureResolution;
    return textureLod(tex, uv, layer).rgb;
}

void main() {
//    fragColor = texture(texture0, uv0);
    fragColor = vec4(filter_fetch(texture0, uv0, 1), 1.0);
}

#endif
