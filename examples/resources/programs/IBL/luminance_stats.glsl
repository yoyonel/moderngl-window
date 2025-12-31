#version 450 core

// ============================================================================
// COMPUTE SHADER: Calcul statistiques luminance d'une HDR environment map
// Méthode: Réduction hiérarchique (sans atomics)
// ============================================================================

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Input: HDR equirectangular texture
layout(binding = 0) uniform sampler2D hdrTexture;

// Output: Buffer avec somme des luminances
layout(std430, binding = 1) buffer LuminanceBuffer {
    float luminances[];  // Chaque thread écrit sa contribution
};

void main() {
    ivec2 coords = ivec2(gl_GlobalInvocationID.xy);
    ivec2 texSize = textureSize(hdrTexture, 0);
    
    // Vérification bounds
    if (coords.x >= texSize.x || coords.y >= texSize.y) {
        return;
    }
    
    // Lecture pixel HDR
    vec3 color = texelFetch(hdrTexture, coords, 0).rgb;
    
    // Calcul luminance Rec.709
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    
    // Index linéaire dans le buffer
    uint index = gl_GlobalInvocationID.y * uint(texSize.x) + gl_GlobalInvocationID.x;
    
    // Écriture dans le buffer
    luminances[index] = lum;
}
