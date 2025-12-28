"""
Generate LTC (Linearly Transformed Cosines) Look-Up Tables (LUTs)
Based on Eric Heitz et al. "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"
SIGGRAPH 2016

This script generates two 64x64 textures:
- ltc_mat.bin: 4-channel (RGBA32F) containing the inverse M matrix coefficients
- ltc_amp.bin: 2-channel (RG32F) containing amplitude and Fresnel terms
"""

import numpy as np
import struct

def fresnel_schlick(F0, cosTheta):
    """Schlick's Fresnel approximation"""
    return F0 + (1.0 - F0) * np.power(1.0 - cosTheta, 5.0)

def generate_ltc_luts(size=64):
    """
    Generate LTC LUTs for GG X BRDF
    
    LUT coordinates:
    - U (x-axis): sqrt(roughness) [0, 1]
    - V (y-axis): sqrt(1 - cos(theta)) where theta is angle between normal and view [0, 1]
    
    The fitted parameters are based on the paper's reference implementation.
    """
    # Allocate arrays
    ltc_mat = np.zeros((size, size, 4), dtype=np.float32)
    ltc_amp = np.zeros((size, size, 2), dtype=np.float32)
    
    for y in range(size):
        for x in range(size):
            # Normalize coordinates to [0, 1]
            roughness = (x + 0.5) / size
            theta_coord = (y + 0.5) / size
            
            # Reconstruct view angle
            cos_theta = 1.0 - theta_coord * theta_coord
            cos_theta = max(cos_theta, 0.001)  # Avoid division by zero
            theta = np.arccos(np.sqrt(cos_theta))
            
            # Simplified fitted coefficients (based on reference tables)
            # For a full implementation, these would be loaded from precomputed tables
            # Here we use a simplified analytical approximation
            
            alpha = roughness * roughness  # GGX alpha = roughness^2
            
            # Inverse LTC matrix coefficients (simplified fit)
            # M^-1 is parameterized to approximate the GGX distribution
            a = 1.0 / (1.0 + alpha * (1.0 - cos_theta))
            b = 0.0
            c = alpha * np.sqrt((1.0 - cos_theta) / cos_theta)
            d = 1.0
            
            # Store in RGBA format: M^-1 = [[a, b], [c, d]]
            # We store as (a, b, c, d) but often b=0 for isotropic BRDFs
            ltc_mat[y, x, 0] = a
            ltc_mat[y, x, 1] = b
            ltc_mat[y, x, 2] = c
            ltc_mat[y, x, 3] = d
            
            # Amplitude / normalization term
            magnitude = 1.0 / np.sqrt(a * d - b * c)
            ltc_amp[y, x, 0] = magnitude
            
            #Fresnel term (Schlick approximation with F0=0.04 for dielectrics)
            F0 = 0.04
            fresnel = fresnel_schlick(F0, np.sqrt(cos_theta))
            ltc_amp[y, x, 1] = fresnel
    
    return ltc_mat, ltc_amp

def save_lut_as_binary(data, filename):
    """Save LUT as raw binary float32 data"""
    with open(filename, 'wb') as f:
        f.write(data.tobytes())
    print(f"Saved {filename}: shape={data.shape}, dtype={data.dtype}")

if __name__ == "__main__":
    print("Generating LTC LUTs (64x64)...")
    ltc_mat, ltc_amp = generate_ltc_luts(size=64)
    
    # Save as binary files
    save_lut_as_binary(ltc_mat, "ltc_mat.bin")
    save_lut_as_binary(ltc_amp, "ltc_amp.bin")
    
    print("LTC LUTs generated successfully!")
    print("ltc_mat.bin: 64x64x4 (RGBA32F) - Inverse LTC matrix")
    print("ltc_amp.bin: 64x64x2 (RG32F) - Amplitude and Fresnel")
