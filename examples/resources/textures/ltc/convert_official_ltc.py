"""
Convert LTC data from Eric Heitz's ltc.js to binary LUT textures
"""

import re
import numpy as np

def parse_js_array(js_content, var_name):
    """Extract JavaScript array data"""
    # Find the array declaration
    pattern = rf'var {var_name} = \[(.*?)\];'
    match = re.search(pattern, js_content, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not find variable {var_name}")
    
    # Extract numbers
    array_str = match.group(1)
    # Remove comments and clean up
    array_str = re.sub(r'//.*?\n', '', array_str)
    # Split by commas and filter out empty strings
    numbers = [float(x.strip()) for x in array_str.split(',') if x.strip()]
    
    return np.array(numbers, dtype=np.float32)

def reshape_ltc_data(data, channels):
    """Reshape flat array to 64x64xC texture"""
    size = 64
    total_elements = size * size * channels
    
    if len(data) != total_elements:
        raise ValueError(f"Expected {total_elements} elements, got {len(data)}")
    
    # Reshape to (height=64, width=64, channels)
    return data.reshape((size, size, channels))

# Read the JavaScript file
with open('/tmp/ltc.js', 'r') as f:
    js_content = f.read()

print("Parsing g_ltc_1 (matrix data)...")
ltc_1_data = parse_js_array(js_content, 'g_ltc_1')
print(f"Found {len(ltc_1_data)} values")

print("Parsing g_ltc_2 (amplitude data)...")
ltc_2_data = parse_js_array(js_content, 'g_ltc_2')
print(f"Found {len(ltc_2_data)} values")

# g_ltc_1 contains 4 channels (matrix inverse coefficients)
ltc_mat = reshape_ltc_data(ltc_1_data, 4)
print(f"ltc_mat reshaped to: {ltc_mat.shape}")

# g_ltc_2 also contains 4 channels (amplitude, fresnel, etc.)
ltc_amp = reshape_ltc_data(ltc_2_data, 4)
print(f"ltc_amp reshaped to: {ltc_amp.shape}")

# Save as binary
output_dir = "examples/resources/textures/ltc/"
ltc_mat.tofile(output_dir + "ltc_mat.bin")
ltc_amp.tofile(output_dir + "ltc_amp.bin")

print(f"\\nSaved official LTC LUTs to {output_dir}")
print(f"ltc_mat.bin: 64x64x4 (RGBA32F)")
print(f"ltc_amp.bin: 64x64x2 (RG32F)")
print("\\nDone!")
