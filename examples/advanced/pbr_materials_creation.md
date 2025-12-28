# PBR Material Presets Creation Guide

This document explains how the `pbr_materials.json` file was generated and how to translate scientific data from databases like [physicallybased.info](https://physicallybased.info) or the [Google Filament Material Guide](https://google.github.io/filament/Material%20Guide.html) into PBR parameters.

## 1. Material Workflow Logic (Metal/Roughness)

The rendering system uses the **Metallic/Roughness workflow**. In this model, the `Albedo` (color) and `Metallic` parameters behave differently depending on whether the material is a conductor (metal) or a dielectric (non-metal).

### A. Metallic Materials (Conductors)
*   **Metallic** = `1.0`
*   **Albedo** = Physical Reflectance (**F0**).
*   **Scientific Source**: Look for "Specular Color" or "Reflectance at 0°".
*   **Logic**: Metals have no diffuse color. All visible light is reflected. The "color" we see is the color of the reflection.
*   **Example (Gold)**: F0 is `[1.00, 0.71, 0.29]`.

### B. Dielectric Materials (Non-metals)
*   **Metallic** = `0.0`
*   **Albedo** = Diffuse Color (**Base Color**).
*   **Scientific Source**: Look for "Base Color" or "Diffuse Color".
*   **Specular (F0)**: Usually assumed to be a constant **0.04** (4%) for most plastics, glass, and wood.
*   **Example (Plastic)**: Albedo is the pigment color (e.g., Red `[1.0, 0.0, 0.0]`).

---

## 2. Parameter Mapping

| JSON Property | Scientific Term / Source | Logic |
| :--- | :--- | :--- |
| `name` | Material Identifier | Descriptive name for the UI. |
| `albedo` | **F0** (for Metals) / **Base Color** (for Dielectrics) | Normalized RGB values [0.0 - 1.0]. |
| `metallic` | Conductivity | `1.0` for pure metals, `0.0` for others. |
| `roughness` | State of Surface | Defined by the physical finish (see below). |

---

## 3. Estimating Roughness

Unlike Albedo or Metallic, **Roughness** is not a fixed physical constant of the material itself, but of its **surface finish**.

| Visual Finish | Typical Roughness | Examples |
| :--- | :--- | :--- |
| **Mirror / Polished** | 0.02 - 0.1 | Chrome, Polished Gold, Clean Glass. |
| **Satin / Semi-gloss** | 0.2 - 0.3 | Machined Steel, New Plastic, Car Paint. |
| **Matte / Brossé** | 0.4 - 0.6 | Brushed Aluminum, Sanded Wood, Matte Paint. |
| **Rough / Raw** | 0.7 - 0.9 | Concrete, Dry Soil, Cast Iron, Rust. |

---

## 4. Conversion Formulas

### Index of Refraction (IOR) to F0
If a source only provides the IOR (e.g., Water = 1.33), use this formula to find the F0 reflectance:
$$F0 = \left( \frac{IOR - 1}{IOR + 1} \right)^2$$

### Linear vs sRGB
The shaders in this project expect **Linear** color space (0.0 to 1.0). If you have sRGB (0-255) values:
$$Linear = \left( \frac{sRGB}{255} \right)^{2.2}$$

---

## 5. Summary of the JSON Generation process
1. **Source**: Pulled base values from the *Filament* and *Substance* PBR charts.
2. **Expansion**: Created variations (e.g., "Rusty Iron" vs "Iron") by increasing roughness and adjusting albedo.
3. **Validation**: Grouped materials into 10 categories (Metals, Woods, Tech, etc.) to ensure a balanced 10x10 grid coverage.
