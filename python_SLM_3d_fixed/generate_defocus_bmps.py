"""
Generate multiple BMP files with different defocus values added.
Load these onto the real SLM to test which defocus brings edges into focus.
"""

import numpy as np
from PIL import Image
from pathlib import Path

# Input BMP (the phase mask to modify)
INPUT_BMP = r"c:\Users\srtwe\Box\EndresLab\z_Second Experiment\Code\SLM simulation\nadine\DMD_SLM\slm_output_paraxial\adaptive_test_fixed\_adaptive_80x80_tilt-13deg_3planes_10iter_scal4_waist9.0_20260120_011439_blazepd7.bmp"

# Output directory
OUTPUT_DIR = Path(INPUT_BMP).parent / "defocus_sweep"
OUTPUT_DIR.mkdir(exist_ok=True)

# Optical parameters
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689
PIXEL_UM = 3.74  # SLM pixel pitch

# Defocus values to generate (in um)
# Negative = move focus closer, Positive = move focus farther
DEFOCUS_VALUES = [-5000, -3000, -2000, -1000, -500, 0, 500, 1000, 2000, 3000, 5000]


def add_defocus_to_bmp(input_path: str, defocus_um: float, output_path: str):
    """Add defocus (quadratic phase) to an existing phase mask BMP."""
    
    # Load phase from BMP
    img = Image.open(input_path)
    phase = np.array(img, dtype=np.float64) / 255.0 * 2 * np.pi
    
    h, w = phase.shape
    print(f"  Loaded {w} x {h} phase mask")
    
    # Create coordinate grids (in um)
    x = (np.arange(w) - w/2) * PIXEL_UM
    y = (np.arange(h) - h/2) * PIXEL_UM
    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    
    # Compute defocus phase: phi = k/(2f^2) * z * r^2
    k = 2 * np.pi / WAVELENGTH_UM
    f = FOCAL_LENGTH_UM
    defocus_phase = (k / (2.0 * f * f)) * defocus_um * R2
    
    # Add defocus to existing phase
    new_phase = np.mod(phase + defocus_phase, 2 * np.pi)
    
    # Save as 8-bit BMP
    img8 = (new_phase / (2 * np.pi) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(output_path)
    
    print(f"  Max defocus phase: {np.max(np.abs(defocus_phase)):.1f} rad")


def main():
    print(f"Input BMP: {INPUT_BMP}")
    print(f"Output dir: {OUTPUT_DIR}")
    print()
    
    input_name = Path(INPUT_BMP).stem
    
    for defocus in DEFOCUS_VALUES:
        sign = "plus" if defocus >= 0 else "minus"
        out_name = f"{input_name}_defocus_{sign}{abs(defocus)}um.bmp"
        out_path = OUTPUT_DIR / out_name
        
        print(f"Generating defocus = {defocus:+d} um -> {out_name}")
        add_defocus_to_bmp(INPUT_BMP, defocus, str(out_path))
    
    print()
    print(f"Generated {len(DEFOCUS_VALUES)} BMP files in:")
    print(f"  {OUTPUT_DIR}")
    print()
    print("Load these onto the SLM one by one and check which defocus")
    print("brings the LEFT and RIGHT edges into focus at the same z-plane.")


if __name__ == "__main__":
    main()
