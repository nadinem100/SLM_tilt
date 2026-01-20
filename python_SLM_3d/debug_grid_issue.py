"""Debug script to understand the grid boundary issue."""

from pathlib import Path
import numpy as np
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# Same config as test_adaptive_gs.py
YAML_PATH = "../slm_parameters.yml"
N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30
REDSLM = 1
SCAL = 4
WAIST_UM = 9
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689

print("Initializing SLM...")
slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)

print(f"\nSLM dimensions after redSLM={REDSLM}:")
print(f"  x_pixels1: {slm.x_pixels1}")
print(f"  y_pixels1: {slm.y_pixels1}")

slm.init_fields(waist_um=WAIST_UM)

print(f"\nField dimensions (scal={SCAL}):")
print(f"  A_in shape: {slm.A_in.shape}")

print("\nSetting target grid...")
slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM,
                    odd_tw=1, box1=2)

print(f"\nA_target shape: {slm.A_target.shape}")
print(f"Number of valid tweezers: {len(slm.tweezlist)}")
print(f"Target positions shape: {slm.target_xy_um.shape if slm.target_xy_um is not None else 'None'}")

if slm.target_xy_um is not None and len(slm.target_xy_um) > 0:
    print(f"\nValid tweezer positions (µm):")
    print(f"  x range: [{slm.target_xy_um[:, 0].min():.2f}, {slm.target_xy_um[:, 0].max():.2f}]")
    print(f"  y range: [{slm.target_xy_um[:, 1].min():.2f}, {slm.target_xy_um[:, 1].max():.2f}]")
else:
    print("\nERROR: No valid tweezers!")

print("\nSetting optics...")
slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

print("\nAttempting to assign planes from tilt...")
try:
    slm.assign_planes_from_tilt(tilt_x_deg=-13, n_planes=3)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED with error: {e}")
    print(f"\nDiagnostics:")
    print(f"  target_xy_um exists: {slm.target_xy_um is not None}")
    if slm.target_xy_um is not None:
        print(f"  target_xy_um shape: {slm.target_xy_um.shape}")
        print(f"  target_xy_um dtype: {slm.target_xy_um.dtype}")
