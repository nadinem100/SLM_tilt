"""Detailed debug script to understand spacing calculation."""

from pathlib import Path
import numpy as np
from numpy.fft import fft2, ifftshift, fftshift
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# Same config as test_adaptive_gs.py
YAML_PATH = "../slm_parameters.yml"
N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30
REDSLM = 1
SCAL = 4
WAIST_UM = 9

print("Initializing SLM...")
slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
slm.init_fields(waist_um=WAIST_UM)

print(f"\nA_in shape: {slm.A_in.shape}")

# Manually compute what set_target_grid does
xpix, ypix = slm.x_pixels1, slm.y_pixels1
print(f"\nx_pixels1: {xpix}, y_pixels1: {ypix}")
print(f"scal: {SCAL}")

# Fourier plane of a single input for center finding
A_single = fftshift(fft2(ifftshift(slm.A_in)))
print(f"\nA_single shape: {A_single.shape}")

power = np.abs(A_single) ** 2
center_idx = np.argmax(power)
center_row, center_col = np.unravel_index(center_idx, A_single.shape)
print(f"Center found at: row={center_row}, col={center_col}")

# Calculate spacing
spacing_h = slm._calc_spacing_pixels(SPACING_UM, A_single)
spacing_v = spacing_h * ypix / xpix
print(f"\nSpacing in pixels:")
print(f"  spacing_h: {spacing_h}")
print(f"  spacing_v: {spacing_v}")

# Generate centered tweezers
h_offset_curr = np.round(spacing_h * (np.arange(N_HORIZ) - (N_HORIZ - 1) / 2))
v_offset_curr = np.round(spacing_v * (np.arange(N_VERT) - (N_VERT - 1) / 2))

print(f"\nHorizontal offsets (first 5): {h_offset_curr[:5]}")
print(f"Vertical offsets (first 5): {v_offset_curr[:5]}")
print(f"Horizontal offset range: [{h_offset_curr.min()}, {h_offset_curr.max()}]")
print(f"Vertical offset range: [{v_offset_curr.min()}, {v_offset_curr.max()}]")

target_rows_ref = np.repeat(v_offset_curr, N_HORIZ)
target_cols_ref = np.tile(h_offset_curr, N_VERT)

target_rows = np.round(center_row + target_rows_ref).astype(int)
target_cols = np.round(center_col + target_cols_ref).astype(int)

print(f"\nTarget positions:")
print(f"  Row range: [{target_rows.min()}, {target_rows.max()}]")
print(f"  Col range: [{target_cols.min()}, {target_cols.max()}]")
print(f"  Array shape: {A_single.shape}")

# Check boundary
box1 = 2
valid = (target_rows >= box1) & (target_rows < A_single.shape[0] - box1) & \
        (target_cols >= box1) & (target_cols < A_single.shape[1] - box1)

print(f"\nBoundary check (box1={box1}):")
print(f"  Row boundary: [{box1}, {A_single.shape[0] - box1})")
print(f"  Col boundary: [{box1}, {A_single.shape[1] - box1})")
print(f"  Number of valid tweezers: {valid.sum()} / {len(valid)}")

# Show which boundaries are violated
rows_too_low = target_rows < box1
rows_too_high = target_rows >= A_single.shape[0] - box1
cols_too_low = target_cols < box1
cols_too_high = target_cols >= A_single.shape[1] - box1

print(f"\nBoundary violations:")
print(f"  Rows too low (<{box1}): {rows_too_low.sum()}")
print(f"  Rows too high (>={A_single.shape[0] - box1}): {rows_too_high.sum()}")
print(f"  Cols too low (<{box1}): {cols_too_low.sum()}")
print(f"  Cols too high (>={A_single.shape[1] - box1}): {cols_too_high.sum()}")

# Show a few example positions
print(f"\nFirst 5 tweezer positions:")
for i in range(min(5, len(target_rows))):
    print(f"  [{i}] row={target_rows[i]}, col={target_cols[i]}, valid={valid[i]}")
