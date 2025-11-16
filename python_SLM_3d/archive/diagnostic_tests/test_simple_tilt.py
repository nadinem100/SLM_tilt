"""
Ultra-simple test: Run the algorithm with smaller SLM and check if tilt works.
"""

import numpy as np
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

print("="*70)
print("SIMPLE TILT TEST (smaller SLM)")
print("="*70)

# Smaller test for speed
N_HORIZ = 5
N_VERT = 5
SPACING_UM = 50.0  # Larger spacing for easier visualization
FOCAL_LENGTH_UM = 200000.0
TILT_ANGLE_X = 10  # Larger tilt for easier detection

print(f"\nTest parameters:")
print(f"  Grid: {N_HORIZ}×{N_VERT}, spacing={SPACING_UM} µm")
print(f"  Focal length: {FOCAL_LENGTH_UM/1000:.0f} mm")
print(f"  Tilt: {TILT_ANGLE_X}°")

# Setup
slm = SLMTweezers(yaml_path="../slm_parameters.yml", redSLM=1, scal=4)
slm.init_fields(waist_um=4500)
slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, odd_tw=1, box1=2)
slm.set_optics(wavelength_um=0.689, focal_length_um=FOCAL_LENGTH_UM)

print(f"\n--- Assigning tilted planes ---")
slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=5)

# Check the results
print(f"\n--- Results ---")
print(f"Z-planes: {slm._z_planes}")
print(f"Z per spot (all {len(slm._z_per_spot)} tweezers):")

# Expected: tweezers arranged in 5 rows
# For each row (constant y), tweezers should have linearly increasing z
for row in range(N_VERT):
    start_idx = row * N_HORIZ
    end_idx = start_idx + N_HORIZ
    z_row = slm._z_per_spot[start_idx:end_idx]
    x_row = slm.target_xy_um[start_idx:end_idx, 0]
    print(f"  Row {row}: x from {x_row.min():.1f} to {x_row.max():.1f} µm")
    print(f"          z from {z_row.min():.1f} to {z_row.max():.1f} µm")

# Calculate expected z-range
x_extent = (N_HORIZ - 1) * SPACING_UM
z_expected = x_extent * np.tan(np.deg2rad(TILT_ANGLE_X))
print(f"\n  Expected z range: ±{z_expected/2:.1f} µm")
print(f"  Actual z range: [{np.min(slm._z_per_spot):.1f}, {np.max(slm._z_per_spot):.1f}] µm")

# Check phase corrections
print(f"\n--- Phase corrections ---")
for i, phi in enumerate(slm._phi_planes):
    rms = np.sqrt(np.mean(phi**2))
    maxval = np.max(np.abs(phi))
    print(f"  Plane {i} (z={slm._z_planes[i]:+.2f} µm): RMS={rms:.4f} rad, max={maxval:.4f} rad")

# Quick GS run
print(f"\n--- Running 3 iterations of GS (just to test) ---")
slm.run_gs_multiplane_v3(iterations=3, Gg=0.6, verbose=True, tol=1e-4)

print(f"\n--- Final phase mask stats ---")
phase = slm.phase_mask
print(f"  Shape: {phase.shape}")
print(f"  Range: [{phase.min():.4f}, {phase.max():.4f}] rad")
print(f"  Mean: {phase.mean():.4f} rad")
print(f"  Std: {phase.std():.4f} rad")

# Check for spatial structure in phase
grad_x = np.gradient(phase, axis=1)
grad_y = np.gradient(phase, axis=0)
print(f"  Mean gradient (x): {np.mean(grad_x):.6f} rad/px")
print(f"  Mean gradient (y): {np.mean(grad_y):.6f} rad/px")
print(f"  Std gradient (x): {np.std(grad_x):.4f} rad/px")
print(f"  Std gradient (y): {np.std(grad_y):.4f} rad/px")

print("\n" + "="*70)
print("INTERPRETATION:")
if np.max(np.abs([rms for phi in slm._phi_planes for rms in [np.sqrt(np.mean(phi**2))]])) > 0.01:
    print("✓ Phase corrections are non-zero (tilt is encoded in algorithm)")
else:
    print("✗ Phase corrections are near zero (tilt NOT working!)")

if phase.std() > 1.0:
    print("✓ Final phase has significant variation (hologram created)")
else:
    print("✗ Final phase is nearly flat (something wrong!)")

print("="*70)
