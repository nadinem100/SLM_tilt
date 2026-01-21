"""
Test the adaptive multi-plane GS algorithm with aspect ratio corrected spacing.
X spacing stays as input, Y spacing is multiplied by aspect ratio (4000/2464).
"""

import os
os.environ['PYDEVD_USE_FAST_XML'] = '1'

from pathlib import Path
import numpy as np
from PIL import Image
import time
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# ================================ CONFIG ================================
YAML_PATH = "../slm_parameters.yml"

# Grid configuration (small test for speed)
N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30 # 30 #4 #30.0

# GS algorithm
ITERATIONS = 40
GG = 0.6
REDSLM = 1
SCAL = 2 # 4
WAIST_UM = 9 /2 * 1e3 # 2.6 / 2 * 1e3  # microns 2.6 -> 9
TOL = 5e-3

# Optics
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689

# Tilt configuration
TILT_ANGLE_X = -13  # degrees
N_Z_PLANES = 5

# Adaptive parameters
Z_SCAN_EVERY = 5  # Scan every 5 iterations
Z_SCAN_RANGE_UM = 50.0  # ±50 µm around target
Z_SCAN_STEPS = 11  # Coarse scan for speed
PEAK_SHARPNESS_THRESHOLD = 2.0
Z_CORRECTION_FACTOR = 0.3  # Gradual correction

# Aspect ratio correction
ASPECT_RATIO = 4000.0 / 2464.0  # x_pixels / y_pixels

# ================================ BMP EXPORT ================================

def add_blazed_grating(phase_mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Add blazed grating with spatial frequencies (fx, fy) to phase mask."""
    H, W = phase_mask.shape
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)
    gr = (2*np.pi*fx*xx)[None, :] + (2*np.pi*fy*yy)[:, None]
    return np.mod(phase_mask + (gr % (2*np.pi)), 2*np.pi).astype(np.float32, copy=False)


def save_phase_bmp(phase: np.ndarray, out_path: Path) -> None:
    """Save phase mask as 8-bit BMP (0-255 maps to 0-2π)."""
    img8 = (np.clip(phase/(2*np.pi), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)

# ================================ MAIN ================================

def main():
    # Start total timer
    total_start_time = time.time()

    print("="*70)
    print("ADAPTIVE GS TEST - ASPECT RATIO CORRECTED SPACING")
    print("="*70)

    OUT_DIR = Path("slm_output_paraxial/adaptive_test")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== SETUP ==========
    print("\n--- Setting up SLM ---")
    setup_start_time = time.time()
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)

    # Calculate aspect-corrected spacing
    spacing_x = SPACING_UM
    spacing_y = SPACING_UM * ASPECT_RATIO

    print(f"\n--- Aspect ratio corrected spacing ---")
    print(f"  X spacing: {spacing_x:.2f} µm")
    print(f"  Y spacing: {spacing_y:.2f} µm (x {ASPECT_RATIO:.4f})")
    print(f"  Aspect ratio: {ASPECT_RATIO:.4f} (4000/2464)")

    # Use set_target_grid with x spacing first
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=spacing_x,
                        odd_tw=1, box1=2)

    # Now manually correct the y-coordinates to use aspect-corrected spacing
    # The class applies ypix/xpix = 2464/4000 = 0.616 to the PIXEL spacing
    # This makes vertical pixel spacing SMALLER than horizontal
    # But we want EQUAL physical spacing, so we need to UNDO this by dividing by (ypix/xpix)
    # Which is the same as multiplying by ASPECT_RATIO = xpix/ypix = 4000/2464
    if hasattr(slm, 'target_xy_um') and slm.target_xy_um is not None:
        print(f"\n--- Adjusting Y positions for aspect ratio ---")
        print(f"  Original X range: [{slm.target_xy_um[:, 0].min():.2f}, {slm.target_xy_um[:, 0].max():.2f}] µm")
        print(f"  Original Y range: [{slm.target_xy_um[:, 1].min():.2f}, {slm.target_xy_um[:, 1].max():.2f}] µm")

        # The class makes Y spacing smaller by factor ypix/xpix
        # We undo this by multiplying by xpix/ypix = ASPECT_RATIO
        slm.target_xy_um[:, 1] = slm.target_xy_um[:, 1] * ASPECT_RATIO

        print(f"  Adjusted X range: [{slm.target_xy_um[:, 0].min():.2f}, {slm.target_xy_um[:, 0].max():.2f}] µm")
        print(f"  Adjusted Y range: [{slm.target_xy_um[:, 1].min():.2f}, {slm.target_xy_um[:, 1].max():.2f}] µm")
        print(f"  Total tweezers: {len(slm.target_xy_um)}")

        # Also print the expected spacing
        if len(slm.target_xy_um) > 1:
            x_spacing_measured = slm.target_xy_um[1, 0] - slm.target_xy_um[0, 0]
            y_spacing_measured = slm.target_xy_um[N_HORIZ, 1] - slm.target_xy_um[0, 1] if len(slm.target_xy_um) > N_HORIZ else 0
            print(f"  Measured X spacing: {x_spacing_measured:.2f} µm")
            print(f"  Measured Y spacing: {y_spacing_measured:.2f} µm")

    # CRITICAL FIX: Also need to modify the PIXEL positions in tweezlist
    # The GS algorithm uses tweezlist (pixel coords), not target_xy_um
    # We need to scale the row positions (vertical) by ASPECT_RATIO
    if hasattr(slm, 'tweezlist') and slm.tweezlist is not None and len(slm.tweezlist) > 0:
        print(f"\n--- CRITICAL: Adjusting pixel positions (tweezlist) ---")
        print(f"  Original pixel rows range: [{slm.tweezlist[:, 0].min()}, {slm.tweezlist[:, 0].max()}]")
        print(f"  Original pixel cols range: [{slm.tweezlist[:, 1].min()}, {slm.tweezlist[:, 1].max()}]")

        # Get the center
        center_row = slm.center_row
        center_col = slm.center_col
        print(f"  Center: row={center_row}, col={center_col}")

        # Scale rows (vertical positions) by ASPECT_RATIO relative to center
        slm.tweezlist[:, 0] = center_row + (slm.tweezlist[:, 0] - center_row) * ASPECT_RATIO

        print(f"  Adjusted pixel rows range: [{slm.tweezlist[:, 0].min():.1f}, {slm.tweezlist[:, 0].max():.1f}]")
        print(f"  Adjusted pixel cols range: [{slm.tweezlist[:, 1].min():.1f}, {slm.tweezlist[:, 1].max():.1f}]")

        # Also need to rebuild A_target and coordinates with new positions
        print(f"  Rebuilding A_target and coordinates...")
        from scipy.fft import fft2, fftshift, ifftshift
        A_target = np.zeros_like(slm.A_target, dtype=np.float32)
        box1 = slm.box1

        # Round to integer positions and check bounds
        target_rows = np.round(slm.tweezlist[:, 0]).astype(int)
        target_cols = np.round(slm.tweezlist[:, 1]).astype(int)
        valid = (target_rows >= box1) & (target_rows < A_target.shape[0] - box1) & \
                (target_cols >= box1) & (target_cols < A_target.shape[1] - box1)

        if not np.all(valid):
            print(f"  Warning: {(~valid).sum()} tweezer(s) out of bounds after aspect correction")
            # Keep only valid ones
            slm.tweezlist = slm.tweezlist[valid]
            slm.target_xy_um = slm.target_xy_um[valid]
            target_rows = target_rows[valid]
            target_cols = target_cols[valid]

        # Set target positions
        A_target[target_rows, target_cols] = 1.0
        slm.A_target = A_target
        slm.tweezer_mask = (A_target > 0).astype(np.uint8)

        # Rebuild coordinates for each tweezer
        side = 2 * box1 + 1
        coords = []
        centers = []
        for r, c in zip(target_rows, target_cols):
            centers.append([r, c])
            rows_local = np.arange(r - box1, r + box1 + 1)
            cols_local = np.arange(c - box1, c + box1 + 1)
            grid_rows = np.repeat(rows_local, side)
            grid_cols = np.tile(cols_local, side)
            flat_idx = np.ravel_multi_index((grid_rows, grid_cols), A_target.shape)
            coords.extend(flat_idx.tolist())

        slm.tweezlist = np.array(centers, dtype=int)
        slm.coordinates = np.array(coords, dtype=int)

        # Update height corrections
        slm.height_corr = np.ones((len(slm.tweezlist), 1), dtype=np.float64)
        slm.height_corr2 = np.repeat(slm.height_corr, (2 * box1 + 1) ** 2, axis=0)

        print(f"  Rebuilt successfully: {len(slm.tweezlist)} tweezers")

    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

    print(f"\n--- Assigning planes with {TILT_ANGLE_X}° tilt ---")
    slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=N_Z_PLANES)

    if hasattr(slm, '_z_planes') and hasattr(slm, '_z_per_spot'):
        print(f"  Z-planes: {slm._z_planes}")
        print(f"  Z per spot range: [{np.min(slm._z_per_spot):.2f}, {np.max(slm._z_per_spot):.2f}] µm")
        print(f"  Number of tweezers: {len(slm._z_per_spot)}")

    setup_time = time.time() - setup_start_time
    print(f"\n[TIMING] Setup completed in {setup_time:.2f} seconds")

    # ========== RUN ADAPTIVE GS ==========
    print("\n" + "="*70)
    print("RUNNING ADAPTIVE MULTI-PLANE GS")
    print("="*70)

    gs_start_time = time.time()

    slm.run_gs_multiplane_adaptive(
        iterations=ITERATIONS,
        Gg=GG,
        z_scan_every=Z_SCAN_EVERY,
        z_scan_range_um=Z_SCAN_RANGE_UM,
        z_scan_steps=Z_SCAN_STEPS,
        peak_sharpness_threshold=PEAK_SHARPNESS_THRESHOLD,
        z_correction_factor=Z_CORRECTION_FACTOR,
        verbose=True,
        tol=TOL
    )

    gs_time = time.time() - gs_start_time
    print(f"\n[TIMING] GS algorithm completed in {gs_time:.2f} seconds")

    # ========== SAVE RESULTS ==========
    print("\n--- Saving results ---")
    label = f"_adaptive_{N_HORIZ}x{N_VERT}_spX{spacing_x:.1f}umY{spacing_y:.1f}um_tilt{TILT_ANGLE_X}deg_{N_Z_PLANES}planes_{ITERATIONS}iter_scal{SCAL}_waist{WAIST_UM/1e3*2:.1f}"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"✓ Saved: {bundle.file}")

    # Print final z-positions
    if hasattr(slm, '_z_per_spot'):
        z_final = slm._z_per_spot
        print(f"\nFinal z-positions after adaptation:")
        print(f"  Range: [{np.min(z_final):.2f}, {np.max(z_final):.2f}] µm")
        print(f"  Mean: {np.mean(z_final):.2f} µm")
        print(f"  Std: {np.std(z_final):.2f} µm")

    # ========== EXPORT BMP WITH BLAZED GRATING ==========
    print("\n--- Exporting BMP with blazed grating ---")

    # Get phase mask
    phase_mask = slm.phase_mask.copy()

    # Add blazed grating (fx=1/7, fy=0)
    fx, fy = 1.0 / 7.0, 0.0
    phase_blazed = add_blazed_grating(phase_mask, fx=fx, fy=fy)

    # Output path: same directory as pickle, add "_blazepd7.bmp" suffix
    pkl_path = Path(bundle.file)
    stem = pkl_path.stem  # Filename without extension
    out_bmp = pkl_path.parent / f"{stem}_blazepd7.bmp"

    # Save BMP
    save_phase_bmp(phase_blazed, out_bmp)
    print(f"✓ Saved BMP: {out_bmp.name}")

    # ========== RUN DIAGNOSTIC VISUALIZATION ==========
    print("\n" + "="*70)
    print("GENERATING DIAGNOSTIC VISUALIZATIONS")
    print("="*70)

    import subprocess
    import sys

    # Run the diagnostic script with the pickle path as argument
    # This will generate both the z-profiles and xy-grid figures
    try:
        result = subprocess.run(
            [sys.executable, "diagnose_tweezer_xz_profiles.py", str(bundle.file)],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not generate diagnostic: {e}")
        print(e.stdout)
        print(e.stderr)


    total_time = time.time() - total_start_time

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print("\n" + "="*70)
    print("TIMING SUMMARY")
    print("="*70)
    print(f"Setup time:      {setup_time:8.2f} seconds")
    print(f"GS algorithm:    {gs_time:8.2f} seconds")
    print(f"Total time:      {total_time:8.2f} seconds")
    print(f"\nConfiguration: {N_HORIZ}x{N_VERT} grid, X={spacing_x:.1f}µm Y={spacing_y:.1f}µm, {ITERATIONS} iterations, {N_Z_PLANES} z-planes")
    print("="*70)


if __name__ == "__main__":
    main()
