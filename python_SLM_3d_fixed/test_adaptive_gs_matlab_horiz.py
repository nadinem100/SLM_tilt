"""
Test the adaptive multi-plane GS algorithm with MATLAB-style HORIZONTAL spacing.
- Horizontal spacing: calculated dynamically from beam waist FFT (like MATLAB)
- Tilt/z-plane spacing: unchanged from working version
"""

import os
os.environ['PYDEVD_USE_FAST_XML'] = '1'

from pathlib import Path
import numpy as np
from PIL import Image
import time
import sys

# Import the FIXED class from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# ================================ CONFIG ================================
YAML_PATH = "../slm_parameters.yml"

# Grid configuration
N_HORIZ = 20
N_VERT = 20
# SPACING_UM is NOT used - calculated dynamically like MATLAB

# MATLAB spacing factor: 4/(2*0.77) = 2.597
SPACING_FACTOR = 4.0 / (2.0 * 0.77)

# GS algorithm
GG = 0.6
REDSLM = 1
TOL = 1e-4

# Optics
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689

# Tilt configuration (UNCHANGED from working version)
TILT_ANGLE_X = -13  # degrees

# Command-line arguments
N_Z_PLANES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ITERATIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SCAL = int(sys.argv[3]) if len(sys.argv) > 3 else 4
WAIST_COEFF = float(sys.argv[4]) if len(sys.argv) > 4 else 9.0
WAIST_UM = WAIST_COEFF / 2 * 1e3  # microns

# Adaptive parameters
Z_SCAN_EVERY = 5
Z_SCAN_RANGE_UM = 50.0
Z_SCAN_STEPS = 11
PEAK_SHARPNESS_THRESHOLD = 2.0
Z_CORRECTION_FACTOR = 0.3

# ================================ MATLAB-STYLE SPACING ================================

def calc_spacing_matlab_style(A_single: np.ndarray) -> int:
    """
    Calculate spacing like MATLAB's calculate_spacing function.
    Finds the 1/e^2 radius of the FFT and uses spacing_factor.
    """
    intensity = np.abs(A_single) ** 2
    max_val = np.max(intensity)
    max_idx = np.argmax(intensity.ravel())
    
    # Find where intensity drops to exp(-2) of max (1/e^2)
    stop_idx = max_idx
    flat_intensity = intensity.ravel()
    threshold = np.exp(-2) * max_val
    
    while stop_idx < len(flat_intensity) - 1 and flat_intensity[stop_idx] >= threshold:
        stop_idx += 1
    
    # Calculate spacing using MATLAB formula
    spacing = int(np.ceil(SPACING_FACTOR * 2 * (stop_idx - max_idx)))
    return max(1, spacing)


def set_target_grid_matlab_style(slm, n_horiz: int, n_vert: int):
    """
    Set up target grid using MATLAB-style spacing calculation.
    This replaces slm.set_target_grid() with dynamic spacing.
    """
    # Get A_single (FFT of input beam)
    A_in = slm.A_in
    A_single = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(A_in)))
    
    # Calculate spacing like MATLAB
    spacing_h = calc_spacing_matlab_style(A_single)
    
    # Vertical spacing scaled by aspect ratio (like MATLAB)
    spacing_v = int(round(spacing_h * slm.params.y_pixels / slm.params.x_pixels))
    
    print(f"MATLAB-style spacing calculation:")
    print(f"  spacing_h = {spacing_h} pixels")
    print(f"  spacing_v = {spacing_v} pixels")
    
    # Find center of A_single
    intensity = np.abs(A_single) ** 2
    max_idx = np.argmax(intensity)
    center_row, center_col = np.unravel_index(max_idx, intensity.shape)
    
    # Calculate target positions (like MATLAB create_full_target_array)
    hh = np.arange(1, n_horiz + 1)
    vv = np.arange(1, n_vert + 1)
    
    h_offset_curr = np.round(spacing_h * (hh - n_horiz / 2)).astype(int)
    v_offset_curr = np.round(spacing_v * (vv - n_vert / 2)).astype(int)
    
    # Build target grid
    target_rows = []
    target_cols = []
    for v_off in v_offset_curr:
        for h_off in h_offset_curr:
            target_rows.append(center_row + v_off)
            target_cols.append(center_col + h_off)
    
    target_rows = np.array(target_rows)
    target_cols = np.array(target_cols)
    
    # Store in SLM object (same attributes as set_target_grid)
    slm.center_row = center_row
    slm.center_col = center_col
    slm.A_target = np.zeros_like(A_in)
    slm.box1 = 2  # Set box1 BEFORE computing coordinates
    
    H, W = slm.A_target.shape
    box1 = slm.box1
    
    # Set target amplitudes - only for positions with valid box1 margin
    valid_mask = (target_rows >= box1) & (target_rows < H - box1) & \
                 (target_cols >= box1) & (target_cols < W - box1)
    
    for i in range(len(target_rows)):
        if valid_mask[i]:
            slm.A_target[target_rows[i], target_cols[i]] = 1.0
    
    # Use the class method to compute tweezlist and coordinates properly
    valid_rows = target_rows[valid_mask]
    valid_cols = target_cols[valid_mask]
    slm.tweezlist, slm.coordinates = slm._compute_tweezer_centers_and_coords(
        slm.A_target, valid_rows, valid_cols, box1
    )
    
    # Height corrections (must match expected shapes)
    num_tweezers = len(slm.tweezlist)
    slm.height_corr = np.ones((num_tweezers, 1), dtype=np.float64)
    slm.height_corr2 = np.repeat(slm.height_corr, (2 * box1 + 1) ** 2, axis=0)
    
    # Create tweezer mask
    slm.tweezer_mask = np.zeros_like(slm.A_target, dtype=bool)
    for r, c in slm.tweezlist:
        slm.tweezer_mask[r, c] = True
    
    # Store reference positions for assign_planes_from_tilt
    slm._target_cols_ref = valid_cols.copy()
    slm._target_rows_ref = valid_rows.copy()
    slm._valid_mask = valid_mask
    slm.target_xy_um = None
    
    print(f"  Created {len(slm.tweezlist)} tweezers")
    print(f"  Grid spans rows [{target_rows.min()}, {target_rows.max()}]")
    print(f"  Grid spans cols [{target_cols.min()}, {target_cols.max()}]")
    
    # Calculate physical spacing for filename
    pixel_um = slm.params.pixel_um
    focal_pixel_um = (WAVELENGTH_UM * FOCAL_LENGTH_UM) / (slm.params.x_pixels * slm.config.scal * pixel_um)
    physical_spacing_um = spacing_h * focal_pixel_um
    print(f"  Physical spacing: ~{physical_spacing_um:.1f} um")
    
    return spacing_h, physical_spacing_um


# ================================ BMP EXPORT ================================

def add_blazed_grating(phase_mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    H, W = phase_mask.shape
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)
    gr = (2*np.pi*fx*xx)[None, :] + (2*np.pi*fy*yy)[:, None]
    return np.mod(phase_mask + (gr % (2*np.pi)), 2*np.pi).astype(np.float32, copy=False)


def save_phase_bmp(phase: np.ndarray, out_path: Path) -> None:
    img8 = (np.clip(phase/(2*np.pi), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)


# ================================ MAIN ================================

def main():
    total_start_time = time.time()

    print("="*70)
    print("ADAPTIVE GS - MATLAB HORIZONTAL SPACING")
    print("="*70)

    script_dir = Path(__file__).parent
    OUT_DIR = script_dir / "slm_output_paraxial" / "adaptive_test_matlab_spacing"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== SETUP ==========
    print("\n--- Setting up SLM ---")
    setup_start_time = time.time()
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    
    # Use MATLAB-style spacing for horizontal
    spacing_pixels, spacing_um = set_target_grid_matlab_style(slm, N_HORIZ, N_VERT)
    
    # Set optics (same as before)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

    # Assign planes from tilt (UNCHANGED - this is the z-spacing)
    print(f"\n--- Assigning planes with {TILT_ANGLE_X} deg tilt ---")
    slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=N_Z_PLANES)

    if hasattr(slm, '_z_planes') and hasattr(slm, '_z_per_spot'):
        print(f"  Z-planes: {slm._z_planes}")
        print(f"  Z per spot range: [{np.min(slm._z_per_spot):.2f}, {np.max(slm._z_per_spot):.2f}] um")
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
    label = f"_matlab_horiz_{N_HORIZ}x{N_VERT}_sp{spacing_um:.1f}um_tilt{TILT_ANGLE_X}deg_{N_Z_PLANES}planes_{ITERATIONS}iter_scal{SCAL}_waist{WAIST_COEFF}"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"[OK] Saved: {bundle.file}")

    # Print final z-positions
    if hasattr(slm, '_z_per_spot'):
        z_final = slm._z_per_spot
        print(f"\nFinal z-positions after adaptation:")
        print(f"  Range: [{np.min(z_final):.2f}, {np.max(z_final):.2f}] um")
        print(f"  Mean: {np.mean(z_final):.2f} um")
        print(f"  Std: {np.std(z_final):.2f} um")

    # ========== EXPORT BMP ==========
    print("\n--- Exporting BMP with blazed grating ---")
    phase_mask = slm.phase_mask.copy()
    fx, fy = 1.0 / 7.0, 0.0
    phase_blazed = add_blazed_grating(phase_mask, fx=fx, fy=fy)
    
    pkl_path = Path(bundle.file)
    out_bmp = pkl_path.parent / f"{pkl_path.stem}_blazepd7.bmp"
    save_phase_bmp(phase_blazed, out_bmp)
    print(f"[OK] Saved BMP: {out_bmp}")

    # ========== DIAGNOSTIC ==========
    print("\n" + "="*70)
    print("GENERATING DIAGNOSTIC VISUALIZATIONS")
    print("="*70)

    import subprocess
    script_dir = Path(__file__).parent
    diagnostic_script = script_dir / "diagnose_tweezer_xz_profiles.py"
    try:
        result = subprocess.run(
            [sys.executable, str(diagnostic_script), str(bundle.file)],
            check=True, capture_output=True, text=True, cwd=str(script_dir)
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not generate diagnostic: {e}")
        if e.stdout: print(e.stdout)
        if e.stderr: print(e.stderr)

    total_time = time.time() - total_start_time

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nConfiguration: {N_HORIZ}x{N_VERT}, MATLAB spacing={spacing_um:.1f}um, {ITERATIONS} iter, {N_Z_PLANES} z-planes")
    print(f"Total time: {total_time:.2f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()
