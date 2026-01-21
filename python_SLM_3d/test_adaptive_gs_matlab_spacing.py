"""
Test the adaptive multi-plane GS algorithm using MATLAB-style spacing calculation.
This version calculates spacing exactly like modularized_WGS.m to ensure symmetric tweezers.
"""

import os
os.environ['PYDEVD_USE_FAST_XML'] = '1'

from pathlib import Path
import numpy as np
from PIL import Image
import time
from scipy.fft import fft2, fftshift, ifftshift
from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# ================================ CONFIG ================================
YAML_PATH = "../slm_parameters.yml"

# Grid configuration
N_HORIZ = 50
N_VERT = 50

# MATLAB spacing parameters (from modularized_WGS.m)
# spacing_factor = 4/(2*0.77) in MATLAB
SPACING_FACTOR = 4 # 4 / (2 * 0.77)  # ≈ 2.597

# GS algorithm
ITERATIONS = 100
GG = 0.6
REDSLM = 1
SCAL = 4
WAIST_UM = 9 / 2 * 1e3  # microns
TOL = 1e-3

# Optics
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689

# Tilt configuration
TILT_ANGLE_X = -13  # degrees
N_Z_PLANES = 5

# Adaptive parameters (improved for tighter spacing)
Z_SCAN_EVERY = 5  # More frequent z-scans for tighter spacing
Z_SCAN_RANGE_UM = 75.0  # Larger range to catch shifted peaks
Z_SCAN_STEPS = 11  # More steps for better z-resolution
PEAK_SHARPNESS_THRESHOLD = 2.5  # Higher threshold for Gaussian quality
Z_CORRECTION_FACTOR = 0.4  # Slightly more aggressive correction


# ================================ MATLAB SPACING CALCULATION ================================

def calculate_matlab_spacing(A_in: np.ndarray) -> int:
    """
    Calculate spacing exactly like modularized_WGS.m does.
    
    From MATLAB:
        A_single = fftshift(fft2(ifftshift(A_in)));
        [max_val, max_idx] = max(abs(A_single(:)).^2);
        stop_idx = max_idx;
        val = max_val;
        
        while val >= exp(-2) * max_val
            stop_idx = stop_idx + 1;
            val = abs(A_single(stop_idx)).^2;
        end
        
        spacing = ceil(spacing_factor * 2 * (stop_idx - max_idx));
    """
    # FFT of input beam
    A_single = fftshift(fft2(ifftshift(A_in)))
    power = np.abs(A_single.ravel()) ** 2
    
    # Find maximum
    max_idx = np.argmax(power)
    max_val = power[max_idx]
    
    # Find 1/e² radius by scanning linearly from max
    stop_idx = max_idx
    val = max_val
    threshold = np.exp(-2) * max_val
    
    while val >= threshold and stop_idx < len(power) - 1:
        stop_idx += 1
        val = power[stop_idx]
    
    # Calculate spacing using MATLAB formula
    spacing = int(np.ceil(SPACING_FACTOR * 2 * (stop_idx - max_idx)))
    
    print(f"  MATLAB spacing calculation:")
    print(f"    max_idx = {max_idx}")
    print(f"    stop_idx = {stop_idx}")
    print(f"    1/e² radius (pixels) = {stop_idx - max_idx}")
    print(f"    spacing_factor = {SPACING_FACTOR:.4f}")
    print(f"    spacing (h) = {spacing} pixels")
    
    return max(1, spacing)


# ================================ BMP EXPORT ================================

def add_blazed_grating(phase_mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Add blazed grating with spatial frequencies (fx, fy) to phase mask."""
    H, W = phase_mask.shape
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)
    gr = (2 * np.pi * fx * xx)[None, :] + (2 * np.pi * fy * yy)[:, None]
    return np.mod(phase_mask + (gr % (2 * np.pi)), 2 * np.pi).astype(np.float32, copy=False)


def save_phase_bmp(phase: np.ndarray, out_path: Path) -> None:
    """Save phase mask as 8-bit BMP (0-255 maps to 0-2π)."""
    img8 = (np.clip(phase / (2 * np.pi), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)


# ================================ MAIN ================================

def main():
    total_start_time = time.time()

    print("=" * 70)
    print("ADAPTIVE GS TEST WITH MATLAB-STYLE SPACING")
    print("=" * 70)

    OUT_DIR = Path("slm_output_paraxial/adaptive_test_matlab_spacing")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========== SETUP ==========
    print("\n--- Setting up SLM ---")
    setup_start_time = time.time()
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)

    # Calculate spacing using MATLAB method
    print("\n--- Calculating MATLAB-style spacing ---")
    spacing_h_pixels = calculate_matlab_spacing(slm.A_in)
    
    # Calculate vertical spacing with aspect ratio correction (like MATLAB)
    # spacing_v = spacing_h * y_pixels / x_pixels
    xpix = slm.x_pixels1  # scal * x_pixels
    ypix = slm.y_pixels1  # scal * y_pixels
    spacing_v_pixels = spacing_h_pixels * ypix / xpix
    
    print(f"    spacing (v) = {spacing_v_pixels:.2f} pixels (aspect ratio corrected)")
    print(f"    x_pixels1 = {xpix}, y_pixels1 = {ypix}")
    print(f"    aspect ratio = {ypix/xpix:.4f}")
    
    # Convert pixel spacing to physical spacing (for reference)
    # The physical spacing in the focal plane depends on the magnification
    # but internally the SLM class uses pixel spacing
    
    # Now we need to pass a spacing_um value that will result in
    # spacing_h_pixels from the _calc_spacing_pixels method
    # This is tricky because the Python method has a different formula
    
    # Instead, let's directly set the target grid using the pixel spacing
    # by modifying the class or using a workaround
    
    # Workaround: Calculate what spacing_um would give us this pixel spacing
    # From Python: spacing_factor = 1.14 * spacing_um / (10.2 / 4.1)
    # And: spacing = ceil(spacing_factor * 2 * (stop_idx - max_idx))
    # So: spacing_um = spacing_factor * (10.2 / 4.1) / 1.14
    #                = spacing / (2 * (stop_idx - max_idx)) * (10.2 / 4.1) / 1.14
    
    # Actually, let's just directly set the grid using the MATLAB calculation
    # We'll modify set_target_grid_matlab to use our calculated spacing
    
    # For now, let's use the MATLAB spacing directly by patching the method
    slm._matlab_spacing_h = spacing_h_pixels
    slm._matlab_spacing_v = spacing_v_pixels
    
    # Call set_target_grid_matlab which we'll add
    set_target_grid_matlab(slm, n_horiz=N_HORIZ, n_vert=N_VERT, 
                           spacing_h=spacing_h_pixels, spacing_v=spacing_v_pixels,
                           box1=2)
    
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
    print("\n" + "=" * 70)
    print("RUNNING ADAPTIVE MULTI-PLANE GS")
    print("=" * 70)

    gs_start_time = time.time()

    slm.run_gs_multiplane_adaptive(
        iterations=ITERATIONS,
        Gg=GG,
        z_scan_every=Z_SCAN_EVERY,
        z_scan_range_um=Z_SCAN_RANGE_UM,
        z_scan_steps=Z_SCAN_STEPS,
        peak_sharpness_threshold=PEAK_SHARPNESS_THRESHOLD,
        z_correction_factor=Z_CORRECTION_FACTOR,
        spatial_search_radius_um=None,  # Auto-calculate from spacing
        verbose=True,
        tol=TOL
    )

    gs_time = time.time() - gs_start_time
    print(f"\n[TIMING] GS algorithm completed in {gs_time:.2f} seconds")

    # ========== SAVE RESULTS ==========
    print("\n--- Saving results ---")
    label = f"_matlab_spacing_{N_HORIZ}x{N_VERT}_tilt{TILT_ANGLE_X}deg_{N_Z_PLANES}planes_{ITERATIONS}iter_scal{SCAL}_waist{WAIST_UM/1e3*2:.1f}"
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
    phase_mask = slm.phase_mask.copy()
    fx, fy = 1.0 / 7.0, 0.0
    phase_blazed = add_blazed_grating(phase_mask, fx=fx, fy=fy)

    pkl_path = Path(bundle.file)
    stem = pkl_path.stem
    out_bmp = pkl_path.parent / f"{stem}_blazepd7.bmp"
    save_phase_bmp(phase_blazed, out_bmp)
    print(f"✓ Saved BMP: {out_bmp.name}")

    # ========== RUN DIAGNOSTIC VISUALIZATION ==========
    print("\n" + "=" * 70)
    print("GENERATING DIAGNOSTIC VISUALIZATIONS")
    print("=" * 70)

    import subprocess
    import sys

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

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"Setup time:      {setup_time:8.2f} seconds")
    print(f"GS algorithm:    {gs_time:8.2f} seconds")
    print(f"Total time:      {total_time:8.2f} seconds")
    print(f"\nConfiguration: {N_HORIZ}x{N_VERT} grid, {ITERATIONS} iterations, {N_Z_PLANES} z-planes")
    print(f"MATLAB spacing: h={spacing_h_pixels} px, v={spacing_v_pixels:.2f} px")
    print("=" * 70)


def set_target_grid_matlab(slm, *, n_horiz: int, n_vert: int, 
                           spacing_h: float, spacing_v: float, box1: int = 1):
    """
    Set target grid using MATLAB-calculated spacing directly in pixels.
    This bypasses the Python spacing calculation and uses the exact
    MATLAB formula for symmetric tweezers.
    
    Based on modularized_WGS.m create_full_target_array()
    """
    assert slm.A_in is not None, "Call init_fields() first."
    
    # FFT for center finding
    A_single = fftshift(fft2(ifftshift(slm.A_in)))
    power = np.abs(A_single) ** 2
    center_idx = np.argmax(power)
    center_row, center_col = np.unravel_index(center_idx, A_single.shape)
    slm.center_row = center_row
    slm.center_col = center_col
    
    print(f"  Center: row={center_row}, col={center_col}")
    print(f"  Using spacing: h={spacing_h:.2f} px, v={spacing_v:.2f} px")
    
    # Generate centered tweezers (MATLAB style)
    # h_offset_curr = round(spacing_h*(hh-n_horiz/2));
    # v_offset_curr = round(spacing_v*(vv-n_vert/2));
    h_offset_curr = np.round(spacing_h * (np.arange(n_horiz) - (n_horiz - 1) / 2))
    v_offset_curr = np.round(spacing_v * (np.arange(n_vert) - (n_vert - 1) / 2))
    
    # Match MATLAB enumeration: target_rows = repmat(...), target_cols = repelem(...)
    target_rows = np.repeat(center_row + v_offset_curr, n_horiz).astype(int)
    target_cols = np.tile(center_col + h_offset_curr, n_vert).astype(int)
    
    # Build target array
    A_target = np.zeros_like(A_single, dtype=np.float32)
    valid = ((target_rows >= box1) & (target_rows < A_target.shape[0] - box1) &
             (target_cols >= box1) & (target_cols < A_target.shape[1] - box1))
    
    if not np.all(valid):
        print(f"  Warning: {(~valid).sum()} tweezer(s) outside boundary")
    
    A_target[target_rows[valid], target_cols[valid]] = 1.0
    
    slm.A_target = A_target
    slm.box1 = int(box1)
    
    # Build per-tweezer pixel blocks and coordinates
    slm.tweezlist, slm.coordinates = slm._compute_tweezer_centers_and_coords(
        A_target, target_rows[valid], target_cols[valid], slm.box1
    )
    
    # Height corrections
    slm.height_corr = np.ones((len(slm.tweezlist), 1), dtype=np.float64)
    slm.height_corr2 = np.repeat(slm.height_corr, (2 * slm.box1 + 1) ** 2, axis=0)
    
    # Binary mask
    slm.tweezer_mask = (A_target > 0).astype(np.uint8)
    
    # Physical target positions in focal plane (µm)
    # Since we used pixel spacing directly, we need to convert back to physical units
    # The conversion depends on the SLM pixel pitch and optics
    # For now, use the pixel spacing as a proxy for physical spacing
    pixel_um = slm.params.pixel_um
    scal = int(slm.config.scal)
    
    # Physical spacing in image plane ≈ spacing_pixels * pixel_um / scal
    # But actual focal plane spacing depends on magnification
    # We'll store the pixel-based positions for now
    x_um_axis = (np.arange(n_horiz) - (n_horiz - 1) / 2.0) * spacing_h * pixel_um / scal
    y_um_axis = (np.arange(n_vert) - (n_vert - 1) / 2.0) * spacing_v * pixel_um / scal
    
    x_um_list = np.tile(x_um_axis, n_vert)
    y_um_list = np.repeat(y_um_axis, n_horiz)
    
    xy_um = np.stack([x_um_list[valid], y_um_list[valid]], axis=1).astype(np.float32)
    xy_um[:, 0] -= np.mean(xy_um[:, 0])
    xy_um[:, 1] -= np.mean(xy_um[:, 1])
    
    slm.target_xy_um = xy_um
    
    print(f"  Created {len(slm.tweezlist)} tweezers")
    print(f"  Physical spacing estimate: x={spacing_h * pixel_um / scal:.2f} µm, y={spacing_v * pixel_um / scal:.2f} µm")
    
    return 0


if __name__ == "__main__":
    main()
