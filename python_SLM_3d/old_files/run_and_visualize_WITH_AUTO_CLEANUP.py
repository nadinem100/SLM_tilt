"""
Fixed version with PROPER memory cleanup
"""

import numpy as np
import matplotlib
from datetime import datetime

from python_SLM_3d.old_files.view_3d_pkl import TILT_ANGLE_X
from python_SLM_3d.old_files.view_pkl_class import N_HORIZ

matplotlib.use('Agg')  # Use non-interactive backend (saves memory!)
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path
import sys
sys.path.insert(0, '/mnt/project')
from slm_tweezers_class_WITH_AUTO_CLEANUP import SLMTweezers
import gc

# CRITICAL: Import this BEFORE anything else
import os
os.environ['PYDEVD_USE_FAST_XML'] = '1'  # Disable PyCharm's slow variable inspection

YAML_PATH = "../slm_parameters.yml"

N_HORIZ = 20
N_VERT = 20
SPACING_UM = 10.0
ITERATIONS = 200
GG = 0.6
REDSLM = 1
SCAL = 2
WAIST_UM = 9 / 2 * 1e3

# TILT_ANGLE_X = 0
FOCAL_LENGTH_UM = 100000.0
WAVELENGTH_UM = 0.689

# Z_RANGE_UM = 0
# N_Z_PLANES = 1
# N_Z_STEPS = 1
DOWNSAMPLE_XZ = 1
DPI = 150

TILT_ANGLE_X = 5                     # was 0; make the plot expectations match the solver
# Z_RANGE_UM   = 100                    # scan ±100 µm around best focus
N_Z_PLANES   = 5
N_Z_STEPS    = 21                    # dense X–Z for a robust slope fit

ts = datetime.now().strftime("%y%m%d-%H%M%S")
OUT_DIR = Path(f"slm_output/tilt_{TILT_ANGLE_X}_tw_{N_HORIZ}_{ts}")


def focal_pixel_size_um(wavelength_um, focal_length_um, slm_pixel_size_um, N_pixels):
    """
    Pixel pitch in the focal (Fourier) plane for a 2f system using an N-point FFT.
    Why: visualization must propagate in *focal-plane* coordinates, not SLM pixels.
    """
    return (wavelength_um * focal_length_um) / (N_pixels * slm_pixel_size_um)

def fresnel_propagate(field, distance_um, wavelength_um, pixel_size_um):
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=pixel_size_um)
    fy = np.fft.fftfreq(ny, d=pixel_size_um)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * np.pi * wavelength_um * distance_um * (FX**2 + FY**2))
    result = np.fft.ifft2(np.fft.fft2(field) * H)

    # CRITICAL: Delete intermediate arrays immediately
    del fx, fy, FX, FY, H
    return result

def visualize_results_memory_safe(phase, A_in, wavelength_um, pixel_size_um,
                                  expected_tilt, z_range_um, n_z_planes, n_z_steps, downsample=4):
    """
    Memory-safe visualization that auto-zooms to include ALL tweezers.
    - Uses focal-plane pixel size (not SLM pitch).
    - Crops once using a robust bbox over reference z-slices and reuses it.
    - Measures tilt by fitting z_best vs x (in µm) over the cropped columns.
    """
    import gc
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.fft import fft2, ifft2, fftshift, ifftshift

    print("\n" + "="*60)
    print("VISUALIZING (memory-safe mode)")
    print("="*60)

    # ---- Build focal field once ----
    H, W = A_in.shape
    h, w = phase.shape
    psi_full = np.zeros((H, W), dtype=np.float32)
    y_start = (H - h) // 2
    x_start = (W - w) // 2
    psi_full[y_start:y_start+h, x_start:x_start+w] = phase

    field_slm = A_in * np.exp(1j * psi_full)
    field_focal = fftshift(fft2(ifftshift(field_slm)))

    del psi_full, field_slm

    # Downsample AFTER FFT
    if downsample > 1:
        field_focal = field_focal[::downsample, ::downsample]

    # ---- Focal-plane sampling (NOT SLM pixel size) ----
    Ny, Nx = field_focal.shape
    px_focal_um = focal_pixel_size_um(
        wavelength_um=wavelength_um,
        focal_length_um=FOCAL_LENGTH_UM,   # constant in your script
        slm_pixel_size_um=PIXEL_SIZE_UM,   # SLM pitch
        N_pixels=A_in.shape[1]             # pre-downsample FFT length
    )
    pixel_size_um = px_focal_um * downsample

    # Helper
    def norm01(I):
        I = np.asarray(I, dtype=np.float64)
        return (I - I.min()) / (I.max() - I.min() + 1e-12)

    # ---- Build a single bbox that covers ALL tweezers ----
    I0 = norm01(np.abs(field_focal)**2)
    I_ref = I0.copy()
    if z_range_um > 0:
        for z_um in (-0.5 * z_range_um, +0.5 * z_range_um):
            Iz = norm01(np.abs(fresnel_propagate(field_focal, z_um, wavelength_um, pixel_size_um))**2)
            I_ref = np.maximum(I_ref, Iz)
            del Iz

    thr = 0.25 * float(I_ref.max())  # inclusive threshold
    ys, xs = np.where(I_ref > thr)

    if ys.size == 0:
        cy, cx = Ny // 2, Nx // 2
        half = min(Ny, Nx) // 2 - 2
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
    else:
        pad = max(20, int(0.05 * min(Ny, Nx)))
        y0, y1 = max(0, ys.min() - pad), min(Ny, ys.max() + pad + 1)
        x0, x1 = max(0, xs.min() - pad), min(Nx, xs.max() + pad + 1)
        # Make square to comfortably include full lattice
        hbox, wbox = (y1 - y0), (x1 - x0)
        side = max(hbox, wbox)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        y0 = max(0, cy - side // 2); y1 = min(Ny, y0 + side)
        x0 = max(0, cx - side // 2); x1 = min(Nx, x0 + side)

    # -------------------------------
    # 1) Multiple z-planes (consistent bbox)
    # -------------------------------
    print("1. Multiple z-planes...")
    z_planes = np.linspace(-z_range_um, z_range_um, n_z_planes)
    fig1, axes1 = plt.subplots(1, n_z_planes, figsize=(4 * n_z_planes, 4))
    if n_z_planes == 1:
        axes1 = [axes1]

    for idx, z_um in enumerate(z_planes):
        field_z = fresnel_propagate(field_focal, z_um, wavelength_um, pixel_size_um)
        I_z = norm01(np.abs(field_z)**2)
        I_zoom = I_z[y0:y1, x0:x1]
        axes1[idx].imshow(I_zoom, origin='lower', vmin=0, vmax=1)
        axes1[idx].set_title(f'z = {z_um:+.1f} µm')
        axes1[idx].axis('off')
        del field_z, I_z, I_zoom

    fig1.suptitle(f'Z-Planes (Tilt: {expected_tilt:.1f}°)')
    plt.tight_layout()

    # -------------------------------
    # 2) X–Z cross-section (full width)
    # -------------------------------
    print("2. X-Z cross-section...")
    z_positions = np.linspace(-z_range_um, z_range_um, n_z_steps)
    center_y = field_focal.shape[0] // 2
    xz_intensity = np.zeros((n_z_steps, field_focal.shape[1]), dtype=np.float32)

    for i, z_um in enumerate(z_positions):
        if i % 5 == 0:
            print(f"   {i+1}/{n_z_steps}", end='\r')
        field_z = fresnel_propagate(field_focal, z_um, wavelength_um, pixel_size_um)
        xz_intensity[i, :] = np.abs(field_z[center_y, :])**2
        del field_z
    print()
    xz_intensity = norm01(xz_intensity)

    # Use µm on X axis (easier, unit-consistent slope)
    x_um = (np.arange(field_focal.shape[1]) - field_focal.shape[1]/2) * pixel_size_um

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    extent = [x_um[0], x_um[-1], z_positions.min(), z_positions.max()]
    ax2.imshow(xz_intensity, aspect='auto', origin='lower', extent=extent)

    # Expected tilt line across cropped span in µm
    if expected_tilt != 0:
        theta = np.deg2rad(expected_tilt)
        x_span_um = (x1 - x0) * pixel_size_um / 2.0
        x_center_um = 0.0
        x_line_um = np.array([-x_span_um, +x_span_um]) + x_center_um
        z_line_um = x_line_um * np.tan(theta)
        ax2.plot(x_line_um, z_line_um, linewidth=2.0, label=f'Expected {expected_tilt:.1f}°')

    # ---- Measured tilt (fit over cropped columns; FIXED the slice bug) ----
    z_best = z_positions[np.argmax(xz_intensity, axis=0)]      # length = Nx (downsampled)
    col_idx = np.arange(x0, x1)                                # columns we want to fit over
    x_fit_um = (col_idx - field_focal.shape[1]/2) * pixel_size_um
    z_fit = z_best[x0:x1]
    if x_fit_um.size >= 2:
        coeff = np.polyfit(x_fit_um - x_fit_um.mean(), z_fit, 1)
        slope = coeff[0]  # µm z per µm x
        measured_angle = np.rad2deg(np.arctan(slope))
        ax2.plot(x_fit_um, np.polyval(coeff, x_fit_um - x_fit_um.mean()),
                 linewidth=2, label=f'Measured {measured_angle:.2f}°')

    ax2.legend()
    ax2.set_xlabel('x [µm]')
    ax2.set_ylabel('z [µm]')
    ax2.set_title('X–Z Cross-Section')
    plt.colorbar(ax2.images[0], ax=ax2)
    plt.tight_layout()

    # Cleanup
    del field_focal, xz_intensity, z_positions, I0, I_ref
    gc.collect()

    return fig1, fig2

def main():
    print("="*60)
    print("MEMORY-SAFE MULTI-PLANE GS")
    print("="*60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup SLM
    print("\n--- Setting up SLM ---")
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, odd_tw=1, box1=1)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

    # Assign planes
    print(f"\n--- Assigning planes {TILT_ANGLE_X}---")
    slm.assign_planes_from_tilt(
        tilt_x_deg=TILT_ANGLE_X,
        n_planes=N_Z_PLANES
    )

    # Run GS
    print("\n--- Running multi-plane GS v3 ---")
    # slm.run_gs(iterations=ITERATIONS, Gg=GG)
    slm.run_gs_multiplane_v3(iterations=ITERATIONS, Gg=GG, verbose=True)

    # Save results
    print("\n--- Saving results ---")
    label = f"_{N_HORIZ}x{N_VERT}_10deg_v3"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"✓ Saved: {bundle.file}")

    # CRITICAL: Copy arrays to local vars BEFORE clearing
    phase_copy = slm.phase_mask.copy()
    A_in_copy = slm.A_in.copy()

    # CRITICAL: Clear SLM's large arrays
    print("\n--- Clearing SLM memory ---")
    slm.A_in = None
    slm.A_target = None
    slm.pad = None
    slm.psi0 = None
    slm._pupil_X2 = None
    slm._pupil_Y2 = None
    slm._pupil_XY = None
    slm._phi_planes = None

    # Visualize (using copies)
    print("\n--- Visualizing results ---")
    fig1, fig2 = visualize_results_memory_safe(
        phase=phase_copy,
        A_in=A_in_copy,
        wavelength_um=WAVELENGTH_UM,
        pixel_size_um=slm.pixel_size_um,
        expected_tilt=TILT_ANGLE_X,
        z_range_um=max(slm._z_per_spot),
        n_z_planes=N_Z_PLANES,
        n_z_steps=N_Z_STEPS,
        downsample=DOWNSAMPLE_XZ
    )

    del slm
    gc.collect()
    print("✓ SLM memory cleared")


    # Save figures
    base_name = OUT_DIR / f"tilt_{TILT_ANGLE_X:.0f}deg_v3"
    fig1.savefig(f"{base_name}_planes.png", dpi=DPI)
    fig2.savefig(f"{base_name}_xz.png", dpi=DPI)
    print(f"✓ Saved figures to {base_name}_*.png")

    # CRITICAL: Close figures and clear arrays
    plt.close('all')
    del fig1, fig2, phase_copy, A_in_copy
    gc.collect()

    print("\n" + "="*60)
    print("DONE!")
    print("✓ All memory cleaned up")
    print("="*60)


if __name__ == "__main__":
    main()

    # Final cleanup
    plt.close('all')
    gc.collect()
    print("\n🧹 Final cleanup complete")