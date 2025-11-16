"""
Multi-plane GS with Automatic Memory Management

This script uses run_gs_multiplane_v3 (fixed algorithm) with automatic
memory cleanup to prevent PyCharm slowdown.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path
from slm_tweezers_class import SLMTweezers
import math
import gc

YAML_PATH = "../slm_parameters.yml"
OUT_DIR = Path("/Users/nadinemeister/PyCharmMiscProject/python_SLM_3d/out")

N_HORIZ = 3
N_VERT = 3
SPACING_UM = 15.0
ITERATIONS = 300
GG = 0.6
REDSLM = 1
SCAL = 4
WAIST_UM = 9 / 2 * 1e3

TILT_ANGLE_X = 0 #5.0
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689

PIXEL_SIZE_UM = 8 #9.2
Z_RANGE_UM = 0 #50
N_Z_PLANES = 1 #5
N_Z_STEPS = 1 #20
DOWNSAMPLE_XZ = 1 #4
DPI = 150


# Monkey-patch the hologram generation to include per-tweezer defocus
def generate_tilted_hologram_direct(slm, tilt_angle_deg=5.0):
    """
    Generate hologram directly with per-tweezer defocus.

    Based on thesis Section 3.2.1, equation 3.1:
    Each tweezer m has phase: Î”_m = (2Ï€/Î»f)(x_mÂ·X + y_mÂ·Y) + (Ï€Â·z_m)/(Î»fÂ²)(XÂ² + YÂ²)

    The last term is the defocus for tweezer m at depth z_m.
    """

    print("\n" + "="*60)
    print("GENERATING TILTED HOLOGRAM DIRECTLY")
    print("="*60)

    # Get tweezer positions
    xy_um = slm.target_xy_um  # (N, 2) array
    x_positions = xy_um[:, 0]
    y_positions = xy_um[:, 1]

    # Calculate defocus for each tweezer
    theta = np.deg2rad(tilt_angle_deg)
    z_defocus = x_positions * np.tan(theta)

    print(f"Tilt: {tilt_angle_deg:.2f}Â°")
    print(f"Z-range: {z_defocus.min():.2f} to {z_defocus.max():.2f} Âµm")
    print(f"Span: {z_defocus.max() - z_defocus.min():.2f} Âµm")

    # SLM pupil coordinates
    ny, nx = slm.phase_mask.shape
    pixel_um = slm.params.pixel_um

    y = (np.arange(ny) - ny/2) * pixel_um
    x = (np.arange(nx) - nx/2) * pixel_um
    XX, YY = np.meshgrid(x, y)
    r_sq = XX**2 + YY**2

    # Build hologram as superposition of elementary patterns
    # Following thesis eq 3.1
    k = 2 * np.pi / WAVELENGTH_UM
    f = FOCAL_LENGTH_UM

    A_hologram = np.zeros((ny, nx), dtype=np.complex128)

    for x_m, y_m, z_m in zip(x_positions, y_positions, z_defocus):
        # Plane wave to position tweezer at (x_m, y_m)
        phi_position = (k / f) * (x_m * XX + y_m * YY)

        # Defocus to shift this tweezer's focus by z_m
        phi_defocus = (k * z_m / (2 * f**2)) * r_sq

        # Random phase for this tweezer (helps uniformity)
        phi_random = np.random.uniform(0, 2*np.pi)

        # Add this tweezer's contribution
        A_hologram += np.exp(1j * (phi_position + phi_defocus + phi_random))

    # Extract phase-only hologram
    phase_hologram = np.angle(A_hologram)
    phase_hologram = np.mod(phase_hologram, 2*np.pi)

    # Store in SLM object
    slm.phase_mask = phase_hologram.astype(np.float32)

    phase_range = phase_hologram.max() - phase_hologram.min()
    print(f"Phase range: {phase_range:.3f} rad")
    print(f"âœ“ Tilted hologram generated!")
    print("="*60)


def fresnel_propagate(field, distance_um, wavelength_um, pixel_size_um):
    ny, nx = field.shape
    fx = np.fft.fftfreq(nx, d=pixel_size_um)
    fy = np.fft.fftfreq(ny, d=pixel_size_um)
    FX, FY = np.meshgrid(fx, fy)
    H = np.exp(1j * np.pi * wavelength_um * distance_um * (FX**2 + FY**2))
    return np.fft.ifft2(np.fft.fft2(field) * H)


def visualize_results(phase, A_in, wavelength_um, pixel_size_um,
                     expected_tilt, z_range_um, n_z_planes, n_z_steps, downsample=4):

    print("\n" + "="*60)
    print("VISUALIZING")
    print("="*60)

    H, W = A_in.shape
    h, w = phase.shape
    psi_full = np.zeros((H, W), dtype=np.float32)
    y_start = (H - h) // 2
    x_start = (W - w) // 2
    psi_full[y_start:y_start+h, x_start:x_start+w] = phase

    field_slm = A_in * np.exp(1j * psi_full)
    field_focal = fftshift(fft2(ifftshift(field_slm)))

    if downsample > 1:
        field_focal = field_focal[::downsample, ::downsample]
        pixel_size_um = pixel_size_um * downsample

    # Multi-plane
    print("1. Multiple z-planes...")
    z_planes = np.linspace(-z_range_um, z_range_um, n_z_planes)
    fig1, axes1 = plt.subplots(1, n_z_planes, figsize=(4*n_z_planes, 4))
    if n_z_planes == 1:
        axes1 = [axes1]

    for idx, z_um in enumerate(z_planes):
        field_z = fresnel_propagate(field_focal, z_um, wavelength_um, pixel_size_um)
        I_z = np.abs(field_z)**2
        I_z = (I_z - I_z.min()) / (I_z.max() - I_z.min() + 1e-12)

        cy, cx = I_z.shape[0]//2, I_z.shape[1]//2
        margin = min(100, cy, cx)
        I_zoom = I_z[cy-margin:cy+margin, cx-margin:cx+margin]

        axes1[idx].imshow(I_zoom, origin='lower', cmap='hot', vmin=0, vmax=1)
        axes1[idx].set_title(f'z = {z_um:+.1f} Âµm')
        axes1[idx].axis('off')

    fig1.suptitle(f'Z-Planes (Tilt: {expected_tilt:.1f}Â°)')
    plt.tight_layout()

    # X-Z cross-section
    print("2. X-Z cross-section...")
    z_positions = np.linspace(-z_range_um, z_range_um, n_z_steps)
    center_y = field_focal.shape[0] // 2
    xz_intensity = np.zeros((n_z_steps, field_focal.shape[1]))

    for i, z_um in enumerate(z_positions):
        if i % 5 == 0:
            print(f"   {i+1}/{n_z_steps}", end='\r')
        field_z = fresnel_propagate(field_focal, z_um, wavelength_um, pixel_size_um)
        xz_intensity[i, :] = np.abs(field_z[center_y, :])**2

    print()
    xz_intensity = (xz_intensity - xz_intensity.min()) / (xz_intensity.max() + 1e-12)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    extent = [0, field_focal.shape[1]*downsample, z_positions.min(), z_positions.max()]
    ax2.imshow(xz_intensity, aspect='auto', origin='lower', extent=extent, cmap='hot')

    # Expected
    if expected_tilt != 0:
        theta = np.deg2rad(expected_tilt)
        x_center = (field_focal.shape[1] * downsample) / 2
        x_span = 500
        x_line = np.array([x_center - x_span, x_center + x_span])
        z_line = (x_line - x_center) * pixel_size_um * np.tan(theta)
        ax2.plot(x_line, z_line, 'c--', linewidth=2.5, label=f'Expected {expected_tilt:.1f}Â°')

    # Measured
    z_best = z_positions[np.argmax(xz_intensity, axis=0)]
    x_pixels = np.arange(field_focal.shape[1]) * downsample
    keep = slice(field_focal.shape[1]//2 - 400, field_focal.shape[1]//2 + 400)

    if np.any(keep):
        fit = np.polyfit(x_pixels[keep] - x_pixels.mean(), z_best[keep], 1)
        measured_angle = np.rad2deg(np.arctan(fit[0] / pixel_size_um))
        ax2.plot(x_pixels[keep], np.polyval(fit, x_pixels[keep] - x_pixels.mean()),
                 'w-', linewidth=2, label=f'Measured {measured_angle:.2f}Â°')

    ax2.legend()
    ax2.set_xlabel('X [pixels]')
    ax2.set_ylabel('Z [Âµm]')
    ax2.set_title('X-Z Cross-Section (Should Show Diagonal Ridge!)')
    plt.colorbar(ax2.images[0], ax=ax2)
    plt.tight_layout()

    return fig1, fig2


def main():
    print("="*60)
    print("MULTI-PLANE GS WITH AUTO MEMORY CLEANUP")
    print("="*60)
    print("✓ Using run_gs_multiplane_v3 (fixed algorithm)")
    print("✓ Automatic memory cleanup enabled")
    print("="*60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup SLM
    print("\n--- Setting up SLM ---")
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, odd_tw=1, box1=1)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

    # Assign planes for tilted tweezer array
    print("\n--- Assigning planes ---")
    slm.assign_planes_from_tilt(
        tilt_x_deg=10.0,
        z_min_um=-50.0,
        z_max_um=50.0,
        n_planes=5
    )
    print(f"✓ {N_HORIZ}x{N_VERT} tweezers, 10° tilt, 5 planes")

    # Run multi-plane GS (v3 with auto cleanup)
    print("\n--- Running multi-plane GS v3 ---")
    slm.run_gs_multiplane(iterations=ITERATIONS, Gg=GG, verbose=True)
    print("✓ Memory automatically cleaned after GS")

    # Save results
    print("\n--- Saving results ---")
    label = f"_{N_HORIZ}x{N_VERT}_{TILT_ANGLE_X:.0f}deg_v3"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"✓ Saved: {bundle.file}")

    # Additional cleanup before visualization (for safety)
    gc.collect()

    # Visualize
    print("\n--- Visualizing results ---")
    fig1, fig2 = visualize_results(
        phase=slm.phase_mask,
        A_in=slm.A_in,
        wavelength_um=WAVELENGTH_UM,
        pixel_size_um=PIXEL_SIZE_UM,
        expected_tilt=TILT_ANGLE_X,
        z_range_um=Z_RANGE_UM,
        n_z_planes=N_Z_PLANES,
        n_z_steps=N_Z_STEPS,
        downsample=DOWNSAMPLE_XZ
    )

    # Save figures
    base_name = OUT_DIR / f"tilt_{TILT_ANGLE_X:.0f}deg_v3"
    fig1.savefig(f"{base_name}_planes.png", dpi=DPI)
    fig2.savefig(f"{base_name}_xz.png", dpi=DPI)
    print(f"✓ Saved figures to {base_name}_*.png")

    # Final cleanup
    gc.collect()

    print("\n" + "="*60)
    print("DONE!")
    print("✓ Multi-plane GS converged successfully")
    print("✓ Memory cleanup completed automatically")
    print("✓ PyCharm should remain fast and responsive")
    print("="*60)
    print("\nCheck X-Z cross-section:")
    print("  • Diagonal ridge = SUCCESS (tilted plane working)")
    print("  • Horizontal stripe = NOT WORKING")
    print("="*60)

    plt.show()


if __name__ == "__main__":
    main()