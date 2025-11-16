"""
Visualization-only script: Load saved pickle and experiment with different plotting strategies.
No GS computation - just visualization!
"""

import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# ================================ CONFIG ================================

# Path to saved pickle file
PICKLE_PATH = "slm_output_paraxial/251106-110355_f200000um_sp30.0um_planes5_tilt_5_tw_20/_20x20_tol5.0e-05_v3_20251106_110448.pkl"

# Optics parameters (must match the run that created the pickle!)
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689
PIXEL_UM = 8.0  # SLM pixel size

# Visualization parameters
N_Z_STEPS = 61  # More z-steps for smoother visualization
Y_BAND_HALFWIDTH_PX = 5

# Tweezer info (from the run)
TILT_ANGLE_X = 5
SPACING_UM = 30.0
N_HORIZ = 20
N_VERT = 20

# Multiple visualization strategies to try
VIZ_STRATEGIES = [
    {"name": "strategy1_gamma0.3", "gamma": 0.3, "log_scale": False, "percentile_clip": 99.5},
    {"name": "strategy2_gamma0.2", "gamma": 0.2, "log_scale": False, "percentile_clip": 99.0},
    {"name": "strategy3_log", "gamma": 1.0, "log_scale": True, "percentile_clip": 99.5},
    {"name": "strategy4_gamma0.5_clip99", "gamma": 0.5, "log_scale": False, "percentile_clip": 99.0},
    {"name": "strategy5_sqrt", "gamma": 0.5, "log_scale": False, "percentile_clip": 99.9},
]

# ================================ LOAD PICKLE ================================

def load_pickle(pkl_path: str):
    """Load the saved ResultBundle from pickle."""
    with open(pkl_path, 'rb') as f:
        bundle = pickle.load(f)
    print(f"✓ Loaded pickle: {pkl_path}")
    print(f"  Phase mask shape: {bundle.phase_mask.shape}")
    print(f"  A_in shape: {bundle.A_in.shape}")
    print(f"  Number of tweezers: {len(bundle.tweezlist)}")
    return bundle


# ================================ FOCAL PLANE PROPAGATION ================================

def compute_focal_plane_intensity(
    phase: np.ndarray,
    A_in: np.ndarray,
    z_um: float,
    wavelength_um: float,
    pixel_size_slm_um: float,
    focal_length_um: float
) -> np.ndarray:
    """Compute intensity at focal plane with defocus z_um using paraxial model."""
    # Build full-size phase on SLM pupil
    H, W = A_in.shape
    h, w = phase.shape
    psi_full = np.zeros((H, W), dtype=np.float32)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    psi_full[y0:y0 + h, x0:x0 + w] = phase

    # Complex pupil field
    A_pupil = (A_in * np.exp(1j * psi_full)).astype(np.complex128)

    # Pupil coordinates (µm)
    yy = (np.arange(H) - H / 2) * pixel_size_slm_um
    xx = (np.arange(W) - W / 2) * pixel_size_slm_um
    X, Y = np.meshgrid(xx, yy)
    R2 = X**2 + Y**2

    # Paraxial defocus formula
    lam = float(wavelength_um)
    f = float(focal_length_um)
    k = 2.0 * np.pi / lam
    # Paraxial defocus: positive z needs positive (diverging) phase
    phase_defocus = +(k / (2.0 * f * f)) * z_um * R2

    # Propagate to focal plane
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I = (A_out.conj() * A_out).real

    return I


# ================================ X-Z SLICE GENERATION ================================

def build_xz_slice(
    phase: np.ndarray,
    A_in: np.ndarray,
    z_array: np.ndarray,
    crop_bounds: tuple,
    wavelength_um: float,
    pixel_size_slm_um: float,
    focal_length_um: float,
    y_band_halfwidth_px: int = 5
):
    """Build X-Z intensity slice."""
    r1, r2, c1, c2 = crop_bounds
    H = A_in.shape[0]

    # Y-band center at ROI center
    y_center = (r1 + r2) // 2
    yL = max(0, y_center - y_band_halfwidth_px)
    yR = min(H, y_center + y_band_halfwidth_px + 1)

    ncols = c2 - c1
    xz_raw = np.zeros((len(z_array), ncols), dtype=np.float64)

    print(f"Building X-Z slice with {len(z_array)} z-steps...")
    for zi, z_um_val in enumerate(z_array):
        if zi % 10 == 0:
            print(f"  Step {zi+1}/{len(z_array)}")

        I = compute_focal_plane_intensity(
            phase, A_in, float(z_um_val),
            wavelength_um, pixel_size_slm_um, focal_length_um
        )

        # Extract y-band and take max across y
        band = I[yL:yR, c1:c2]
        prof = band.max(axis=0)
        xz_raw[zi, :] = prof

    print("✓ X-Z slice complete")
    return xz_raw


# ================================ AUTO-ROI ================================

def auto_detect_roi(I_focal, px_focal_um, roi_thresh_p=99.5, roi_pad_um=200.0):
    """Automatically detect ROI around bright tweezers."""
    H, W = I_focal.shape
    thr = np.percentile(I_focal, roi_thresh_p)
    mask = I_focal > thr

    if mask.sum() < 20:
        hw = 50
        r_mid, c_mid = H // 2, W // 2
        return (max(0, r_mid - hw), min(H, r_mid + hw),
                max(0, c_mid - hw), min(W, c_mid + hw))

    ys, xs = np.where(mask)
    r1_raw, r2_raw = ys.min(), ys.max() + 1
    c1_raw, c2_raw = xs.min(), xs.max() + 1

    pad_px = int(np.round(roi_pad_um / max(px_focal_um, 1e-12)))
    r1 = max(0, r1_raw - pad_px)
    r2 = min(H, r2_raw + pad_px)
    c1 = max(0, c1_raw - pad_px)
    c2 = min(W, c2_raw + pad_px)

    return r1, r2, c1, c2


# ================================ PLOTTING STRATEGIES ================================

def plot_xz_with_strategy(
    xz_intensity: np.ndarray,
    z_array: np.ndarray,
    x_um_crop: np.ndarray,
    strategy: dict,
    tilt_deg: float,
    target_xy_um: np.ndarray,
    z_per_spot: np.ndarray,
    out_path: Path
):
    """Plot X-Z slice with a specific visualization strategy."""

    # Apply strategy transformations
    xz_norm = xz_intensity / (xz_intensity.max() + 1e-12)

    # Percentile clipping
    vmax_val = np.percentile(xz_norm, strategy["percentile_clip"])
    xz_clip = np.clip(xz_norm, 0, vmax_val) / vmax_val

    # Log scale
    if strategy["log_scale"]:
        xz_display = np.log10(xz_clip + 1e-3)
        xz_display = (xz_display - xz_display.min()) / (xz_display.max() - xz_display.min() + 1e-12)
    else:
        xz_display = xz_clip

    # Gamma correction
    gamma = strategy["gamma"]
    xz_display = np.power(xz_display, gamma)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    extent = [x_um_crop[0], x_um_crop[-1], z_array.min(), z_array.max()]
    im = ax.imshow(xz_display, aspect='auto', origin='lower',
                   extent=extent, vmin=0, vmax=1, cmap='hot')

    ax.set_xlabel('x [µm]', fontsize=14)
    ax.set_ylabel('z [µm]', fontsize=14)

    title_parts = [
        f"Strategy: {strategy['name']}",
        f"γ={gamma:.2f}",
        f"clip={strategy['percentile_clip']:.1f}%",
    ]
    if strategy["log_scale"]:
        title_parts.append("log-scale")
    ax.set_title(" | ".join(title_parts), fontsize=13, fontweight='bold')

    # Overlay lines
    ax.axhline(y=0.0, color='cyan', linestyle='--', alpha=0.8, linewidth=2,
               label='z=0 focal plane')

    if target_xy_um is not None and z_per_spot is not None:
        # Theoretical tilt line
        x_min = float(np.min(target_xy_um[:, 0]))
        x_max = float(np.max(target_xy_um[:, 0]))
        x_line = np.array([x_min, x_max])
        z_line = np.tan(np.deg2rad(tilt_deg)) * x_line

        ax.plot(x_line, z_line, color='magenta', linewidth=2.5, alpha=0.7,
                linestyle='--', label=f'Ideal tilt ({tilt_deg:.1f}°)')

        # Actual tweezers (middle row)
        y_coords = target_xy_um[:, 1]
        mid_row_mask = np.abs(y_coords) < 50.0

        if mid_row_mask.sum() > 0:
            x_tweezers = target_xy_um[mid_row_mask, 0]
            z_tweezers = z_per_spot[mid_row_mask]
            sort_idx = np.argsort(x_tweezers)
            x_tweezers = x_tweezers[sort_idx]
            z_tweezers = z_tweezers[sort_idx]

            ax.scatter(x_tweezers, z_tweezers, c='yellow', s=100,
                       marker='o', edgecolors='orange', linewidths=2,
                       label='Actual tweezers (y≈0)', zorder=5)
            ax.plot(x_tweezers, z_tweezers, 'orange', linewidth=2,
                    alpha=0.7, zorder=4)

    ax.legend(loc='upper right', fontsize=11)
    plt.colorbar(im, ax=ax, label='Intensity (processed)')
    plt.tight_layout()

    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"  ✓ Saved: {out_path.name}")


# ================================ MAIN ================================

def main():
    print("="*70)
    print("VISUALIZATION FROM PICKLE - Multiple Strategies")
    print("="*70)

    # Load pickle
    pkl_path = Path(PICKLE_PATH)
    if not pkl_path.exists():
        print(f"ERROR: Pickle file not found: {pkl_path}")
        print("Please update PICKLE_PATH in the script")
        return

    bundle = load_pickle(pkl_path)

    # Setup coordinates
    H, W = bundle.A_in.shape
    px_focal_um = (WAVELENGTH_UM * FOCAL_LENGTH_UM) / (W * PIXEL_UM)
    x_um = (np.arange(W) - W / 2) * px_focal_um
    y_um = (np.arange(H) - H / 2) * px_focal_um

    print(f"\n  Focal plane pixel size: {px_focal_um:.2f} µm/pixel")

    # Auto-detect ROI
    I0 = compute_focal_plane_intensity(
        bundle.phase_mask, bundle.A_in, 0.0,
        WAVELENGTH_UM, PIXEL_UM, FOCAL_LENGTH_UM
    )
    crop_bounds = auto_detect_roi(I0, px_focal_um, roi_thresh_p=99.5, roi_pad_um=200.0)
    r1, r2, c1, c2 = crop_bounds
    x_um_crop = x_um[c1:c2]

    x_extent = x_um[c2-1] - x_um[c1]
    print(f"  ROI x-extent: {x_extent:.0f} µm ({c2-c1} pixels)")

    # Compute tweezer positions (reconstruct from grid)
    x_um_axis = (np.arange(N_HORIZ) - (N_HORIZ - 1) / 2.0) * SPACING_UM
    y_um_axis = (np.arange(N_VERT) - (N_VERT - 1) / 2.0) * SPACING_UM
    x_um_list = np.tile(x_um_axis, N_VERT)
    y_um_list = np.repeat(y_um_axis, N_HORIZ)
    target_xy_um = np.stack([x_um_list, y_um_list], axis=1)

    # Compute z positions from tilt
    tx = np.tan(np.deg2rad(TILT_ANGLE_X))
    z_per_spot = tx * target_xy_um[:, 0]

    z_min = float(np.min(z_per_spot))
    z_max = float(np.max(z_per_spot))
    print(f"  Tweezer z-range: [{z_min:.1f}, {z_max:.1f}] µm")

    # Build z-array for X-Z slice
    z_span = z_max - z_min
    z_pad = z_span * 1.0  # 100% padding
    z_array = np.linspace(z_min - z_pad, z_max + z_pad, N_Z_STEPS)
    print(f"  X-Z plot z-range: [{z_array.min():.1f}, {z_array.max():.1f}] µm")

    # Build X-Z slice (this takes time, but only once!)
    xz_raw = build_xz_slice(
        bundle.phase_mask, bundle.A_in, z_array, crop_bounds,
        WAVELENGTH_UM, PIXEL_UM, FOCAL_LENGTH_UM,
        Y_BAND_HALFWIDTH_PX
    )

    # Try all visualization strategies
    out_dir = pkl_path.parent / "viz_strategies"
    out_dir.mkdir(exist_ok=True)

    print(f"\n--- Generating {len(VIZ_STRATEGIES)} visualization strategies ---")
    for strategy in VIZ_STRATEGIES:
        out_path = out_dir / f"{strategy['name']}_xz.png"
        plot_xz_with_strategy(
            xz_raw, z_array, x_um_crop, strategy,
            TILT_ANGLE_X, target_xy_um, z_per_spot,
            out_path
        )

    print(f"\n{'='*70}")
    print(f"✓ DONE! Check output directory:")
    print(f"  {out_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
