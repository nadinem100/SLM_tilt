# ============================== final_run_visualize.py ==============================
"""
Memory-safe GS runner with blazed BMP export and modular visualization.

Features:
- Configurable GS tolerance (TOL) for early stopping
- Multi-plane GS with tilt support
- Modular X-Z visualization using thin-lens approximation
- BMP export with blazed grating
"""

import os
os.environ['PYDEVD_USE_FAST_XML'] = '1'

from pathlib import Path
from datetime import datetime
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from slm_tweezers_class_WITH_AUTO_CLEANUP_paraxial import SLMTweezers

# ================================ CONFIG ================================
YAML_PATH = "../slm_parameters.yml"

# Grid configuration
N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30.0

# GS algorithm
ITERATIONS = 250  # Quick test
GG = 0.6
REDSLM = 1
SCAL = 4
WAIST_UM = 9 / 2 * 1e3  # microns
TOL = 5e-5  # Early-stop tolerance

# Optics
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689     # 689 nm

# Tilt configuration
TILT_ANGLE_X = 30  # degrees
N_Z_PLANES = 5

# Visualization
N_Z_STEPS = 41
Z_RANGE_UM = 500.0  # Range for visualization (fallback if no tweezers)
DPI = 300

# Auto-ROI settings
AUTO_ROI = True
ROI_THRESH_P = 99.5   # Percentile threshold for bright spots
ROI_MIN_AREA_PX = 20  # Minimum area to accept as valid ROI
ROI_PAD_UM = 200.0    # Symmetric pad around bbox (microns)

# X-Z slice settings
Y_BAND_HALFWIDTH_PX = 5     # Half-band width for X-Z aggregation
AGG_MODE = "max"             # "max" or "sum" across y-band
Z_EXTRA_FOR_XZ_UM = 0.3     # 30% padding beyond actual tweezer z-range (interpreted as fraction if <1)

BBOX = 2

# Improved visualization settings
GAMMA_CORRECTION = 0.4     # Gamma for brightening dim features (< 1 brightens, > 1 darkens)
VMAX_PERCENTILE = 99.5     # Clip bright outliers to improve contrast

# ================================ UTILITIES ================================

def make_out_dir() -> Path:
    """Build output directory with timestamp and configuration info."""
    ts = datetime.now().strftime("%y%m%d-%H%M%S")
    return Path(
        "slm_output_paraxial/"  # New directory!
        f"{ts}"
        f"_f{FOCAL_LENGTH_UM:.0f}um"
        f"_sp{SPACING_UM:.1f}um"
        f"_planes{N_Z_PLANES}"
        f"_tilt_{TILT_ANGLE_X}"
        f"_tw_{N_HORIZ}"
    )


def normalize01(arr):
    """Normalize array to [0, 1] range."""
    arr = np.asarray(arr, dtype=np.float64)
    arr -= arr.min()
    m = arr.max()
    if m > 0:
        arr /= m
    return arr


# ================================ BLAZED GRATING ================================

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


# ================================ FOCAL PLANE PROPAGATION ================================

def compute_focal_plane_intensity(
    phase: np.ndarray,
    A_in: np.ndarray,
    z_um: float,
    wavelength_um: float,
    pixel_size_slm_um: float,
    focal_length_um: float
) -> np.ndarray:
    """
    Compute intensity at focal plane with defocus z_um using thin-lens model.
    
    Args:
        phase: Phase mask on SLM (H_small, W_small)
        A_in: Input amplitude field (H_large, W_large) - padded canvas
        z_um: Defocus distance in microns
        wavelength_um: Wavelength in microns
        pixel_size_slm_um: SLM pixel size in microns
        focal_length_um: Focal length in microns
    
    Returns:
        Intensity array at focal plane (H_large, W_large)
    """
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
    
    # Optics parameters
    lam = float(wavelength_um)
    f = float(focal_length_um)
    k = 2.0 * np.pi / lam

    # PARAXIAL defocus formula: φ(X,Y) = +(k/(2f²)) * z * R²
    # SIGN: Positive z (focus farther) needs POSITIVE (diverging) phase
    # This matches the GS algorithm in slm_tweezers_class_paraxial
    phase_defocus = +(k / (2.0 * f * f)) * z_um * R2
    
    # Propagate to focal plane
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I = (A_out.conj() * A_out).real
    
    return I


# ================================ AUTO-ROI DETECTION ================================

def auto_detect_roi(
    I_focal: np.ndarray,
    px_focal_um: float,
    roi_thresh_p: float = 99.5,
    roi_min_area_px: int = 20,
    roi_pad_um: float = 200.0
) -> tuple[int, int, int, int]:
    """
    Automatically detect ROI around bright tweezers at focal plane.
    
    Args:
        I_focal: Intensity at focal plane
        px_focal_um: Pixel size at focal plane (microns)
        roi_thresh_p: Percentile threshold for bright-spot mask
        roi_min_area_px: Minimum area to accept as valid ROI
        roi_pad_um: Symmetric padding around bbox in microns
    
    Returns:
        (r1, r2, c1, c2): Row and column indices for cropping
    """
    H, W = I_focal.shape
    thr = np.percentile(I_focal, roi_thresh_p)
    mask = I_focal > thr
    
    # Fallback: small centered box if detection fails
    if mask.sum() < roi_min_area_px:
        hw = max(32, int(120.0 / max(px_focal_um, 1e-12)))
        r_mid, c_mid = H // 2, W // 2
        return (max(0, r_mid - hw), min(H, r_mid + hw),
                max(0, c_mid - hw), min(W, c_mid + hw))
    
    # Find bounding box of bright spots
    ys, xs = np.where(mask)
    r1_raw, r2_raw = ys.min(), ys.max() + 1
    c1_raw, c2_raw = xs.min(), xs.max() + 1
    
    # Add padding
    pad_px = int(np.round(roi_pad_um / max(px_focal_um, 1e-12)))
    r1 = max(0, r1_raw - pad_px)
    r2 = min(H, r2_raw + pad_px)
    c1 = max(0, c1_raw - pad_px)
    c2 = min(W, c2_raw + pad_px)
    
    # Ensure non-degenerate box
    if r2 - r1 < 4:
        r1, r2 = max(0, H // 2 - 4), min(H, H // 2 + 4)
    if c2 - c1 < 4:
        c1, c2 = max(0, W // 2 - 4), min(W, W // 2 + 4)
    
    return r1, r2, c1, c2


# ================================ X-Z SLICE GENERATION ================================

def build_xz_slice(
    phase: np.ndarray,
    A_in: np.ndarray,
    z_array: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    wavelength_um: float,
    pixel_size_slm_um: float,
    focal_length_um: float,
    y_band_halfwidth_px: int = 5,
    agg_mode: str = "max"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X-Z intensity slice by aggregating across y-band.
    
    Args:
        phase: Phase mask on SLM
        A_in: Input amplitude field
        z_array: Array of z positions (microns)
        crop_bounds: (r1, r2, c1, c2) for x-axis cropping
        wavelength_um: Wavelength
        pixel_size_slm_um: SLM pixel size
        focal_length_um: Focal length
        y_band_halfwidth_px: Half-width of y-band for aggregation
        agg_mode: "max" or "sum" across y-band
    
    Returns:
        (xz_intensity, plane_max): X-Z slice array and max per plane
    """
    r1, r2, c1, c2 = crop_bounds
    H = A_in.shape[0]
    
    # Y-band center at ROI center
    y_center = (r1 + r2) // 2
    yL = max(0, y_center - y_band_halfwidth_px)
    yR = min(H, y_center + y_band_halfwidth_px + 1)
    
    ncols = c2 - c1
    xz_raw = np.zeros((len(z_array), ncols), dtype=np.float64)
    plane_max = np.zeros(len(z_array), dtype=np.float64)
    
    for zi, z_um_val in enumerate(z_array):
        I = compute_focal_plane_intensity(
            phase, A_in, float(z_um_val),
            wavelength_um, pixel_size_slm_um, focal_length_um
        )
        
        # Extract y-band and aggregate
        band = I[yL:yR, c1:c2]
        if agg_mode.lower() == "sum":
            prof = band.sum(axis=0)
        else:
            prof = band.max(axis=0)
        
        xz_raw[zi, :] = prof
        plane_max[zi] = prof.max()
    
    return xz_raw, plane_max


# ================================ PLOTTING ================================

def plot_three_planes(
    I_triplet: list[np.ndarray],
    z_values: list[float],
    x_um: np.ndarray,
    y_um: np.ndarray,
    crop_bounds: tuple[int, int, int, int],
    normalize: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0
) -> plt.Figure:
    """
    Create figure with three focal plane intensity panels.
    
    Args:
        I_triplet: List of 3 intensity arrays (cropped)
        z_values: List of 3 z positions
        x_um, y_um: Full coordinate arrays
        crop_bounds: (r1, r2, c1, c2)
        normalize: Whether intensities are normalized
        vmin, vmax: Color scale limits
    
    Returns:
        Matplotlib figure
    """
    r1, r2, c1, c2 = crop_bounds
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, z_um_val, I_view in zip(axes, z_values, I_triplet):
        extent = [x_um[c1], x_um[c2 - 1], y_um[r1], y_um[r2 - 1]]
        im = ax.imshow(I_view, origin='lower', extent=extent,
                       vmin=vmin, vmax=vmax, cmap='hot')
        ax.set_title(f'z = {z_um_val:+.1f} µm', fontsize=14)
        ax.set_xlabel('x [µm]')
    
    axes[0].set_ylabel('y [µm]')
    
    for ax in axes:
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label='Normalized intensity' if normalize else 'Intensity (a.u.)')
    
    fig.suptitle('Focal Plane Intensity at Three Z Positions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_xz_slice(
    xz_intensity: np.ndarray,
    z_array: np.ndarray,
    x_um_crop: np.ndarray,
    tilt_deg: float,
    y_center_px: int,
    y_band_halfwidth_px: int,
    agg_mode: str,
    target_xy_um: np.ndarray | None = None,
    z_per_spot: np.ndarray | None = None,
    normalize: bool = True,
    vmin: float = 0.0,
    vmax: float = 1.0,
    gamma: float = 0.4
) -> plt.Figure:
    """
    Create X-Z slice figure with tilt overlay and improved visibility.

    Args:
        xz_intensity: X-Z intensity array (n_z, n_x)
        z_array: Z positions
        x_um_crop: X coordinates (cropped)
        tilt_deg: Tilt angle in degrees
        y_center_px: Y-band center pixel
        y_band_halfwidth_px: Y-band half-width
        agg_mode: Aggregation mode ("max" or "sum")
        target_xy_um: (K, 2) array of tweezer positions
        z_per_spot: (K,) array of z positions for each tweezer
        normalize: Whether intensity is normalized
        vmin, vmax: Color scale limits
        gamma: Gamma correction factor (< 1 brightens, > 1 darkens)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Apply gamma correction to improve visibility of dim features
    xz_display = np.power(np.clip(xz_intensity, 0, 1), gamma)

    extent = [x_um_crop[0], x_um_crop[-1], z_array.min(), z_array.max()]
    im = ax.imshow(xz_display, aspect='auto', origin='lower',
                   extent=extent, vmin=0, vmax=1, cmap='hot')

    ax.set_xlabel('x [µm]', fontsize=13)
    ax.set_ylabel('z [µm]', fontsize=13)
    ax.set_title(f'X–Z slice (y-band center={y_center_px}, ±{y_band_halfwidth_px}px, {agg_mode}, γ={gamma:.2f})',
                 fontsize=14, fontweight='bold')
    
    # Mark z=0 focal plane
    ax.axhline(y=0.0, color='cyan', linestyle='--', alpha=0.8, linewidth=2,
               label='z=0 focal plane')

    # Plot theoretical tilt line AND actual tweezer positions
    if abs(tilt_deg) > 1e-6 and target_xy_um is not None and z_per_spot is not None:
        # Theoretical tilt line (limited to tweezer extent)
        x_min = float(np.min(target_xy_um[:, 0]))
        x_max = float(np.max(target_xy_um[:, 0]))
        x_line = np.array([x_min, x_max])  # NO PADDING!
        z_line = np.tan(np.deg2rad(tilt_deg)) * x_line

        ax.plot(x_line, z_line, color='magenta', linewidth=2.5, alpha=0.7,
                linestyle='--', label=f'Ideal tilt ({tilt_deg:.1f}°)')

        # Actual tweezer positions (discretized by GS algorithm)
        # Filter to middle row (y ≈ 0) for clarity
        y_coords = target_xy_um[:, 1]
        y_tol = 50.0  # µm tolerance for "middle row"
        mid_row_mask = np.abs(y_coords) < y_tol

        if mid_row_mask.sum() > 0:
            x_tweezers = target_xy_um[mid_row_mask, 0]
            z_tweezers = z_per_spot[mid_row_mask]

            # Sort by x for clean plotting
            sort_idx = np.argsort(x_tweezers)
            x_tweezers = x_tweezers[sort_idx]
            z_tweezers = z_tweezers[sort_idx]

            # Plot actual positions as scatter + line
            ax.scatter(x_tweezers, z_tweezers, c='yellow', s=80,
                       marker='o', edgecolors='orange', linewidths=1.5,
                       label='Actual tweezers (y≈0)', zorder=5)
            ax.plot(x_tweezers, z_tweezers, 'orange', linewidth=1.5,
                    alpha=0.6, zorder=4)

    ax.legend(loc='upper right', fontsize=10)

    plt.colorbar(im, ax=ax,
                 label='Normalized intensity' if normalize else 'Intensity (a.u.)')
    plt.tight_layout()
    
    return fig


# ================================ MAIN VISUALIZATION ================================

def visualize_focal_planes(
    phase: np.ndarray,
    A_in: np.ndarray,
    wavelength_um: float,
    pixel_size_slm_um: float,
    focal_length_um: float,
    z_range_um: float,
    n_z_steps: int,
    tilt_deg: float,
    *,
    target_xy_um: np.ndarray | None = None,
    z_per_spot: np.ndarray | None = None,
    auto_roi: bool = True,
    roi_thresh_p: float = 99.5,
    roi_min_area_px: int = 20,
    roi_pad_um: float = 200.0,
    y_band_halfwidth_px: int = 5,
    agg_mode: str = "max",
    z_extra_for_xz_um: float = 50.0,
    normalize: bool = True
) -> tuple[plt.Figure, plt.Figure, int, dict]:
    """
    Main visualization function: creates 3-plane and X-Z slice figures.
    
    Args:
        phase: Phase mask on SLM
        A_in: Input amplitude field
        wavelength_um: Wavelength
        pixel_size_slm_um: SLM pixel size
        focal_length_um: Focal length
        z_range_um: Fallback z-range if no tweezer positions provided (µm)
        n_z_steps: Number of z steps
        tilt_deg: Tilt angle
        target_xy_um: (K, 2) array of tweezer (x, y) positions in microns
        z_per_spot: (K,) array of z positions for each tweezer in microns
        auto_roi: Enable auto-ROI detection
        roi_thresh_p: ROI detection threshold percentile
        roi_min_area_px: Minimum ROI area
        roi_pad_um: ROI padding
        y_band_halfwidth_px: Y-band half-width for X-Z
        agg_mode: "max" or "sum" for X-Z aggregation
        z_extra_for_xz_um: If <1, interpreted as fraction (e.g., 0.3 = 30% padding).
                          If >=1, interpreted as absolute padding in µm.
                          Applied to auto-calculated z-range from actual tweezers.
        normalize: Normalize intensities to [0,1]
    
    Returns:
        (fig_3planes, fig_xz, best_z_index, info_dict)
    """

    H, W = A_in.shape

    # Focal-plane pixel size
    px_focal_um = (wavelength_um * focal_length_um) / (W * pixel_size_slm_um)
    x_um = (np.arange(W) - W / 2) * px_focal_um
    y_um = (np.arange(H) - H / 2) * px_focal_um

    print(f"\n  Focal plane pixel size: {px_focal_um:.2f} µm/pixel")
    print(f"  Full FOV: {x_um.max() - x_um.min():.0f} × {y_um.max() - y_um.min():.0f} µm²")
    
    # Auto-detect ROI from z=0 plane
    I0 = compute_focal_plane_intensity(
        phase, A_in, 0.0, wavelength_um, pixel_size_slm_um, focal_length_um
    )
    
    if auto_roi:
        crop_bounds = auto_detect_roi(
            I0, px_focal_um, roi_thresh_p, roi_min_area_px, roi_pad_um
        )
    else:
        crop_bounds = (0, H, 0, W)

    r1, r2, c1, c2 = crop_bounds
    x_um_crop = x_um[c1:c2]

    x_roi_extent = x_um[c2-1] - x_um[c1]
    y_roi_extent = y_um[r2-1] - y_um[r1]
    print(f"  ROI extent: {x_roi_extent:.0f} × {y_roi_extent:.0f} µm²")
    print(f"  ROI pixels: {c2-c1} × {r2-r1} px")
    
    # Z grids - use ACTUAL tweezer positions if provided, else fallback
    if z_per_spot is not None and len(z_per_spot) > 0:
        z_min_actual = float(np.min(z_per_spot))
        z_max_actual = float(np.max(z_per_spot))
        z_span = z_max_actual - z_min_actual
        print(f"  Using actual tweezer z range: [{z_min_actual:.1f}, {z_max_actual:.1f}] µm (span={z_span:.1f} µm)")
        z_panel = np.array([z_min_actual, 0.0, z_max_actual])
        
        # Auto-scale X-Z range based on actual tweezer span
        # Add padding as a fraction of the span (z_extra_for_xz_um is now interpreted as fraction)
        z_padding_fraction = 1.0
        z_pad = z_span * z_padding_fraction
        z_xz_min = z_min_actual - z_pad
        z_xz_max = z_max_actual + z_pad
        print(f"  X-Z plot z-range: [{z_xz_min:.1f}, {z_xz_max:.1f}] µm ({z_padding_fraction*100:.0f}% padding)")
    else:
        print(f"  Using default z range: ±{z_range_um:.1f} µm")
        z_panel = np.linspace(-abs(z_range_um), abs(z_range_um), max(3, int(n_z_steps)))
        # Fallback: use old fixed range
        z_xz_min = -(abs(z_range_um) + abs(z_extra_for_xz_um))
        z_xz_max = +(abs(z_range_um) + abs(z_extra_for_xz_um))
    
    # X-Z z-array with more samples for smooth visualization
    z_xz = np.linspace(z_xz_min, z_xz_max, max(31, int(n_z_steps)))
    
    # Build X-Z slice
    xz_raw, plane_max = build_xz_slice(
        phase, A_in, z_xz, crop_bounds,
        wavelength_um, pixel_size_slm_um, focal_length_um,
        y_band_halfwidth_px, agg_mode
    )
    
    best_z_index = int(np.argmax(plane_max))
    global_max = plane_max.max()
    
    # Normalize if requested
    if normalize and global_max > 0:
        xz_disp = xz_raw / global_max
        vmin, vmax = 0.0, 1.0
    else:
        xz_disp = xz_raw
        vmin, vmax = 0.0, max(global_max, 1.0)
    
    # Compute 3-plane intensities
    z_triplet = [z_panel[0], 0.0, z_panel[-1]]
    I_triplet_cropped = []
    I_triplet_full = []
    
    for z_um_val in z_triplet:
        I = compute_focal_plane_intensity(
            phase, A_in, float(z_um_val),
            wavelength_um, pixel_size_slm_um, focal_length_um
        )
        I_triplet_full.append(I)
        I_view = I[r1:r2, c1:c2]
        if normalize and global_max > 0:
            I_view = I_view / global_max
        I_triplet_cropped.append(I_view)
    
    # Create figures
    fig_3planes = plot_three_planes(
        I_triplet_cropped, z_triplet, x_um, y_um,
        crop_bounds, normalize, vmin, vmax
    )
    
    y_center = (r1 + r2) // 2
    fig_xz = plot_xz_slice(
        xz_disp, z_xz, x_um_crop, tilt_deg,
        y_center, y_band_halfwidth_px, agg_mode,
        target_xy_um, z_per_spot,
        normalize, vmin, vmax,
        gamma=GAMMA_CORRECTION
    )
    
    # Info dict
    info_dict = {
        "z_min": I_triplet_full[0],
        "z0": I_triplet_full[1],
        "z_max": I_triplet_full[2],
        "roi": crop_bounds,
        "x_um": x_um,
        "y_um": y_um
    }
    
    return fig_3planes, fig_xz, best_z_index, info_dict


# ================================ MAIN EXECUTION ================================

def main():
    print("="*60)
    print("MEMORY-SAFE MULTI-PLANE GS + VISUALIZATION")
    print("="*60)
    
    OUT_DIR = make_out_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n--- Setting up SLM ---")
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, 
                        odd_tw=1, box1=BBOX)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)
    
    print(f"\n--- Assigning planes with {TILT_ANGLE_X}° tilt ---")
    slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=N_Z_PLANES)

    # DIAGNOSTIC: Print z-plane information
    if hasattr(slm, '_z_planes') and hasattr(slm, '_z_per_spot'):
        print(f"\n  DEBUG: Z-planes created: {slm._z_planes}")
        print(f"  DEBUG: Z per spot range: [{np.min(slm._z_per_spot):.2f}, {np.max(slm._z_per_spot):.2f}] µm")
        print(f"  DEBUG: Number of tweezers: {len(slm._z_per_spot)}")
    else:
        print("\n  WARNING: _z_planes or _z_per_spot not found in slm object!")
    
    print("\n--- Running multi-plane GS ---")
    slm.run_gs_multiplane_v3(iterations=ITERATIONS, Gg=GG, verbose=True, tol=TOL)
    
    print("\n--- Saving results ---")
    label = f"_{N_HORIZ}x{N_VERT}_tol{TOL:.1e}_v3"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"✓ Saved: {bundle.file}")
    
    # Snapshots for visualization
    phase_copy = slm.phase_mask.copy()
    A_in_copy = slm.A_in.copy()
    pixel_um = slm.params.pixel_um
    
    # Extract actual tweezer positions (critical for correct tilt line!)
    target_xy_um = slm.target_xy_um.copy() if hasattr(slm, 'target_xy_um') else None
    z_per_spot = slm._z_per_spot.copy() if hasattr(slm, '_z_per_spot') else None
    
    if z_per_spot is not None:
        print(f"\n✓ Using actual tweezer z positions:")
        print(f"  Range: [{np.min(z_per_spot):.1f}, {np.max(z_per_spot):.1f}] µm")
        print(f"  Mean: {np.mean(z_per_spot):.1f} µm")
    
    # ---------- BMP with blazed grating ----------
    fx, fy = 1.0 / 7.0, 0.0
    phase_blazed = add_blazed_grating(phase_copy, fx=fx, fy=fy)
    
    pkl_path = Path(bundle.file)
    stem = pkl_path.with_suffix("").name
    
    suffix = (
        f"_f{FOCAL_LENGTH_UM:.0f}um"
        f"_sp{SPACING_UM:.1f}um"
        f"_planes{N_Z_PLANES}"
        f"_tol{TOL:.1e}"
        f"_tilt{TILT_ANGLE_X:.0f}deg"
        f"_bbox{BBOX}"
    )
    
    out_bmp = pkl_path.parent / f"{stem}{suffix}_blazepd7.bmp"
    save_phase_bmp(phase_blazed, out_bmp)
    print(f"✓ Phase+blaze BMP: {out_bmp}")
    
    # Try to save to alternative location (may fail, that's OK)
    try:
        out2_bmp = pkl_path.parent / f"../../../phase_masks_tilt/{stem}{suffix}_blazepd7.bmp"
        out2_bmp.parent.mkdir(parents=True, exist_ok=True)
        save_phase_bmp(phase_blazed, out2_bmp)
        print(f"✓ Also saved to: {out2_bmp}")
    except Exception as e:
        print(f"  (Could not save to alternative location: {e})")
    
    # ---------- Visualization ----------
    print("\n--- Generating visualizations ---")
    
    fig_3planes, fig_xz, best_idx, I_dict = visualize_focal_planes(
        phase=phase_copy,
        A_in=A_in_copy,
        wavelength_um=WAVELENGTH_UM,
        pixel_size_slm_um=pixel_um,
        focal_length_um=FOCAL_LENGTH_UM,
        z_range_um=Z_RANGE_UM,
        n_z_steps=N_Z_STEPS,
        tilt_deg=TILT_ANGLE_X,
        target_xy_um=target_xy_um,
        z_per_spot=z_per_spot,
        auto_roi=AUTO_ROI,
        roi_thresh_p=ROI_THRESH_P,
        roi_min_area_px=ROI_MIN_AREA_PX,
        roi_pad_um=ROI_PAD_UM,
        y_band_halfwidth_px=Y_BAND_HALFWIDTH_PX,
        agg_mode=AGG_MODE,
        z_extra_for_xz_um=Z_EXTRA_FOR_XZ_UM,
        normalize=True
    )
    
    # Clean up SLM memory
    del slm
    gc.collect()
    print("✓ SLM memory cleared")
    
    # ---------- Save figures ----------
    base_name = OUT_DIR / f"{TILT_ANGLE_X:.0f}deg_tilt{suffix}"
    fig_3planes.savefig(f"{base_name}_3planes.png", dpi=DPI)
    fig_xz.savefig(f"{base_name}_xz.png", dpi=DPI)
    print(f"✓ Saved figures:")
    print(f"  - {base_name}_3planes.png")
    print(f"  - {base_name}_xz.png")
    
    # Cleanup
    plt.close('all')
    del fig_3planes, fig_xz, phase_copy, A_in_copy, phase_blazed
    gc.collect()
    
    print("\n" + "="*60)
    print("DONE!")
    print("✓ All memory cleaned up")
    print("="*60)


if __name__ == "__main__":
    main()
    plt.close('all')
    gc.collect()
    print("\n🧹 Final cleanup complete")