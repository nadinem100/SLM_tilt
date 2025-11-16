# ============================== final_run_visualize.py ==============================
# file: final_run_visualize.py
"""
Memory-safe GS runner with blazed BMP export.

Adds:
- Configurable GS tolerance (TOL) for early stopping.
- OUT_DIR encodes focal length, spacing, n_z_planes, and tilt.
- Pickle filename stem includes tolerance.
- PNG and BMP filenames append f/sp/planes/tol/tilt for traceability.
- NEW: Simple XÃ¢â‚¬â€œZ visualization using pure defocus on the pupil (quad phase -> FFT).
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

from slm_tweezers_class_WITH_AUTO_CLEANUP import SLMTweezers  # requires TOL support

# ----------------- CONFIG -----------------
YAML_PATH = "../slm_parameters.yml"

N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30.0
ITERATIONS = 100
GG = 0.6
REDSLM = 1
SCAL = 4
WAIST_UM = 9 / 2 * 1e3  # microns

FOCAL_LENGTH_UM = 7000.0   # 200 mm
WAVELENGTH_UM = 0.689        # 689 nm

DOWNSAMPLE_XZ = 1
DPI = 300

TILT_ANGLE_X = 5
N_Z_PLANES   = 5
N_Z_STEPS    = 21

Z_RANGE_UM   = 3000.0   # <<< try 3000Ã¢â‚¬â€œ5000 Ã‚Âµm to see the tweezer blur evolve


# Configurable GS early-stop tolerance
TOL = 5e-3 # if error_signal < TOL for 3 consecutive iters -> stop

# CONFIG knobs (put near your other CONFIG):
PAD_FFT   = 4          # oversample factor in FFT (4x Ã¢â€ â€™ 16x pixels per area)
CROP_PIX  = 128        # half-width crop around the brightest tweezer (in focal pixels)

# Add these configurable knobs near your config:
AUTO_ROI_3PLANES = True   # turn on auto-crop around tweezers
ROI_THRESH_P     = 99.5   # percentile threshold for bright-spot mask
ROI_MIN_AREA_PX  = 20     # min area to accept as valid ROI; else fallback
ROI_PAD_UM       = 200.0  # symmetric pad around bbox (physical units)

# --- Helper to compute bounding box from a z=0 plane ---
def _auto_roi_from_plane(I0: np.ndarray,
                         pad_um: float,
                         px_focal_um: float,
                         min_area_px: int,
                         thresh_p: float) -> tuple[int,int,int,int]:
    """
    Returns (r1, r2, c1, c2) cropping indices from a bright-spot mask on I0.
    Falls back to center crop if detection fails.
    """
    H, W = I0.shape
    thr = np.percentile(I0, thresh_p)
    mask = I0 > thr

    # fail-safes: remove tiny speckles
    if mask.sum() < min_area_px:
        # fallback: centered small box
        box_half_w = max(32, int(120.0 / px_focal_um))  # ~120 Âµm half-width fallback
        r_mid, c_mid = H // 2, W // 2
        r1 = max(0, r_mid - box_half_w)
        r2 = min(H, r_mid + box_half_w)
        c1 = max(0, c_mid - box_half_w)
        c2 = min(W, c_mid + box_half_w)
        return r1, r2, c1, c2

    ys, xs = np.where(mask)
    r1_raw, r2_raw = ys.min(), ys.max() + 1
    c1_raw, c2_raw = xs.min(), xs.max() + 1

    # pad in Âµm â†’ pixels
    pad_px = int(np.round(pad_um / max(px_focal_um, 1e-12)))
    r1 = max(0, r1_raw - pad_px)
    r2 = min(H, r2_raw + pad_px)
    c1 = max(0, c1_raw - pad_px)
    c2 = min(W, c2_raw + pad_px)

    # ensure non-degenerate box
    if r2 - r1 < 4 or c2 - c1 < 4:
        r1 = max(0, (H // 2) - 4); r2 = min(H, (H // 2) + 4)
        c1 = max(0, (W // 2) - 4); c2 = min(W, (W // 2) + 4)
    return r1, r2, c1, c2



def make_out_dir() -> Path:
    """
    Build OUT_DIR from current runtime globals so overrides (tilt, focal, etc.)
    are reflected per run. Called inside main().
    """
    ts = datetime.now().strftime("%y%m%d-%H%M%S")  # WHY: per-run timestamp
    return Path(
        "slm_output/"
        f"{ts}"
        f"_f{FOCAL_LENGTH_UM:.0f}um"
        f"_sp{SPACING_UM:.1f}um"
        f"_planes{N_Z_PLANES}"
        f"_tilt_{TILT_ANGLE_X}"
        f"_tw_{N_HORIZ}"
    )


# ----------------- helpers -----------------
def _pad_center(arr, pad_factor: int):
    """Center-pad to (pad_factor*H, pad_factor*W). WHY: oversample focal plane."""
    H, W = arr.shape
    Hp, Wp = pad_factor*H, pad_factor*W
    out = np.zeros((Hp, Wp), dtype=arr.dtype)
    y0 = (Hp - H)//2; x0 = (Wp - W)//2
    out[y0:y0+H, x0:x0+W] = arr
    return out

def focal_pixel_size_um(wavelength_um, focal_length_um, slm_pixel_size_um, N_pixels):
    # ÃŽÂ» f / (N * ÃŽâ€x_slm)
    return (wavelength_um * focal_length_um) / (N_pixels * slm_pixel_size_um)

def normalize01(arr):
    arr = np.asarray(arr, dtype=np.float64)
    arr -= arr.min()
    m = arr.max()
    if m > 0:
        arr /= m
    return arr

def build_pupil_coords_um(ny, nx, pixel_um):
    """Helper for old visualize_xz_simple - builds pupil coordinates in microns."""
    yy = (np.arange(ny) - ny/2) * pixel_um
    xx = (np.arange(nx) - nx/2) * pixel_um
    X, Y = np.meshgrid(xx, yy)
    return X, Y

def defocus_phase_quadratic(z_um, f_um, k_um, X, Y):
    """Helper for old visualize_xz_simple - quadratic defocus phase (geometric optics approximation)."""
    denom = max(1e-9, float(2.0 * f_um * (f_um - z_um)))
    quad = (X*X + Y*Y) / denom
    return np.exp(-1j * k_um * z_um * quad, dtype=np.complex128)

def propagate_angular_spectrum(A_field, z_um, wavelength_um, pixel_size_um):
    """
    Propagate complex field using Angular Spectrum method.
    
    This properly accounts for both phase curvature AND diffraction spreading.
    
    Args:
        A_field: Complex field at current plane (H, W)
        z_um: Propagation distance in microns (positive = away from pupil)
        wavelength_um: Wavelength in microns
        pixel_size_um: Pixel size at current plane
    
    Returns:
        Complex field at plane z_um away
    """
    from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
    
    H, W = A_field.shape
    k = 2 * np.pi / wavelength_um
    
    # Spatial frequency coordinates (cycles per micron)
    fy = fftfreq(H, d=pixel_size_um)
    fx = fftfreq(W, d=pixel_size_um)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    
    # Transverse wavevector components
    k_perp_sq = (2*np.pi*FX)**2 + (2*np.pi*FY)**2
    
    # Longitudinal wavevector: k_z = sqrt(k^2 - k_perp^2)
    # Use complex sqrt to handle evanescent waves (when k_perp > k)
    k_sq = k**2
    k_z = np.sqrt(k_sq - k_perp_sq + 0j, dtype=np.complex128)
    
    # Angular Spectrum transfer function: H(fx,fy) = exp(i * k_z * z)
    H_transfer = np.exp(1j * k_z * float(z_um), dtype=np.complex128)
    
    # Propagate: FFT -> multiply by transfer function -> IFFT
    spectrum = fft2(A_field)
    spectrum_prop = spectrum * H_transfer
    A_propagated = ifft2(spectrum_prop)
    
    return A_propagated

def visualize_xz_matlab_style(
    phase: np.ndarray,
    A_in: np.ndarray,
    wavelength_um: float,
    pixel_size_slm_um: float,
    focal_length_um: float,
    z_range_um: float,
    n_z_steps: int,
    tilt_deg: float,
    *,
    # Xâ€“Z construction
    agg_mode: str = "max",            # "max" or "sum" across the y-band
    y_band_halfwidth_px: int = 5,     # half-band for Xâ€“Z aggregation
    z_extra_for_xz_um: float = 50.0,  # Xâ€“Z spans (z_range_um + this)
    # Auto-ROI around tweezers at z=0
    auto_roi: bool = True,
    roi_thresh_p: float = 99.5,       # percentile for bright mask at z=0
    roi_min_area_px: int = 20,
    roi_pad_um: float = 200.0,        # physical pad around bbox (Âµm)
    # Intensity scaling
    normalize: bool = True            # if True, both figs share [0,1] scale
):
    """
    MATLAB-style: pupil defocus phase + FFT to focal plane.
    - Auto-detect bright tweezers at z=0, crop 3-plane panels around them,
      and reuse the same x-crop for the Xâ€“Z slice.
    - Xâ€“Z uses a y-band around the ROI center (max or sum aggregation).
    - If normalize=True, both figures share the same [0,1] scale.

    Returns:
        fig_3planes, fig_xz, best_z_index, {
            "z_min": I_at_-z, "z0": I_at_0, "z_max": I_at_+z,
            "roi": (r1, r2, c1, c2), "x_um": x_um, "y_um": y_um
        }
    """

    # ---------- Build full-size phase on the SLM pupil ----------
    H, W = A_in.shape
    h, w = phase.shape
    psi_full = np.zeros((H, W), dtype=np.float32)
    y0 = (H - h) // 2
    x0 = (W - w) // 2
    psi_full[y0:y0 + h, x0:x0 + w] = phase

    # Complex pupil field
    A_pupil = (A_in * np.exp(1j * psi_full)).astype(np.complex128)

    # Pupil coordinates (Âµm)
    yy = (np.arange(H) - H / 2) * pixel_size_slm_um
    xx = (np.arange(W) - W / 2) * pixel_size_slm_um
    X, Y = np.meshgrid(xx, yy)
    R2 = X**2 + Y**2

    # Optics
    lam = float(wavelength_um)
    f   = float(focal_length_um)
    k   = 2.0 * np.pi / lam

    # Focal-plane sampling (per pixel, Âµm)
    # (Use W along x for Î”x_focal; using H instead changes only aspect)
    px_focal_um = (lam * f) / (W * pixel_size_slm_um)
    x_um = (np.arange(W) - W / 2) * px_focal_um
    y_um = (np.arange(H) - H / 2) * px_focal_um

    # Defocus propagator â†’ focal plane intensity
    def plane_intensity_at(z_um: float) -> np.ndarray:
        denom = 2.0 * f * (f - z_um)
        if abs(denom) < 1e-12:        # guard against zâ‰ˆf
            denom = 2.0 * f * f
        phase_defocus = -k * z_um * R2 / denom
        A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
        I = (A_out.conj() * A_out).real
        return I

    # ------------------- Auto-ROI from z = 0 plane -------------------
    I0_raw = plane_intensity_at(0.0)

    def _auto_roi_from_plane(I0: np.ndarray) -> tuple[int, int, int, int]:
        H0, W0 = I0.shape
        thr = np.percentile(I0, roi_thresh_p)
        mask = I0 > thr
        if mask.sum() < roi_min_area_px:
            # Fallback: small centered box
            hw = max(32, int(120.0 / max(px_focal_um, 1e-12)))
            r_mid, c_mid = H0 // 2, W0 // 2
            return (max(0, r_mid - hw), min(H0, r_mid + hw),
                    max(0, c_mid - hw), min(W0, c_mid + hw))
        ys, xs = np.where(mask)
        r1_raw, r2_raw = ys.min(), ys.max() + 1
        c1_raw, c2_raw = xs.min(), xs.max() + 1
        pad_px = int(np.round(roi_pad_um / max(px_focal_um, 1e-12)))
        r1 = max(0, r1_raw - pad_px); r2 = min(H0, r2_raw + pad_px)
        c1 = max(0, c1_raw - pad_px); c2 = min(W0, c2_raw + pad_px)
        # Ensure non-degenerate
        if r2 - r1 < 4: r1, r2 = max(0, H0 // 2 - 4), min(H0, H0 // 2 + 4)
        if c2 - c1 < 4: c1, c2 = max(0, W0 // 2 - 4), min(W0, W0 // 2 + 4)
        return r1, r2, c1, c2

    if auto_roi:
        r1, r2, c1, c2 = _auto_roi_from_plane(I0_raw)
    else:
        r1, r2, c1, c2 = 0, H, 0, W

    # ------------------------ Z grids ------------------------
    # For the 3 side-by-side planes, honor user z_range_um
    z_panel = np.linspace(-abs(z_range_um), abs(z_range_um), max(3, int(n_z_steps)))
    # For Xâ€“Z, expand range by +z_extra_for_xz_um (your earlier request)
    z_xz = np.linspace(-(abs(z_range_um) + abs(z_extra_for_xz_um)),
                        +(abs(z_range_um) + abs(z_extra_for_xz_um)),
                        max(11, int(n_z_steps)))

    # ----------------- Build Xâ€“Z (use cropped columns) -----------------
    ncols = c2 - c1
    x_um_crop = x_um[c1:c2]
    xz_raw = np.zeros((len(z_xz), ncols), dtype=np.float64)
    plane_max = np.zeros(len(z_xz), dtype=np.float64)

    # y-band center at ROI center; clamp to image
    y_center = (r1 + r2) // 2
    yL = max(0, y_center - y_band_halfwidth_px)
    yR = min(H, y_center + y_band_halfwidth_px + 1)

    global_max = 0.0
    for zi, z_um_val in enumerate(z_xz):
        I = plane_intensity_at(float(z_um_val))
        band = I[yL:yR, c1:c2]
        if agg_mode.lower() == "sum":
            prof = band.sum(axis=0)
        else:
            prof = band.max(axis=0)
        xz_raw[zi, :] = prof
        val = prof.max()
        plane_max[zi] = val
        if val > global_max:
            global_max = val

    best_z_index = int(np.argmax(plane_max))

    # Consistent scaling for both figures
    if normalize and global_max > 0:
        xz_disp = xz_raw / global_max
        vmin, vmax = 0.0, 1.0
    else:
        xz_disp = xz_raw
        vmin, vmax = 0.0, max(global_max, 1.0)

    # ----------------- 3-plane panels (using same scale) -----------------
    z_triplet = [z_panel[0], 0.0, z_panel[-1]]
    I_triplet_cropped = []
    for z_um_val in z_triplet:
        I = plane_intensity_at(float(z_um_val))
        I_view = I[r1:r2, c1:c2]
        if normalize and global_max > 0:
            I_view = I_view / global_max
        I_triplet_cropped.append(I_view)

    # ---------------------------- Plotting ----------------------------
    # Figure A: three planes
    fig_3, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, z_um_val, I_view in zip(axes, z_triplet, I_triplet_cropped):
        extent = [x_um[c1], x_um[c2 - 1], y_um[r1], y_um[r2 - 1]]
        im = ax.imshow(I_view, origin='lower', extent=extent,
                       vmin=vmin, vmax=vmax, cmap='hot')
        ax.set_title(f'z = {z_um_val:+.1f} Âµm', fontsize=14)
        ax.set_xlabel('x [Âµm]')
    axes[0].set_ylabel('y [Âµm]')
    for ax in axes:
        plt.colorbar(im, ax=ax, fraction=0.046,
                     label='Normalized intensity' if normalize else 'Intensity (a.u.)')
    fig_3.suptitle('Focal Plane Intensity at Three Z Positions',
                   fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Figure B: Xâ€“Z slice (cropped x, same scale)
    fig_xz, ax2 = plt.subplots(figsize=(14, 5))
    extent_xz = [x_um_crop[0], x_um_crop[-1], z_xz.min(), z_xz.max()]
    im2 = ax2.imshow(xz_disp, aspect='auto', origin='lower',
                     extent=extent_xz, vmin=vmin, vmax=vmax, cmap='hot')
    ax2.set_xlabel('x [Âµm]', fontsize=13)
    ax2.set_ylabel('z [Âµm]', fontsize=13)
    ax2.set_title(f'Xâ€“Z slice (picked y-band center={y_center}, Â±{y_band_halfwidth_px}px, {agg_mode})',
                  fontsize=14, fontweight='bold')
    # z=0 focal plane
    ax2.axhline(y=0.0, color='cyan', linestyle='--', alpha=0.8, linewidth=2,
                label='z=0 focal plane')
    # Tilt line: z(x) = tan(theta)*x (through x=0)
    if abs(tilt_deg) > 1e-6:
        tilt_rad = np.deg2rad(tilt_deg)
        x_line = np.array([x_um_crop[0], x_um_crop[-1]])
        z_line = np.tan(tilt_rad) * (x_line - 0.0)
        ax2.plot(x_line, z_line, color='magenta', linewidth=2.5,
                 label=f'Tilted plane ({tilt_deg:.1f}Â°)')
    ax2.legend(loc='upper right')
    plt.colorbar(im2, ax=ax2,
                 label='Normalized intensity' if normalize else 'Intensity (a.u.)')
    plt.tight_layout()

    # Return the *uncropped* z=Â±range and 0 images too (for debugging)
    I_minus = plane_intensity_at(float(z_panel[0]))
    I_zero  = I0_raw
    I_plus  = plane_intensity_at(float(z_panel[-1]))

    # Cleanup big arrays
    del psi_full, A_pupil, X, Y, R2
    gc.collect()

    return (fig_3, fig_xz, best_z_index,
            {"z_min": I_minus, "z0": I_zero, "z_max": I_plus,
             "roi": (r1, r2, c1, c2), "x_um": x_um, "y_um": y_um})


# --------- blaze helpers (BMP only) ---------
def add_blazed_grating(phase_mask: np.ndarray, fx: float, fy: float) -> np.ndarray:
    H, W = phase_mask.shape
    xx = np.arange(W, dtype=np.float32)
    yy = np.arange(H, dtype=np.float32)
    gr = (2*np.pi*fx*xx)[None, :] + (2*np.pi*fy*yy)[:, None]
    return np.mod(phase_mask + (gr % (2*np.pi)), 2*np.pi).astype(np.float32, copy=False)

def save_phase_bmp(phase: np.ndarray, out_path: Path) -> None:
    img8 = (np.clip(phase/(2*np.pi), 0, 1) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img8, mode="L").save(out_path)

# ----------------- main -----------------
def main():
    print("="*60)
    print("MEMORY-SAFE MULTI-PLANE GS + BMP (blaze only in BMP)")
    print("="*60)

    OUT_DIR = make_out_dir()           # <<< recompute with current TILT_ANGLE_X & FOCAL_LENGTH_UM
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Setting up SLM ---")
    slm = SLMTweezers(yaml_path=YAML_PATH, redSLM=REDSLM, scal=SCAL)
    slm.init_fields(waist_um=WAIST_UM)
    slm.set_target_grid(n_horiz=N_HORIZ, n_vert=N_VERT, spacing_um=SPACING_UM, odd_tw=1, box1=1)
    slm.set_optics(wavelength_um=WAVELENGTH_UM, focal_length_um=FOCAL_LENGTH_UM)

    print(f"\n--- Assigning planes {TILT_ANGLE_X} ---")
    slm.assign_planes_from_tilt(tilt_x_deg=TILT_ANGLE_X, n_planes=N_Z_PLANES)

    print("\n--- Running multi-plane GS v3 ---")
    # slm.run_gs(iterations=ITERATIONS, Gg=GG, useFilter=False, tol=TOL)
    slm.run_gs_multiplane_v3(iterations=ITERATIONS, Gg=GG, verbose=True, tol=TOL)

    print("\n--- Saving results ---")
    label = f"_{N_HORIZ}x{N_VERT}_tol{TOL:.1e}_v3"
    bundle = slm.save_pickle(out_dir=str(OUT_DIR), label=label)
    print(f"Ã¢Å“â€œ Saved: {bundle.file}")

    # snapshots for viz
    phase_copy = slm.phase_mask.copy()
    A_in_copy = slm.A_in.copy()
    pixel_um = slm.params.pixel_um

    # ---------- BMP with blazed grating ----------
    fx = 1.0 / 7.0
    fy = 0.0
    phase_blazed = add_blazed_grating(phase_copy, fx=fx, fy=fy)

    pkl_path = Path(bundle.file)
    stem = pkl_path.with_suffix("").name

    # unified suffix for PNG & BMP, includes tilt
    suffix = (
        f"_f{FOCAL_LENGTH_UM:.0f}um"
        f"_sp{SPACING_UM:.1f}um"
        f"_planes{N_Z_PLANES}"
        f"_tol{TOL:.1e}"
        f"_tilt{TILT_ANGLE_X:.0f}deg"
    )

    out_bmp = pkl_path.parent / f"{stem}{suffix}_blazepd7.bmp"
    out2_bmp = pkl_path.parent / f"../../../phase_masks_tilt/{stem}{suffix}_blazepd7.bmp"
    save_phase_bmp(phase_blazed, out_bmp)
    save_phase_bmp(phase_blazed, out2_bmp)
    print(f"Ã¢Å“â€œ Phase+blaze BMP: {out_bmp}, {out2_bmp}")

    # ---------- MATLAB-STYLE visualization (UNBLAZED) ----------
    print("\n--- Visualizing results (MATLAB-STYLE: THIN LENS + FFT, UNBLAZED) ---")
    # Use a reasonable z-range for visualization
    # The tilt creates only ~Â±11 Âµm defocus, so scan Â±100 Âµm to see beam evolution
    z_range = 100.0  # Âµm - enough to see diffraction spreading
    
    fig_3planes, fig_xz, best_idx, I_dict = visualize_xz_matlab_style(
        phase=phase_copy,
        A_in=A_in_copy,
        wavelength_um=WAVELENGTH_UM,
        pixel_size_slm_um=pixel_um,
        focal_length_um=FOCAL_LENGTH_UM,
        z_range_um=max(slm._z_per_spot),
        n_z_steps=N_Z_STEPS,
        tilt=TILT_ANGLE_X,
        row_mode=ROW_MODE
    )

    del slm
    gc.collect()
    print("Ã¢Å“â€œ SLM memory cleared")

    # ---------- Save figures ----------
    base_name = OUT_DIR / f"{TILT_ANGLE_X:.0f}deg_tilt{suffix}"
    fig_3planes.savefig(f"{base_name}_3planes.png", dpi=DPI)
    fig_xz.savefig(f"{base_name}_xz.png", dpi=DPI)
    print(f"Ã¢Å“â€œ Saved figures to {base_name}_3planes.png and {base_name}_xz.png")

    plt.close('all')
    del fig_3planes, fig_xz, phase_copy, A_in_copy, phase_blazed
    gc.collect()

    print("\n" + "="*60)
    print("DONE!")
    print("Ã¢Å“â€œ All memory cleaned up")
    print("="*60)

if __name__ == "__main__":
    main()
    plt.close('all')
    gc.collect()
    print("\nÃ°Å¸Â§Â¹ Final cleanup complete")