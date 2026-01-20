#!/usr/bin/env python3
"""
Remove blazed grating from BMP and regenerate zoomed 3-plane plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path

# ============================================================================
# PARAMETERS (extracted from filename)
# ============================================================================

BMP_PATH = "~/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/_20x20_tol1.0e-05_v3_20251102_122628_f200000um_sp30.0um_planes5_tol1.0e-05_tilt10deg_bbox1.0_blazepd7.bmp"
OUTPUT_DIR = "~/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/"

# System parameters
N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30.0
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689  # 689 nm
TILT_DEG = 10.0
N_PLANES = 5

# Blaze grating parameters (from code: fx=1/7, fy=0)
BLAZE_FX = 1.0 / 7.0
BLAZE_FY = 0.0

# SLM parameters (from YAML)
SLM_X_PIXELS = 4000
SLM_Y_PIXELS = 2464
PIXEL_UM = 3.74

# Gaussian beam waist (from code)
WAIST_UM = 9 / 2 * 1e3  # 4500 µm

# Visualization zoom (adjust these!)
ZOOM_FACTOR = 4  # How much to zoom in (1 = full FOV, 4 = 1/4 width)
DPI = 300

# ============================================================================
# STEP 1: Load BMP and remove blaze
# ============================================================================

print("=" * 70)
print("STEP 1: Loading BMP and removing blazed grating")
print("=" * 70)

# Load BMP (8-bit grayscale, 0-255 maps to 0-2π)
# Expand ~ to full user path
BMP_PATH = Path(BMP_PATH).expanduser()
OUTPUT_DIR = Path(OUTPUT_DIR).expanduser()

print("Expanded BMP_PATH:", BMP_PATH)
print("Expanded OUTPUT_DIR:", OUTPUT_DIR)

img = Image.open(BMP_PATH)
phase_blazed_uint8 = np.array(img)
print(f"Loaded BMP: shape {phase_blazed_uint8.shape}, range [{phase_blazed_uint8.min()}, {phase_blazed_uint8.max()}]")

# Convert to phase (0-2π)
phase_blazed = (phase_blazed_uint8.astype(np.float32) / 255.0) * 2 * np.pi

# Remove blaze grating
H, W = phase_blazed.shape
xx = np.arange(W, dtype=np.float32)
yy = np.arange(H, dtype=np.float32)
blaze_grating = (2 * np.pi * BLAZE_FX * xx)[None, :] + (2 * np.pi * BLAZE_FY * yy)[:, None]

# Subtract blaze and wrap to [0, 2π]
phase_mask = np.mod(phase_blazed - blaze_grating, 2 * np.pi)

print(f"✓ Removed blaze grating (fx={BLAZE_FX:.4f}, fy={BLAZE_FY:.4f})")
print(f"  Phase range: [{phase_mask.min():.3f}, {phase_mask.max():.3f}] rad")

# ============================================================================
# STEP 2: Build input field (Gaussian beam)
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Building input Gaussian field")
print("=" * 70)

# Gaussian waist in pixels
waist_px = WAIST_UM / PIXEL_UM

# Build Gaussian on SLM
y_slm, x_slm = np.ogrid[0:SLM_Y_PIXELS, 0:SLM_X_PIXELS]
cy = SLM_Y_PIXELS / 2.0
cx = SLM_X_PIXELS / 2.0

A_in = np.exp(-(((x_slm - cx) ** 2 + (y_slm - cy) ** 2) / (waist_px ** 2))).astype(np.float32)

print(f"✓ Gaussian beam: waist = {WAIST_UM:.1f} µm ({waist_px:.1f} px)")
print(f"  Input field shape: {A_in.shape}")

# ============================================================================
# STEP 3: Compute focal plane intensity at three z-positions
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Computing focal plane intensity")
print("=" * 70)


def compute_focal_intensity(phase, A_in, z_um, wavelength_um, pixel_um, focal_length_um):
    """Compute intensity at focal plane with defocus z_um."""
    # Complex pupil field
    A_pupil = (A_in * np.exp(1j * phase)).astype(np.complex128)

    # Pupil coordinates (µm)
    H, W = A_pupil.shape
    yy = (np.arange(H) - H / 2) * pixel_um
    xx = (np.arange(W) - W / 2) * pixel_um
    X, Y = np.meshgrid(xx, yy)
    R2 = X ** 2 + Y ** 2

    # Defocus phase
    k = 2.0 * np.pi / wavelength_um
    f = float(focal_length_um)
    denom = 2.0 * f * (f - z_um)
    if abs(denom) < 1e-6:
        denom = 2.0 * f * f
    phase_defocus = -k * z_um * R2 / denom

    # Propagate to focal plane
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I = (A_out.conj() * A_out).real

    return I


# Calculate z-range from tilt
x_extent = (N_HORIZ - 1) * SPACING_UM
z_range = x_extent * np.tan(np.deg2rad(TILT_DEG))

print(f"Tilt: {TILT_DEG}°")
print(f"X-extent: {x_extent:.1f} µm")
print(f"Expected z-range: ±{z_range / 2:.1f} µm")

# Three z-positions
z_positions = [-z_range / 2, 0.0, +z_range / 2]

# Focal plane coordinates
px_focal_um = (WAVELENGTH_UM * FOCAL_LENGTH_UM) / (SLM_X_PIXELS * PIXEL_UM)
x_um = (np.arange(SLM_X_PIXELS) - SLM_X_PIXELS / 2) * px_focal_um
y_um = (np.arange(SLM_Y_PIXELS) - SLM_Y_PIXELS / 2) * px_focal_um

print(f"Focal plane pixel size: {px_focal_um:.3f} µm/px")

# Compute intensity at 3 z-positions
I_triplet = []
for z_val in z_positions:
    print(f"  Computing z = {z_val:+.1f} µm...")
    I = compute_focal_intensity(phase_mask, A_in, z_val, WAVELENGTH_UM, PIXEL_UM, FOCAL_LENGTH_UM)
    I_triplet.append(I)
    print(f"    Max intensity: {I.max():.2e}")

# ============================================================================
# STEP 4: Auto-detect ROI and zoom
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Detecting tweezers and setting zoom")
print("=" * 70)

# Find bright spots in central plane
I_central = I_triplet[1]
threshold = np.percentile(I_central, 99.5)
mask = I_central > threshold

if mask.sum() > 20:
    ys, xs = np.where(mask)
    r1_raw, r2_raw = ys.min(), ys.max() + 1
    c1_raw, c2_raw = xs.min(), xs.max() + 1

    # Add padding based on zoom factor
    pad_r = int((r2_raw - r1_raw) * (ZOOM_FACTOR - 1) / 2)
    pad_c = int((c2_raw - c1_raw) * (ZOOM_FACTOR - 1) / 2)

    r1 = max(0, r1_raw - pad_r)
    r2 = min(SLM_Y_PIXELS, r2_raw + pad_r)
    c1 = max(0, c1_raw - pad_c)
    c2 = min(SLM_X_PIXELS, c2_raw + pad_c)

    print(f"✓ Detected tweezer array:")
    print(f"    Raw bounds: rows {r1_raw}:{r2_raw}, cols {c1_raw}:{c2_raw}")
    print(f"    With zoom padding: rows {r1}:{r2}, cols {c1}:{c2}")
    print(f"    FOV: x = [{x_um[c1]:.1f}, {x_um[c2 - 1]:.1f}] µm")
    print(f"    FOV: y = [{y_um[r1]:.1f}, {y_um[r2 - 1]:.1f}] µm")
else:
    print("⚠ Could not detect tweezers, using center crop")
    r1, r2 = SLM_Y_PIXELS // 2 - 500, SLM_Y_PIXELS // 2 + 500
    c1, c2 = SLM_X_PIXELS // 2 - 500, SLM_X_PIXELS // 2 + 500

# ============================================================================
# STEP 5: Plot zoomed 3-plane figure
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Creating zoomed 3-plane plot")
print("=" * 70)

# Normalize across all three planes
I_all = np.concatenate([I.ravel() for I in I_triplet])
global_max = np.max(I_all)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, z_val, I_full in zip(axes, z_positions, I_triplet):
    # Crop and normalize
    I_crop = I_full[r1:r2, c1:c2] / global_max

    extent = [x_um[c1], x_um[c2 - 1], y_um[r1], y_um[r2 - 1]]
    im = ax.imshow(I_crop, origin='lower', extent=extent,
                   vmin=0.0, vmax=1.0, cmap='hot')
    ax.set_title(f'z = {z_val:+.1f} µm', fontsize=14, fontweight='bold')
    ax.set_xlabel('x [µm]', fontsize=12)

    plt.colorbar(im, ax=ax, fraction=0.046, label='Normalized intensity')

axes[0].set_ylabel('y [µm]', fontsize=12)

fig.suptitle(f'Focal Plane Intensity at Three Z Positions (Zoom {ZOOM_FACTOR}x)',
             fontsize=16, fontweight='bold')
plt.tight_layout()

# Save
output_path = f"{OUTPUT_DIR}/3planes_zoomed_{ZOOM_FACTOR}x_tilt{TILT_DEG:.0f}deg.png"
fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# ============================================================================
# STEP 6: Also save unblazed phase mask
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Saving unblazed phase mask")
print("=" * 70)

# Convert back to uint8
phase_uint8 = (np.clip(phase_mask / (2 * np.pi), 0, 1) * 255.0 + 0.5).astype(np.uint8)
output_bmp = f"{OUTPUT_DIR}/phase_mask_no_blaze.bmp"
Image.fromarray(phase_uint8, mode="L").save(output_bmp)
print(f"✓ Saved unblazed phase mask: {output_bmp}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
print(f"✓ Removed blaze grating from phase mask")
print(f"✓ Generated zoomed 3-plane plot ({ZOOM_FACTOR}x zoom)")
print(f"✓ Output files in: {OUTPUT_DIR}")

plt.close('all')