#!/usr/bin/env python3
"""
Create multiple zoom levels to find the tweezers.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

BMP_PATH = "/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/_20x20_tol1.0e-05_v3_20251102_122628_f200000um_sp30.0um_planes5_tol1.0e-05_tilt10deg_bbox1.0_blazepd7.bmp"
OUTPUT_DIR = "/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/"

N_HORIZ = 20
N_VERT = 20
SPACING_UM = 30.0
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689
TILT_DEG = 10.0
SLM_X_PIXELS = 4000
SLM_Y_PIXELS = 2464
PIXEL_UM = 3.74
BLAZE_FX = 1.0 / 7.0
BLAZE_FY = 0.0
WAIST_UM = 9 / 2 * 1e3
DPI = 300

# Different zoom levels to try
ZOOM_CONFIGS = [
    {"halfwidth": 2000.0, "name": "wide"},
    {"halfwidth": 1000.0, "name": "medium"},
    {"halfwidth": 500.0, "name": "close"},
    {"halfwidth": 300.0, "name": "tight"},
]

print("=" * 70)
print("CREATING MULTIPLE ZOOM LEVELS")
print("=" * 70)

# Load and remove blaze
img = Image.open(BMP_PATH)
phase_blazed_uint8 = np.array(img)
phase_blazed = (phase_blazed_uint8.astype(np.float32) / 255.0) * 2 * np.pi
H, W = phase_blazed.shape
xx = np.arange(W, dtype=np.float32)
yy = np.arange(H, dtype=np.float32)
blaze_grating = (2 * np.pi * BLAZE_FX * xx)[None, :] + (2 * np.pi * BLAZE_FY * yy)[:, None]
phase_mask = np.mod(phase_blazed - blaze_grating, 2 * np.pi)

# Build input
waist_px = WAIST_UM / PIXEL_UM
y_slm, x_slm = np.ogrid[0:SLM_Y_PIXELS, 0:SLM_X_PIXELS]
cy, cx = SLM_Y_PIXELS / 2.0, SLM_X_PIXELS / 2.0
A_in = np.exp(-(((x_slm - cx) ** 2 + (y_slm - cy) ** 2) / (waist_px ** 2))).astype(np.float32)


# Focal plane
def compute_focal_intensity(phase, A_in, z_um, wavelength_um, pixel_um, focal_length_um):
    A_pupil = (A_in * np.exp(1j * phase)).astype(np.complex128)
    H, W = A_pupil.shape
    yy = (np.arange(H) - H / 2) * pixel_um
    xx = (np.arange(W) - W / 2) * pixel_um
    X, Y = np.meshgrid(xx, yy)
    R2 = X ** 2 + Y ** 2
    k = 2.0 * np.pi / wavelength_um
    f = float(focal_length_um)
    denom = 2.0 * f * (f - z_um)
    if abs(denom) < 1e-6:
        denom = 2.0 * f * f
    phase_defocus = -k * z_um * R2 / denom
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I = (A_out.conj() * A_out).real
    return I


x_extent = (N_HORIZ - 1) * SPACING_UM
z_range = x_extent * np.tan(np.deg2rad(TILT_DEG))
z_positions = [-z_range / 2, 0.0, +z_range / 2]

px_focal_um = (WAVELENGTH_UM * FOCAL_LENGTH_UM) / (SLM_X_PIXELS * PIXEL_UM)
x_um = (np.arange(SLM_X_PIXELS) - SLM_X_PIXELS / 2) * px_focal_um
y_um = (np.arange(SLM_Y_PIXELS) - SLM_Y_PIXELS / 2) * px_focal_um

print(f"Computing intensity at 3 z-positions...")
I_triplet = []
for z_val in z_positions:
    I = compute_focal_intensity(phase_mask, A_in, z_val, WAVELENGTH_UM, PIXEL_UM, FOCAL_LENGTH_UM)
    I_triplet.append(I)

I_all = np.concatenate([I.ravel() for I in I_triplet])
global_max = np.max(I_all)

# Create plots at different zoom levels
for config in ZOOM_CONFIGS:
    hw = config["halfwidth"]
    name = config["name"]

    c1 = np.argmin(np.abs(x_um - (-hw)))
    c2 = np.argmin(np.abs(x_um - (hw)))
    r1 = np.argmin(np.abs(y_um - (-hw)))
    r2 = np.argmin(np.abs(y_um - (hw)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, z_val, I_full in zip(axes, z_positions, I_triplet):
        I_crop = I_full[r1:r2, c1:c2] / global_max
        extent = [x_um[c1], x_um[c2 - 1], y_um[r1], y_um[r2 - 1]]
        im = ax.imshow(I_crop, origin='lower', extent=extent,
                       vmin=0.0, vmax=1.0, cmap='hot')
        ax.set_title(f'z = {z_val:+.1f} µm', fontsize=14, fontweight='bold')
        ax.set_xlabel('x [µm]', fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, label='Norm. intensity')

    axes[0].set_ylabel('y [µm]', fontsize=12)
    fig.suptitle(f'Focal Plane Intensity ({name} zoom, ±{hw:.0f} µm)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = f"{OUTPUT_DIR}/3planes_{name}_zoom_tilt{TILT_DEG:.0f}deg.png"
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"✓ Saved {name} zoom: {output_path}")
    plt.close('all')

print("\n" + "=" * 70)
print("DONE! Created 4 zoom levels")
print("=" * 70)