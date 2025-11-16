#!/usr/bin/env python3
"""
ULTRA zoom to actually see the individual tweezer spots.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft2, ifft2, fftshift, ifftshift

BMP_PATH = "/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/_20x20_tol1.0e-05_v3_20251102_122628_f200000um_sp30.0um_planes5_tol1.0e-05_tilt10deg_bbox1.0_blazepd7.bmp"
OUTPUT_DIR = "/Users/nadinemeister/Library/CloudStorage/Box-Box/EndresLab/z_Second Experiment/Code/SLM simulation/nadine/DMD_SLM/python_SLM_3d/slm_output/251102-015626_f200000um_sp30.0um_planes5_tilt_10.0_tw_20/"

# Parameters
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

# ULTRA TIGHT zoom - just show a few spots
X_HALFWIDTH = 100.0  # Only ±100 µm (tiny FOV!)
Y_HALFWIDTH = 100.0

print("="*70)
print("ULTRA ZOOM - SHOWING INDIVIDUAL SPOTS")
print("="*70)

# Load and remove blaze
img = Image.open(BMP_PATH)
phase_blazed = (np.array(img).astype(np.float32) / 255.0) * 2 * np.pi
H, W = phase_blazed.shape
xx = np.arange(W, dtype=np.float32)
yy = np.arange(H, dtype=np.float32)
blaze = (2*np.pi*BLAZE_FX*xx)[None, :] + (2*np.pi*BLAZE_FY*yy)[:, None]
phase_mask = np.mod(phase_blazed - blaze, 2*np.pi)

# Input field
waist_px = WAIST_UM / PIXEL_UM
y_slm, x_slm = np.ogrid[0:SLM_Y_PIXELS, 0:SLM_X_PIXELS]
cy, cx = SLM_Y_PIXELS / 2.0, SLM_X_PIXELS / 2.0
A_in = np.exp(-(((x_slm - cx)**2 + (y_slm - cy)**2) / (waist_px**2))).astype(np.float32)

# Compute focal plane
def compute_focal(phase, A_in, z_um):
    A_pupil = (A_in * np.exp(1j * phase)).astype(np.complex128)
    H, W = A_pupil.shape
    yy = (np.arange(H) - H / 2) * PIXEL_UM
    xx = (np.arange(W) - W / 2) * PIXEL_UM
    X, Y = np.meshgrid(xx, yy)
    R2 = X**2 + Y**2
    k = 2.0 * np.pi / WAVELENGTH_UM
    f = FOCAL_LENGTH_UM
    denom = 2.0 * f * (f - z_um)
    if abs(denom) < 1e-6:
        denom = 2.0 * f * f
    phase_defocus = -k * z_um * R2 / denom
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    return (A_out.conj() * A_out).real

x_extent = (N_HORIZ - 1) * SPACING_UM
z_range = x_extent * np.tan(np.deg2rad(TILT_DEG))
z_positions = [0.0]  # Just show focal plane

px_focal_um = (WAVELENGTH_UM * FOCAL_LENGTH_UM) / (SLM_X_PIXELS * PIXEL_UM)
x_um = (np.arange(SLM_X_PIXELS) - SLM_X_PIXELS / 2) * px_focal_um
y_um = (np.arange(SLM_Y_PIXELS) - SLM_Y_PIXELS / 2) * px_focal_um

print(f"Computing focal plane (z=0)...")
I = compute_focal(phase_mask, A_in, 0.0)

# Ultra tight crop
c1 = np.argmin(np.abs(x_um - (-X_HALFWIDTH)))
c2 = np.argmin(np.abs(x_um - (X_HALFWIDTH)))
r1 = np.argmin(np.abs(y_um - (-Y_HALFWIDTH)))
r2 = np.argmin(np.abs(y_um - (Y_HALFWIDTH)))

print(f"Ultra zoom window: ±{X_HALFWIDTH} µm")
print(f"  Pixels: {r2-r1} × {c2-c1}")
print(f"  Should show ~{2*X_HALFWIDTH/SPACING_UM:.0f} spots in each direction")

I_crop = I[r1:r2, c1:c2]
I_norm = I_crop / I_crop.max()

# Plot with high interpolation for visibility
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

extent = [x_um[c1], x_um[c2-1], y_um[r1], y_um[r2-1]]
im = ax.imshow(I_norm, origin='lower', extent=extent,
               vmin=0.0, vmax=1.0, cmap='hot', interpolation='bilinear')

ax.set_title(f'Ultra Zoom: Focal Plane at z=0 µm (±{X_HALFWIDTH:.0f} µm FOV)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('x [µm]', fontsize=12)
ax.set_ylabel('y [µm]', fontsize=12)

# Add grid lines at expected tweezer positions
n_spots = int(2*X_HALFWIDTH / SPACING_UM) + 1
x_grid = np.linspace(-X_HALFWIDTH, X_HALFWIDTH, n_spots)
y_grid = np.linspace(-Y_HALFWIDTH, Y_HALFWIDTH, n_spots)

for xg in x_grid:
    ax.axvline(xg, color='cyan', alpha=0.3, linewidth=0.5, linestyle='--')
for yg in y_grid:
    ax.axhline(yg, color='cyan', alpha=0.3, linewidth=0.5, linestyle='--')

plt.colorbar(im, ax=ax, label='Normalized intensity')

output_path = f"{OUTPUT_DIR}/focal_plane_ULTRA_ZOOM_z0.png"
fig.savefig(output_path, dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: {output_path}")

# Also create a version with spot size annotation
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))

im2 = ax2.imshow(I_norm, origin='lower', extent=extent,
                 vmin=0.0, vmax=1.0, cmap='hot', interpolation='nearest')

ax2.set_title(f'Individual Spots (±{X_HALFWIDTH:.0f} µm, nearest-neighbor interpolation)',
             fontsize=14, fontweight='bold')
ax2.set_xlabel('x [µm]', fontsize=12)
ax2.set_ylabel('y [µm]', fontsize=12)

# Annotate spot size
spot_size_um = WAVELENGTH_UM * FOCAL_LENGTH_UM / (2 * WAIST_UM)
ax2.text(0.05, 0.95, f'Expected spot size: ~{spot_size_um:.1f} µm\n'
                      f'Focal plane pixel: {px_focal_um:.2f} µm\n'
                      f'Spot width: ~{spot_size_um/px_focal_um:.1f} pixels',
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
         color='white')

plt.colorbar(im2, ax=ax2, label='Normalized intensity')

output_path2 = f"{OUTPUT_DIR}/focal_plane_ULTRA_ZOOM_annotated.png"
fig2.savefig(output_path2, dpi=DPI, bbox_inches='tight')
print(f"✓ Saved: {output_path2}")

plt.close('all')

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"The spots are ~{spot_size_um:.1f} µm wide = ~{spot_size_um/px_focal_um:.1f} pixels")
print(f"This is why they look tiny - each spot is only 1-2 pixels!")