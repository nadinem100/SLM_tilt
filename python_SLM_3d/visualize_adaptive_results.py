"""
Visualize the adaptive GS results from saved pickle file.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
from pathlib import Path

# Path to the adaptive GS result
PKL_PATH = "slm_output_paraxial/adaptive_test/_adaptive_5x5_tilt10deg_20251108_102106.pkl"

# Optics parameters
FOCAL_LENGTH_UM = 200000.0
WAVELENGTH_UM = 0.689

print("="*70)
print("VISUALIZING ADAPTIVE GS RESULTS")
print("="*70)

# Load pickle
print(f"\nLoading: {PKL_PATH}")
with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

phase_mask = bundle.phase_mask
A_in = bundle.A_in

print(f"  Phase mask shape: {phase_mask.shape}")
print(f"  Input field shape: {A_in.shape}")

# Compute focal plane intensity at z=0
print("\nComputing focal plane intensity...")

H, W = A_in.shape
h, w = phase_mask.shape

# Build full-size phase
psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0 + h, x0:x0 + w] = phase_mask

# Complex pupil field
A_pupil = (A_in * np.exp(1j * psi_full)).astype(np.complex128)

# Propagate to focal plane (z=0)
A_focal = fftshift(fft2(ifftshift(A_pupil)))
I_focal = np.abs(A_focal)**2

# Compute focal plane pixel size
# Assuming SCAL=4 from test script
pixel_um = 9.74 * 4  # From slm_parameters.yml
lam = WAVELENGTH_UM
f = FOCAL_LENGTH_UM
px_focal_um = (lam * f) / (W * pixel_um)

x_focal_um = (np.arange(W) - W/2) * px_focal_um
y_focal_um = (np.arange(H) - H/2) * px_focal_um

print(f"  Focal plane pixel size: {px_focal_um:.2f} µm/pixel")
print(f"  FOV: {x_focal_um.max() - x_focal_um.min():.0f} × {y_focal_um.max() - y_focal_um.min():.0f} µm²")

# Auto-detect ROI
thr = np.percentile(I_focal, 99.5)
mask = I_focal > thr

if mask.sum() > 20:
    ys, xs = np.where(mask)
    r1, r2 = ys.min(), ys.max() + 1
    c1, c2 = xs.min(), xs.max() + 1

    # Add padding
    pad_px = int(200 / px_focal_um)
    r1 = max(0, r1 - pad_px)
    r2 = min(H, r2 + pad_px)
    c1 = max(0, c1 - pad_px)
    c2 = min(W, c2 + pad_px)
else:
    # Fallback: center crop
    hw = 100
    r1, r2 = H//2 - hw, H//2 + hw
    c1, c2 = W//2 - hw, W//2 + hw

I_crop = I_focal[r1:r2, c1:c2]
I_crop_norm = I_crop / I_crop.max()

x_crop = x_focal_um[c1:c2]
y_crop = y_focal_um[r1:r2]

print(f"  ROI: {c2-c1} × {r2-r1} pixels")
print(f"  ROI extent: {x_crop.max()-x_crop.min():.0f} × {y_crop.max()-y_crop.min():.0f} µm²")

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: Focal plane intensity
ax = axes[0]
extent = [x_crop.min(), x_crop.max(), y_crop.min(), y_crop.max()]
im = ax.imshow(I_crop_norm, origin='lower', extent=extent, cmap='hot',
               vmin=0, vmax=1, interpolation='bilinear')
ax.set_xlabel('x [µm]', fontsize=14, fontweight='bold')
ax.set_ylabel('y [µm]', fontsize=14, fontweight='bold')
ax.set_title('Focal Plane Intensity (z=0, normalized)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Normalized Intensity')

# Right: Phase mask
ax = axes[1]
im = ax.imshow(phase_mask, cmap='twilight', vmin=0, vmax=2*np.pi)
ax.set_xlabel('x [pixels]', fontsize=14, fontweight='bold')
ax.set_ylabel('y [pixels]', fontsize=14, fontweight='bold')
ax.set_title('Phase Mask (0 to 2π)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, label='Phase [rad]')
cbar.set_ticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

plt.tight_layout()

out_path = Path(PKL_PATH).parent / "adaptive_visualization.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved: {out_path}")

plt.show()

print("\n" + "="*70)
print("DONE!")
print("="*70)
