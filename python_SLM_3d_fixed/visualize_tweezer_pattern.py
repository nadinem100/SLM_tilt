"""
Visualize the tweezer pattern from a saved pickle file.
Shows the focal plane intensity (what you should see on camera).
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - just save files
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
from pathlib import Path
import sys

# Get pickle path from command line or use default
if len(sys.argv) > 1:
    PKL_PATH = sys.argv[1]
else:
    # Default to most recent 100x100 file
    PKL_PATH = str(Path(__file__).parent.parent / 
                   "slm_output_paraxial/adaptive_test_matlab_spacing/_matlab_spacing_100x100_tilt0deg_1planes_10iter_scal4_waist8.1_20260119_205742.pkl")

print("="*70)
print("TWEEZER PATTERN VISUALIZATION")
print("="*70)
print(f"\nLoading: {PKL_PATH}")

# Load the pickle
with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

phase_mask = bundle.phase_mask
A_in = bundle.A_in

print(f"  Phase mask shape: {phase_mask.shape}")
print(f"  Input field shape: {A_in.shape}")

# Embed phase mask into the full padded array
H, W = A_in.shape
h, w = phase_mask.shape

psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0+h, x0:x0+w] = phase_mask

# Create the modulated field
A_mod = A_in * np.exp(1j * psi_full)

# Fourier transform to get focal plane
A_focal = fftshift(fft2(ifftshift(A_mod)))
I_focal = np.abs(A_focal)**2

# Normalize
I_focal_norm = I_focal / I_focal.max()

# Count tweezers by finding peaks
from scipy.ndimage import maximum_filter
I_maxfilter = maximum_filter(I_focal, size=10)
threshold = 0.05  # 5% of max intensity
peaks = (I_focal == I_maxfilter) & (I_focal > threshold * I_focal.max())
n_peaks = np.sum(peaks)

print(f"\n  Detected approximately {n_peaks} tweezer peaks above {threshold*100:.0f}% threshold")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. Phase mask
ax1 = axes[0]
im1 = ax1.imshow(phase_mask, cmap='twilight', vmin=0, vmax=2*np.pi)
ax1.set_title(f'Phase Mask ({phase_mask.shape[1]}x{phase_mask.shape[0]} pixels)')
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
plt.colorbar(im1, ax=ax1, label='Phase (rad)')

# 2. Full focal plane (log scale to see structure)
ax2 = axes[1]
I_log = np.log10(I_focal_norm + 1e-6)
im2 = ax2.imshow(I_log, cmap='hot', vmin=-4, vmax=0)
ax2.set_title(f'Focal Plane Intensity (log scale)\n~{n_peaks} peaks detected')
ax2.set_xlabel('x (pixels)')
ax2.set_ylabel('y (pixels)')
plt.colorbar(im2, ax=ax2, label='log10(I/I_max)')

# 3. Zoomed center region (linear scale)
ax3 = axes[2]
# Zoom to center region where tweezers should be
center_y, center_x = H//2, W//2
# Estimate zoom region based on detected peaks
if n_peaks > 0:
    peak_coords = np.where(peaks)
    y_min, y_max = peak_coords[0].min(), peak_coords[0].max()
    x_min, x_max = peak_coords[1].min(), peak_coords[1].max()
    # Add 10% margin
    margin_y = int(0.1 * (y_max - y_min))
    margin_x = int(0.1 * (x_max - x_min))
    y_min = max(0, y_min - margin_y)
    y_max = min(H, y_max + margin_y)
    x_min = max(0, x_min - margin_x)
    x_max = min(W, x_max + margin_x)
else:
    # Default zoom
    zoom = min(H, W) // 4
    y_min, y_max = center_y - zoom, center_y + zoom
    x_min, x_max = center_x - zoom, center_x + zoom

I_zoom = I_focal_norm[y_min:y_max, x_min:x_max]
im3 = ax3.imshow(I_zoom, cmap='hot', vmin=0, vmax=1)
ax3.set_title(f'Zoomed Tweezer Region (linear scale)\nRegion: [{x_min}:{x_max}, {y_min}:{y_max}]')
ax3.set_xlabel('x (pixels)')
ax3.set_ylabel('y (pixels)')
plt.colorbar(im3, ax=ax3, label='I/I_max')

plt.tight_layout()

# Save figure
out_path = Path(PKL_PATH).with_suffix('.png').name.replace('.png', '_visualization.png')
out_full = Path(PKL_PATH).parent / out_path
plt.savefig(out_full, dpi=150, bbox_inches='tight')
print(f"\n  Saved visualization: {out_full}")

# Also save a high-res version of just the focal plane
fig2, ax = plt.subplots(figsize=(12, 12))
ax.imshow(I_focal_norm[y_min:y_max, x_min:x_max], cmap='hot', vmin=0, vmax=1)
ax.set_title(f'Tweezer Array (~{n_peaks} spots)')
ax.axis('off')
out_full2 = Path(PKL_PATH).parent / out_path.replace('_visualization.png', '_tweezers_only.png')
plt.savefig(out_full2, dpi=200, bbox_inches='tight')
print(f"  Saved tweezer-only image: {out_full2}")

# Don't show interactively - just save the files
# plt.show()

print("\n" + "="*70)
print("DONE!")
print("="*70)
