"""
Visualize tweezer array at multiple z-planes.
Shows what the focal plane looks like at different defocus positions.
"""

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
from pathlib import Path
import yaml
import sys

# Get pickle path from command line or use default
if len(sys.argv) > 1:
    PKL_PATH = sys.argv[1]
else:
    # Default to most recent file
    script_dir = Path(__file__).parent
    PKL_PATH = str(script_dir / "slm_output_paraxial/adaptive_test_matlab_spacing/_matlab_spacing_100x100_tilt-13deg_5planes_50iter_scal4_waist8.1_20260119_231914.pkl")

print("="*70)
print("MULTI-PLANE VISUALIZATION")
print("="*70)
print(f"\nLoading: {PKL_PATH}")

# Load the pickle
with open(PKL_PATH, 'rb') as f:
    bundle = pickle.load(f)

phase_mask = bundle.phase_mask
A_in = bundle.A_in

# Load SLM parameters
script_dir = Path(__file__).parent
yaml_path = script_dir / "../slm_parameters.yml"
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)
pixel_um = config['slm_parameters']['pixel_um']

print(f"  Phase mask shape: {phase_mask.shape}")
print(f"  Input field shape: {A_in.shape}")
print(f"  Pixel size: {pixel_um} um")

# Parameters
FOCAL_LENGTH_UM = 200000.0  # 200 mm
WAVELENGTH_UM = 0.689
SCAL = 4  # padding factor

# Embed phase mask into the full padded array
H, W = A_in.shape
h, w = phase_mask.shape

psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0+h, x0:x0+w] = phase_mask

# Create the modulated field
A_mod = A_in * np.exp(1j * psi_full)

# Compute physical parameters
pixel_um_padded = pixel_um * SCAL
k = 2 * np.pi / WAVELENGTH_UM
f = FOCAL_LENGTH_UM

# Pupil coordinates for defocus
yy = (np.arange(H) - H/2) * pixel_um_padded
xx = (np.arange(W) - W/2) * pixel_um_padded
X_pupil, Y_pupil = np.meshgrid(xx, yy)
R2_pupil = X_pupil**2 + Y_pupil**2

# Focal plane coordinates
px_focal_um = (WAVELENGTH_UM * f) / (W * pixel_um_padded)
py_focal_um = (WAVELENGTH_UM * f) / (H * pixel_um_padded)
x_focal_um = (np.arange(W) - W/2) * px_focal_um
y_focal_um = (np.arange(H) - H/2) * py_focal_um

print(f"  Focal plane pixel size: {px_focal_um:.3f} x {py_focal_um:.3f} um")

# Z-planes to visualize
# For a -13 degree tilt over ~6000 um, z-range is about +/- 700 um
Z_PLANES = [-800, -400, -200, 0, 200, 400, 800]  # um
N_PLANES = len(Z_PLANES)

print(f"\nComputing focal planes at z = {Z_PLANES} um...")

# Compute intensity at each z-plane
I_planes = []
for zi, z_val in enumerate(Z_PLANES):
    print(f"  Computing z = {z_val:+.0f} um...", end='', flush=True)
    
    # Defocus phase
    phase_defocus = (k / (2.0 * f * f)) * z_val * R2_pupil
    
    # Propagate to focal plane with defocus
    A_focal = fftshift(fft2(ifftshift(A_mod * np.exp(1j * phase_defocus))))
    I_focal = np.abs(A_focal)**2
    I_planes.append(I_focal)
    
    print(" Done")

# Find the region containing tweezers (from z=0 plane)
I_z0 = I_planes[Z_PLANES.index(0)] if 0 in Z_PLANES else I_planes[len(Z_PLANES)//2]
threshold = 0.01 * I_z0.max()

# Find bounds of tweezer region
rows_with_signal = np.any(I_z0 > threshold, axis=1)
cols_with_signal = np.any(I_z0 > threshold, axis=0)

if np.any(rows_with_signal) and np.any(cols_with_signal):
    row_min = np.argmax(rows_with_signal)
    row_max = H - np.argmax(rows_with_signal[::-1])
    col_min = np.argmax(cols_with_signal)
    col_max = W - np.argmax(cols_with_signal[::-1])
    
    # Add margin
    margin = 50
    row_min = max(0, row_min - margin)
    row_max = min(H, row_max + margin)
    col_min = max(0, col_min - margin)
    col_max = min(W, col_max + margin)
else:
    # Default to center region
    row_min, row_max = H//4, 3*H//4
    col_min, col_max = W//4, 3*W//4

print(f"  Cropping to region: rows [{row_min}:{row_max}], cols [{col_min}:{col_max}]")

# Create figure
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Global max for consistent color scaling
I_max_global = max(I.max() for I in I_planes)

for i, (z_val, I_focal) in enumerate(zip(Z_PLANES, I_planes)):
    if i >= len(axes):
        break
        
    ax = axes[i]
    
    # Crop to tweezer region
    I_crop = I_focal[row_min:row_max, col_min:col_max]
    I_norm = I_crop / I_max_global
    
    # Display
    extent = [x_focal_um[col_min], x_focal_um[col_max-1],
              y_focal_um[row_min], y_focal_um[row_max-1]]
    
    im = ax.imshow(I_norm, cmap='hot', vmin=0, vmax=1,
                   extent=extent, origin='lower', aspect='equal')
    
    ax.set_title(f'z = {z_val:+.0f} um', fontsize=14, fontweight='bold')
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    
    # Add intensity stats
    peak_intensity = I_crop.max() / I_max_global
    ax.text(0.02, 0.98, f'Peak: {peak_intensity:.2f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Hide unused subplot
if len(Z_PLANES) < len(axes):
    for i in range(len(Z_PLANES), len(axes)):
        axes[i].axis('off')

# Add colorbar
fig.colorbar(im, ax=axes, label='I / I_max', shrink=0.8)

plt.suptitle(f'Tweezer Array at Different Z-Planes\n{Path(PKL_PATH).stem}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save
out_path = Path(PKL_PATH).with_suffix('.png').name.replace('.png', '_z_planes.png')
out_full = Path(PKL_PATH).parent / out_path
plt.savefig(out_full, dpi=150, bbox_inches='tight')
print(f"\n[OK] Saved: {out_full}")

# Also create a figure showing the INPUT beam profile
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Input beam intensity
ax1 = axes2[0]
I_in = np.abs(A_in)**2
I_in_norm = I_in / I_in.max()
im1 = ax1.imshow(I_in_norm, cmap='viridis', vmin=0, vmax=1)
ax1.set_title('Input Gaussian Beam (A_in)', fontsize=14, fontweight='bold')
ax1.set_xlabel('x [pixels]')
ax1.set_ylabel('y [pixels]')
plt.colorbar(im1, ax=ax1, label='I / I_max')

# Cross-section through center
ax2 = axes2[1]
center_row = H // 2
profile = I_in_norm[center_row, :]
x_pixels = np.arange(W)
ax2.plot(x_pixels, profile, 'b-', linewidth=2)
ax2.axhline(y=np.exp(-2), color='r', linestyle='--', label='1/e^2 level')
ax2.axhline(y=0.01, color='orange', linestyle='--', label='1% level')
ax2.set_xlabel('x [pixels]', fontsize=12)
ax2.set_ylabel('Normalized Intensity', fontsize=12)
ax2.set_title('Input Beam Profile (horizontal cross-section)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.set_ylim([0, 1.1])
ax2.grid(True, alpha=0.3)

# Mark the SLM active region
slm_start = (W - w) // 2
slm_end = slm_start + w
ax2.axvline(x=slm_start, color='green', linestyle='-', alpha=0.5, label='SLM edge')
ax2.axvline(x=slm_end, color='green', linestyle='-', alpha=0.5)

# Find where intensity drops to 1%
above_1pct = profile > 0.01
if np.any(above_1pct):
    first_1pct = np.argmax(above_1pct)
    last_1pct = W - np.argmax(above_1pct[::-1])
    usable_width = last_1pct - first_1pct
    print(f"\n  Input beam >1% intensity: pixels {first_1pct} to {last_1pct} (width: {usable_width} pixels)")
    print(f"  SLM active region: pixels {slm_start} to {slm_end} (width: {w} pixels)")
    
    # Check overlap
    overlap_start = max(first_1pct, slm_start)
    overlap_end = min(last_1pct, slm_end)
    if overlap_end > overlap_start:
        overlap_pct = (overlap_end - overlap_start) / w * 100
        print(f"  Usable overlap: {overlap_pct:.1f}% of SLM")
    else:
        print(f"  WARNING: Input beam doesn't cover SLM!")

plt.tight_layout()
out_path2 = Path(PKL_PATH).with_suffix('.png').name.replace('.png', '_input_beam.png')
out_full2 = Path(PKL_PATH).parent / out_path2
plt.savefig(out_full2, dpi=150, bbox_inches='tight')
print(f"[OK] Saved: {out_full2}")

print("\n" + "="*70)
print("DONE!")
print("="*70)
