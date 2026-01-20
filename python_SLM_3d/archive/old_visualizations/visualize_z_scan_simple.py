"""
Simple: Load most recent pickle and scan intensity vs z over a large range.
Extract parameters from filename.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
import pickle
from pathlib import Path
import re

print("="*70)
print("EXPANDED Z-SCAN (from pickle, parameters from filename)")
print("="*70)

# Find most recent pickle
output_dir = Path("slm_output_paraxial")
pickle_files = list(output_dir.glob("**/*.pkl"))
if not pickle_files:
    print("ERROR: No pickles found")
    exit(1)

pickle_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
pickle_path = pickle_files[0]

# Extract parameters from filename
parent_name = pickle_path.parent.name
print(f"\nLoading: {parent_name}")

# Extract f, spacing, tilt, n_tw from filename like: 251106-142433_f200000um_sp30.0um_planes5_tilt_30_tw_20
f_match = re.search(r'f(\d+)um', parent_name)
sp_match = re.search(r'sp([\d.]+)um', parent_name)
tilt_match = re.search(r'tilt_(\d+)', parent_name)
tw_match = re.search(r'tw_(\d+)', parent_name)

focal_length_um = float(f_match.group(1)) if f_match else 200000.0
spacing_um = float(sp_match.group(1)) if sp_match else 30.0
tilt_deg = float(tilt_match.group(1)) if tilt_match else 5.0
n_tw = int(tw_match.group(1)) if tw_match else 20

print(f"  f = {focal_length_um/1000:.0f} mm")
print(f"  spacing = {spacing_um} µm")
print(f"  tilt = {tilt_deg}°")
print(f"  n_tweezers = {n_tw}")

# Calculate expected z-range from geometry
# For a square grid, extent is approximately sqrt(n_tw) * spacing
n_side = int(np.sqrt(n_tw))
extent_um = (n_side - 1) * spacing_um
z_expected_range = extent_um * np.tan(np.deg2rad(tilt_deg))

print(f"  Expected z-range: ±{z_expected_range/2:.1f} µm")

# Load pickle
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

phase_mask = data.phase_mask if hasattr(data, 'phase_mask') else None
A_in = data.A_in if hasattr(data, 'A_in') else None

if phase_mask is None or A_in is None:
    print("ERROR: Could not load phase_mask or A_in from pickle")
    exit(1)

# Setup
wavelength_um = 0.689
pixel_um = 3.74
k = 2 * np.pi / wavelength_um
f = focal_length_um

H, W = A_in.shape
h, w = phase_mask.shape

# Build full phase field
psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0 + h, x0:x0 + w] = phase_mask

# Pupil
A_pupil = A_in * np.exp(1j * psi_full)

# Pupil coordinates
yy = (np.arange(H) - H / 2) * pixel_um
xx = (np.arange(W) - W / 2) * pixel_um
X, Y = np.meshgrid(xx, yy)
R2 = X**2 + Y**2

# Focal plane coordinates
px_focal_um = (wavelength_um * f) / (W * pixel_um)
x_focal = (np.arange(W) - W / 2) * px_focal_um

# Z-scan range: center around expected z-range, but go much wider
z_scan_range = max(500, 10 * z_expected_range)  # At least ±500 µm
z_array = np.linspace(-z_scan_range, +z_scan_range, 501)

print(f"\nScanning z from {-z_scan_range:.0f} to {+z_scan_range:.0f} µm...")
print(f"Measuring center pixel only...")

# Scan through z, measure intensity at center
intensities = []

for i, z_um in enumerate(z_array):
    if i % 100 == 0:
        print(f"  {i}/{len(z_array)}...", end='\r')

    # Paraxial propagation
    phase_defocus = +(k / (2 * f * f)) * z_um * R2
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I = np.abs(A_out)**2

    # Measure at center
    intensity_center = I[H//2, W//2]
    intensities.append(intensity_center)

print(f"  Done!{' '*30}")

intensities = np.array(intensities)

# Find peak
peak_idx = np.argmax(intensities)
z_peak = z_array[peak_idx]

print(f"\n✓ Peak intensity at z = {z_peak:+.1f} µm")
print(f"  (Expected tweezers to span ±{z_expected_range/2:.1f} µm)")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

# Normalize intensity
norm_int = intensities / intensities.max()

ax.plot(z_array, norm_int, color='blue', linewidth=2.5, label='Center pixel intensity')
ax.axvline(z_peak, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label=f'Peak at z={z_peak:+.0f}µm')

# Mark expected z-range
ax.axvspan(-z_expected_range/2, +z_expected_range/2, alpha=0.2, color='green',
           label=f'Expected tweezer z-range (±{z_expected_range/2:.0f}µm)')

ax.set_xlabel('z [µm]', fontsize=15, fontweight='bold')
ax.set_ylabel('Normalized Intensity', fontsize=15, fontweight='bold')
ax.set_title(f'Intensity vs z (f={f/1000:.0f}mm, tilt={tilt_deg}°, {n_tw} tweezers)',
             fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12, loc='best')
ax.set_xlim(z_array.min(), z_array.max())

plt.tight_layout()
out_path = output_dir / "intensity_vs_z_EXPANDED.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved: {out_path}")

print("="*70)
