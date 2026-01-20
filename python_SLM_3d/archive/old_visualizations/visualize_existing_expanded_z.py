"""
Load existing pickle and visualize intensity vs z over an EXPANDED range.
This will show the full tweezer intensity profile without re-running WGS.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
import pickle
from pathlib import Path

print("="*70)
print("EXPANDED Z-SCAN OF EXISTING WGS RESULT")
print("="*70)

# Find the most recent pickle file
output_dir = Path("slm_output_paraxial")
pickle_files = list(output_dir.glob("**/*.pkl"))
if not pickle_files:
    print("ERROR: No pickle files found in slm_output_paraxial/")
    exit(1)

# Sort by modification time and get the most recent
pickle_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
pickle_path = pickle_files[0]
print(f"\nLoading: {pickle_path}")

# Load the pickle
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

# Handle different pickle formats
if hasattr(data, 'phase_mask'):
    # It's a ResultBundle
    phase_mask = data.phase_mask
    A_in = data.A_in
    # Extract params from filename
    parent_name = pickle_path.parent.name
    import re
    f_match = re.search(r'f(\d+)um', parent_name)
    focal_length_um = float(f_match.group(1)) if f_match else 200000.0
    wavelength_um = 0.689  # Default
    pixel_um = 3.74  # From slm_parameters.yml
    # Tweezers info not in bundle, need to load from somewhere else
    print("  WARNING: This is an old ResultBundle format without tweezer positions")
    print("  Will use phase mask but cannot compare to expected positions")
    target_xy_um = None
    z_per_spot = None
else:
    # It's a dict
    phase_mask = data['phase_mask']
    A_in = data['A_in']
    pixel_um = data.get('pixel_um', 3.74)
    wavelength_um = data.get('wavelength_um', 0.689)
    focal_length_um = data.get('focal_length_um', 200000.0)
    target_xy_um = data.get('target_xy_um')
    z_per_spot = data.get('z_per_spot')

print(f"  Loaded result from: {pickle_path.parent.name}")
print(f"  Focal length: {focal_length_um/1000:.0f} mm")
print(f"  Wavelength: {wavelength_um} µm")

if z_per_spot is not None:
    print(f"  Number of tweezers: {len(z_per_spot)}")
    print(f"  Z-range: [{z_per_spot.min():.1f}, {z_per_spot.max():.1f}] µm")
else:
    # Extract from phase mask - just use center pixel for single tweezer test
    print("  No tweezer position data - will measure center only")
    target_xy_um = np.array([[0.0, 0.0]])
    z_per_spot = np.array([0.0])

# Build full phase field
H, W = A_in.shape
h, w = phase_mask.shape
psi_full = np.zeros((H, W), dtype=np.float32)
y0 = (H - h) // 2
x0 = (W - w) // 2
psi_full[y0:y0 + h, x0:x0 + w] = phase_mask

# Pupil field
A_pupil = A_in * np.exp(1j * psi_full)

# Pupil coordinates
yy = (np.arange(H) - H / 2) * pixel_um
xx = (np.arange(W) - W / 2) * pixel_um
X, Y = np.meshgrid(xx, yy)
R2 = X**2 + Y**2

# Focal plane coordinates
k = 2 * np.pi / wavelength_um
f = focal_length_um
px_focal_um = (wavelength_um * f) / (W * pixel_um)
x_focal = (np.arange(W) - W / 2) * px_focal_um
y_focal = (np.arange(H) - H / 2) * px_focal_um

# EXPANDED Z range
z_min = -500  # Much larger range!
z_max = +500
z_array = np.linspace(z_min, z_max, 251)

print(f"\nScanning z from {z_min} to {z_max} µm ({len(z_array)} points)...")

# For each tweezer, measure intensity vs z
x_tweezers = target_xy_um[:, 0]
focal_positions = []
intensity_curves = []

# Limit to first 5 tweezers for speed (or adjust as needed)
n_tweezers_to_plot = min(5, len(x_tweezers))

for tw_idx in range(n_tweezers_to_plot):
    x_tw = x_tweezers[tw_idx]

    # Find focal plane pixel closest to this x position
    x_px_idx = np.argmin(np.abs(x_focal - x_tw))

    # Scan through z
    intensities = []

    for i, z_um in enumerate(z_array):
        if i % 50 == 0:
            print(f"  Tweezer {tw_idx}: {i}/{len(z_array)} z-positions scanned...", end='\r')

        # Paraxial propagation
        phase_defocus = +(k / (2 * f * f)) * z_um * R2
        A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
        I = np.abs(A_out)**2

        # Measure intensity at tweezer position (center row, specific x)
        intensity_at_tw = I[H//2, x_px_idx]
        intensities.append(intensity_at_tw)

    print(f"  Tweezer {tw_idx}: Done!{' '*30}")

    intensities = np.array(intensities)
    intensity_curves.append(intensities)

    # Find peak
    peak_idx = np.argmax(intensities)
    z_peak = z_array[peak_idx]
    focal_positions.append(z_peak)

    print(f"    Peak at z={z_peak:+.1f} µm (expected {z_per_spot[tw_idx]:+.1f} µm)")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Intensity vs z for each tweezer
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, n_tweezers_to_plot))

for i in range(n_tweezers_to_plot):
    x_tw = x_tweezers[i]
    z_exp = z_per_spot[i]
    intensities = intensity_curves[i]

    norm_int = intensities / intensities.max()
    ax.plot(z_array, norm_int, color=colors[i], linewidth=2.5,
            label=f'x={x_tw:+.0f}µm (expect z={z_exp:+.1f}µm)')

    # Mark expected position
    ax.axvline(z_exp, color=colors[i], linestyle='--', alpha=0.6, linewidth=1.5)

ax.set_xlabel('z [µm]', fontsize=14, fontweight='bold')
ax.set_ylabel('Normalized Intensity', fontsize=14, fontweight='bold')
ax.set_title(f'Intensity vs z (EXPANDED range, f={focal_length_um/1000:.0f}mm)',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc='best')
ax.set_xlim(z_min, z_max)

# Right: Measured vs expected focal positions
ax = axes[1]
ax.scatter(x_tweezers[:n_tweezers_to_plot], focal_positions,
           s=120, c='blue', marker='o',
           label='Measured peak', zorder=3, edgecolors='black', linewidths=2)
ax.scatter(x_tweezers[:n_tweezers_to_plot], z_per_spot[:n_tweezers_to_plot],
           s=120, c='red', marker='x',
           label='Expected (from algorithm)', zorder=3, linewidths=3)

# Ideal tilt line (estimate from data)
if len(z_per_spot) > 1:
    x_line = np.array([x_tweezers.min(), x_tweezers.max()])
    z_line_min = z_per_spot[np.argmin(x_tweezers)]
    z_line_max = z_per_spot[np.argmax(x_tweezers)]
    z_line = np.interp(x_line, [x_tweezers.min(), x_tweezers.max()],
                       [z_line_min, z_line_max])
    ax.plot(x_line, z_line, 'k--', linewidth=2, alpha=0.7, label='Ideal tilt')

ax.set_xlabel('x position [µm]', fontsize=14, fontweight='bold')
ax.set_ylabel('z focal position [µm]', fontsize=14, fontweight='bold')
ax.set_title('Focal positions: Measured vs Expected', fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)

plt.tight_layout()
out_path = output_dir / "tweezer_focal_positions_EXPANDED.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved plot: {out_path}")

# Error analysis
errors = np.array(focal_positions) - z_per_spot[:n_tweezers_to_plot]
rms_error = np.sqrt(np.mean(errors**2))
max_error = np.max(np.abs(errors))
z_range = z_per_spot.max() - z_per_spot.min()

print(f"\n--- Error Analysis ---")
print(f"  RMS error: {rms_error:.2f} µm")
print(f"  Max error: {max_error:.2f} µm")
print(f"  Z-range: {z_range:.2f} µm")
print(f"  Error as % of z-range: {100*rms_error/z_range:.1f}%")

print("="*70)
plt.show()
