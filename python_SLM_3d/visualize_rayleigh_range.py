"""
Visualize the Rayleigh range by creating a single focused spot (uniform phase)
and plotting the X-Z cross-section from the focal plane all the way back to the SLM.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift
import yaml

print("="*70)
print("RAYLEIGH RANGE VISUALIZATION")
print("="*70)

# Load SLM parameters
with open("../slm_parameters.yml", 'r') as f:
    config = yaml.safe_load(f)

pixel_um = config['slm_parameters']['pixel_um']
x_pixels = config['slm_parameters']['x_pixels']
y_pixels = config['slm_parameters']['y_pixels']

# Optical parameters
wavelength_um = 0.689
focal_length_um = 200000.0  # 200 mm
waist_um = 4500  # Gaussian beam waist at SLM

print(f"\nParameters:")
print(f"  Focal length: {focal_length_um/1000:.0f} mm")
print(f"  Wavelength: {wavelength_um} µm")
print(f"  SLM: {x_pixels}×{y_pixels} pixels, {pixel_um} µm/pixel")
print(f"  Input beam waist: {waist_um} µm")

# Reduce resolution for speed (use every 4th pixel)
scal = 4
H = y_pixels // scal
W = x_pixels // scal

print(f"  Reduced array: {H}×{W} pixels")

# Create Gaussian input beam
yy = (np.arange(H) - H/2) * pixel_um * scal
xx = (np.arange(W) - W/2) * pixel_um * scal
X, Y = np.meshgrid(xx, yy)
R2 = X**2 + Y**2

# Gaussian amplitude
A_in = np.exp(-R2 / (2 * waist_um**2))

# UNIFORM phase (no hologram) - just a single focused spot
phase_mask = np.zeros((H, W), dtype=np.float32)

# Pupil field
A_pupil = A_in * np.exp(1j * phase_mask)

print(f"\nPupil field:")
print(f"  Peak amplitude: {np.max(np.abs(A_pupil)):.3f}")
print(f"  Total power: {np.sum(np.abs(A_pupil)**2):.1e}")

# Focal plane coordinates
k = 2 * np.pi / wavelength_um
f = focal_length_um
px_focal_um = (wavelength_um * f) / (W * pixel_um * scal)
x_focal = (np.arange(W) - W/2) * px_focal_um

print(f"\nFocal plane:")
print(f"  Pixel size: {px_focal_um:.2f} µm/pixel")
print(f"  FOV: ±{x_focal.max():.0f} µm")

# Z range: from focal plane (z=0) all the way back to SLM (z=f)
z_min = 0
z_max = focal_length_um  # All the way to SLM plane
n_z = 501  # Number of z slices

z_array = np.linspace(z_min, z_max, n_z)

print(f"\nPropagating through z = {z_min/1000:.0f} to {z_max/1000:.0f} mm ({n_z} slices)...")

# Storage for X-Z intensity map (center row, y=0)
I_xz = np.zeros((n_z, W))

for i, z_um in enumerate(z_array):
    if i % 100 == 0:
        print(f"  {i}/{n_z}... (z={z_um/1000:.1f} mm)", end='\r')

    # Paraxial propagation to this z
    phase_defocus = +(k / (2 * f * f)) * z_um * R2
    A_out = fftshift(fft2(ifftshift(A_pupil * np.exp(1j * phase_defocus))))
    I = np.abs(A_out)**2

    # Store center row (y=0)
    I_xz[i, :] = I[H//2, :]

print(f"  Done!{' '*30}")

# Calculate Rayleigh range from theory
# For Gaussian beam: z_R = π w₀² / λ
# where w₀ is spot size at focus
spot_size_um = wavelength_um * f / (waist_um * np.pi)  # Rough estimate
z_R_theory = np.pi * spot_size_um**2 / wavelength_um

print(f"\nTheoretical Rayleigh range:")
print(f"  Spot size w₀ ≈ {spot_size_um:.1f} µm")
print(f"  Rayleigh range z_R ≈ {z_R_theory/1000:.1f} mm")

# Find where intensity drops to 50% of peak
I_center = I_xz[:, W//2]  # Intensity on axis
peak_I = I_center.max()
half_max_idx = np.where(I_center < 0.5 * peak_I)[0]
if len(half_max_idx) > 0:
    z_halfmax = z_array[half_max_idx[0]]
    print(f"  Measured half-max range: {z_halfmax/1000:.1f} mm")

# Plot - INTERACTIVE
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: X-Z intensity map with LINEAR scale (normalized)
ax = axes[0]
extent = [x_focal.min(), x_focal.max(), z_array.min()/1000, z_array.max()/1000]

# Normalize for better visualization
I_xz_normalized = I_xz / I_xz.max()
im = ax.imshow(I_xz_normalized, aspect='auto', origin='lower', extent=extent,
               cmap='hot', interpolation='bilinear', vmin=0, vmax=1)

ax.set_xlabel('x [µm]', fontsize=14, fontweight='bold')
ax.set_ylabel('z [mm]', fontsize=14, fontweight='bold')
title_text = (f'X-Z Intensity (LINEAR scale, normalized)\n'
              f'Use toolbar: 🔍 to zoom, 🏠 to reset, ⬅➡⬆⬇ to pan')
ax.set_title(title_text, fontsize=14, fontweight='bold')
ax.axhline(0, color='cyan', linestyle='--', linewidth=1, alpha=0.7, label='Focal plane (z=0)')
if z_R_theory < z_max:
    ax.axhline(z_R_theory/1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Rayleigh range ({z_R_theory/1000:.1f} mm)')
ax.set_xlim(-500, 500)  # Focus on central ±500 µm in x
ax.legend(fontsize=11, loc='upper right')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Normalized Intensity', fontsize=12)

# Right: On-axis intensity vs z
ax = axes[1]
ax.plot(z_array/1000, I_center / peak_I, linewidth=2.5, color='blue')
ax.axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='50% of peak')
ax.axvline(0, color='cyan', linestyle='--', linewidth=1.5, alpha=0.7, label='Focal plane')
if z_R_theory < z_max:
    ax.axvline(z_R_theory/1000, color='yellow', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Rayleigh range ({z_R_theory/1000:.1f} mm)')

ax.set_xlabel('z [mm]', fontsize=14, fontweight='bold')
ax.set_ylabel('Normalized Intensity (on axis)', fontsize=14, fontweight='bold')
ax.set_title('On-axis intensity from focal plane to SLM',
             fontsize=15, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_ylim(0, 1.1)

plt.tight_layout()
out_path = "slm_output_paraxial/rayleigh_range_visualization.png"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✓ Saved: {out_path}")
print(f"\nShowing interactive plot - you can zoom in with the toolbar!")
print("  Close the plot window to exit.")

# Show interactive plot
plt.show()

print("="*70)
